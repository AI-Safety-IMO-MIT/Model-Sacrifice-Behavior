"""Which attention heads and MLPs carry the strong-versus-weak favour?
Attribution patching on the favour pairs (experiment 4).

    python -m project.self_sacrifice.interp.favour_components sample
    python -m project.self_sacrifice.interp.favour_components attribute --layer 30 --window 10-30
    python -m project.self_sacrifice.interp.favour_components table

Needs the GPU, except `table`. Output: `run_result/self_sacrifice/interp/
favour_components/`. Pairs: `favour_patching.load_pairs(one_per_environment=
True)` in the 5 settings and the first 20 environments (100 pairs).

Components, at the positions after the favour line only ("suffix"): each
attention head's output (the input of `o_proj`, split by head: 32 heads per
layer) and each layer's MLP output. For a component c and a metric m:

    score(c) = (c in the strong run - c in the weak run) . dm/dc

the first-order estimate of how much m changes when c is copied from the
strong run into the weak run (summed over the suffix positions). It is
evaluated at both ends, with the gradient taken in the weak run and in the
strong run, and the two are averaged.

metric d   the projection, averaged over the suffix, of the residual after
           layer `--layer` onto the strong-minus-weak direction of
           `favour_steering` (unit length). Only layers below `--layer` can
           change it. No replies are involved.
metric bb  log p(R_give) - log p(R_keep), where R_give is the strong prompt's
           reply that gives and R_keep the weak prompt's reply that keeps,
           both written with the reply blocked from the favour line
           (`favour_attention`, check A) so that they speak of the favour
           only in general terms, and scored with the same block. Replies
           are cut at MAX_REPLY tokens. The prompt
           pass is unchanged by the block, so the components are the same as
           in the normal model.
metric bp  log p(R_give) - log p(R_keep) on the weak prompt, where both are
           the weak prompt's own replies: R_keep unpatched, R_give with the
           strong prompt's residual patched in at `--layer` (experiment 1).
           Both describe the weak favour, so they differ mainly in the
           decision. Run with `attribute --only bp` after the others.
metric b   log P(give) - log P(keep) for the prompt, where P(give) sums over
           every reply the model could write. Its gradient is estimated from
           the model's own replies (`sample`: 16 per prompt at temperature
           1): mean of grad log p(reply) over the replies that give, minus
           the same over the replies that keep. A prompt with fewer than 3
           replies of either kind gives no estimate at that end.

`sample` is deterministic: a fixed seed per batch, a fixed batch layout and
PyTorch's deterministic algorithms. Every sampled reply is saved, and the
gradients are computed from the saved replies.
"""

# pyright: reportMissingImports=false

import argparse
import json
import os
from pathlib import Path
from typing import Any

from project.self_sacrifice.games import favour_pairs
from project.self_sacrifice.interp import (
    common,
    favour_attention,
    favour_baseline,
    favour_patching,
    favour_steering,
)

__all__ = [
    "OUT",
    "SAMPLES",
    "sample",
    "attribute",
    "validate",
    "heads_where",
    "table",
    "main",
]

OUT = common.OUT / "favour_components"
SAMPLES = 16
MIN_EACH = 3
BATCH = 16
SEED = 20261122
# longest reply scored with gradients (memory: run on two GPUs with FAVOUR_DEVICE_MAP=auto)
MAX_REPLY = 1000


def _pairs() -> list[favour_patching.Pair]:
    return favour_patching.load_pairs(
        one_per_environment=True,
        environments=favour_steering.ENVIRONMENTS,
        settings=favour_steering.SETTINGS,
    )


def sample(
    samples: int = SAMPLES,
    max_new: int = favour_baseline.MAX_NEW,
    limit: int | None = None,
    name: str = "samples",
) -> Path:
    """`samples` replies at temperature 1 for the strong and the weak prompt
    of every pair. Resumable at batch boundaries."""
    import torch

    tokenizer, model = favour_baseline.load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    done = (
        {json.loads(line)["batch"] for line in path.read_text().splitlines() if line}
        if path.exists()
        else set()
    )
    jobs = [
        (pair, member, k)
        for pair in _pairs()[:limit]
        for member in ("big", "small")
        for k in range(samples)
    ]
    for index, start in enumerate(range(0, len(jobs), BATCH)):
        if index in done:
            continue
        batch = jobs[start : start + BATCH]
        trials = [getattr(pair, member) for pair, member, _ in batch]
        encoded = tokenizer(
            [common.prompt_text(tokenizer, t) for t in trials],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(0)
        torch.manual_seed(SEED + index)
        with torch.inference_mode():
            written = model.generate(
                **encoded,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=1.0,
                top_p=None,
                top_k=0,
            )
        length = encoded["input_ids"].shape[1]
        with path.open("a") as out:
            for (pair, member, k), tokens in zip(batch, written[:, length:]):
                reply = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                decided, receiver = favour_pairs.decision(reply)
                in_need = pair.big["environment"]["in_need"]
                out.write(
                    json.dumps(
                        {
                            "batch": index,
                            "pair": pair.key,
                            "member": member,
                            "k": k,
                            "reply": reply,
                            "decision": decided,
                            "gives": decided == "give" and receiver == in_need,
                        }
                    )
                    + "\n"
                )
            out.flush()
            os.fsync(out.fileno())
        print("batch", index + 1, "of", (len(jobs) + BATCH - 1) // BATCH, flush=True)
    return path


def attribute(
    layer: int,
    window: tuple[int, int],
    only: str | None = None,
    positions: bool = False,
) -> Path:
    """Scores of every head and MLP in layers window[0] .. window[1] - 1 for
    metrics d and b, for every pair. Saves `scores.pt` (one entry per pair)."""
    import torch

    tokenizer, model = favour_baseline.load_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stack = common.decoder_layers(model)
    lo, hi = window
    wanted = range(lo, hi)
    unit = torch.load(favour_steering.OUT / "directions.pt")["directions"]["suffix"][
        layer
    ].to(0)
    unit = unit / unit.norm()
    state: dict[str, Any] = {"mode": None}
    got: dict[str, Any] = {}

    def keep(name: str, tensor: Any) -> None:
        if state["mode"] == "cache":
            got[name] = tensor[0, -state["k"] :].detach().float()
        elif state["mode"] == "grad":
            tensor.retain_grad()
            got[name] = tensor

    def head_hook(i: int):
        def run(_module: Any, inputs: Any) -> None:
            keep(f"attn{i}", inputs[0])

        return run

    def mlp_hook(i: int):
        def run(_module: Any, _inputs: Any, output: Any) -> None:
            keep(f"mlp{i}", output)

        return run

    def resid_hook(_module: Any, _inputs: Any, output: Any) -> None:
        if state["mode"] == "grad":
            got["resid"] = favour_patching._hidden(output)

    def with_grad(_module: Any, _inputs: Any, output: Any) -> Any:
        return output.requires_grad_(True) if state["mode"] == "grad" else output

    handles = [model.get_input_embeddings().register_forward_hook(with_grad)]
    for i in wanted:
        handles.append(
            stack[i].self_attn.o_proj.register_forward_pre_hook(head_hook(i))
        )
        handles.append(stack[i].mlp.register_forward_hook(mlp_hook(i)))
    handles.append(stack[layer].register_forward_hook(resid_hook))

    samples: dict[tuple[str, str], list[dict]] = {}
    if (OUT / "samples.jsonl").exists():
        for line in (OUT / "samples.jsonl").read_text().splitlines():
            row = json.loads(line)
            samples.setdefault((row["pair"], row["member"]), []).append(row)
    blocked = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (favour_attention.OUT / "blocked.jsonl")
            .read_text()
            .splitlines()
        )
    }

    def favour_positions(trial: dict) -> list[int]:
        text = common.prompt_text(tokenizer, trial)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        starts = [a if b > a else -1 for a, b in encoded["offset_mapping"]]
        return favour_attention.favour_span(text, starts, trial["favour_line"])

    def run_prompt(ids: list[int]) -> None:
        model(input_ids=torch.tensor([ids], device=0), logits_to_keep=1)

    def component_grad(
        ids: list[int],
        k: int,
        reply_ids: list[int] | None,
        block: list[int] | None = None,
    ) -> dict:
        """Gradient at the suffix positions of the prompt of either the
        projection (reply_ids None) or log p(reply | prompt). With `block`
        (prompt positions), the prompt is read normally and the reply tokens
        may not attend to those positions."""
        got.clear()
        state["mode"] = "grad"
        if reply_ids and block is not None:
            n = len(ids)
            first = model(
                input_ids=torch.tensor([ids], device=0),
                position_ids=torch.arange(n, device=0)[None],
                cache_position=torch.arange(n, device=0),
                use_cache=True,
                logits_to_keep=1,
            )
            state["mode"] = "reply"  # components are only taken from the prompt
            mask = torch.ones(1, n + len(reply_ids) - 1, device=0, dtype=torch.long)
            mask[0, block] = 0
            rest = model(
                input_ids=torch.tensor([reply_ids[:-1]], device=0),
                attention_mask=mask,
                position_ids=torch.arange(n, n + len(reply_ids) - 1, device=0)[None],
                cache_position=torch.arange(n, n + len(reply_ids) - 1, device=0),
                past_key_values=first.past_key_values,
                use_cache=True,
            )
            logits = torch.cat([first.logits[0, -1:], rest.logits[0]])
            del first, rest
            target = torch.tensor(reply_ids, device=0)
            # cross-entropy in chunks keeps the float32 copy of the logits small
            metric = sum(
                -torch.nn.functional.cross_entropy(
                    logits[i : i + 128].float(), target[i : i + 128], reduction="sum"
                )
                for i in range(0, len(reply_ids), 128)
            )
        elif reply_ids:
            out = model(
                input_ids=torch.tensor([ids + reply_ids], device=0),
                logits_to_keep=len(reply_ids) + 1,
            )
            logits = out.logits[0, : len(reply_ids)].float()
            target = torch.tensor(reply_ids, device=0)[:, None]
            metric = torch.log_softmax(logits, -1).gather(1, target).sum()
        else:
            model(input_ids=torch.tensor([ids], device=0), logits_to_keep=1)
            resid = got["resid"][0, len(ids) - k : len(ids)].float()
            metric = (resid.mean(0) * unit.to(resid.device)).sum()
        metric.backward()
        state["mode"] = None
        end = len(ids)
        result = {
            name: tensor.grad[0, end - k : end].float()
            for name, tensor in got.items()
            if name != "resid" and tensor.grad is not None
        }
        got.clear()
        model.zero_grad(set_to_none=True)
        return result

    def scores(delta: dict, grad: dict) -> dict[str, Any]:
        heads = torch.zeros(hi - lo, 32)
        mlps = torch.zeros(hi - lo)
        for i in wanted:
            dz, gz = delta[f"attn{i}"], grad[f"attn{i}"]
            per = (dz * gz).sum(0)  # [heads * head_dim]
            heads[i - lo] = per.view(32, -1).sum(-1).cpu()
            mlps[i - lo] = (delta[f"mlp{i}"] * grad[f"mlp{i}"]).sum().cpu()
        out = {"heads": heads, "mlp": mlps}
        if positions:
            # the same, per position after the favour line (index k - 1 is
            # the last prompt token), as float16
            k = next(iter(delta.values())).shape[0]
            hp = torch.zeros(hi - lo, 32, k)
            mp = torch.zeros(hi - lo, k)
            for i in wanted:
                dz, gz = delta[f"attn{i}"], grad[f"attn{i}"]
                hp[i - lo] = (dz * gz).view(k, 32, -1).sum(-1).T.cpu()
                mp[i - lo] = (delta[f"mlp{i}"] * grad[f"mlp{i}"]).sum(-1).cpu()
            out["heads_pos"], out["mlp_pos"] = hp.half(), mp.half()
        return out

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"scores_L{layer}_{lo}-{hi}{'_pos' if positions else ''}.pt"
    results = torch.load(path) if path.exists() else []
    done = {r["pair"] for r in results}
    by_key = {r["pair"]: r for r in results}
    # metric bp: the weak prompt's own replies, unpatched (keeps) and with the
    # strong prompt's residual patched in at `layer` (gives) - the location run
    location: dict[str, dict[str, Any]] = {}
    loc_path = favour_patching.OUT / f"location_L{layer}.jsonl"
    if loc_path.exists():
        for line in loc_path.read_text().splitlines():
            r = json.loads(line)
            if r["from"] is None:
                location.setdefault(r["pair"], {})[r["label"]] = r
            elif r["label"] == "denoise" and r["group"] == "all":
                location.setdefault(r["pair"], {})["patched"] = r
    try:
        for pair in _pairs():
            if only == "bp":
                if pair.key not in by_key or "bp" in by_key[pair.key]:
                    continue
            elif pair.key in done:
                continue
            entry = favour_patching.prepare(tokenizer, pair)
            k = entry["suffix"]
            cache = {}
            with torch.no_grad():
                for member in ("big", "small"):
                    got.clear()
                    state.update(mode="cache", k=k)
                    run_prompt(entry["ids"][member])
                    cache[member] = dict(got)
                    state["mode"] = None
            delta = {
                name: cache["big"][name] - cache["small"][name] for name in cache["big"]
            }
            if only == "bp":
                row = by_key[pair.key]
                loc = location.get(pair.key, {})
                row["bp"] = None
                if (
                    loc.get("patched", {}).get("gives")
                    and loc.get("base_small", {}).get("gives") is False
                ):
                    ids_give = tokenizer(
                        loc["patched"]["reply"], add_special_tokens=False
                    )["input_ids"][:MAX_REPLY]
                    ids_keep = tokenizer(
                        loc["base_small"]["reply"], add_special_tokens=False
                    )["input_ids"][:MAX_REPLY]
                    g_give = component_grad(entry["ids"]["small"], k, ids_give)
                    g_keep = component_grad(entry["ids"]["small"], k, ids_keep)
                    row["bp"] = scores(
                        delta, {n: g_give[n] - g_keep[n] for n in g_give}
                    )
                torch.save(results, path)
                print(pair.key, "bp", row["bp"] is not None, flush=True)
                torch.cuda.empty_cache()
                continue
            row: dict[str, Any] = {"pair": pair.key, "suffix": k}
            # metric d at both ends
            ends = []
            for member in ("small", "big"):
                ends.append(
                    scores(delta, component_grad(entry["ids"][member], k, None))
                )
            row["d"] = {
                part: (ends[0][part] + ends[1][part]) / 2 for part in ("heads", "mlp")
            }
            # metric b at each end with enough of both decisions
            ends_b, counts = [], {}
            for member in ("small", "big") if not positions else ():
                rows = samples.get((pair.key, member), [])
                gives = [r for r in rows if r["gives"]]
                keeps = [
                    r for r in rows if not r["gives"] and r["decision"] is not None
                ]
                counts[member] = [len(gives), len(keeps)]
                if len(gives) < MIN_EACH or len(keeps) < MIN_EACH:
                    continue
                mean: dict[str, dict[str, Any]] = {}
                for label, group in (("give", gives), ("keep", keeps)):
                    total: dict[str, Any] = {}
                    for r in group:
                        reply_ids = tokenizer(r["reply"], add_special_tokens=False)[
                            "input_ids"
                        ][:MAX_REPLY]
                        g = component_grad(entry["ids"][member], k, reply_ids)
                        for name, value in g.items():
                            total[name] = total.get(name, 0) + value / len(group)
                    mean[label] = total
                g = {
                    name: mean["give"][name] - mean["keep"][name]
                    for name in mean["give"]
                }
                ends_b.append(scores(delta, g))
            row["counts"] = counts
            row["b"] = (
                {
                    part: sum(e[part] for e in ends_b) / len(ends_b)
                    for part in ("heads", "mlp")
                }
                if ends_b
                else None
            )
            # metric b, blocked replies: the strong prompt's reply that gives
            # and the weak prompt's reply that keeps, both written with the
            # reply blocked from the favour line (check A), scored with the
            # same block
            give_row = blocked.get(pair.big["id"])
            keep_row = blocked.get(pair.small["id"])
            row["bb"] = None
            if give_row and keep_row and give_row["gives"] and not keep_row["gives"]:
                ids_give = tokenizer(give_row["reply"], add_special_tokens=False)[
                    "input_ids"
                ][:MAX_REPLY]
                ids_keep = tokenizer(keep_row["reply"], add_special_tokens=False)[
                    "input_ids"
                ][:MAX_REPLY]
                ends_bb = []
                for member, trial in (("small", pair.small), ("big", pair.big)):
                    span = favour_positions(trial)
                    g_give = component_grad(entry["ids"][member], k, ids_give, span)
                    g_keep = component_grad(entry["ids"][member], k, ids_keep, span)
                    ends_bb.append(
                        scores(delta, {n: g_give[n] - g_keep[n] for n in g_give})
                    )
                row["bb"] = {
                    part: (ends_bb[0][part] + ends_bb[1][part]) / 2
                    for part in ("heads", "mlp")
                }
            if positions:
                loc = location.get(pair.key, {})
                row["bp"] = None
                if (
                    loc.get("patched", {}).get("gives")
                    and loc.get("base_small", {}).get("gives") is False
                ):
                    ids_give = tokenizer(
                        loc["patched"]["reply"], add_special_tokens=False
                    )["input_ids"][:MAX_REPLY]
                    ids_keep = tokenizer(
                        loc["base_small"]["reply"], add_special_tokens=False
                    )["input_ids"][:MAX_REPLY]
                    g_give = component_grad(entry["ids"]["small"], k, ids_give)
                    g_keep = component_grad(entry["ids"]["small"], k, ids_keep)
                    row["bp"] = scores(
                        delta, {n: g_give[n] - g_keep[n] for n in g_give}
                    )
                row["groups"] = entry["groups"]
                row["text"] = entry["text"]["small"]
            results.append(row)
            torch.save(results, path)
            print(pair.key, counts, flush=True)
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
    return path


def validate(
    layer: int,
    window: tuple[int, int],
    metric: str = "bb",
    sizes: tuple[int, ...] = (5, 20, 40),
    name: str = "validate",
    pairs: int = 40,
) -> Path:
    """Real patching of the top-ranked components (by `metric`, averaged
    over pairs): copy their outputs from the strong run into the weak run at
    the suffix positions, during the prompt pass, write the reply greedily,
    and record whether the model now gives. Top-k sets for each k in `sizes`,
    and the same number of components picked at random (control)."""
    import random

    import torch

    lo, hi = window
    results = torch.load(OUT / f"scores_L{layer}_{lo}-{hi}.pt")
    rows = [r[metric] for r in results if r.get(metric) is not None]
    heads = torch.stack([r["heads"] for r in rows]).mean(0)
    mlp = torch.stack([r["mlp"] for r in rows]).mean(0)
    ranked = [
        (heads[i, h].item(), ("attn", lo + i, h))
        for i in range(hi - lo)
        for h in range(32)
    ]
    ranked += [(mlp[i].item(), ("mlp", lo + i, None)) for i in range(hi - lo)]
    ranked.sort(key=lambda x: -x[0])
    order = [c for _, c in ranked]
    rng = random.Random(0)
    sets = {}
    for k in sizes:
        sets[f"top{k}"] = order[:k]
        sets[f"random{k}"] = rng.sample(order, k)

    tokenizer, model = favour_baseline.load_model()
    stack = common.decoder_layers(model)
    state: dict[str, Any] = {"mode": None}
    store: dict[str, Any] = {}

    def attn_pre(i: int):
        def run(_module: Any, inputs: Any) -> Any:
            x = inputs[0]
            if x.shape[1] == 1:
                return None
            if state["mode"] == "cache":
                store[f"attn{i}"] = x[0, -state["k"] :].detach().clone()
                return None
            if state["mode"] == "patch" and f"attn{i}" in state["src"]:
                x = x.clone()
                k_, src, cols = state["k"], state["src"][f"attn{i}"], state["cols"][i]
                x[0, -k_:, cols] = src[:, cols].to(x.dtype)
                return (x,) + tuple(inputs[1:])
            return None

        return run

    def mlp_post(i: int):
        def run(_module: Any, _inputs: Any, output: Any) -> Any:
            if output.shape[1] == 1:
                return output
            if state["mode"] == "cache":
                store[f"mlp{i}"] = output[0, -state["k"] :].detach().clone()
                return output
            if state["mode"] == "patch" and f"mlp{i}" in state["src"]:
                output = output.clone()
                output[0, -state["k"] :] = state["src"][f"mlp{i}"].to(output.dtype)
            return output

        return run

    handles = []
    for i in range(lo, hi):
        handles.append(stack[i].self_attn.o_proj.register_forward_pre_hook(attn_pre(i)))
        handles.append(stack[i].mlp.register_forward_hook(mlp_post(i)))
    path = OUT / f"{name}_L{layer}_{lo}-{hi}_{metric}.jsonl"
    done = (
        {json.loads(line)["job"] for line in path.read_text().splitlines() if line}
        if path.exists()
        else set()
    )
    width = {i: stack[i].self_attn.o_proj.in_features for i in range(lo, hi)}
    try:
        with path.open("a") as out:
            for pair in _pairs()[::2][:pairs]:  # every other pair: all settings
                entry = favour_patching.prepare(tokenizer, pair)
                k = entry["suffix"]
                with torch.no_grad():
                    store.clear()
                    state.update(mode="cache", k=k)
                    model(
                        input_ids=torch.tensor([entry["ids"]["big"]], device=0),
                        logits_to_keep=1,
                    )
                    strong = dict(store)
                state["mode"] = None
                for label in ["none", *sets]:
                    job = f"{pair.key}|{label}"
                    if job in done:
                        continue
                    chosen = sets.get(label, [])
                    src, cols = {}, {}
                    for kind, i, h in chosen:
                        key = f"{kind}{i}"
                        src[key] = strong[key]
                        if kind == "attn":
                            per = width[i] // 32
                            cols.setdefault(i, []).extend(range(h * per, (h + 1) * per))
                    state.update(mode="patch", src=src, cols=cols, k=k)
                    with torch.inference_mode():
                        written = model.generate(
                            input_ids=torch.tensor([entry["ids"]["small"]], device=0),
                            max_new_tokens=favour_baseline.MAX_NEW,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                        )
                    state["mode"] = None
                    reply = tokenizer.decode(
                        written[0, len(entry["ids"]["small"]) :],
                        skip_special_tokens=True,
                    ).strip()
                    decided, receiver = favour_pairs.decision(reply)
                    out.write(
                        json.dumps(
                            {
                                "job": job,
                                "pair": pair.key,
                                "set": label,
                                "components": [list(c) for c in chosen],
                                "gives": decided == "give"
                                and receiver == pair.big["environment"]["in_need"],
                                "decision": decided,
                                "reply": reply,
                            }
                        )
                        + "\n"
                    )
                    out.flush()
                print(pair.key, flush=True)
    finally:
        for handle in handles:
            handle.remove()
    rows_out = [json.loads(line) for line in path.read_text().splitlines() if line]
    for label in ["none", *sets]:
        picked = [r for r in rows_out if r["set"] == label]
        if picked:
            rate = 100 * sum(r["gives"] for r in picked) / len(picked)
            print(f"{label:9s} {rate:5.1f}% of {len(picked)}")
    return path


def heads_where(
    layer: int,
    window: tuple[int, int],
    metric: str = "bb",
    top: int = 12,
    pairs: int = 40,
) -> Path:
    """Where do the top heads (by |metric|) read from and write to? For each
    head, in the strong and in the weak prompt: for each part after the
    favour line (destination), the share of its attention that goes to each
    part of the prompt (source): the rules (system turn), round 1, round 2
    up to the favour line, the favour line itself, X's name elsewhere, and
    the parts after the favour line. Averaged over `pairs` pairs. Saves
    `heads_where_L<layer>_<metric>.json`."""
    import re

    import torch

    lo, hi = window
    results = torch.load(OUT / f"scores_L{layer}_{lo}-{hi}.pt")
    rows = [r[metric] for r in results if r.get(metric) is not None]
    heads = torch.stack([r["heads"] for r in rows]).mean(0)
    flat = sorted(
        ((heads[i, h].item(), lo + i, h) for i in range(hi - lo) for h in range(32)),
        key=lambda x: -abs(x[0]),
    )[:top]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    tokenizer, model = favour_baseline.load_model()
    model.set_attn_implementation("eager")
    sources = [
        "rules",
        "round 1",
        "round 2 before",
        "favour line",
        "X's name",
        *favour_patching.GROUPS,
    ]
    total: dict[str, Any] = {}
    count = 0
    for pair in _pairs()[::2][:pairs]:
        entry = favour_patching.prepare(tokenizer, pair)
        k = entry["suffix"]
        for member, trial in (("big", pair.big), ("small", pair.small)):
            text = entry["text"][member]
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            starts = [a for a, _ in enc["offset_mapping"]]
            n = len(starts)
            label = ["rules"] * n
            first_user = text.index("<|turn>user")
            second_user = text.index("<|turn>user", first_user + 1)
            fav = set(
                favour_attention.favour_span(
                    text,
                    [a if b > a else -1 for a, b in enc["offset_mapping"]],
                    trial["favour_line"],
                )
            )
            x = trial["environment"]["in_need"]
            names = [m.span() for m in re.finditer(rf"\b{x}\b", text)]
            for i, a in enumerate(starts):
                if a >= second_user:
                    label[i] = "round 2 before"
                elif a >= first_user:
                    label[i] = "round 1"
                if i in fav:
                    label[i] = "favour line"
                elif any(s0 <= a < s1 for s0, s1 in names):
                    label[i] = "X's name"
            for group, distances in entry["groups"].items():
                if group == "last":
                    continue
                for d in distances:
                    if label[n - d] != "X's name":
                        label[n - d] = group
            with torch.inference_mode():
                out = model(
                    input_ids=torch.tensor([enc["input_ids"]], device=0),
                    output_attentions=True,
                    logits_to_keep=1,
                )
            for _value, li, h in flat:
                att = out.attentions[li][0, h].float()  # [n, n]
                key = f"L{li}.H{h}"
                for dest in favour_patching.GROUPS:
                    rows_ = [n - d for d in entry["groups"][dest]]
                    if not rows_:
                        continue
                    a = att[rows_].mean(0)  # attention from this part, per source
                    for src in sources:
                        cols = [i for i in range(n) if label[i] == src]
                        mass = a[cols].sum().item() if cols else 0.0
                        cell = (
                            total.setdefault(key, {})
                            .setdefault(dest, {})
                            .setdefault(src, [0.0, 0.0])
                        )
                        cell[0 if member == "big" else 1] += mass
            del out
        count += 1
        print(pair.key, flush=True)
    for key in total:
        for dest in total[key]:
            for src in total[key][dest]:
                total[key][dest][src] = [
                    round(v / count, 4) for v in total[key][dest][src]
                ]
    result = {
        "pairs": count,
        "metric": metric,
        "heads": [{"head": f"L{li}.H{h}", "score": v} for v, li, h in flat],
        "sources": sources,
        "attention": total,
    }
    path = OUT / f"heads_where_L{layer}_{metric}.json"
    common.save_json(path, result)
    return path


def table(layer: int, window: tuple[int, int], top: int = 20) -> dict[str, Any]:
    import torch

    lo, hi = window
    results = torch.load(OUT / f"scores_L{layer}_{lo}-{hi}.pt")
    summary: dict[str, Any] = {"pairs": len(results)}
    for metric in ("d", "b", "bb", "bp"):
        rows = [r.get(metric) for r in results if r.get(metric) is not None]
        if not rows:
            continue
        heads = torch.stack([r["heads"] for r in rows]).mean(0)
        mlp = torch.stack([r["mlp"] for r in rows]).mean(0)
        flat = [
            (heads[i, h].item(), f"L{lo + i}.H{h}")
            for i in range(hi - lo)
            for h in range(32)
        ]
        flat += [(mlp[i].item(), f"L{lo + i}.MLP") for i in range(hi - lo)]
        flat.sort(key=lambda x: -abs(x[0]))
        summary[metric] = {
            "pairs": len(rows),
            "top": flat[:top],
            "by_layer_heads": heads.sum(1).tolist(),
            "by_layer_mlp": mlp.tolist(),
        }
        print(f"metric {metric} ({len(rows)} pairs): top components")
        for value, name in flat[:top]:
            print(f"  {name:10s} {value:9.4f}")
    common.save_json(OUT / f"table_L{layer}_{lo}-{hi}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "what", choices=("sample", "attribute", "validate", "where", "table")
    )
    parser.add_argument("--metric", default="bb")
    parser.add_argument(
        "--positions", action="store_true", help="attribute: keep per-position scores"
    )
    parser.add_argument(
        "--only", default=None, help="attribute: add only this metric (bp)"
    )
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--window", default="10-30")
    parser.add_argument("--pairs", type=int, default=None, help="`sample`: first n")
    parser.add_argument("--name", default="samples")
    args = parser.parse_args()
    window = tuple(int(x) for x in args.window.split("-"))
    if args.what == "sample":
        sample(limit=args.pairs, name=args.name)
    elif args.what == "validate":
        validate(args.layer, window, args.metric)
    elif args.what == "where":
        heads_where(args.layer, window, args.metric)
    elif args.what == "attribute":
        attribute(args.layer, window, args.only, args.positions)
        table(args.layer, window)
    else:
        table(args.layer, window)


if __name__ == "__main__":
    main()
