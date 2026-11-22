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
    favour_baseline,
    favour_patching,
    favour_steering,
)

__all__ = ["OUT", "SAMPLES", "sample", "attribute", "table", "main"]

OUT = common.OUT / "favour_components"
SAMPLES = 16
MIN_EACH = 3
BATCH = 16
SEED = 20261122


def _pairs() -> list[favour_patching.Pair]:
    return favour_patching.load_pairs(
        one_per_environment=True,
        environments=favour_steering.ENVIRONMENTS,
        settings=favour_steering.SETTINGS,
    )


def sample(samples: int = SAMPLES, max_new: int = favour_baseline.MAX_NEW) -> Path:
    """`samples` replies at temperature 1 for the strong and the weak prompt
    of every pair. Resumable at batch boundaries."""
    import torch

    tokenizer, model = favour_baseline.load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "samples.jsonl"
    done = (
        {json.loads(line)["batch"] for line in path.read_text().splitlines() if line}
        if path.exists()
        else set()
    )
    jobs = [
        (pair, member, k)
        for pair in _pairs()
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


def attribute(layer: int, window: tuple[int, int]) -> Path:
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
    for line in (OUT / "samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        samples.setdefault((row["pair"], row["member"]), []).append(row)

    def run_prompt(ids: list[int]) -> None:
        model(input_ids=torch.tensor([ids], device=0), logits_to_keep=1)

    def component_grad(ids: list[int], k: int, reply_ids: list[int] | None) -> dict:
        """Gradient at the suffix positions of the prompt of either the
        projection (reply_ids None) or log p(reply | prompt)."""
        got.clear()
        state["mode"] = "grad"
        full = ids + (reply_ids or [])
        out = model(
            input_ids=torch.tensor([full], device=0),
            logits_to_keep=(len(reply_ids) + 1) if reply_ids else 1,
        )
        if reply_ids:
            logits = out.logits[0, : len(reply_ids)].float()
            target = torch.tensor(reply_ids, device=0)[:, None]
            metric = torch.log_softmax(logits, -1).gather(1, target).sum()
        else:
            resid = got["resid"][0, len(ids) - k : len(ids)].float()
            metric = (resid.mean(0) * unit).sum()
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
        return {"heads": heads, "mlp": mlps}

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"scores_L{layer}_{lo}-{hi}.pt"
    results = torch.load(path) if path.exists() else []
    done = {r["pair"] for r in results}
    try:
        for pair in _pairs():
            if pair.key in done:
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
            for member in ("small", "big"):
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
                        ][: favour_baseline.MAX_NEW]
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
            results.append(row)
            torch.save(results, path)
            print(pair.key, counts, flush=True)
            torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()
    return path


def table(layer: int, window: tuple[int, int], top: int = 20) -> dict[str, Any]:
    import torch

    lo, hi = window
    results = torch.load(OUT / f"scores_L{layer}_{lo}-{hi}.pt")
    summary: dict[str, Any] = {"pairs": len(results)}
    for metric in ("d", "b"):
        rows = [r[metric] for r in results if r[metric] is not None]
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
    parser.add_argument("what", choices=("sample", "attribute", "table"))
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--window", default="10-30")
    args = parser.parse_args()
    window = tuple(int(x) for x in args.window.split("-"))
    if args.what == "sample":
        sample()
    elif args.what == "attribute":
        attribute(args.layer, window)
        table(args.layer, window)
    else:
        table(args.layer, window)


if __name__ == "__main__":
    main()
