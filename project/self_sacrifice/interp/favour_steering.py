"""The strong-minus-weak direction: compute it, steer with it, read it.

    python -m project.self_sacrifice.interp.favour_steering direction
    python -m project.self_sacrifice.interp.favour_steering steer --layer 28
    python -m project.self_sacrifice.interp.favour_steering lens --layer 28
    python -m project.self_sacrifice.interp.favour_steering table --layer 28

Needs the GPU, except `table`. Output: `run_result/self_sacrifice/interp/
favour_steering/`.

direction
    For each pair of `favour_patching.load_pairs(one_per_environment=True)`
    in the 5 settings and first 20 environments, run the strong and the weak
    prompt and take, at every layer, the difference of the residuals averaged
    over the positions after the favour line (the "suffix"), and at the last
    prompt token alone. The direction is the mean over pairs of the first
    half of the environments (w000-w009): `directions.pt` holds both kinds,
    for all 60 layers. On the other half (w010-w019) it reports how well the
    projection on the direction separates strong from weak, and where the
    no-favour prompts fall (`separation.json`).

groups
    The same difference at one layer, but separately for each part of the
    suffix (event lines, holdings, ...): `groups_L<layer>.pt`. `steer --kind
    groups` adds each part's own vector at that part's positions.

steer
    At one layer, during the prompt pass only, add alpha * direction to every
    suffix position, then let the model write its reply. Held-out pairs only.
        weak + alpha*d      does the model now give?    (alpha > 0)
        strong - alpha*d    does it now keep?
        weak + alpha*r      the same with a random direction r of the same
                            norm (control)
    alpha = 1 adds exactly the mean strong-minus-weak difference. Every reply
    is saved, so one can read whether a steered reply now invents a rescue,
    or gives for another stated reason.

free
    Steer ordinary chat prompts (a JSON list): add alpha * direction at every
    position, prompt and reply, and save the reply for each alpha, alpha = 0
    included. For experiment 3: what does the direction do outside the game?

lens
    Logit lens of the direction at one layer: the final norm's scale and the
    unembedding applied to the direction; the tokens it raises and lowers
    most.
"""

# pyright: reportMissingImports=false

import argparse
import json
import os
from pathlib import Path
from typing import Any

from project.self_sacrifice.games import favour_pairs
from project.self_sacrifice.interp import common, favour_baseline, favour_patching

__all__ = [
    "OUT",
    "SETTINGS",
    "ALPHAS",
    "direction",
    "steer",
    "free",
    "lens",
    "table",
    "main",
]

OUT = common.OUT / "favour_steering"
SETTINGS = ("lives", "credits", "cells", "permits", "canisters")
ENVIRONMENTS = 20
TRAIN = 10  # w000-w009 give the direction; w010-w019 are held out
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
BATCH = 12


def _pairs(train: bool) -> list[favour_patching.Pair]:
    pairs = favour_patching.load_pairs(
        one_per_environment=True, environments=ENVIRONMENTS, settings=SETTINGS
    )
    return [
        pair
        for pair in pairs
        if (int(pair.big["environment"]["env"][1:]) < TRAIN) == train
    ]


def _all_layers(tokenizer: Any, model: Any, trial: dict, keep: int) -> Any:
    """Outputs of all 60 decoder layers at the last `keep` prompt positions:
    [layers, keep, width], float32 on the CPU."""
    import torch

    stack = common.decoder_layers(model)
    got: dict[int, Any] = {}

    def hook(index: int):
        def run(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = favour_patching._hidden(output)
            got[index] = hidden[0, -keep:].float().cpu()

        return run

    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(stack)]
    ids = tokenizer(common.prompt_text(tokenizer, trial), add_special_tokens=False)[
        "input_ids"
    ]
    try:
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=0), logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack([got[i] for i in range(len(stack))])


def direction() -> Path:
    import torch

    tokenizer, model = favour_baseline.load_model()
    OUT.mkdir(parents=True, exist_ok=True)

    def diffs(pairs: list) -> dict[str, Any]:
        suffix_d, last_d = [], []
        for pair in pairs:
            entry = favour_patching.prepare(tokenizer, pair)
            k = entry["suffix"]
            big = _all_layers(tokenizer, model, pair.big, k)
            small = _all_layers(tokenizer, model, pair.small, k)
            suffix_d.append((big - small).mean(1))
            last_d.append(big[:, -1] - small[:, -1])
        return {"suffix": torch.stack(suffix_d), "last": torch.stack(last_d)}

    train, test = _pairs(True), _pairs(False)
    d_train = diffs(train)
    directions = {kind: d.mean(0) for kind, d in d_train.items()}  # [layers, width]
    torch.save(
        {"directions": directions, "train": [p.key for p in train]},
        OUT / "directions.pt",
    )
    # held-out separation: project strong, weak and no-favour prompts
    report: dict[str, Any] = {"train_pairs": len(train), "test_pairs": len(test)}
    proj: dict[str, dict[str, list]] = {
        kind: {"big": [], "small": [], "none": []} for kind in directions
    }
    for pair in test:
        entry = favour_patching.prepare(tokenizer, pair)
        k = entry["suffix"]
        none_id = pair.big["id"].rsplit("-", 1)[0] + "-none"
        none = next(
            t
            for t in favour_baseline.load_trials(pair.big["version"])
            if t["id"] == none_id
        )
        for name, trial in (("big", pair.big), ("small", pair.small), ("none", none)):
            # the no-favour prompt is shorter: its last k positions are the same
            # text as the pair's suffix
            h = _all_layers(tokenizer, model, trial, k)
            for kind, d in directions.items():
                unit = d / d.norm(dim=-1, keepdim=True)
                x = h.mean(1) if kind == "suffix" else h[:, -1]
                proj[kind][name].append((x * unit).sum(-1))
    for kind in directions:
        big = torch.stack(proj[kind]["big"])
        small = torch.stack(proj[kind]["small"])
        none = torch.stack(proj[kind]["none"])
        layers = []
        for layer in range(big.shape[1]):
            b, s, n = big[:, layer], small[:, layer], none[:, layer]
            auroc = (b[:, None] > s[None, :]).float().mean().item()
            layers.append(
                {
                    "layer": layer,
                    "norm": directions[kind][layer].norm().item(),
                    "auroc_strong_vs_weak": auroc,
                    "mean_strong": b.mean().item(),
                    "mean_weak": s.mean().item(),
                    "mean_none": n.mean().item(),
                    "cos_pairs": torch.nn.functional.cosine_similarity(
                        d_train[kind][:, layer],
                        directions[kind][layer][None],
                        dim=-1,
                    )
                    .mean()
                    .item(),
                }
            )
        report[kind] = layers
    common.save_json(OUT / "separation.json", report)
    for kind in directions:
        print(kind)
        for row in report[kind][::4] + [report[kind][-1]]:
            print(
                f"  L{row['layer']:2d} norm {row['norm']:7.1f} auroc {row['auroc_strong_vs_weak']:.2f}"
                f"  strong {row['mean_strong']:8.1f} weak {row['mean_weak']:8.1f}"
                f" none {row['mean_none']:8.1f}  cos {row['cos_pairs']:.2f}"
            )
    return OUT / "directions.pt"


def group_directions(layer: int) -> Path:
    """Position-specific version of the direction at one layer: for each part
    of the suffix (`favour_patching.GROUPS`), the strong-minus-weak residual
    averaged over that part's positions and over the training pairs. Saves
    `groups_L<layer>.pt` {group: vector}."""
    import torch

    tokenizer, model = favour_baseline.load_model()
    sums: dict[str, Any] = {}
    count = 0
    for pair in _pairs(True):
        entry = favour_patching.prepare(tokenizer, pair)
        k = entry["suffix"]
        big = _all_layers(tokenizer, model, pair.big, k)[layer]
        small = _all_layers(tokenizer, model, pair.small, k)[layer]
        diff = big - small  # [k, width]; row k - distance
        for group in favour_patching.GROUPS:
            rows = [k - distance for distance in entry["groups"][group]]
            if rows:
                sums[group] = sums.get(group, 0) + diff[rows].mean(0)
        count += 1
    vectors = {group: total / count for group, total in sums.items()}
    path = OUT / f"groups_L{layer}.pt"
    torch.save(vectors, path)
    for group, v in vectors.items():
        print(group, round(v.norm().item(), 2))
    return path


def steer(layer: int, kind: str = "suffix", alphas: tuple[float, ...] = ALPHAS) -> Path:
    import torch

    tokenizer, model = favour_baseline.load_model()
    generator = torch.Generator().manual_seed(0)
    if kind == "groups":
        vectors = {
            g: v.to(0) for g, v in torch.load(OUT / f"groups_L{layer}.pt").items()
        }
        randoms = {}
        for g, v in vectors.items():
            r = torch.randn(v.shape, generator=generator).to(0)
            randoms[g] = r / r.norm() * v.norm()
        d = next(iter(vectors.values()))
    else:
        d = torch.load(OUT / "directions.pt")["directions"][kind][layer].to(0)
        r = torch.randn(d.shape, generator=generator).to(0)
        r = r / r.norm() * d.norm()
    module = favour_patching._layers(model, (layer,))[layer]
    state: dict[str, Any] = {"add": None}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = favour_patching._hidden(output)
        if state["add"] is None or hidden.shape[1] == 1:
            return output
        hidden = (hidden.float() + state["add"]).to(hidden.dtype)
        return favour_patching._with_hidden(output, hidden)

    handle = module.register_forward_hook(hook)
    path = OUT / f"steer_L{layer}_{kind}.jsonl"
    done = (
        {json.loads(line)["job"] for line in path.read_text().splitlines() if line}
        if path.exists()
        else set()
    )
    jobs = []
    for pair in _pairs(False):
        entry = favour_patching.prepare(tokenizer, pair)
        jobs += [
            (pair, entry, "small", 0.0, "base_weak"),
            (pair, entry, "big", 0.0, "base_strong"),
        ]
        for alpha in alphas:
            jobs += [
                (pair, entry, "small", alpha, "weak_plus_d"),
                (pair, entry, "big", -alpha, "strong_minus_d"),
                (pair, entry, "small", alpha, "weak_plus_random"),
            ]
    jobs = [j for j in jobs if f"{j[0].key}|{j[4]}|{j[3]}" not in done]
    try:
        for start in range(0, len(jobs), BATCH):
            batch = jobs[start : start + BATCH]
            trials = [getattr(j[0], j[2]) for j in batch]
            encoded = tokenizer(
                [common.prompt_text(tokenizer, t) for t in trials],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(0)
            length = encoded["input_ids"].shape[1]
            add = torch.zeros(len(batch), length, d.shape[0], device=0)
            for row, (_pair, entry, _member, alpha, label) in enumerate(batch):
                if kind == "groups":
                    table_ = randoms if label == "weak_plus_random" else vectors
                    for group, vec in table_.items():
                        for distance in entry["groups"][group]:
                            add[row, length - distance] = alpha * vec
                else:
                    vec = r if label == "weak_plus_random" else d
                    add[row, length - entry["suffix"] :] = alpha * vec
            state["add"] = add
            with torch.inference_mode():
                written = model.generate(
                    **encoded,
                    max_new_tokens=favour_baseline.MAX_NEW,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            state["add"] = None
            with path.open("a") as handle_out:
                for (pair, _entry, member, alpha, label), tokens in zip(
                    batch, written[:, length:]
                ):
                    reply = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    decided, receiver = favour_pairs.decision(reply)
                    in_need = pair.big["environment"]["in_need"]
                    handle_out.write(
                        json.dumps(
                            {
                                "job": f"{pair.key}|{label}|{alpha}",
                                "pair": pair.key,
                                "label": label,
                                "alpha": alpha,
                                "layer": layer,
                                "kind": kind,
                                "prompt": member,
                                "decision": decided,
                                "gives": decided == "give" and receiver == in_need,
                                "reply": reply,
                            }
                        )
                        + "\n"
                    )
                handle_out.flush()
                os.fsync(handle_out.fileno())
            print(start + len(batch), "/", len(jobs), flush=True)
    finally:
        handle.remove()
    return path


def lens(layer: int, kind: str = "suffix", top: int = 40) -> dict[str, Any]:
    import torch

    tokenizer, model = favour_baseline.load_model()
    d = torch.load(OUT / "directions.pt")["directions"][kind][layer].to(0)
    norm = next(
        m for name, m in model.named_modules() if name.endswith("language_model.norm")
    )
    unembed = model.get_output_embeddings().weight  # [vocab, width]
    with torch.inference_mode():
        # the final norm rescales each coordinate by its weight before the
        # unembedding; the overall RMS factor does not change the ranking
        logits = unembed.float() @ (norm.weight.float() * d)
    up = torch.topk(logits, top).indices.tolist()
    down = torch.topk(-logits, top).indices.tolist()
    result = {
        "layer": layer,
        "kind": kind,
        "raises": [tokenizer.decode([i]) for i in up],
        "lowers": [tokenizer.decode([i]) for i in down],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    common.save_json(OUT / f"lens_L{layer}_{kind}.json", result)
    print("raises:", result["raises"])
    print("lowers:", result["lowers"])
    return result


def free(
    layer: int,
    prompts_path: Path,
    alphas: tuple[float, ...],
    kind: str = "suffix",
    name: str = "free",
    max_new: int = 300,
) -> Path:
    """Steer ordinary chat prompts: add alpha * direction at `layer` to every
    position, in the prompt and in each written token, and save the reply for
    every alpha (0 = unsteered). `prompts_path` is a JSON list of
    {"id", "group", "prompt"}; each prompt is one user turn, no system turn."""
    import torch

    tokenizer, model = favour_baseline.load_model()
    d = torch.load(OUT / "directions.pt")["directions"][kind][layer].to(0)
    module = favour_patching._layers(model, (layer,))[layer]
    state: dict[str, Any] = {"add": None}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        if state["add"] is None:
            return output
        hidden = favour_patching._hidden(output)
        hidden = (hidden.float() + state["add"]).to(hidden.dtype)
        return favour_patching._with_hidden(output, hidden)

    handle = module.register_forward_hook(hook)
    prompts = json.loads(prompts_path.read_text())
    path = OUT / f"{name}_L{layer}_{kind}.jsonl"
    done = (
        {json.loads(line)["job"] for line in path.read_text().splitlines() if line}
        if path.exists()
        else set()
    )
    jobs = [(p, a) for a in (0.0, *alphas) for p in prompts]
    jobs = [j for j in jobs if f"{j[0]['id']}|{j[1]}" not in done]
    try:
        for start in range(0, len(jobs), BATCH):
            batch = jobs[start : start + BATCH]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for p, _ in batch
            ]
            encoded = tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(0)
            # the same shift at every position, scaled per row; the norm of the
            # mean strong-minus-weak difference is the unit of alpha
            scale = torch.tensor([a for _, a in batch], device=0)[:, None, None]
            state["add"] = scale * d[None, None, :]
            with torch.inference_mode():
                written = model.generate(
                    **encoded,
                    max_new_tokens=max_new,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            state["add"] = None
            length = encoded["input_ids"].shape[1]
            with path.open("a") as out:
                for (p, a), tokens in zip(batch, written[:, length:]):
                    out.write(
                        json.dumps(
                            {
                                "job": f"{p['id']}|{a}",
                                "id": p["id"],
                                "group": p["group"],
                                "prompt": p["prompt"],
                                "alpha": a,
                                "layer": layer,
                                "kind": kind,
                                "norm": d.norm().item(),
                                "reply": tokenizer.decode(
                                    tokens, skip_special_tokens=True
                                ).strip(),
                            }
                        )
                        + "\n"
                    )
                out.flush()
                os.fsync(out.fileno())
            print(start + len(batch), "/", len(jobs), flush=True)
    finally:
        handle.remove()
    return path


def table(layer: int, kind: str = "suffix") -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (OUT / f"steer_L{layer}_{kind}.jsonl").read_text().splitlines()
        if line
    ]
    base = {
        (r["pair"], r["label"]): r["gives"]
        for r in rows
        if r["label"].startswith("base")
    }
    pairs = sorted({p for p, _ in base})
    differ = {
        p
        for p in pairs
        if base.get((p, "base_strong")) and base.get((p, "base_weak")) is False
    }
    print(
        f"layer {layer} ({kind}): {len(pairs)} held-out pairs, {len(differ)} still differ"
    )
    print(" alpha   weak+d gives   strong-d keeps   weak+random gives")
    out: dict[str, Any] = {"pairs": len(pairs), "differ": len(differ), "alphas": {}}
    for alpha in sorted({abs(r["alpha"]) for r in rows if r["alpha"]}):
        cells = []
        for label, want in (
            ("weak_plus_d", True),
            ("strong_minus_d", False),
            ("weak_plus_random", True),
        ):
            picked = [
                r
                for r in rows
                if r["label"] == label
                and abs(r["alpha"]) == alpha
                and r["pair"] in differ
            ]
            cells.append([sum(r["gives"] is want for r in picked), len(picked)])
        out["alphas"][alpha] = cells
        print(
            f"{alpha:6.1f}  "
            + "  ".join(f"{h:4d}/{n:<4d} {100 * h / max(n, 1):5.1f}%" for h, n in cells)
        )
    common.save_json(OUT / f"steer_L{layer}_{kind}_summary.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("direction", "steer", "lens", "table", "free"))
    parser.add_argument("--prompts", default="", help="JSON prompt list for `free`")
    parser.add_argument("--name", default="free")
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument(
        "--kind", default="suffix", choices=("suffix", "last", "groups")
    )
    parser.add_argument("--alphas", default=",".join(map(str, ALPHAS)))
    args = parser.parse_args()
    if args.what == "direction":
        direction()
    elif args.what == "groups":
        group_directions(args.layer)
    elif args.what == "steer":
        steer(args.layer, args.kind, tuple(float(a) for a in args.alphas.split(",")))
        table(args.layer, args.kind)
    elif args.what == "lens":
        lens(args.layer, args.kind)
    elif args.what == "free":
        free(
            args.layer,
            Path(args.prompts),
            tuple(float(a) for a in args.alphas.split(",")),
            args.kind,
            args.name,
        )
    else:
        table(args.layer, args.kind)


if __name__ == "__main__":
    main()
