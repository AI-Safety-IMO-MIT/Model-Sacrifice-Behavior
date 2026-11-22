"""Dose-response interventions over a layer window (default 25-37), at the
positions after the favour line, during the prompt pass, on the weak prompt
of each pair; the reply is then written freely and greedily.

    python -m project.self_sacrifice.interp.favour_dose run --part 0 --parts 2
    python -m project.self_sacrifice.interp.favour_dose table

Needs the GPU, except `table`, and `favour_steering direction` (directions.pt).
Output: `run_result/self_sacrifice/interp/favour_dose/dose.jsonl`.

At every layer j of the window, with w_j and s_j the weak and strong prompt's
residuals of the same pair (cached, unpatched runs) and d_j the strong-minus-
weak direction of `favour_steering` (unit length u_j):

    mix t        replace the residual by (1 - t) w_j + t s_j
                 (t = 1 is the range patch of experiment 1.2)
    along_d t    add t ((s_j - w_j) . u_j) u_j: move the weak prompt t of the
                 way to the strong prompt along d only, nothing else
    game t       add t (mean of the game's 10 strong favours - mean of its 10
                 weak favours): the game's own direction
    steer a      add a d_j (the averaged direction, in units of its own size)
    steer_reply a  the same, also at every token of the reply

Base runs: the weak and the strong prompt unpatched.
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

__all__ = ["OUT", "CONDITIONS", "run", "table", "main"]

OUT = common.OUT / "favour_dose"
WINDOW = (25, 37)
CONDITIONS = (
    [("mix", t) for t in (0.2, 0.4, 0.6, 0.8)]
    + [("along_d", t) for t in (0.2, 0.4, 0.6, 0.8, 1.0)]
    + [("game", t) for t in (0.5, 1.0)]
    + [("steer", a) for a in (1.0, 4.0, 16.0)]
    + [("steer_reply", 4.0)]
)
BATCH = 12


def run(part: int = 0, parts: int = 1) -> Path:
    import torch

    tokenizer, model = favour_baseline.load_model()
    lo, hi = WINDOW
    layers = list(range(lo, hi + 1))
    stack = common.decoder_layers(model)
    d_all = torch.load(favour_steering.OUT / "directions.pt")["directions"]["suffix"]
    d = {j: d_all[j].to(0) for j in layers}
    unit = {j: d[j] / d[j].norm() for j in layers}
    state: dict[str, Any] = {"mode": None}
    cache: dict[int, Any] = {}

    def hook(j: int):
        def fn(_m: Any, _i: Any, output: Any) -> Any:
            h = favour_patching._hidden(output)
            if state["mode"] == "cache":
                cache[j] = h[0, -state["k"] :].detach().float().cpu()
                return output
            if state["mode"] != "patch":
                return output
            if h.shape[1] == 1:
                add = state["reply_add"].get(j)
                if add is None:
                    return output
                return favour_patching._with_hidden(
                    output, (h.float() + add).to(h.dtype)
                )
            # all rows of a batch belong to one pair: the last k positions are
            # the suffix in every weak row
            k = state["k"]
            replace, target, add = state["edit"][j]
            tail = h[:, -k:].float() + add
            tail = torch.where(replace[:, None, None], target, tail)
            h = h.clone()
            h[:, -k:] = tail.to(h.dtype)
            return favour_patching._with_hidden(output, h)

        return fn

    handles = [stack[j].register_forward_hook(hook(j)) for j in layers]
    pairs = favour_patching.load_pairs(
        one_per_environment=True,
        environments=favour_steering.ENVIRONMENTS,
        settings=favour_steering.SETTINGS,
    )[part::parts]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"dose_{part}.jsonl"
    done = (
        {json.loads(x)["job"] for x in path.read_text().splitlines() if x}
        if path.exists()
        else set()
    )

    def residuals(trial: dict, k: int) -> dict[int, Any]:
        ids = tokenizer(common.prompt_text(tokenizer, trial), add_special_tokens=False)[
            "input_ids"
        ]
        cache.clear()
        state.update(mode="cache", k=k)
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=0), logits_to_keep=1)
        state["mode"] = None
        return dict(cache)

    try:
        for pair in pairs:
            entry = favour_patching.prepare(tokenizer, pair)
            k = entry["suffix"]
            strong, weak = residuals(pair.big, k), residuals(pair.small, k)
            same = {
                t["history"]: t
                for t in favour_baseline.load_trials(pair.big["version"])
                if t["environment"]["env"] == pair.big["environment"]["env"]
            }
            game = {j: 0 for j in layers}
            for keys, sign in ((favour_pairs.BIG, 1), (favour_pairs.SMALL, -1)):
                for key in keys:
                    r = residuals(same[key], k)
                    for j in layers:
                        game[j] = game[j] + sign * r[j] / len(keys)
            jobs = [("base_weak", 0.0, "small"), ("base_strong", 0.0, "big")]
            jobs += [(c, t, "small") for c, t in CONDITIONS]
            jobs = [j for j in jobs if f"{pair.key}|{j[0]}|{j[1]}" not in done]
            for start in range(0, len(jobs), BATCH):
                batch = jobs[start : start + BATCH]
                members = [getattr(pair, m) for _, _, m in batch]
                enc = tokenizer(
                    [common.prompt_text(tokenizer, t) for t in members],
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ).to(0)
                n = enc["input_ids"].shape[1]
                width = d[lo].shape[0]
                edit, reply_add = {}, {}
                for j in layers:
                    replace = torch.zeros(len(batch), dtype=torch.bool, device=0)
                    target = torch.zeros(len(batch), k, width, device=0)
                    add = torch.zeros(len(batch), k, width, device=0)
                    diff = (strong[j] - weak[j]).to(0)  # [k, width]
                    for row, (cond, t, _m) in enumerate(batch):
                        if cond == "mix":
                            replace[row] = True
                            target[row] = (1 - t) * weak[j].to(0) + t * strong[j].to(0)
                        elif cond == "along_d":
                            add[row] = t * (diff @ unit[j])[:, None] * unit[j]
                        elif cond == "game":
                            add[row] = t * game[j].to(0)
                        elif cond.startswith("steer"):
                            add[row] = t * d[j]
                    edit[j] = (replace, target, add)
                    radd = torch.zeros(len(batch), 1, width, device=0)
                    for row, (cond, t, _m) in enumerate(batch):
                        if cond == "steer_reply":
                            radd[row, 0] = t * d[j]
                    reply_add[j] = radd
                state.update(mode="patch", edit=edit, reply_add=reply_add, k=k)
                with torch.inference_mode():
                    written = model.generate(
                        **enc,
                        max_new_tokens=favour_baseline.MAX_NEW,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )
                state["mode"] = None
                with path.open("a") as out:
                    for (cond, t, member), tokens in zip(batch, written[:, n:]):
                        reply = tokenizer.decode(
                            tokens, skip_special_tokens=True
                        ).strip()
                        decided, receiver = favour_pairs.decision(reply)
                        out.write(
                            json.dumps(
                                {
                                    "job": f"{pair.key}|{cond}|{t}",
                                    "pair": pair.key,
                                    "condition": cond,
                                    "dose": t,
                                    "prompt": member,
                                    "gives": decided == "give"
                                    and receiver == pair.big["environment"]["in_need"],
                                    "decision": decided,
                                    "reply": reply,
                                }
                            )
                            + "\n"
                        )
                    out.flush()
                    os.fsync(out.fileno())
            print(pair.key, flush=True)
    finally:
        for handle in handles:
            handle.remove()
    return path


def table() -> dict[str, Any]:
    rows = [
        json.loads(x)
        for f in sorted(OUT.glob("dose_*.jsonl"))
        for x in f.read_text().splitlines()
        if x
    ]
    base: dict[str, dict[str, bool]] = {}
    for r in rows:
        if r["condition"].startswith("base"):
            base.setdefault(r["pair"], {})[r["condition"]] = r["gives"]
    differ = {
        p
        for p, v in base.items()
        if v.get("base_strong") and v.get("base_weak") is False
    }
    summary: dict[str, Any] = {"pairs": len(base), "differ": len(differ)}
    print(
        f"{len(differ)} of {len(base)} pairs differ unpatched; weak prompt now gives:"
    )
    for cond, t in CONDITIONS:
        picked = [
            r
            for r in rows
            if r["condition"] == cond and r["dose"] == t and r["pair"] in differ
        ]
        if picked:
            n = sum(r["gives"] for r in picked)
            summary[f"{cond}|{t}"] = [n, len(picked)]
            print(
                f"  {cond:12s} {t:5.2f}  {n:3d}/{len(picked)}  {100 * n / len(picked):5.1f}%"
            )
    common.save_json(OUT / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("run", "table"))
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    args = parser.parse_args()
    if args.what == "run":
        run(args.part, args.parts)
    table()


if __name__ == "__main__":
    main()
