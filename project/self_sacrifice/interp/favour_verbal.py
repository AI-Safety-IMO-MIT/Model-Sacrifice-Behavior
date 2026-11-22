"""Does patching raise the favour's weight directly, or does it work by making
the model talk about the favour, which then leads it to give?

    python -m project.self_sacrifice.interp.favour_verbal run --layer 29
    python -m project.self_sacrifice.interp.favour_verbal table --layer 29

Needs the GPU, except `table`, and the replies of the location experiment at
the same layer (`favour_patching activation --name location_L<layer>`).
Output: `run_result/self_sacrifice/interp/favour_verbal/L<layer>.jsonl`.

Pairs: those of the location run whose two prompts still decide differently
unpatched. Patch: the strong prompt's residual at every position after the
favour line, at one layer, during the prompt pass, as in experiment 1.

Possibility 1, magnitude: the patch raises how much the favour counts, and
that moves the decision whatever the reply says first.
Possibility 2, verbalisation: the patch makes the reply start by talking
about the favour, and talking about it is what leads to giving.

Four conditions, each the weak prompt with a fixed opening of the reply
(prefilled) and the rest written freely:
    patch_neutral   patched, the reply opens with the first sentence of the
                    unpatched weak reply (which does not dwell on the favour)
    patch_favour    patched, opens with the first sentence of the patched
                    reply (control: should match the patched rate)
    plain_favour    not patched, opens with the first sentence of the
                    patched reply (the favour put into words)
    plain_neutral   not patched, opens with the unpatched first sentence
                    (control: should match the unpatched rate)
Possibility 1 predicts patch_neutral stays high and plain_favour stays low;
possibility 2 predicts the reverse.

And one readout at the decision itself: the unpatched weak reply is fed in
whole up to its final "$", and the log-odds of " give" against " pass" for
the next token is read with and without the patch. A shift there, with the
reasoning held fixed, is the patch acting on the decision directly.
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

__all__ = ["OUT", "first_sentence", "run", "table", "main"]

OUT = common.OUT / "favour_verbal"
BATCH = 12


def first_sentence(reply: str) -> str:
    """The reply up to and including its first full stop after 20
    characters (the whole first paragraph if there is none)."""
    for i in range(20, len(reply)):
        if reply[i] == "." and (i + 1 == len(reply) or reply[i + 1] in " \n"):
            return reply[: i + 1]
    return reply.split("\n\n")[0]


def run(layer: int) -> Path:
    import torch

    rows = [
        json.loads(line)
        for line in (favour_patching.OUT / f"location_L{layer}.jsonl")
        .read_text()
        .splitlines()
    ]
    replies: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["from"] is None:
            replies.setdefault(r["pair"], {})[r["label"]] = r
        elif r["label"] == "denoise" and r["group"] == "all":
            replies.setdefault(r["pair"], {})["patched"] = r
    tokenizer, model = favour_baseline.load_model()
    module = favour_patching._layers(model, (layer,))[layer]
    state: dict[str, Any] = {"source": None}

    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = favour_patching._hidden(output)
        if state["source"] is None or hidden.shape[1] == 1:
            return output
        k, src, end = state["k"], state["source"], state["end"]
        hidden = hidden.clone()
        hidden[0, end - k : end] = src.to(hidden.dtype)
        return favour_patching._with_hidden(output, hidden)

    def cache(ids: list[int], k: int) -> Any:
        got = {}

        def grab(_m: Any, _i: Any, output: Any) -> None:
            got["h"] = favour_patching._hidden(output)[0, -k:].detach().clone()

        h = module.register_forward_hook(grab)
        with torch.inference_mode():
            model(input_ids=torch.tensor([ids], device=0), logits_to_keep=1)
        h.remove()
        return got["h"]

    handle = module.register_forward_hook(hook)
    give = tokenizer.encode("$ give", add_special_tokens=False)[-1]
    keep = tokenizer.encode("$ pass", add_special_tokens=False)[-1]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"L{layer}.jsonl"
    pairs = {
        p.key: p
        for p in favour_patching.load_pairs(
            one_per_environment=True,
            environments=favour_steering.ENVIRONMENTS,
            settings=favour_steering.SETTINGS,
        )
    }
    try:
        with path.open("w") as out:
            for key, r in sorted(replies.items()):
                if not (
                    r.get("base_big", {}).get("gives")
                    and r.get("base_small", {}).get("gives") is False
                    and "patched" in r
                ):
                    continue
                pair = pairs[key]
                entry = favour_patching.prepare(tokenizer, pair)
                k = entry["suffix"]
                ids = entry["ids"]["small"]
                source = cache(entry["ids"]["big"], k)
                neutral = first_sentence(r["base_small"]["reply"])
                favour = first_sentence(r["patched"]["reply"])
                in_need = pair.big["environment"]["in_need"]
                result: dict[str, Any] = {
                    "pair": key,
                    "patched_gives": r["patched"]["gives"],
                    "neutral_opening": neutral,
                    "favour_opening": favour,
                }
                for label, patched, opening in (
                    ("patch_neutral", True, neutral),
                    ("patch_favour", True, favour),
                    ("plain_favour", False, favour),
                    ("plain_neutral", False, neutral),
                ):
                    prefix = tokenizer(opening, add_special_tokens=False)["input_ids"]
                    full = ids + prefix
                    state.update(source=source if patched else None, k=k, end=len(ids))
                    with torch.inference_mode():
                        written = model.generate(
                            input_ids=torch.tensor([full], device=0),
                            max_new_tokens=favour_baseline.MAX_NEW,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            top_k=None,
                        )
                    state["source"] = None
                    reply = opening + tokenizer.decode(
                        written[0, len(full) :], skip_special_tokens=True
                    )
                    decided, receiver = favour_pairs.decision(reply)
                    result[label] = {
                        "gives": decided == "give" and receiver == in_need,
                        "decision": decided,
                        "reply": reply.strip(),
                    }
                # decision readout with the unpatched reasoning held fixed
                text = r["base_small"]["reply"]
                cut = text.rfind("$")
                if cut > 0:
                    prefix = tokenizer(text[: cut + 1], add_special_tokens=False)[
                        "input_ids"
                    ]
                    full = ids + prefix
                    odds = {}
                    for label, patched in (("plain", False), ("patched", True)):
                        state.update(
                            source=source if patched else None, k=k, end=len(ids)
                        )
                        with torch.inference_mode():
                            logits = (
                                model(
                                    input_ids=torch.tensor([full], device=0),
                                    logits_to_keep=1,
                                )
                                .logits[0, -1]
                                .float()
                            )
                        state["source"] = None
                        odds[label] = (logits[give] - logits[keep]).item()
                    result["decision_logodds"] = odds
                out.write(json.dumps(result) + "\n")
                out.flush()
                os.fsync(out.fileno())
                print(key, flush=True)
    finally:
        handle.remove()
    return path


def table(layer: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in (OUT / f"L{layer}.jsonl").read_text().splitlines()
        if line
    ]
    summary: dict[str, Any] = {"pairs": len(rows)}
    print(f"layer {layer}: {len(rows)} pairs; weak prompt gives")
    for label in ("plain_neutral", "plain_favour", "patch_neutral", "patch_favour"):
        n = sum(r[label]["gives"] for r in rows)
        summary[label] = [n, len(rows)]
        print(f"  {label:14s} {n:3d}/{len(rows)}  {100 * n / len(rows):5.1f}%")
    with_odds = [r for r in rows if "decision_logodds" in r]
    if with_odds:
        plain = sum(r["decision_logodds"]["plain"] for r in with_odds) / len(with_odds)
        patched = sum(r["decision_logodds"]["patched"] for r in with_odds) / len(
            with_odds
        )
        moved = sum(
            r["decision_logodds"]["patched"] > r["decision_logodds"]["plain"]
            for r in with_odds
        )
        summary["decision_logodds"] = {
            "plain": plain,
            "patched": patched,
            "raised_in": [moved, len(with_odds)],
        }
        print(
            f"  decision log-odds give vs pass, unpatched reasoning held fixed: "
            f"plain {plain:.2f}, patched {patched:.2f}; raised in {moved}/{len(with_odds)}"
        )
    common.save_json(OUT / f"L{layer}_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("run", "table"))
    parser.add_argument("--layer", type=int, default=29)
    args = parser.parse_args()
    if args.what == "run":
        run(args.layer)
    table(args.layer)


if __name__ == "__main__":
    main()
