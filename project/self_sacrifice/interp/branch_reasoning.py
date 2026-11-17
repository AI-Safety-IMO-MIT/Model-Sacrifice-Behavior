"""Where does the decision lock in? Branch from the model's reasoning and
resample what comes next.

Fixed-first-response version. Takes gives in helped, bystander and none, with
the same number of random keeps from that history. Each reply is cut after 0
to 5 sentences of its reasoning and continued several times at temperature
0.7. This is the one step that samples: greedy decoding would return the same
continuation.

    python -m project.self_sacrifice.interp.branch_reasoning

Needs the GPU. Output: `interp/branch_reasoning.jsonl`, one record per
conversation and cut. This is jrwong's `runs/main/branch_reasoning.py`. He
took every give and 10 samples; this takes at most `MAX_GIVES` gives per
history and `SAMPLES` samples, to keep the run to a few GPU hours.
"""

# pyright: reportMissingImports=false

import json
import random

import torch

from project.self_sacrifice.interp import common

__all__ = ["SAMPLES", "TEMPERATURE", "MAX_CUT", "MAX_GIVES", "MAX_NEW", "main"]

SAMPLES, TEMPERATURE, MAX_CUT = 8, 0.7, 5
MAX_GIVES = 30
MAX_NEW = 450
VERSION = "main-fixed-first-response"


def chosen_trials() -> list[dict]:
    trials = common.load_trials(VERSION)
    rng = random.Random(0)
    chosen = []
    for history in ("helped", "bystander", "none"):
        mine = [t for t in trials if t["history"] == history and t["reasoning"]]
        gave = [t for t in mine if t["gives_to_in_need"]][:MAX_GIVES]
        kept = [t for t in mine if not t["gives_to_in_need"]]
        chosen += gave + rng.sample(kept, min(len(gave), len(kept)))
    return chosen


def main() -> None:
    tokenizer, model = common.load_model()
    out = common.OUT / "branch_reasoning.jsonl"
    done = set()
    if out.exists():
        rows = map(json.loads, out.read_text().splitlines())
        done = {(row["id"], row["k"]) for row in rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    trials = chosen_trials()
    print("conversations", len(trials), flush=True)
    with out.open("a") as handle:
        for n, trial in enumerate(trials):
            parts = common.sentences(trial["reasoning"])
            for k in range(min(MAX_CUT, len(parts) - 1) + 1):
                if (trial["id"], k) in done:
                    continue
                prefix = (
                    common.reply_start(" ".join(parts[:k]), closed=False) if k else ""
                )
                encoded = tokenizer(
                    common.prompt_text(tokenizer, trial) + prefix,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(0)
                torch.manual_seed(1000 + k)
                with torch.no_grad():
                    sampled = model.generate(
                        **encoded,
                        max_new_tokens=MAX_NEW,
                        do_sample=True,
                        temperature=TEMPERATURE,
                        top_p=1.0,
                        top_k=0,
                        num_return_sequences=SAMPLES,
                    )
                new = sampled[:, encoded["input_ids"].shape[1] :]
                results = [
                    common.outcome(
                        trial, prefix + tokenizer.decode(ids, skip_special_tokens=True)
                    )
                    for ids in new
                ]
                row = {
                    "id": trial["id"],
                    "env": trial["environment"]["env"],
                    "history": trial["history"],
                    "gave": trial["gives_to_in_need"],
                    "k": k,
                    "n_sentences": len(parts),
                    "results": results,
                } | {
                    name: results.count(name)
                    for name in ("give", "keep", "other", "none")
                }
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            print(n + 1, "/", len(trials), flush=True)


if __name__ == "__main__":
    main()
