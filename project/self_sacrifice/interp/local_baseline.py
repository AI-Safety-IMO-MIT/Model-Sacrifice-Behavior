"""Does the model on this GPU decide like the model on the study's host?

The steering experiments generate replies locally, without a schema enforced
on the output. This replays the fixed-first-response conversations (helped
and none, the first 100 environments) unsteered and compares the local
decisions with the saved ones.

    python -m project.self_sacrifice.interp.local_baseline

Needs the GPU. Output: `interp/local_baseline.jsonl`, one reply per line.
"""

# pyright: reportMissingImports=false

import json

from project.self_sacrifice.interp import common
from project.self_sacrifice.interp.steering import BATCH, MAX_NEW, test_trials

__all__ = ["main"]


def main() -> None:
    tokenizer, model = common.load_model()
    trials = test_trials()
    out = common.OUT / "local_baseline.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w") as handle:
        for start in range(0, len(trials), BATCH):
            batch = trials[start : start + BATCH]
            for trial, reply in zip(
                batch, common.generate(tokenizer, model, batch, MAX_NEW)
            ):
                row = {
                    "id": trial["id"],
                    "history": trial["history"],
                    "condition": "unsteered",
                    "reply": reply,
                    "gives_to_in_need": common.read_reply(trial, reply),
                    "original_gave": trial["gives_to_in_need"],
                }
                rows.append(row)
                handle.write(json.dumps(row) + "\n")
            print(start + len(batch), "/", len(trials), flush=True)
    for history in ("helped", "none"):
        mine = [row for row in rows if row["history"] == history]
        print(
            history,
            "local gives",
            sum(row["gives_to_in_need"] is True for row in mine),
            "saved gives",
            sum(row["original_gave"] for row in mine),
            "same",
            sum(row["gives_to_in_need"] == row["original_gave"] for row in mine),
            "unreadable",
            sum(row["gives_to_in_need"] is None for row in mine),
            "of",
            len(mine),
        )


if __name__ == "__main__":
    main()
