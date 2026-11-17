"""How does the decision form across the reasoning?

Fixed-first-response version, all 800 conversations. The model's own
reasoning is cut after each sentence, including after zero sentences, the
reply is closed up to its list of gifts, and the log-odds of giving against
passing are read there. The model's own text is replayed unchanged, so
nothing is resampled.

    python -m project.self_sacrifice.interp.reasoning_trajectory

Needs the GPU. Output: `interp/reasoning_trajectory.jsonl`, one record per
conversation. This is jrwong's `runs/main/reasoning_trajectory.py`, which
appended `Decision: $` after each cut.
"""

# pyright: reportMissingImports=false

import json

from project.self_sacrifice.interp import common

__all__ = ["VERSION", "main"]

VERSION = "main-fixed-first-response"


def main() -> None:
    tokenizer, model = common.load_model()
    odds = common.GiftOdds(tokenizer, model)
    out = common.OUT / "reasoning_trajectory.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for n, trial in enumerate(common.load_trials(VERSION)):
            base = common.prompt_text(tokenizer, trial)
            parts = common.sentences(trial["reasoning"] or "")
            scores = [
                round(
                    odds(base + common.reply_start(" ".join(parts[:k]) if k else None)),
                    3,
                )
                for k in range(len(parts) + 1)
            ]
            handle.write(
                json.dumps(
                    {
                        "id": trial["id"],
                        "env": trial["environment"]["env"],
                        "history": trial["history"],
                        "gave": trial["gives_to_in_need"],
                        "sentences": parts,
                        "scores": scores,
                    }
                )
                + "\n"
            )
            handle.flush()
            if n % 50 == 0:
                print(n, len(parts), flush=True)


if __name__ == "__main__":
    main()
