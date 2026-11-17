"""Before any reasoning: prefill the reply up to its list of gifts and read
the log-odds of giving against passing there, for every saved conversation.

    python -m project.self_sacrifice.interp.prefill_logprobs

Needs the GPU. Output: `interp/prefill_logprobs.jsonl`. This is jrwong's
`runs/main/prefill_logprobs.py`, which prefilled `$`.
"""

# pyright: reportMissingImports=false

import json
import math

from project.self_sacrifice.interp import common

__all__ = ["VERSIONS", "main"]

VERSIONS = ("main", "main-fixed-first-response")


def main() -> None:
    tokenizer, model = common.load_model()
    odds = common.GiftOdds(tokenizer, model)
    out = common.OUT / "prefill_logprobs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for version in VERSIONS:
            for i, trial in enumerate(common.load_trials(version)):
                log_odds = odds(
                    common.prompt_text(tokenizer, trial) + common.reply_start()
                )
                handle.write(
                    json.dumps(
                        {
                            "id": trial["id"],
                            "version": version,
                            "env": trial["environment"]["env"],
                            "history": trial["history"],
                            "gave": trial["gives_to_in_need"],
                            "log_odds": log_odds,
                            "p_give": 1 / (1 + math.exp(-log_odds)),
                        }
                    )
                    + "\n"
                )
                if i % 100 == 0:
                    print(version, i, flush=True)


if __name__ == "__main__":
    main()
