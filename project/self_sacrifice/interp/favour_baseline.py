"""The model's reply to every favour-pair conversation, written locally and
greedily: how often does it give its last unit after a big favour, a small
favour, and no favour?

    python -m project.self_sacrifice.interp.favour_baseline
    python -m project.self_sacrifice.interp.favour_baseline --settings lives,cells --environments 8
    python -m project.self_sacrifice.interp.favour_baseline --table

Needs the GPU, except `--table`. Output: `run_result/self_sacrifice/
favour_pairs/<setting>.jsonl` (`held_out/<setting>.jsonl` with `--wordings fresh`), one trial per line (the conversation, the reply
and the decision). A run resumes where it stopped: trials already in the file
are skipped.

The replies are free text and nothing is enforced on them; `favour_pairs.
decision` reads the last `$ give <agent> <n>` or `$ pass`. Decoding is greedy
with PyTorch's deterministic algorithms on, in fixed batches sorted by id, so a
second run on the same GPU type writes the same replies. A different GPU type,
batch size or attention implementation changes a few of them: greedy replies
to these prompts are that fragile, which is why every patching result in
`favour_patching` is compared with a placebo.

Set `GEMMA_PATH` to a local copy of the weights to avoid a download.
"""

# pyright: reportMissingImports=false

import argparse
import json
import os
from pathlib import Path
from typing import Any

from project.self_sacrifice.config import FAVOUR_PAIRS
from project.self_sacrifice.games import favour_pairs
from project.self_sacrifice.interp import common

__all__ = [
    "BATCH",
    "MAX_NEW",
    "load_model",
    "path_of",
    "load_trials",
    "replies",
    "run",
    "rates",
    "main",
]

BATCH = 16
MAX_NEW = 1000


def load_model() -> tuple[Any, Any]:
    """(tokenizer, model) in bf16 on one GPU, set up so that greedy decoding
    repeats exactly: deterministic algorithms, no TF32, SDPA attention."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = os.environ.get("GEMMA_PATH", common.MODEL)
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        device_map=os.environ.get("FAVOUR_DEVICE_MAP") or {"": 0},
    ).eval()
    model.set_attn_implementation("sdpa")
    return tokenizer, model


def path_of(setting: str, wordings: str = "new") -> Path:
    """Saved trials: new wordings at the top level, the others in a folder
    (fresh: `held_out/`)."""
    folder = {"new": FAVOUR_PAIRS, "fresh": FAVOUR_PAIRS / "held_out"}.get(
        wordings, FAVOUR_PAIRS / wordings
    )
    return folder / f"{setting}.jsonl"


def load_trials(setting: str, wordings: str = "new") -> list[dict]:
    """The saved trials of one setting, in file order."""
    path = path_of(setting, wordings)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def replies(tokenizer: Any, model: Any, trials: list[dict]) -> list[str]:
    """Each trial's reply, written greedily."""
    return [
        reply.strip() for reply in common.generate(tokenizer, model, trials, MAX_NEW)
    ]


def run(
    tokenizer: Any,
    model: Any,
    setting: str,
    favours: tuple[str, ...],
    environments: int,
    wordings: str,
) -> None:
    """Write the missing trials of one setting."""
    path = path_of(setting, wordings)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in load_trials(setting, wordings)}
    todo = [
        favour_pairs.trial(conversation)
        for conversation in favour_pairs.items(
            setting, favours, range(environments), wordings
        )
    ]
    todo = sorted((t for t in todo if t["id"] not in done), key=lambda t: t["id"])
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        with path.open("a") as handle:
            for trial, reply in zip(batch, replies(tokenizer, model, batch)):
                decided, receiver = favour_pairs.decision(reply)
                trial.update(
                    model=common.MODEL,
                    reply=reply,
                    decision=decided,
                    receiver=receiver,
                    gives_to_in_need=decided == "give"
                    and receiver == trial["environment"]["in_need"],
                    decoding={"greedy": True, "batch": BATCH, "attention": "sdpa"},
                )
                handle.write(json.dumps(trial) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(setting, start + len(batch), "/", len(todo), flush=True)


def rates(setting: str, wordings: str = "new") -> dict[str, Any]:
    """Give rates of one setting from its saved trials: per favour, pooled by
    size, and the share of big-small pairs of the same environment where the
    model gives after the big favour and keeps after the small one."""
    trials = load_trials(setting, wordings)
    by_favour: dict[str, list[dict]] = {}
    for trial in trials:
        by_favour.setdefault(trial["history"], []).append(trial)

    def rate(rows: list[dict]) -> float | None:
        if not rows:
            return None
        return round(100 * sum(row["gives_to_in_need"] for row in rows) / len(rows), 1)

    gave = {
        (trial["history"], trial["environment"]["env"]): trial["gives_to_in_need"]
        for trial in trials
    }
    pairs = [
        (gave[(big, env)], gave[(small, env)])
        for big in favour_pairs.BIG
        for small in favour_pairs.SMALL
        for env in sorted({env for _, env in gave})
        if (big, env) in gave and (small, env) in gave
    ]
    return {
        "trials": len(trials),
        "favour": {favour: rate(rows) for favour, rows in by_favour.items()},
        "size": {
            size: rate([trial for trial in trials if trial["size"] == size])
            for size in ("big", "small", "none")
        },
        "usable_pairs": round(
            100 * sum(big and not small for big, small in pairs) / len(pairs), 1
        )
        if pairs
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--settings", default=",".join(favour_pairs.SETTINGS))
    parser.add_argument("--favours", default=",".join(favour_pairs.FAVOURS))
    parser.add_argument("--environments", type=int, default=favour_pairs.N_ENVIRONMENTS)
    parser.add_argument("--wordings", default="new", choices=favour_pairs.WORDINGS)
    parser.add_argument(
        "--table", action="store_true", help="print the saved give rates and stop"
    )
    args = parser.parse_args()
    settings = args.settings.split(",")
    if not args.table:
        tokenizer, model = load_model()
        for setting in settings:
            run(
                tokenizer,
                model,
                setting,
                tuple(args.favours.split(",")),
                args.environments,
                args.wordings,
            )
    for setting in settings:
        summary = rates(setting, args.wordings)
        print(setting, summary["trials"], summary["size"], summary["usable_pairs"])


if __name__ == "__main__":
    main()
