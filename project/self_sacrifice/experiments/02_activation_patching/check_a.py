"""How much does the written reply read the favour line directly?

    python -m project.self_sacrifice.experiments.02_activation_patching.check_a run --name blocked
    python -m project.self_sacrifice.experiments.02_activation_patching.check_a run --name normal --unblocked
    python -m project.self_sacrifice.experiments.02_activation_patching.check_a table --name blocked

Needs the GPU, except `table`. Output: `run_result/self_sacrifice/interp/
favour_attention/<name>.jsonl`, one reply per line.

The prompt is read as usual, so by the time the reply starts, whatever the
model took from the favour line during the prompt is already in the later
positions (and in the key-value cache). Then, while the reply is written, no
reply token may attend to the favour line's tokens, at any layer. Everything
else stays visible: the rest of the prompt and the reply so far.

If giving after strong and weak favours stays where it was, the decision is
carried by what the prompt pass moved out of the favour line; if it moves
towards the no-favour rate, the reply needs to reread the favour itself.

Decoding is greedy and written out step by step (not `model.generate`) so
that the attention mask can differ between the prompt pass and the reply:
the prompt pass uses the ordinary mask, every reply step a mask with the
favour line's positions set to 0. Position ids are those of the unmasked
sequence, so blocking does not shift positions. With nothing blocked, the
loop writes the same replies as `model.generate` (`check`).
"""

# pyright: reportMissingImports=false

import argparse
import json
import os
from pathlib import Path
from typing import Any

from project.self_sacrifice.games import favour_pairs
from project.self_sacrifice.interp import common
from project.self_sacrifice.lib import favour_baseline
from project.self_sacrifice.lib import pairs as favour_patching

__all__ = ["OUT", "write", "run", "check", "table", "main"]

OUT = common.OUT / "favour_attention"
BATCH = 16


def write(
    tokenizer: Any,
    model: Any,
    trials: list[dict],
    block: bool,
    max_new: int = favour_baseline.MAX_NEW,
) -> list[str]:
    """Greedy replies to `trials`, written token by token; with `block`, no
    reply token attends to the favour line."""
    import torch

    texts = [common.prompt_text(tokenizer, trial) for trial in trials]
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = encoded["input_ids"].to(0)
    mask = encoded["attention_mask"].to(0)
    rows, length = ids.shape
    blocked = torch.ones_like(mask)
    if block:
        for row, trial in enumerate(trials):
            if not trial["favour_line"]:
                continue
            starts = [
                start if end > start else -1
                for start, end in encoded["offset_mapping"][row].tolist()
            ]
            for i in favour_patching.favour_span(
                texts[row], starts, trial["favour_line"]
            ):
                blocked[row, i] = 0
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    eos = set(
        tokenizer.convert_tokens_to_ids(["<turn|>"])
        + [tokenizer.eos_token_id, tokenizer.pad_token_id]
    )
    out = [[] for _ in range(rows)]
    finished = [False] * rows
    with torch.inference_mode():
        step = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=positions,
            cache_position=torch.arange(length, device=0),
            use_cache=True,
            logits_to_keep=1,
        )
        cache = step.past_key_values
        next_pos = positions[:, -1:] + 1
        reply_mask = mask * blocked
        for t in range(max_new):
            token = step.logits[:, -1].argmax(-1)
            for row in range(rows):
                if not finished[row]:
                    if token[row].item() in eos:
                        finished[row] = True
                    else:
                        out[row].append(token[row].item())
            if all(finished):
                break
            reply_mask = torch.cat([reply_mask, torch.ones_like(mask[:, :1])], -1)
            step = model(
                input_ids=token[:, None],
                attention_mask=reply_mask,
                position_ids=next_pos,
                cache_position=torch.tensor([length + t], device=0),
                past_key_values=cache,
                use_cache=True,
            )
            cache = step.past_key_values
            next_pos = next_pos + 1
    return [
        tokenizer.decode(tokens, skip_special_tokens=True).strip() for tokens in out
    ]


def run(
    name: str,
    block: bool,
    wordings: str = "new",
    settings: tuple[str, ...] = favour_pairs.SETTINGS,
    environments: int = 24,
) -> Path:
    """One reply per saved trial of `wordings` in `settings` and the first
    `environments` environments, with or without blocking. Resumable."""
    tokenizer, model = favour_baseline.load_model()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    done = (
        {json.loads(line)["id"] for line in path.read_text().splitlines() if line}
        if path.exists()
        else set()
    )
    todo = sorted(
        (
            trial
            for setting in settings
            for trial in favour_baseline.load_trials(setting, wordings)
            if trial["id"] not in done
            and int(trial["environment"]["env"][1:]) < environments
        ),
        key=lambda trial: trial["id"],
    )
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        replies = write(tokenizer, model, batch, block)
        with path.open("a") as handle:
            for trial, reply in zip(batch, replies):
                decided, receiver = favour_pairs.decision(reply)
                in_need = trial["environment"]["in_need"]
                handle.write(
                    json.dumps(
                        {
                            "id": trial["id"],
                            "setting": trial["version"],
                            "env": trial["environment"]["env"],
                            "favour": trial["history"],
                            "size": trial["size"],
                            "wordings": wordings,
                            "blocked": block,
                            "reply": reply,
                            "decision": decided,
                            "gives": decided == "give" and receiver == in_need,
                            "saved_gives": trial["gives_to_in_need"],
                        }
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        print(name, start + len(batch), "/", len(todo), flush=True)
    return path


def check(count: int = 32) -> None:
    """With nothing blocked, the step-by-step loop must write what
    `model.generate` writes, and blocking must change the masked run only."""
    tokenizer, model = favour_baseline.load_model()
    trials = favour_baseline.load_trials("lives")[:count]
    loop = []
    for start in range(0, len(trials), BATCH):
        loop += write(tokenizer, model, trials[start : start + BATCH], False)
    reference = []
    for start in range(0, len(trials), BATCH):
        reference += favour_baseline.replies(
            tokenizer, model, trials[start : start + BATCH]
        )
    same = sum(a == b for a, b in zip(loop, reference))
    saved = sum(a == trial["reply"] for a, trial in zip(loop, trials))
    print(f"loop vs generate: {same}/{len(trials)} identical")
    print(f"loop vs saved replies: {saved}/{len(trials)} identical")
    text = common.prompt_text(tokenizer, trials[0])
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    span = favour_patching.favour_span(
        text, [s for s, _ in encoded["offset_mapping"]], trials[0]["favour_line"]
    )
    print(
        "favour line tokens:",
        repr(tokenizer.decode(encoded["input_ids"][span[0] : span[-1] + 1])),
    )


def table(name: str) -> dict[str, Any]:
    """Give rates by size, against the saved (unblocked) replies."""
    rows = [
        json.loads(line)
        for line in (OUT / f"{name}.jsonl").read_text().splitlines()
        if line
    ]
    summary: dict[str, Any] = {}
    print(f"{name}: {len(rows)} replies")
    print("size    n     now gives   saved gives   changed")
    for size in ("big", "small", "none"):
        picked = [row for row in rows if row["size"] == size]
        if not picked:
            continue
        now = sum(row["gives"] for row in picked)
        before = sum(row["saved_gives"] for row in picked)
        changed = sum(row["gives"] != row["saved_gives"] for row in picked)
        summary[size] = {
            "n": len(picked),
            "gives": now,
            "saved_gives": before,
            "changed": changed,
        }
        print(
            f"{size:6s} {len(picked):4d}  {100 * now / len(picked):8.1f}%  "
            f"{100 * before / len(picked):10.1f}%  {changed:6d}"
        )
    common.save_json(OUT / f"{name}_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("run", "check", "table"))
    parser.add_argument("--name", default="blocked")
    parser.add_argument("--unblocked", action="store_true")
    parser.add_argument("--wordings", default="new", choices=favour_pairs.WORDINGS)
    parser.add_argument("--settings", default=",".join(favour_pairs.SETTINGS))
    parser.add_argument(
        "--environments", type=int, default=24, help="use w000 .. w<n-1>"
    )
    args = parser.parse_args()
    if args.what == "run":
        run(
            args.name,
            not args.unblocked,
            args.wordings,
            tuple(args.settings.split(",")),
            args.environments,
        )
        table(args.name)
    elif args.what == "check":
        check()
    else:
        table(args.name)


if __name__ == "__main__":
    main()
