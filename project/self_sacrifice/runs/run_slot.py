"""Run the slot game: two agents, one migration slot, one turn.

    # free: counts the calls and renders one conversation, sends nothing
    .venv/bin/python -m project.self_sacrifice.runs.run_slot base

    # COSTS MONEY: one call per item
    .venv/bin/python -m project.self_sacrifice.runs.run_slot base --run

    --samples K  K sampled replies per item at temperature 0.7 (file
                 `samples/<variant>`)

Dry run is the default and `--run` is required to spend. Finished items are
skipped. This is the runner of jrwong's `slot_game.py`; his sampled runs
were seeded and these are not, because the hosts take no seed.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agent_framework.dto import AgenticChat
from project.self_sacrifice import config, models
from project.self_sacrifice.games import slot_game
from project.self_sacrifice.games.slot_game import SlotItem
from project.self_sacrifice.runs.run_main import ChatFactory, History

__all__ = [
    "MODEL",
    "TEMPERATURE",
    "SAMPLE_TEMPERATURE",
    "MAX_OUTPUT_TOKENS",
    "WORKERS",
    "SlotTrial",
    "chat_for",
    "result_path",
    "run_item",
    "run",
]

MODEL = models.DEFAULT
TEMPERATURE = 0.0
SAMPLE_TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 800
WORKERS = 4


class SlotTrial(BaseModel):
    """One item, answered. The line written to the variant's .jsonl."""

    id: str
    variant: str
    env: str
    history: str
    me: str
    other: str
    model: str
    agent_config: object
    temperature: float
    # Which sampled reply this is. None in a greedy run.
    sample: int | None

    system: str
    user: str
    reply: str
    reasoning: str | None
    decision: Literal["give", "pass"] | None  # None when the reply is unreadable
    gives_slot: bool
    stop_reason: str | None


def chat_for(
    system_prompt: str,
    history: History,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
) -> AgenticChat:
    return models.MODELS[model].build(
        system_prompt, temperature, MAX_OUTPUT_TOKENS, history
    )


def result_path(variant: str, model: str = MODEL, samples: bool = False) -> Path:
    folder = config.SLOT
    if model != MODEL:
        folder = folder / models.file_slug(model)
    if samples:
        folder = folder / "samples"
    return folder / f"{variant}.jsonl"


def run_item(
    item: SlotItem,
    variant: str,
    chat: ChatFactory,
    temperature: float = TEMPERATURE,
    sample: int | None = None,
) -> SlotTrial:
    schema = slot_game.decision_schema(item)
    asked = chat(item.system, ())
    report = asked.call(item.user, output_format=schema)
    content = report.response.content
    reply = (content[-1].content or "") if content else ""
    decision: Literal["give", "pass"] | None = None
    reasoning = None
    try:
        parsed = schema.model_validate_json(reply)
        decision = "give" if str(getattr(parsed, "decision")) == "give" else "pass"
        reasoning = getattr(parsed, "reasoning", None)
    except ValueError:
        pass
    return SlotTrial(
        id=f"{variant}-{item.id}" + ("" if sample is None else f"-s{sample}"),
        variant=variant,
        env=item.env,
        history=item.history,
        me=item.me,
        other=item.other,
        model=asked.model_name,
        agent_config=asked.get_trace().agent_config,
        temperature=temperature,
        sample=sample,
        system=item.system,
        user=item.user,
        reply=reply,
        reasoning=reasoning,
        decision=decision,
        gives_slot=decision == "give",
        stop_reason=report.stop_reason,
    )


def _jobs(variant: str, samples: int) -> list[tuple[str, SlotItem, int | None]]:
    every = slot_game.items(variant)
    if not samples:
        return [(f"{variant}-{item.id}", item, None) for item in every]
    return [
        (f"{variant}-{item.id}-s{sample}", item, sample)
        for item in every
        for sample in range(samples)
    ]


def _saved_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line}


def run(
    variant: str,
    chat: ChatFactory,
    out: Path,
    samples: int = 0,
    limit: int | None = None,
    workers: int = WORKERS,
) -> tuple[int, list[tuple[str, str]]]:
    """Run everything that is not saved yet. Returns how many were saved and
    the (id, error) of each that failed."""
    done = _saved_ids(out)
    todo = [job for job in _jobs(variant, samples) if job[0] not in done]
    if limit is not None:
        todo = todo[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []
    temperature = SAMPLE_TEMPERATURE if samples else TEMPERATURE

    def one(job: tuple[str, SlotItem, int | None]) -> None:
        _, item, sample = job
        try:
            trial = run_item(item, variant, chat, temperature, sample)
        except Exception as error:  # noqa: BLE001 -- one bad item, not the run
            with lock:
                failures.append((item.id, repr(error)[:200]))
            print(f"  FAILED {item.id}: {error!r}"[:200], flush=True)
            return
        with lock, out.open("a") as handle:
            handle.write(trial.model_dump_json() + "\n")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return len(todo) - len(failures), failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=sorted(slot_game.VARIANTS))
    parser.add_argument(
        "--run",
        action="store_true",
        help="actually call the model; COSTS MONEY. Without it nothing is sent.",
    )
    parser.add_argument("--samples", type=int, default=0, help="sampled replies")
    parser.add_argument("--limit", type=int, default=None, help="at most this many")
    parser.add_argument("--model", default=MODEL, choices=sorted(models.MODELS))
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    path = result_path(args.variant, args.model, bool(args.samples))
    if args.run:
        temperature = SAMPLE_TEMPERATURE if args.samples else TEMPERATURE
        saved, failed = run(
            args.variant,
            partial(chat_for, model=args.model, temperature=temperature),
            path,
            args.samples,
            args.limit,
            args.workers,
        )
        print(f"new trials {saved} -> {path}")
        if failed:
            raise SystemExit(f"{len(failed)} items failed; run again to retry them")
        raise SystemExit(0)

    done = _saved_ids(path)
    jobs = _jobs(args.variant, args.samples)
    todo = [job for job in jobs if job[0] not in done][: args.limit]
    chars = sum(len(item.system) + len(item.user) for _, item, _ in todo)
    print(f"{args.variant}: {len(jobs)} trials, {len(done)} saved, {len(todo)} to run")
    print(f"  {len(todo)} calls, roughly {chars // 4:,} input tokens")
    if todo:
        item = todo[0][1]
        print(f"\n--- {item.id} ---\n[system]\n{item.system}\n\n[user]\n{item.user}")
    print(f"\nnothing was sent. Add --run to run for real. Results go to {path}")
