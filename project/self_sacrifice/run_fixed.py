"""Run the fixed pairs: is the decision there before the reasoning?

Each item is a fixed two-round conversation (see `fixed_pairs`), asked in
three reply formats. The measure is P(give) read from the token
log-probabilities at the decision, next to the decision itself.

    fixed    the round-1 reply is fixed: give nothing
    own      the model writes its own round-1 reply, once per item, and every
             format is then asked after that same reply

    # free: counts the calls and renders one conversation, sends nothing
    .venv/bin/python -m project.self_sacrifice.run_fixed fixed

    # COSTS MONEY: one call per item and format, plus one per item for `own`
    .venv/bin/python -m project.self_sacrifice.run_fixed fixed --run

Dry run is the default and `--run` is required to spend. Finished trials are
skipped, so a run that dies on a rate limit is resumed by running it again.
Never run the same version in two processes at once: nothing guards it.

The subject is pinned to a host that returns log-probabilities (see
`models.LOGPROB_MODELS`), which is not the host the rest of the study uses.
jrwong read P(give) at a free-text `$ give` / `$ pass`; here it is read at
the `decision` field of a structured reply, so the rates and probabilities
are not comparable with his.
"""

import argparse
import json
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from agent_framework.dto import AgenticChat
from project.self_sacrifice import config, models
from project.self_sacrifice import fixed_pairs as pairs
from project.self_sacrifice.fixed_pairs import Item, TokenAlternatives

__all__ = [
    "MODEL",
    "TEMPERATURE",
    "MAX_OUTPUT_TOKENS",
    "WORKERS",
    "VERSIONS",
    "History",
    "ChatFactory",
    "Trial",
    "chat_for",
    "version_name",
    "result_path",
    "round1_path",
    "run_item",
    "run",
]

# Key into models.LOGPROB_MODELS.
MODEL = models.DEFAULT
TEMPERATURE = 0.0
# The formats without reasoning need far less, and a low cap stops a reply
# that explains anyway.
MAX_OUTPUT_TOKENS = {"free": 1000, "first": 300, "stated": 300}
# jrwong's cap for the round-1 reply.
ROUND1_TOKENS = 600
WORKERS = 4
# version -> whether the round-1 reply is fixed
VERSIONS = {"fixed": True, "own": False}

History = models.History
# (system prompt, earlier turns, max output tokens) -> a fresh chat
ChatFactory = Callable[[str, History, int], AgenticChat]


class Trial(BaseModel):
    """One item in one reply format, answered. One line of the .jsonl."""

    id: str
    version: str
    format: str
    rules: str
    permission: bool
    names: str
    role: int
    history: str
    blessing: bool
    me: str
    in_need: str
    other: str
    other2: str
    model: str
    agent_config: object

    system: str
    first_message: str
    first_reply: str
    # Whether the round-1 reply gave anything. The histories are only
    # comparable when it did not. None when that reply could not be read.
    first_acted: bool | None
    last_message: str  # as sent, with the format's suffix
    reply: str

    spare_lives: int | None
    reasoning: str | None
    statement: str | None
    decision: Literal["give", "pass"] | None  # None: the reply was not valid
    gifts: list[tuple[str, int]]
    gives_to_in_need: bool
    # P(give) / (P(give) + P(pass)) at the decision token, and the
    # probability the two cover. None when the token was not found.
    p_give: float | None
    mass: float

    stop_reason: str | None
    input_tokens: int | None
    completion_tokens: int | None


def chat_for(
    system_prompt: str, history: History, max_tokens: int, model: str = MODEL
) -> AgenticChat:
    return models.LOGPROB_MODELS[model].build(
        system_prompt, TEMPERATURE, max_tokens, history
    )


def version_name(version: str, blessing: bool) -> str:
    return f"{version}-blessing" if blessing else version


def _folder(model: str) -> Path:
    if model == MODEL:
        return config.FIXED
    return config.FIXED / models.file_slug(model)


def result_path(version: str, blessing: bool = False, model: str = MODEL) -> Path:
    name = version_name(version, blessing).replace("-", "_")
    return _folder(model) / f"{name}.jsonl"


def round1_path(blessing: bool = False, model: str = MODEL) -> Path:
    """The model's own round-1 replies, one per item, written once."""
    return _folder(model) / f"round1{'_blessing' if blessing else ''}.jsonl"


def _lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _reply_text(chat: AgenticChat, user_text: str, schema: type[BaseModel]):
    report = chat.call(user_text, output_format=schema)
    content = report.response.content
    return report, (content[-1].content or "") if content else ""


def _read(schema: type[BaseModel], raw: str) -> BaseModel | None:
    try:
        return schema.model_validate_json(raw)
    except ValueError:
        return None


def _tokens(chat: AgenticChat) -> list[TokenAlternatives]:
    """The last reply's tokens with their alternatives; empty for a chat or a
    host that returns none."""
    response = getattr(chat, "last_response", None)
    choices = getattr(response, "choices", None)
    logprobs = choices[0].logprobs if choices else None
    content = getattr(logprobs, "content", None) or []
    return [
        (token.token, [(a.token, a.logprob) for a in token.top_logprobs])
        for token in content
    ]


def _acted(item: Item, raw: str) -> bool | None:
    move = _read(pairs.move_schema(item), raw)
    if move is None:
        return None
    return bool(getattr(move, "gifts")) or str(getattr(move, "decision")) == "give"


def run_item(
    item: Item,
    version: str,
    reply_format: str,
    first_reply: str,
    chat: ChatFactory,
) -> Trial:
    """Answer one item in one format, after `first_reply`."""
    schema = pairs.move_schema(item, reply_format)
    sent = pairs.last_message(item, reply_format)
    so_far: History = (("user", item.first_message), ("assistant", first_reply))
    decider = chat(item.system, so_far, MAX_OUTPUT_TOKENS[reply_format])
    report, reply = _reply_text(decider, sent, schema)
    parsed = _read(schema, reply)
    p_give, mass = pairs.decision_prob(_tokens(decider))

    gifts: list[tuple[str, int]] = []
    decision: Literal["give", "pass"] | None = None
    if parsed is not None:
        gifts = [
            (str(gift.to), gift.lives)
            for gift in getattr(parsed, "gifts")
            if str(gift.to) != item.me
        ]
        decision = "give" if str(getattr(parsed, "decision")) == "give" else "pass"
    return Trial(
        id=f"{version_name(version, item.blessing)}-{item.id}-{reply_format}",
        version=version_name(version, item.blessing),
        format=reply_format,
        rules=item.rules,
        permission=item.permission,
        names=item.names,
        role=item.role,
        history=item.history,
        blessing=item.blessing,
        me=item.me,
        in_need=item.in_need,
        other=item.other,
        other2=item.other2,
        model=decider.model_name,
        agent_config=decider.get_trace().agent_config,
        system=item.system,
        first_message=item.first_message,
        first_reply=first_reply,
        first_acted=_acted(item, first_reply),
        last_message=sent,
        reply=reply,
        spare_lives=getattr(parsed, "spare_lives", None),
        reasoning=getattr(parsed, "reasoning", None),
        statement=getattr(parsed, "statement", None),
        decision=decision,
        gifts=gifts,
        gives_to_in_need=any(to == item.in_need for to, _ in gifts),
        p_give=p_give,
        mass=mass,
        stop_reason=report.stop_reason,
        input_tokens=report.input_tokens,
        completion_tokens=report.completion_tokens,
    )


def _todo(
    version: str,
    blessing: bool,
    formats: tuple[str, ...],
    path: Path,
    limit: int | None,
) -> list[tuple[Item, list[str]]]:
    """Every item with the formats it still lacks, in item order."""
    done = {row["id"] for row in _lines(path)}
    name = version_name(version, blessing)
    out = []
    for item in pairs.items(blessing):
        missing = [f for f in formats if f"{name}-{item.id}-{f}" not in done]
        if missing:
            out.append((item, missing))
    return out if limit is None else out[:limit]


def run(
    version: str,
    chat: ChatFactory,
    formats: tuple[str, ...] = tuple(pairs.FORMATS),
    blessing: bool = False,
    limit: int | None = None,
    out: Path | None = None,
    round1: Path | None = None,
    workers: int = WORKERS,
) -> tuple[int, list[tuple[str, str]]]:
    """Run every (item, format) of a version that is not saved yet. Returns
    how many trials were saved and the (item id, error) of each item that
    failed.

    One job is one item, so its own round-1 reply is written once and shared
    by its formats. A trial is one appended line; a failed item keeps the
    formats it finished and the next run does the rest.
    """
    fixed = VERSIONS[version]
    path = out or result_path(version, blessing)
    cache_path = round1 or round1_path(blessing)
    cache = {row["id"]: row["reply"] for row in _lines(cache_path)}
    todo = _todo(version, blessing, formats, path, limit)

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    failures: list[tuple[str, str]] = []
    saved = 0

    def first_reply(item: Item) -> str:
        if fixed:
            return pairs.fixed_first_response()
        if item.id in cache:
            return cache[item.id]
        _, reply = _reply_text(
            chat(item.system, (), ROUND1_TOKENS),
            item.first_message,
            pairs.move_schema(item),
        )
        with lock, cache_path.open("a") as handle:
            cache[item.id] = reply
            handle.write(json.dumps({"id": item.id, "reply": reply}) + "\n")
        return reply

    def one(job: tuple[Item, list[str]]) -> None:
        nonlocal saved
        item, missing = job
        try:
            first = first_reply(item)
            for reply_format in missing:
                trial = run_item(item, version, reply_format, first, chat)
                with lock, path.open("a") as handle:
                    handle.write(trial.model_dump_json() + "\n")
                    saved += 1
        except Exception as error:  # noqa: BLE001 -- one bad item, not the run
            with lock:
                failures.append((item.id, repr(error)[:200]))
            print(f"  FAILED {item.id}: {error!r}"[:200], flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return saved, failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=sorted(VERSIONS))
    parser.add_argument(
        "--run",
        action="store_true",
        help="actually call the model; COSTS MONEY. Without it nothing is sent.",
    )
    parser.add_argument(
        "--formats",
        default=",".join(pairs.FORMATS),
        help="comma-separated reply formats (default: all three)",
    )
    parser.add_argument(
        "--blessing",
        action="store_true",
        help="jrwong's earlier variant: the spare life comes from a blessing",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="run at most this many items"
    )
    parser.add_argument("--model", default=MODEL, choices=sorted(models.LOGPROB_MODELS))
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help="items in flight at once; more is faster and draws more 429s",
    )
    args = parser.parse_args()
    chosen = tuple(args.formats.split(","))
    unknown = [f for f in chosen if f not in pairs.FORMATS]
    if unknown:
        raise SystemExit(
            f"unknown formats {unknown}; choose from {list(pairs.FORMATS)}"
        )
    out_path = result_path(args.version, args.blessing, args.model)
    cache_file = round1_path(args.blessing, args.model)

    if args.run:
        count, failed = run(
            args.version,
            lambda system, history, tokens: chat_for(
                system, history, tokens, args.model
            ),
            chosen,
            args.blessing,
            args.limit,
            out=out_path,
            round1=cache_file,
            workers=args.workers,
        )
        print(f"new trials {count} -> {out_path}")
        if failed:
            raise SystemExit(f"{len(failed)} items failed; run again to retry them")
        raise SystemExit(0)

    jobs = _todo(args.version, args.blessing, chosen, out_path, args.limit)
    decisions = sum(len(missing) for _, missing in jobs)
    cached = {row["id"] for row in _lines(cache_file)}
    first_calls = (
        0 if VERSIONS[args.version] else sum(item.id not in cached for item, _ in jobs)
    )
    chars = sum(
        len(missing)
        * (len(item.system) + len(item.first_message) + len(item.last_message))
        for item, missing in jobs
    )
    subject = models.LOGPROB_MODELS[args.model]
    print(
        f"{version_name(args.version, args.blessing)} on {subject.slug} "
        f"({subject.host}): {len(jobs)} items to run"
    )
    print(
        f"  {decisions} decision calls + {first_calls} round-1 calls = "
        f"{decisions + first_calls} calls, roughly {chars // 4:,} input tokens "
        f"(an undercount: schemas and round-1 replies not included)"
    )
    if jobs:
        item, missing = jobs[0]
        print(f"\n--- {item.id}, format {missing[-1]} ---\n[system]\n{item.system}")
        print(f"\n[user]\n{item.first_message}")
        if VERSIONS[args.version]:
            print(f"\n[assistant, fixed]\n{pairs.fixed_first_response()}")
        else:
            print("\n[assistant]\n(the model's own reply)")
        print(f"\n[user]\n{pairs.last_message(item, missing[-1])}")
    print("\nnothing was sent. Add --run to run for real.")
