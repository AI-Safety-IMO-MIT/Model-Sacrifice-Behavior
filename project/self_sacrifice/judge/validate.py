"""Score judge models against the hand labels in validation_set.json.

A judge is an instrument, and an unvalidated instrument produces numbers with
no error bar. jrwong validated his phase-1 judge this way and reported
19/20 for `gemini-2.5-flash`; his lives-game judge
(`runs/games/scripts/judge_game.py`) was never validated at all, and there is
no hand-labelled set in the archive that covers it. So this set is new.

    # free: shows the set and what would be scored, sends nothing
    .venv/bin/python -m project.self_sacrifice.judge.validate

    # FREE: score against codings a judge run already paid for
    .venv/bin/python -m project.self_sacrifice.judge.validate --run --from-saved

    # COSTS MONEY: one judge call per item per model named
    .venv/bin/python -m project.self_sacrifice.judge.validate --run \\
        claude-haiku-5-5 gpt-oss-120b

How the gold labels were made, so their weaknesses are on the record:

- They were assigned by reading each reply, by the agent that wrote this file,
  not by a human and not by a second model. jrwong's set says the same of
  itself. It is a consistency check on the judge, not ground truth about the
  model's motives.
- The set is **deliberately unbalanced**: 9 items are the windfall games where
  an agent gave its only life, which is 9 of 144 trials in the real data.
  Validation needs signal on the rare class, so this is on purpose -- but it
  means the headline fraction here is not the accuracy you would get on a
  trial drawn at random, and must never be quoted as one.
- It spans **two experiments**. The windfall items only ever produce two
  reasons (`reciprocity` when an agent repaid, `own survival` when it refused),
  so a set drawn from them alone could not tell whether a judge uses the other
  seven categories at all. The blessed / blessed-tight items are free gifts and
  bring in `peer's sake` and `fairness`, plus one *free* `reciprocity` -- a
  repayment that cost the giver nothing.
- `gold_reason` is labelled on every item. `gold_understood_cost` is labelled
  where the reply states the consequence, and `gold_self_stake` where it states
  the matter plainly; items without them are skipped for that column rather
  than guessed. **`understood_cost` is a poor fit for a free gift** -- there is
  no cost to understand -- so on those items it is labelled "yes" only when the
  reply explicitly says it keeps enough to migrate, and left off otherwise.
- An item marked `ambiguous` is excluded from the reason score and listed, so
  a judge is not penalised for a reply the labeller could not call either.
"""

import argparse
import json
from pathlib import Path

from project.self_sacrifice import analyze_windfall
from project.self_sacrifice.config import JUDGED, OUT, AgentFactory, PlayerName
from project.self_sacrifice.judge.game import JUDGES, JudgedTrial, judge_trial
from project.self_sacrifice.judge.windfall import channel_of

__all__ = [
    "VALIDATION_SET",
    "load_set",
    "find_game",
    "trial_of",
    "saved_verdicts",
    "judge_item",
    "score",
    "report",
]

VALIDATION_SET = Path(__file__).with_name("validation_set.json")

# The columns scored, as (gold key, verdict field).
COLUMNS = (
    ("gold_reason", "reason"),
    ("gold_understood_cost", "understood_cost"),
    ("gold_self_stake", "self_stake"),
)


def load_set() -> list[dict]:
    return list(json.loads(VALIDATION_SET.read_text()))


def find_game(name: str) -> Path:
    """The result file an item names.

    One directory, because the text-protocol games were deleted when the study
    settled on structured output. An item naming one of them fails here rather
    than being quietly skipped -- a validation set that points at files nobody
    has is worse than one that says so.
    """
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(f"{name} is not in {OUT}")
    return path


def trial_of(item: dict, log: dict) -> tuple[PlayerName, PlayerName, str, int]:
    """(focal, peer, role, round_number) for one item.

    Two item shapes, because the set spans two experiments:

        windfall     `game` + `focal` only. The peer is the benefactor and the
                     round is the last one, both derived from the game's own
                     schedule by `analyze_windfall.roles()` -- the same
                     derivation the p-value uses, so a gold label can never be
                     attached to an agent the analysis would not have counted.
        blessed      `game`, `focal`, `peer` and `round`, spelled out. These
                     games have no benefactor and no last-round dilemma; the
                     decision of interest is one giver's gift in one round, and
                     only the labeller knows which.

    A reply that gave to several agents becomes one item per peer it is
    labelled for, since `reason` is a reason *about that peer* -- the same
    reply can be reciprocity towards the agent that helped it and plain
    concern towards the other two.
    """
    # Through PlayerName, so a typo in the set fails here rather than being
    # judged as an agent that is not in the game.
    focal = PlayerName(item["focal"])
    if "peer" in item:
        return focal, PlayerName(item["peer"]), item.get("role", "giver"), item["round"]

    benefactor, recipient, bystanders = analyze_windfall.roles(log)
    if focal not in (recipient, *bystanders):
        raise ValueError(
            f"{item['game']}: {focal} is the benefactor or not in the game, so it "
            f"has no gift to explain"
        )
    return (
        focal,
        benefactor,
        "recipient" if focal == recipient else "bystander",
        log["rounds"][-1]["round_number"],
    )


def saved_verdicts(judge_name: str) -> dict[tuple[str, str, int], JudgedTrial]:
    """Codings already on disk, keyed by (game file, focal, round).

    A judge run over an experiment covers the same trials the validation set
    draws its items from, so most of the set can be scored for **nothing** --
    the calls were already paid for. Only items whose trial was never judged
    (the blessed games, unless a judge run covered them too) need new calls.
    """
    out: dict[tuple[str, str, int], JudgedTrial] = {}
    for path in sorted(JUDGED.glob(f"*_judged_{judge_name}.json")):
        for row in json.loads(path.read_text()):
            trial = JudgedTrial.model_validate(row)
            out[(trial.game, str(trial.focal), trial.round_number)] = trial
    return out


def judge_item(item: dict, judge: AgentFactory) -> JudgedTrial:
    path = find_game(item["game"])
    log = analyze_windfall.load(path)
    focal, peer, role, round_number = trial_of(item, log)
    return judge_trial(
        log=log,
        game=path.name,
        focal=focal,
        peer=peer,
        role=role,
        round_number=round_number,
        channel=channel_of(path),
        judge=judge,
    )


def score(items: list[dict], verdicts: list[JudgedTrial]) -> dict:
    """Per-column hits, misses and the items that failed to judge at all."""
    out: dict = {"errors": [], "columns": {}}
    for index, (item, trial) in enumerate(zip(items, verdicts)):
        if trial.verdict is None:
            out["errors"].append({"index": index, "item": item, "error": trial.error})
    for gold_key, field in COLUMNS:
        scored, wrong = 0, []
        for index, (item, trial) in enumerate(zip(items, verdicts)):
            if trial.verdict is None or gold_key not in item:
                continue
            if gold_key == "gold_reason" and "ambiguous" in item:
                continue
            scored += 1
            got = getattr(trial.verdict, field)
            if str(got) != item[gold_key]:
                wrong.append(
                    {
                        "index": index,
                        "game": item["game"],
                        "focal": item["focal"],
                        "gold": item[gold_key],
                        "judge": str(got),
                        "quote": trial.verdict.quote,
                    }
                )
        out["columns"][field] = {
            "scored": scored,
            "correct": scored - len(wrong),
            "wrong": wrong,
        }
    out["ambiguous"] = [
        {"index": i, "game": item["game"], "why": item["ambiguous"]}
        for i, item in enumerate(items)
        if "ambiguous" in item
    ]
    return out


def report(judge_name: str, result: dict) -> str:
    lines = [judge_name]
    for _, field in COLUMNS:
        column = result["columns"][field]
        if not column["scored"]:
            lines.append(f"  {field:16s} no gold labels")
            continue
        lines.append(
            f"  {field:16s} {column['correct']}/{column['scored']}"
            f"  ({column['correct'] / column['scored']:.0%})"
        )
        for row in column["wrong"]:
            lines.append(
                f"      item {row['index']} {row['game']} {row['focal']}: "
                f"gold {row['gold']!r}, judge {row['judge']!r}"
            )
            lines.append(f"        judge quoted: {row['quote'][:120]}")
    if result["ambiguous"]:
        lines.append(
            f"  excluded from the reason score: "
            f"{len(result['ambiguous'])} ambiguous item(s)"
        )
    for row in result["errors"]:
        lines.append(
            f"  ERROR item {row['index']} {row['item']['game']}: {row['error']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "judges",
        nargs="*",
        default=["claude-haiku-5-5"],
        choices=sorted(JUDGES),
        help="judge models to score",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="actually call the judges; COSTS MONEY (one call per item per judge)",
    )
    parser.add_argument(
        "--from-saved",
        action="store_true",
        help="score against codings already in run_result/.../judge instead of "
        "calling; FREE. Items never judged are skipped and counted.",
    )
    args = parser.parse_args()

    items = load_set()
    if not args.run:
        reason_scored = sum(1 for item in items if "ambiguous" not in item)
        print(f"{len(items)} items in {VALIDATION_SET.name}")
        print(f"  reason scored on {reason_scored} (the rest are marked ambiguous)")
        for key, field in COLUMNS:
            print(f"  {field:16s} gold on {sum(key in item for item in items)}")
        by_reason: dict[str, int] = {}
        for item in items:
            by_reason[item["gold_reason"]] = by_reason.get(item["gold_reason"], 0) + 1
        print("  gold reasons:", ", ".join(f"{k} {v}" for k, v in by_reason.items()))
        for item in items:
            path = find_game(item["game"])  # fails loudly on a stale item
            print(f"    {path.parent.name}/{item['game']} {item['focal']}")
        print(f"\njudges that would be scored: {', '.join(args.judges)}")
        print("nothing was sent. Add --run to score for real.")
        raise SystemExit(0)

    for judge_name in args.judges:
        saved = saved_verdicts(judge_name) if args.from_saved else {}
        scored, verdicts, unjudged = [], [], 0
        for item in items:
            path = find_game(item["game"])
            focal, _peer, _role, round_number = trial_of(
                item, analyze_windfall.load(path)
            )
            trial_key = (path.name, str(focal), round_number)
            if trial_key in saved:
                scored.append(item)
                verdicts.append(saved[trial_key])
            elif args.from_saved:
                unjudged += 1  # never judged, and --from-saved spends nothing
            else:
                scored.append(item)
                verdicts.append(judge_item(item, JUDGES[judge_name]))
        print(report(judge_name, score(scored, verdicts)))
        if unjudged:
            print(
                f"  {unjudged} item(s) have no saved coding and were skipped; "
                f"drop --from-saved to judge them (costs money)"
            )
        print()
