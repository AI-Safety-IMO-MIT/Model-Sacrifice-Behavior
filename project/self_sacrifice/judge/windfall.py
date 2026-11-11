"""Judge the windfall fault trials: why did each agent in the recipient's
position give, or not give, the benefactor its last life?

The analysis already answers *how often* (`analyze_windfall`). This answers
*why*, over exactly the same trials -- the recipient and the two bystanders in
the last round of every game that passed both gates. The trial set is taken
from `analyze_windfall` rather than rebuilt here, so the judged set and the
set the p-value is computed over cannot drift apart. The benefactor is not a
trial: it is the agent being saved, and has no gift to explain.

Only the json arm exists now: the text-protocol games were deleted when the
study settled on structured output, and the engine lost the protocol
machinery before that. `channel` stays on every row and every report line
anyway, because it records how a result file was produced -- a judged file
that did not say which channel it came from would be silent about the one
variable step 3 found (json 1/24 saves vs text 8/24). The text rows live in
git history, not on disk.

    # free: render every prompt, spend nothing
    .venv/bin/python -m project.self_sacrifice.judge.windfall

    # COSTS MONEY: one judge call per trial
    .venv/bin/python -m project.self_sacrifice.judge.windfall --run

Dry run is the default on purpose. A full pass is 144 trials, and this repo
has lost paid runs to a flag typed the wrong way round.
"""

import argparse
import json
from pathlib import Path

from project.self_sacrifice import analyze_windfall
from project.self_sacrifice.config import JUDGED, OUT, AgentFactory, PlayerName
from project.self_sacrifice.judge.game import (
    JUDGES,
    JudgedTrial,
    Reason,
    judge_prompt,
    judge_trial,
)

__all__ = [
    "DEFAULT_JUDGE",
    "Trial",
    "channel_of",
    "trials",
    "judged_path",
    "run",
    "load_judged",
    "reason_counts",
    "report",
]

DEFAULT_JUDGE = "claude-haiku-5-5"

# (game path, channel, focal, peer, role, round_number)
Trial = tuple[Path, str, PlayerName, PlayerName, str, int]


def channel_of(path: Path) -> str:
    """Which answer channel the game was played through, read off the saved
    config rather than guessed from the replies.

    The field is `protocol` in the file. `GameConfig` drops it on the way in
    (the protocol machinery was deleted from the engine), so the loaded log no
    longer carries the one variable step 3 turned on -- hence reading the raw
    file here. A game saved without the field is a json-arm game: the field
    only ever existed while both arms could run, and the json arm kept the
    bare filenames.
    """
    setup = json.loads(path.read_text())["setup"]
    return str(setup.get("protocol", "json"))


def trials(paths: list[Path]) -> tuple[list[Trial], list[dict]]:
    """Every trial worth judging, and the games that did not qualify.

    A game that failed either gate is not judged. The gates are about whether
    the dilemma was ever posed -- a benefactor that did not share never armed
    the trap -- so coding the reasoning of an agent that faced no choice would
    put noise into the breakdown. They are reported instead.
    """
    out: list[Trial] = []
    skipped: list[dict] = []
    for path in paths:
        log = analyze_windfall.load(path)
        benefactor, recipient, bystanders = analyze_windfall.roles(log)
        if not (analyze_windfall.shared(log) and analyze_windfall.clean(log)):
            skipped.append(
                {
                    "game": path.name,
                    "why": (
                        "the benefactor did not share"
                        if not analyze_windfall.shared(log)
                        else "last round drifted"
                    ),
                }
            )
            continue
        channel = channel_of(path)
        last = log["rounds"][-1]["round_number"]
        for focal in (recipient, *bystanders):
            role = "recipient" if focal == recipient else "bystander"
            out.append((path, channel, focal, benefactor, role, last))
    return out, skipped


def judged_path(path: Path, judge_name: str) -> Path:
    """One file per game, named for the judge. A second judge therefore never
    overwrites the first one's codings, which is what makes judge-vs-judge
    agreement checkable after the fact."""
    return JUDGED / f"{path.stem}_judged_{judge_name}.json"


def run(
    paths: list[Path],
    judge_name: str,
    judge: AgentFactory,
    limit: int | None = None,
) -> list[JudgedTrial]:
    """Judge every trial not already judged, saving one file per game.

    Resume is by file, like the runners: a game whose judged file exists is
    skipped whole, so a re-run after a 429 retries only what is missing.
    """
    wanted, skipped = trials(paths)
    for row in skipped:
        print(f"  skipped {row['game']}: {row['why']}")

    by_game: dict[Path, list[Trial]] = {}
    for trial in wanted:
        if judged_path(trial[0], judge_name).exists():
            continue
        by_game.setdefault(trial[0], []).append(trial)

    if limit is not None:
        by_game = dict(list(by_game.items())[:limit])

    JUDGED.mkdir(parents=True, exist_ok=True)
    done: list[JudgedTrial] = []
    for path, game_trials in by_game.items():
        log = analyze_windfall.load(path)
        rows = [
            judge_trial(
                log=log,
                game=path.name,
                focal=focal,
                peer=peer,
                role=role,
                round_number=round_number,
                channel=channel,
                judge=judge,
            )
            for _, channel, focal, peer, role, round_number in game_trials
        ]
        failed = [row for row in rows if row.verdict is None]
        if failed:
            # Nothing is written, so a re-run retries this game. Saving a
            # partial game would make the resume skip the trials that failed.
            print(f"  FAILED {path.name}: {failed[0].error}", flush=True)
            continue
        out = judged_path(path, judge_name)
        out.write_text(
            json.dumps([row.model_dump(mode="json") for row in rows], indent=2)
        )
        done += rows
        print(f"  done  {out.name}", flush=True)
    return done


def load_judged(judge_name: str) -> list[dict]:
    return [
        row
        for path in sorted(JUDGED.glob(f"*_judged_{judge_name}.json"))
        for row in json.loads(path.read_text())
    ]


def report(rows: list[dict]) -> str:
    """Reason breakdown by channel and role, gifts first."""
    lines = []
    for group in reason_counts(rows):
        lines += [
            f"{group['channel']} arm, {group['role']}, "
            f"{'gave' if group['gave'] else 'passed'} (n={group['n']})",
            "  " + ", ".join(f"{r} {c}" for r, c in group["reasons"]),
            f"  understood the cost {group['understood']}/{group['n']}, "
            f"values own survival {group['values']}/{group['n']}",
            "",
        ]

    misread = [row for row in rows if row["verdict"]["reason"] == Reason.MISREADING]
    if misread:
        lines.append(f"coded as misreading the rules: {len(misread)}/{len(rows)}")
        for row in misread:
            lines.append(f"  {row['game']} {row['focal']} ({row['role']})")
    return "\n".join(lines).rstrip()


def reason_counts(rows: list[dict]) -> list[dict]:
    """One group per (channel, role, gave), with its reason tally. Plain data,
    so `report()` and the HTML display render the same numbers instead of each
    counting for itself -- the rule `analyze_windfall.counts()` already follows.

    Split by channel because that is step 3's variable, by role because the
    experiment is recipient versus bystander, and by gave because the reason a
    refusal gives and the reason a gift gives are not the same question. A
    pooled table would be the one presentation that hides all three.
    """
    groups = []
    for channel in sorted({row["channel"] for row in rows}):
        for role in ("recipient", "bystander"):
            for gave in (True, False):
                subset = [
                    row
                    for row in rows
                    if row["channel"] == channel
                    and row["role"] == role
                    and row["gave"] is gave
                ]
                if not subset:
                    continue
                tally: dict[str, int] = {}
                for row in subset:
                    reason = row["verdict"]["reason"]
                    tally[reason] = tally.get(reason, 0) + 1
                groups.append(
                    {
                        "channel": channel,
                        "role": role,
                        "gave": gave,
                        "n": len(subset),
                        "reasons": sorted(tally.items(), key=lambda kv: -kv[1]),
                        "understood": sum(
                            row["verdict"]["understood_cost"] == "yes" for row in subset
                        ),
                        "values": sum(
                            row["verdict"]["self_stake"] == "values" for row in subset
                        ),
                    }
                )
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="store_true",
        help="actually call the judge; COSTS MONEY. Without it, prompts are "
        "rendered and nothing is sent.",
    )
    parser.add_argument("--judge", default=DEFAULT_JUDGE, choices=sorted(JUDGES))
    parser.add_argument(
        "--limit", type=int, default=None, help="judge at most this many games"
    )
    args = parser.parse_args()

    paths = sorted(OUT.glob("*windfall_fault*.json"))
    if not paths:
        raise SystemExit(f"no windfall fault results in {OUT}")

    if args.run:
        rows = run(paths, args.judge, JUDGES[args.judge], args.limit)
        print(f"judged {len(rows)} trials -> {JUDGED}")
        read = sum(row.input_tokens or 0 for row in rows)
        wrote = sum(row.output_tokens or 0 for row in rows)
        print(f"judge tokens: {read:,} in, {wrote:,} out")
        print()
        print(report(load_judged(args.judge)))
    else:
        wanted, skipped = trials(paths)
        for row in skipped:
            print(f"  skipped {row['game']}: {row['why']}")
        prompts = [
            judge_prompt(
                analyze_windfall.load(path),
                focal,
                peer,
                round_number,
                gave=any(
                    to == peer and lives > 0
                    for to, lives in analyze_windfall.load(path)["rounds"][
                        round_number - 1
                    ]["replies"][focal]["gifts"]
                ),
            )
            for path, _, focal, peer, _, round_number in wanted
        ]
        chars = sum(map(len, prompts))
        print(f"\n{len(wanted)} trials over {len(paths)} games, judge {args.judge}")
        print(f"prompt size: {chars:,} characters, ~{chars // 4:,} tokens of input")
        print(f"\n--- one rendered prompt ({wanted[0][2]}, {wanted[0][4]}) ---")
        print(prompts[0])
        print("\nnothing was sent. Add --run to judge for real.")
