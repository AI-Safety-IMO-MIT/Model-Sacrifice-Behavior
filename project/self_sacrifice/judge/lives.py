"""Judge the decisive reply of the lives games played from .jsonl files: the
scripted-agent games (`run_lives`) and the robustness sweep (`run_robust`).

    run_lives   Atlas's first gift to the peer in need or, if it never gives,
                its last reply while the peer could still be saved
    run_robust  the recipient's last-round reply towards the benefactor, in
                every clean game

One trial per game, coded with the rubric of `judge.game`.

    # free: counts the trials and renders one prompt, sends nothing
    .venv/bin/python -m project.self_sacrifice.judge.lives <results.jsonl> ...

    # COSTS MONEY: one judge call per trial
    .venv/bin/python -m project.self_sacrifice.judge.lives <results.jsonl> ... --run

    # free: the tally of everything judged so far
    .venv/bin/python -m project.self_sacrifice.judge.lives

These are jrwong's `judge_game.py` and `judge_robust.py`. His robustness
judge reworded two sentences of the prompt for a refusal; this one asks the
same question of every trial.
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from project.self_sacrifice import analyze_robust, run_lives
from project.self_sacrifice.config import JUDGED, OUT, AgentFactory, PlayerName
from project.self_sacrifice.judge.game import (
    JUDGES,
    JudgedTrial,
    judge_prompt,
    judge_trial,
)

__all__ = [
    "DEFAULT_JUDGE",
    "WORKERS",
    "Trial",
    "decisive_round",
    "trial_of",
    "trials",
    "judged_path",
    "run",
    "load_judged",
    "counts",
    "report",
]

DEFAULT_JUDGE = "claude-haiku-5-5"
WORKERS = 8
RESULTS = OUT.parent

# (game id, focal, peer, role, round_number)
Trial = tuple[str, PlayerName, PlayerName, str, int]


def decisive_round(log: dict, focal: str, peer: str) -> int:
    """The round of the focal agent's first gift to the peer; if it never
    gives, its last round while the peer was short and still running."""
    asked = [r for r in log["rounds"] if focal in r["replies"]]
    for record in asked:
        if any(to == peer and n > 0 for to, n in record["replies"][focal]["gifts"]):
            return record["round_number"]
    last = log["setup"]["max_rounds"]
    savable = [
        r for r in asked if 0 <= r["before"][peer] < last - r["round_number"] + 1
    ]
    return (savable or asked)[-1]["round_number"]


def trial_of(log: dict, source: str) -> Trial | None:
    """The one reply of this game the judge reads, or None for a robustness
    game whose last round did not open as designed."""
    game = f"{source}/{log['setup']['config_name']}/{log['seed']}"
    if "variant" in log:
        if not analyze_robust.clean(log):
            return None
        benefactor, recipient, _ = analyze_robust.run_robust.roles(
            analyze_robust.Variant(**log["variant"]), log["kind"]
        )
        role = f"robust {log['kind']}{' no points' if log['no_points'] else ''}"
        return f"{game}/{log['repeat']}", recipient, benefactor, role, 2
    cond, names, _ = log["setup"]["config_name"].split("/")
    peer = run_lives.ROLES[names][0]
    focal = run_lives.ATLAS
    return game, focal, peer, cond, decisive_round(log, focal, peer)


def _logs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def trials(path: Path) -> list[tuple[dict, Trial]]:
    """Every (game, trial) of one result file."""
    out = []
    for log in _logs(path):
        trial = trial_of(log, path.stem)
        if trial is not None:
            out.append((log, trial))
    return out


def judged_path(path: Path, judge_name: str) -> Path:
    """Where a result file's codings go: the same place under the judge
    folder, one line per trial."""
    relative = path.resolve().relative_to(RESULTS)
    return JUDGED / relative.with_name(f"{path.stem}_judged_{judge_name}.jsonl")


def run(
    path: Path,
    judge_name: str,
    judge: AgentFactory,
    limit: int | None = None,
    workers: int = WORKERS,
) -> tuple[list[JudgedTrial], int]:
    """Judge every trial of one result file that is not judged yet. Returns
    the new codings and how many calls failed; a failed trial writes nothing
    and is retried by the next run."""
    out = judged_path(path, judge_name)
    done = {row["game"] for row in _logs(out)} if out.exists() else set()
    todo = [(log, trial) for log, trial in trials(path) if trial[0] not in done]
    if limit is not None:
        todo = todo[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    saved: list[JudgedTrial] = []
    failed = 0

    def one(job: tuple[dict, Trial]) -> None:
        nonlocal failed
        log, (game, focal, peer, role, round_number) = job
        row = judge_trial(
            log=log,
            game=game,
            focal=focal,
            peer=peer,
            role=role,
            round_number=round_number,
            channel="json",
            judge=judge,
        )
        with lock:
            if row.verdict is None:
                failed += 1
                print(f"  FAILED {game}: {row.error}"[:200], flush=True)
                return
            with out.open("a") as handle:
                handle.write(row.model_dump_json() + "\n")
            saved.append(row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, todo))
    return saved, failed


def load_judged(judge_name: str = DEFAULT_JUDGE) -> list[dict]:
    return [
        row
        for path in sorted(JUDGED.rglob(f"*_judged_{judge_name}.jsonl"))
        for row in _logs(path)
    ]


def counts(rows: list[dict]) -> list[dict]:
    """One group per (model, condition, gave), with its reason tally."""
    groups: dict[tuple[str, str, bool], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["model"], row["role"], row["gave"]), []).append(row)
    out = []
    for (model, role, gave), subset in groups.items():
        tally: dict[str, int] = {}
        for row in subset:
            reason = row["verdict"]["reason"]
            tally[reason] = tally.get(reason, 0) + 1
        out.append(
            {
                "model": model,
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
    return out


def report(rows: list[dict]) -> str:
    lines = [
        "The judged reply, by condition (json channel)",
        f"  {'model':<24} {'condition':<28} {'did':<7} {'n':>4} {'cost ok':>8} "
        f"{'values':>7}  reasons",
    ]
    for group in counts(rows):
        n = group["n"]
        lines.append(
            f"  {group['model']:<24} {group['role']:<28} "
            f"{'gave' if group['gave'] else 'passed':<7} {n:>4} "
            f"{str(group['understood']) + '/' + str(n):>8} "
            f"{str(group['values']) + '/' + str(n):>7}  "
            + ", ".join(f"{reason} {count}" for reason, count in group["reasons"])
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", type=Path, help="result .jsonl files")
    parser.add_argument(
        "--run",
        action="store_true",
        help="actually call the judge; COSTS MONEY. Without it nothing is sent.",
    )
    parser.add_argument("--judge", default=DEFAULT_JUDGE, choices=sorted(JUDGES))
    parser.add_argument(
        "--limit", type=int, default=None, help="judge at most this many per file"
    )
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    if not args.results:
        judged = load_judged(args.judge)
        if not judged:
            raise SystemExit(f"nothing judged by {args.judge} under {JUDGED}")
        print(report(judged))
        raise SystemExit(0)

    if args.run:
        failures = 0
        for results in args.results:
            new, bad = run(
                results, args.judge, JUDGES[args.judge], args.limit, args.workers
            )
            failures += bad
            read = sum(row.input_tokens or 0 for row in new)
            wrote = sum(row.output_tokens or 0 for row in new)
            print(
                f"{results.name}: judged {len(new)}, failed {bad}, "
                f"{read:,} tokens in, {wrote:,} out"
            )
        if failures:
            raise SystemExit(f"{failures} trials failed; run again to retry them")
        raise SystemExit(0)

    total = chars = 0
    example = ""
    for results in args.results:
        wanted = trials(results)
        prompts = [
            judge_prompt(
                log,
                focal,
                peer,
                round_number,
                gave=any(
                    to == peer and n > 0
                    for to, n in log["rounds"][round_number - 1]["replies"][focal][
                        "gifts"
                    ]
                ),
            )
            for log, (_, focal, peer, _, round_number) in wanted
        ]
        print(f"{results.name}: {len(wanted)} trials")
        total += len(wanted)
        chars += sum(map(len, prompts))
        example = example or (prompts[0] if prompts else "")
    print(f"\n{total} trials, judge {args.judge}")
    print(f"prompt size: {chars:,} characters, ~{chars // 4:,} tokens of input")
    print(f"\n--- one rendered prompt ---\n{example}")
    print("\nnothing was sent. Add --run to judge for real.")
