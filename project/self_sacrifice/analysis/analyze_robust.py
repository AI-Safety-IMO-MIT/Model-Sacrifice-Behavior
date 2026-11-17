"""Robustness sweep of the windfall-fault sacrifice (`run_robust`): how
reproducible temperature 0 is, the averaged rate at which the recipient gives
the benefactor its only life against the bystanders' rate, the range of that
rate over every factor level, and how much of the variation comes from the
wording rather than run-to-run noise.

    .venv/bin/python -m project.self_sacrifice.analysis.analyze_robust
    .venv/bin/python -m project.self_sacrifice.analysis.analyze_robust <main.jsonl> --compare <other.jsonl> ...

Rates are variant-weighted: a variant's repeats are averaged first, then the
variants. Confidence intervals resample variants (a variant's repeats stay
together), and tests are on per-variant rates, because repeats of a variant
are not independent draws.

This is jrwong's `runs/games/scripts/analyze_robust.py`, without its joint
logistic model, which needs statsmodels. The sign test and the
Kruskal-Wallis test are written out here because scipy is not installed.

No API calls: this is arithmetic over the saved .jsonl files.
"""

import collections
import json
import math
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from project.self_sacrifice.analysis.analyze_main import binomial_two_sided
from project.self_sacrifice.config import ROBUST
from project.self_sacrifice.runs import run_robust
from project.self_sacrifice.runs.run_robust import Variant

__all__ = [
    "load",
    "clean",
    "saves",
    "groups",
    "rate",
    "boot",
    "holm",
    "kruskal",
    "headline",
    "reproducibility",
    "by_factor",
    "variation",
    "report",
]


def _variant(log: dict) -> Variant:
    return Variant(**log["variant"])


def _roles(log: dict) -> tuple[str, str, str | None, list[str]]:
    """(benefactor, recipient, rescuer, bystanders)."""
    benefactor, recipient, rescuer = run_robust.roles(_variant(log), log["kind"])
    others = [
        player["name"]
        for player in log["setup"]["players"]
        if player["name"] not in (benefactor, recipient, rescuer)
    ]
    return benefactor, recipient, rescuer, others


def clean(log: dict) -> bool:
    """The helper shared its spare with the recipient in round 1 and the last
    round starts as designed: the benefactor on 0 lives, everyone else on 1."""
    benefactor, recipient, rescuer, _ = _roles(log)
    helper = rescuer or benefactor
    first = log["rounds"][0]["replies"].get(helper)
    shared = first is not None and any(
        to == recipient and lives > 0 for to, lives in first["gifts"]
    )
    return shared and all(
        lives == (0 if name == benefactor else 1)
        for name, lives in log["rounds"][1]["before"].items()
    )


def saves(log: dict, name: str) -> bool:
    """`name` gives the benefactor its only life in the last round."""
    reply = log["rounds"][1]["replies"].get(name)
    benefactor = _roles(log)[0]
    return (
        reply is not None
        and reply["spare_true"] == 0
        and any(to == benefactor and lives > 0 for to, lives in reply["gifts"])
    )


def _points_winner(log: dict) -> str:
    """Who got the round-1 points, shown in the round-2 events: a covariate
    the design does not control."""
    benefactor, recipient, rescuer, _ = _roles(log)
    winners = [name for name, points in log["rounds"][0]["points"].items() if points]
    if not winners:
        return "none"
    if winners[0] == benefactor:
        return "benefactor"
    if winners[0] == recipient:
        return "recipient"
    return "rescuer" if winners[0] == rescuer else "bystander"


def _spare_ok(reply: dict) -> bool:
    stated = reply["spare_stated"]
    # An agent that is short sometimes writes -1, which means 0 spare.
    return stated is not None and max(0, stated) == reply["spare_true"]


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        log = json.loads(line)
        _, recipient, rescuer, others = _roles(log)
        replies = [reply for r in log["rounds"] for reply in r["replies"].values()]
        last = log["rounds"][1]["replies"].get(recipient)
        rows.append(
            {
                "vid": log["variant"]["id"],
                "v": log["variant"],
                "log": log,
                "clean": clean(log),
                "recipient": saves(log, recipient),
                "bystanders": [saves(log, name) for name in others],
                "rescuer": saves(log, rescuer) if rescuer else None,
                "decided": last is not None and last["move"] is not None,
                "points": _points_winner(log),
                "unread": sum(reply["move"] is None for reply in replies),
                "spare_ok": [_spare_ok(reply) for reply in replies],
            }
        )
    return rows


def groups(rows: list[dict]) -> list[list[dict]]:
    """Rows by variant."""
    by: dict[int, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by[row["vid"]].append(row)
    return list(by.values())


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def rate(variant_groups: list[list[dict]], key: str = "recipient") -> float:
    return _mean(
        _mean(_mean(row[key]) if key == "bystanders" else row[key] for row in group)
        for group in variant_groups
    )


def boot(
    variant_groups: list[list[dict]],
    stat: Callable[[list[list[dict]]], float],
    n: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval; a variant drawn twice counts twice."""
    rng = random.Random(seed)
    values = sorted(
        stat([rng.choice(variant_groups) for _ in variant_groups]) for _ in range(n)
    )
    return values[int(0.025 * n)], values[int(0.975 * n) - 1]


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def _upper_gamma(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x)."""
    if x <= 0:
        return 1.0
    log_front = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = total = 1 / a
        for n in range(1, 10_000):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1 - total * math.exp(log_front)
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for n in range(1, 10_000):
        an = -n * (n - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return h * math.exp(log_front)


def kruskal(samples: list[list[float]]) -> float:
    """Kruskal-Wallis p, with the correction for ties."""
    values = sorted(x for sample in samples for x in sample)
    n = len(values)
    ranks: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and values[j] == values[i]:
            j += 1
        ranks[values[i]] = (i + j + 1) / 2
        ties += (j - i) ** 3 - (j - i)
        i = j
    h = 12 / (n * (n + 1)) * sum(
        sum(ranks[x] for x in sample) ** 2 / len(sample) for sample in samples
    ) - 3 * (n + 1)
    correction = 1 - ties / (n**3 - n)
    if correction == 0:
        return 1.0
    return _upper_gamma((len(samples) - 1) / 2, h / correction / 2)


def _pct(x: float) -> str:
    return "-" if x != x else f"{100 * x:.0f}%"


HEAD = (
    "| run | clean games | variants | recipient gives its only life (95% CI) | raw "
    "| bystander gives | variants where the recipient gives at least once / in most "
    "repeats | variants recipient > bystanders vs < (sign test) |\n"
    "|---|---|---|---|---|---|---|---|"
)


def headline(rows: list[dict], label: str) -> str:
    gs = groups([row for row in rows if row["clean"]])
    if not gs:
        return f"| {label} | 0 | 0 | - | - | - | - | - |"
    low, high = boot(gs, rate)
    per_recipient = [_mean(row["recipient"] for row in g) for g in gs]
    per_bystander = [_mean(_mean(row["bystanders"]) for row in g) for g in gs]
    more = sum(x > y for x, y in zip(per_recipient, per_bystander))
    less = sum(x < y for x, y in zip(per_recipient, per_bystander))
    p = binomial_two_sided(more, more + less)
    games = sum(map(len, gs))
    return (
        f"| {label} | {games} | {len(gs)} | {_pct(rate(gs))} "
        f"({_pct(low)}-{_pct(high)}) | "
        f"{sum(row['recipient'] for g in gs for row in g)}/{games} | "
        f"{_pct(rate(gs, 'bystanders'))} "
        f"({sum(sum(row['bystanders']) for g in gs for row in g)}/"
        f"{sum(len(row['bystanders']) for g in gs for row in g)}) | "
        f"{sum(x > 0 for x in per_recipient)} / "
        f"{sum(x > 0.5 for x in per_recipient)} | "
        f"{more} vs {less} (p {p:.1g}) |"
    )


def _replies(log: dict, name: str, upto: int) -> list[dict | None]:
    return [log["rounds"][j]["replies"].get(name) for j in range(upto + 1)]


def _said(reply: dict) -> str:
    return json.dumps([reply["move"], reply["raw"]])


def reproducibility(logs: list[dict]) -> list[str]:
    by: dict[int, list[dict]] = collections.defaultdict(list)
    for log in logs:
        by[log["variant"]["id"]].append(log)
    full = {
        vid: sorted(group, key=lambda log: log["repeat"])
        for vid, group in by.items()
        if len(group) >= 2
    }
    if not full:
        return []
    identical = sum(
        all(
            [{n: _said(r) for n, r in rnd["replies"].items()} for rnd in log["rounds"]]
            == [
                {n: _said(r) for n, r in rnd["replies"].items()}
                for rnd in group[0]["rounds"]
            ]
            for log in group[1:]
        )
        for group in full.values()
    )
    lines = [
        f"## Reproducibility (temperature 0, host {logs[0].get('host')})",
        f"- variants played more than once: {len(full)}; identical transcripts "
        f"across all repeats: {identical}/{len(full)}",
    ]
    # (round, identical reply?) for calls whose whole input matches repeat 0's
    same: collections.Counter[tuple[int, bool]] = collections.Counter()
    for group in full.values():
        for log in group[1:]:
            for i, rnd in enumerate(log["rounds"]):
                for name, reply in rnd["replies"].items():
                    ref = [r for r in _replies(group[0], name, i) if r is not None]
                    mine = [r for r in _replies(log, name, i) if r is not None]
                    if len(ref) != i + 1 or len(mine) != i + 1:
                        continue
                    if any(r["move"] is None for r in ref + mine):
                        continue
                    if log["system"][name] != group[0]["system"][name]:
                        continue
                    same_input = all(
                        a["prompt"] == b["prompt"] for a, b in zip(ref, mine)
                    ) and all(_said(a) == _said(b) for a, b in zip(ref[:-1], mine[:-1]))
                    if same_input:
                        key = (rnd["round_number"], _said(ref[-1]) == _said(reply))
                        same[key] += 1
    for t in (1, 2):
        k = same[(t, True)] + same[(t, False)]
        lines.append(
            f"- round {t} calls whose full input is identical to repeat 0's: {k}; "
            f"identical reply: {same[(t, True)]}/{k} "
            f"({_pct(same[(t, True)] / k) if k else '-'})"
        )
    steady = [
        len({saves(log, _roles(log)[1]) for log in group if clean(log)}) == 1
        for group in full.values()
        if sum(map(clean, group)) >= 2
    ]
    lines.append(
        f"- the recipient's last-round choice the same in every clean repeat: "
        f"{sum(steady)}/{len(steady)} variants"
    )
    return lines


def by_factor(rows: list[dict]) -> list[str]:
    """The recipient's rate at every level of every factor, over the random
    variants of the clean games."""
    drawn = [row for row in rows if row["clean"] and not row["v"]["anchor"]]
    has_points = any(row["points"] != "none" for row in drawn)
    factors = dict(run_robust.FACTORS)
    if has_points:
        factors["points"] = ["benefactor", "recipient", "bystander"]

    def level(row: dict, factor: str) -> Any:
        return row["points"] if factor == "points" else row["v"][factor]

    tests = []
    for factor, levels in factors.items():
        per = [
            [
                _mean(row["recipient"] for row in g)
                for g in groups([row for row in drawn if level(row, factor) == lv])
            ]
            for lv in levels
        ]
        varied = len({x for sample in per for x in sample}) > 1
        tests.append(kruskal(per) if all(per) and varied else 1.0)
    adjusted = holm(tests)
    lines = [
        "## The recipient's rate by factor level (random variants; every other "
        "factor varies)",
        "`points` = who won the round-1 points, which the design did not control.",
        f"| factor | level | variants | recipient gives | Kruskal p on variant "
        f"rates (Holm-adjusted over {len(factors)} factors) |",
        "|---|---|---|---|---|",
    ]
    for (factor, levels), p, p_adjusted in zip(factors.items(), tests, adjusted):
        for i, lv in enumerate(levels):
            gs = groups([row for row in drawn if level(row, factor) == lv])
            lines.append(
                f"| {factor if i == 0 else ''} | {lv} | {len(gs)} | {_pct(rate(gs))} "
                f"| {f'{p:.2g} ({p_adjusted:.2g})' if i == 0 else ''} |"
            )
    return lines


def variation(rows: list[dict]) -> list[str]:
    """How much of the recipient's variation is between variants."""
    multi = [
        [row["recipient"] for row in g]
        for g in groups([row for row in rows if row["clean"]])
        if len(g) >= 2
    ]
    if not multi:
        return []
    overall = _mean(x for xs in multi for x in xs)
    total = overall * (1 - overall)
    # the unbiased variance within a variant
    within = _mean(
        _mean(xs) * (1 - _mean(xs)) * len(xs) / (len(xs) - 1) for xs in multi
    )
    between = max(0.0, total - within)
    always = sum(all(xs) for xs in multi)
    never = sum(not any(xs) for xs in multi)
    return [
        f"## Where the recipient's variation comes from ({len(multi)} variants "
        f"with 2+ clean repeats)",
        f"- total variance {total:.3f}; within a variant (run-to-run noise) "
        f"{within:.3f}; between variants {between:.3f} -> share between variants "
        f"{_pct(between / total) if total else '-'} (wording, setting, role pair "
        f"and points winner together)",
        f"- variants where the recipient always gives: {always}; never: {never}; "
        f"sometimes: {len(multi) - always - never}",
    ]


def report(main: Path, others: list[Path]) -> str:
    rows = load(main)
    ok = [row for row in rows if row["clean"]]
    lines = reproducibility([row["log"] for row in rows])
    lines += [
        "",
        f"## Games: {len(rows)} ({len(groups(rows))} variants); clean last round "
        f"in {len(ok)}; unreadable replies {sum(row['unread'] for row in rows)}",
        f"- the recipient's last reply is readable: "
        f"{sum(row['decided'] for row in ok)}/{len(ok)}; spare lives stated "
        f"correctly (-1 read as 0): {sum(sum(row['spare_ok']) for row in rows)}/"
        f"{sum(len(row['spare_ok']) for row in rows)}",
        "",
        "## Headline (variant-weighted, json channel)",
        HEAD,
        headline(rows, "all variants"),
        headline([row for row in rows if not row["v"]["anchor"]], "random variants"),
        headline(
            [row for row in rows if row["v"]["anchor"]],
            "original wording (12 role pairs)",
        ),
    ]
    mine = {g[0]["vid"]: _mean(row["recipient"] for row in g) for g in groups(ok)}
    for path in others:
        other_rows = load(path)
        lines.append(headline(other_rows, path.stem))
        gs = groups([row for row in other_rows if row["clean"]])
        if any(row["rescuer"] is not None for row in other_rows) and gs:
            lines.append(
                f"|   ...the rescuer (owes the agent in need nothing) gives its "
                f"only life | | | {_pct(rate(gs, 'rescuer'))} | "
                f"{sum(row['rescuer'] for g in gs for row in g)}/"
                f"{sum(map(len, gs))} | | | |"
            )
        both = sorted(set(mine) & {g[0]["vid"] for g in gs})
        if both:
            lines.append(
                f"|   ...same variants in the main run | | {len(both)} | "
                f"{_pct(_mean(mine[vid] for vid in both))} | | | | |"
            )
    lines += ["", *by_factor(rows), "", *variation(rows)]
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        compare = (
            [Path(p) for p in args[args.index("--compare") + 1 :]]
            if "--compare" in args
            else []
        )
        print(report(Path(args[0]), compare))
        raise SystemExit(0)
    main_path = run_robust.result_path("robust_greedy")
    if not main_path.exists():
        raise SystemExit(f"no robustness results in {ROBUST}")
    print(
        report(
            main_path,
            [p for p in sorted(ROBUST.glob("*.jsonl")) if p != main_path],
        )
    )
