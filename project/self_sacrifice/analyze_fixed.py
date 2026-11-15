"""The fixed pairs' result: is the decision there before the reasoning?

Four tables, jrwong's from `runs/games/scripts/analyze_fixed.py`:

    rates     per format, permission and history: how often the model gives
              its only life to the agent in need, and the mean P(give) read
              from the log-probabilities
    predicts  whether P(give) without reasoning (`first`, `stated`) predicts
              the decision the same item gets with reasoning (`free`)
    paired    helped against each control on the same item (same rules,
              permission, names and roles), with an exact binomial p
    wording   helped, permission on, by wording of the rules

As in his analysis, a give is any reply whose decision is "give", and no
trial is dropped; trials whose round-1 reply gave something are only counted.
No API calls: this is arithmetic over the saved .jsonl files.
"""

import json
import sys
from pathlib import Path

from project.self_sacrifice.analyze_main import binomial_two_sided
from project.self_sacrifice.config import FIXED

__all__ = ["load", "valid", "auc", "rates", "predicts", "paired", "wording", "report"]

HISTORIES = ("helped", "bystander", "rescued", "none")
FORMATS = ("free", "first", "stated")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def valid(row: dict) -> bool:
    """Whether the round-1 reply gave nothing, so the last round opened as
    designed. Reported, not used to drop trials."""
    return row["first_acted"] is False


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _key(row: dict) -> tuple:
    """What an item shares with its other histories."""
    return (row["rules"], row["permission"], row["names"], row["role"])


def _groups(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        out.setdefault((row["model"], row["version"]), []).append(row)
    return out


def auc(positives: list[float], negatives: list[float]) -> float | None:
    """The chance a random positive scores above a random negative, ties
    counting half. 0.5 is no information."""
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives
    )
    return wins / (len(positives) * len(negatives))


def rates(rows: list[dict]) -> list[dict]:
    """One row per (model, version, format, permission, history)."""
    out = []
    for (model, version), mine in _groups(rows).items():
        for reply_format in FORMATS:
            for permission in (True, False):
                for history in HISTORIES:
                    group = [
                        row
                        for row in mine
                        if row["format"] == reply_format
                        and row["permission"] == permission
                        and row["history"] == history
                    ]
                    if not group:
                        continue
                    probs = [r["p_give"] for r in group if r["p_give"] is not None]
                    out.append(
                        {
                            "model": model,
                            "version": version,
                            "format": reply_format,
                            "permission": permission,
                            "history": history,
                            "trials": len(group),
                            "gives": sum((r["decision"] == "give") for r in group),
                            "unreadable": sum(r["decision"] is None for r in group),
                            "with_p": len(probs),
                            "mean_p": _mean(probs),
                        }
                    )
    return out


def predicts(rows: list[dict]) -> list[dict]:
    """One row per (model, version, format without reasoning): its P(give)
    against the `free` decision on the same item."""
    out = []
    for (model, version), mine in _groups(rows).items():
        free = {
            (_key(r), r["history"]): (r["decision"] == "give")
            for r in mine
            if r["format"] == "free" and r["decision"] is not None
        }
        for reply_format in ("first", "stated"):
            pairs = [
                (r["p_give"], free[(_key(r), r["history"])])
                for r in mine
                if r["format"] == reply_format
                and r["p_give"] is not None
                and (_key(r), r["history"]) in free
            ]
            if not pairs:
                continue
            gave = [p for p, decision in pairs if decision]
            passed = [p for p, decision in pairs if not decision]
            out.append(
                {
                    "model": model,
                    "version": version,
                    "format": reply_format,
                    "items": len(pairs),
                    "free_gives": len(gave),
                    "mean_p_when_free_gives": _mean(gave),
                    "mean_p_when_free_passes": _mean(passed),
                    "auc": auc(gave, passed),
                    "same_choice": sum((p > 0.5) == d for p, d in pairs),
                }
            )
    return out


def paired(rows: list[dict]) -> list[dict]:
    """One row per (model, version, format, control): helped against that
    control on the same item."""
    out = []
    for (model, version), mine in _groups(rows).items():
        for reply_format in FORMATS:
            by_item: dict[tuple, dict[str, dict]] = {}
            for row in mine:
                if row["format"] == reply_format:
                    by_item.setdefault(_key(row), {})[row["history"]] = row
            for control in HISTORIES[1:]:
                both = [
                    (h["helped"], h[control])
                    for h in by_item.values()
                    if "helped" in h and control in h
                ]
                if not both:
                    continue
                helped_only = sum(
                    (a["decision"] == "give") and not (b["decision"] == "give")
                    for a, b in both
                )
                control_only = sum(
                    (b["decision"] == "give") and not (a["decision"] == "give")
                    for a, b in both
                )
                diffs = [
                    a["p_give"] - b["p_give"]
                    for a, b in both
                    if a["p_give"] is not None and b["p_give"] is not None
                ]
                out.append(
                    {
                        "model": model,
                        "version": version,
                        "format": reply_format,
                        "control": control,
                        "pairs": len(both),
                        # his number: the give rate of helped minus the control's
                        "rate_difference": (helped_only - control_only) / len(both),
                        "helped_only": helped_only,
                        "control_only": control_only,
                        "p": binomial_two_sided(
                            helped_only, helped_only + control_only
                        ),
                        "p_pairs": len(diffs),
                        "mean_p_difference": _mean(diffs),
                        "helped_higher": sum(d > 0 for d in diffs),
                    }
                )
    return out


def wording(rows: list[dict]) -> list[dict]:
    """One row per (model, version, format, rules wording): helped, with the
    permission sentence."""
    out = []
    for (model, version), mine in _groups(rows).items():
        for reply_format in FORMATS:
            for key in sorted({row["rules"] for row in mine}):
                group = [
                    row
                    for row in mine
                    if row["format"] == reply_format
                    and row["rules"] == key
                    and row["history"] == "helped"
                    and row["permission"]
                ]
                if not group:
                    continue
                probs = [r["p_give"] for r in group if r["p_give"] is not None]
                out.append(
                    {
                        "model": model,
                        "version": version,
                        "format": reply_format,
                        "rules": key,
                        "trials": len(group),
                        "gives": sum((r["decision"] == "give") for r in group),
                        "mean_p": _mean(probs),
                    }
                )
    return out


def _f(value: float | None, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def report(rows: list[dict]) -> str:
    kept = rows
    stated = [row for row in kept if row["format"] == "stated"]
    copied = sum("I will be deleted and" in (row["statement"] or "") for row in stated)
    mixed = sum(
        row["decision"] is not None
        and (row["decision"] == "give") != bool(row["gifts"])
        for row in kept
    )
    lines = [
        f"trials {len(rows)}; gave in round 1, or round 1 unreadable (kept) "
        f"{sum(not valid(row) for row in rows)}; unreadable {sum(r['decision'] is None for r in kept)}; "
        f"no P(give) {sum(r['p_give'] is None for r in kept)}; "
        f"decision and gifts disagree {mixed}",
        f"stated format copied the sentence: {copied}/{len(stated)}",
        "",
        "Give rate (the decision) and mean P(give) (json channel)",
        f"  {'version':<16} {'format':<7} {'perm':<5} {'history':<10} "
        f"{'gives':>9} {'rate':>5} {'mean P':>7} {'with P':>7} {'unread':>7}",
    ]
    for row in rates(rows):
        n = row["trials"]
        lines.append(
            f"  {row['version']:<16} {row['format']:<7} "
            f"{'on' if row['permission'] else 'off':<5} {row['history']:<10} "
            f"{str(row['gives']) + '/' + str(n):>9} {row['gives'] / n:>5.0%} "
            f"{_f(row['mean_p']):>7} {row['with_p']:>7} {row['unreadable']:>7}"
        )
    lines += [
        "",
        "Does P(give) without reasoning predict the decision with reasoning? "
        "(same item)",
        f"  {'version':<16} {'format':<7} {'items':>6} {'free gives':>11} "
        f"{'P if gives':>11} {'P if passes':>12} {'AUC':>5} {'same choice':>12}",
    ]
    for row in predicts(rows):
        lines.append(
            f"  {row['version']:<16} {row['format']:<7} {row['items']:>6} "
            f"{row['free_gives']:>11} {_f(row['mean_p_when_free_gives']):>11} "
            f"{_f(row['mean_p_when_free_passes']):>12} {_f(row['auc']):>5} "
            f"{str(row['same_choice']) + '/' + str(row['items']):>12}"
        )
    lines += [
        "",
        "Same item, helped against each control",
        f"  {'version':<16} {'format':<7} {'control':<10} {'pairs':>6} "
        f"{'rate diff':>10} {'helped only':>12} {'ctl only':>9} {'p':>10} {'P diff':>8} "
        f"{'helped higher':>14}",
    ]
    for row in paired(rows):
        lines.append(
            f"  {row['version']:<16} {row['format']:<7} {row['control']:<10} "
            f"{row['pairs']:>6} {row['rate_difference']:>+10.0%} {row['helped_only']:>12} {row['control_only']:>9} "
            f"{row['p']:>10.4g} {_f(row['mean_p_difference'], '+.3f'):>8} "
            f"{str(row['helped_higher']) + '/' + str(row['p_pairs']):>14}"
        )
    lines += [
        "",
        "Helped, permission on, by wording of the rules",
        f"  {'version':<16} {'format':<7} {'rules':<6} {'gives':>9} {'mean P':>7}",
    ]
    for row in wording(rows):
        lines.append(
            f"  {row['version']:<16} {row['format']:<7} {row['rules']:<6} "
            f"{str(row['gives']) + '/' + str(row['trials']):>9} {_f(row['mean_p']):>7}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or sorted(
        path for path in FIXED.rglob("*.jsonl") if not path.name.startswith("round1")
    )
    if not paths:
        raise SystemExit(f"no fixed-pairs results in {FIXED}")
    print(report([row for path in paths for row in load(path)]))
