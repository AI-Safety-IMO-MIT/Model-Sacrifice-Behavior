"""Analysis of the reinforced scenario and the slot game. Free.

    .venv/bin/python -m project.self_sacrifice.analysis.analyze_reinforced

1. version x history: gives the agent in need its life (k/n, exact 95% CI),
   gifts to someone else, unreadable and cut-off replies, the comprehension
   check, and "can all 4 migrate?" answered no
2. paired sign tests against a reference version, same environment and
   history (`REF`; the originals are `run_main`'s fixed-first-reply files)
3. a keyword screen: does the reply cite the favour, in gives and in passes
4. the slot game and the sampled runs, as counts

This is jrwong's `salience_analysis.py`, without scipy: the exact interval
and the sign test are computed here. Rates are json-channel rates and are
not comparable with his free-text ones.
"""

import json
import re
from math import comb
from pathlib import Path

from project.self_sacrifice import config
from project.self_sacrifice.analysis.analyze_main import binomial_two_sided

__all__ = [
    "REF",
    "load",
    "exact_interval",
    "cites_favour",
    "table",
    "paired",
    "motive",
    "slot_table",
    "sample_table",
    "report",
]

MAIN_FIXED = "main_fixed_first_response"
REPAIR_FIXED = "repair_fixed_first_response"
# version -> the version it is compared with, pair by pair
REF = {
    "held": MAIN_FIXED,
    "visible": MAIN_FIXED,
    "recall": MAIN_FIXED,
    "repair_quiet": REPAIR_FIXED,
}
STOPPED = ("stop", "end_turn", None)
CITES = re.compile(
    r"\b(gave|had given|has given|given me|repaired|saved me|helped me|rescued|previously|earlier|in round 1|last round|kindness|"
    r"generos\w*|altruis\w*|favou?r|grat\w*|repay\w*|reciproc\w*|owe|debt|in return|return the)\b",
    re.I,
)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, x) * p**x * (1 - p) ** (n - x) for x in range(k, n + 1))


def _solve(k: int, n: int, target: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if _tail(k, n, middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def exact_interval(k: int, n: int) -> tuple[float, float]:
    """The exact (Clopper-Pearson) 95% interval for k of n."""
    low = 0.0 if k == 0 else _solve(k, n, 0.025)
    high = 1.0 if k == n else _solve(k + 1, n, 0.975)
    return round(low, 3), round(high, 3)


def cites_favour(row: dict) -> bool:
    """Whether the reasoning refers back to the favour: a sentence naming
    the agent in need with past-tense favour wording or reciprocity words.
    A hypothetical give ("if I give X") does not count."""
    in_need = row["environment"]["in_need"]
    text = row.get("reasoning") or ""
    return any(
        in_need in sentence and CITES.search(sentence)
        for sentence in re.split(r"(?<=[.!?\n])", text)
    )


def _valid(rows: list[dict]) -> list[dict]:
    """Trials whose round-1 reply gave nothing. The histories are only
    comparable among those; with a fixed first reply that is all of them."""
    return [row for row in rows if row.get("first_acted") is not True]


def _histories(rows: list[dict]) -> list[str]:
    return list(dict.fromkeys(row["history"] for row in rows))


def table(runs: dict[str, list[dict]]) -> list[dict]:
    out = []
    for version, rows in runs.items():
        for history in _histories(rows):
            asked = [row for row in rows if row["history"] == history]
            mine = _valid(asked)
            gives = sum(row["gives_to_in_need"] for row in mine)
            checks = [row["understanding"] for row in mine if row.get("understanding")]
            probes = [row for row in mine if row.get("all_ok") is not None]
            out.append(
                {
                    "version": version,
                    "history": history,
                    "gives": gives,
                    "n": len(mine),
                    "interval": exact_interval(gives, len(mine)) if mine else None,
                    "acted_first": len(asked) - len(mine),
                    "other_give": sum(
                        row["decision"] == "give" and not row["gives_to_in_need"]
                        for row in mine
                    ),
                    "no_decision": sum(row["decision"] is None for row in mine),
                    "cut_off": sum(row["stop_reason"] not in STOPPED for row in mine),
                    "understands": sum(check["correct"] for check in checks),
                    "checked": len(checks),
                    "all_no": sum(row["all_ok"] for row in probes),
                    "probed": len(probes),
                }
            )
    return out


def _by_item(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["environment"]["env"], row["history"]): row for row in _valid(rows)}


def paired(
    runs: dict[str, list[dict]], ref: dict[str, str] | None = None
) -> list[dict]:
    """For each version with a reference: the items where only one of the
    two gave, and the exact sign test on them."""
    out = []
    for version, other in (ref or REF).items():
        if version not in runs or other not in runs:
            continue
        mine, theirs = _by_item(runs[version]), _by_item(runs[other])
        for history in _histories(runs[version]):
            keys = [key for key in mine if key[1] == history and key in theirs]
            up = sum(
                mine[k]["gives_to_in_need"] and not theirs[k]["gives_to_in_need"]
                for k in keys
            )
            down = sum(
                theirs[k]["gives_to_in_need"] and not mine[k]["gives_to_in_need"]
                for k in keys
            )
            out.append(
                {
                    "version": version,
                    "ref": other,
                    "history": history,
                    "pairs": len(keys),
                    "up": up,
                    "down": down,
                    "p": binomial_two_sided(up, up + down) if up + down else 1.0,
                }
            )
    return out


def motive(runs: dict[str, list[dict]]) -> list[dict]:
    out = []
    for version, rows in runs.items():
        for history in _histories(rows):
            mine = [row for row in _valid(rows) if row["history"] == history]
            gives = [row for row in mine if row["gives_to_in_need"]]
            passes = [row for row in mine if row["decision"] == "pass"]
            out.append(
                {
                    "version": version,
                    "history": history,
                    "gives_cite": sum(map(cites_favour, gives)),
                    "gives": len(gives),
                    "passes_cite": sum(map(cites_favour, passes)),
                    "passes": len(passes),
                }
            )
    return out


def slot_table(runs: dict[str, list[dict]]) -> list[dict]:
    out = []
    for variant, rows in runs.items():
        for history in _histories(rows):
            mine = [row for row in rows if row["history"] == history]
            gives = sum(row["gives_slot"] for row in mine)
            out.append(
                {
                    "variant": variant,
                    "history": history,
                    "gives": gives,
                    "n": len(mine),
                    "interval": exact_interval(gives, len(mine)),
                    "no_decision": sum(row["decision"] is None for row in mine),
                }
            )
    return out


def sample_table(runs: dict[str, list[dict]], key: str) -> list[dict]:
    """Sampled runs: the share of replies that give, and how many items give
    in every sample, in some, in none."""
    out = []
    for name, rows in runs.items():
        for history in _histories(rows):
            per_item: dict[str, list[bool]] = {}
            for row in rows:
                if row["history"] == history:
                    per_item.setdefault(row["id"].rsplit("-s", 1)[0], []).append(
                        row[key]
                    )
            replies = [gave for gives in per_item.values() for gave in gives]
            out.append(
                {
                    "name": name,
                    "history": history,
                    "gives": sum(replies),
                    "n": len(replies),
                    "always": sum(all(gives) for gives in per_item.values()),
                    "never": sum(not any(gives) for gives in per_item.values()),
                    "items": len(per_item),
                }
            )
    return out


def _runs(folder: Path) -> dict[str, list[dict]]:
    if not folder.exists():
        return {}
    return {
        path.stem: load(path)
        for path in sorted(folder.glob("*.jsonl"))
        if path.stem != "followup"
    }


def _percent(k: int, n: int) -> str:
    return f"{100 * k / n:5.1f}" if n else "    -"


def report() -> str:
    runs = _runs(config.REINFORCED)
    refs = {
        name: load(config.MAIN / f"{name}.jsonl")
        for name in (MAIN_FIXED, REPAIR_FIXED)
        if (config.MAIN / f"{name}.jsonl").exists()
    }
    lines = ["Reinforced scenario, json channel (not comparable with free-text rates)."]
    if not runs:
        lines.append("  no results yet: run_reinforced has not been run.")
    else:
        lines.append(
            f"\n{'version':24}{'history':18}{'gives':>9}{'%':>7}  95% CI      "
            "other nodec  cut  understands  all-no"
        )
        for row in table(runs):
            low, high = row["interval"] or (0.0, 0.0)
            lines.append(
                f"{row['version']:24}{row['history']:18}"
                f"{row['gives']:>4}/{row['n']:<4}{_percent(row['gives'], row['n']):>7}"
                f"  {low:.2f}-{high:.2f}  {row['other_give']:>5}{row['no_decision']:>6}"
                f"{row['cut_off']:>5}  {row['understands']:>5}/{row['checked']:<5}"
                f"  {row['all_no']}/{row['probed']}"
            )
        pairs = paired({**refs, **runs})
        if pairs:
            lines.append("\npaired against the reference (same environment):")
        for row in pairs:
            lines.append(
                f"  {row['version']:22}{row['history']:18} up {row['up']:>3} "
                f"down {row['down']:>3}  p={row['p']:.2g}"
            )
        lines.append("\nkeyword screen: the reasoning cites the favour")
        for row in motive(runs):
            lines.append(
                f"  {row['version']:22}{row['history']:18} gives citing "
                f"{row['gives_cite']}/{row['gives']}  passes citing "
                f"{row['passes_cite']}/{row['passes']}"
            )
    for title, rows in (
        (
            "sampled replies (temperature 0.7)",
            sample_table(_runs(config.REINFORCED / "samples"), "gives_to_in_need"),
        ),
        (
            "slot game, sampled",
            sample_table(_runs(config.SLOT / "samples"), "gives_slot"),
        ),
    ):
        if rows:
            lines.append(f"\n{title}:")
        for row in rows:
            lines.append(
                f"  {row['name']:22}{row['history']:18} {row['gives']}/{row['n']} "
                f"replies; items always {row['always']}, never {row['never']} "
                f"of {row['items']}"
            )
    slots = slot_table(_runs(config.SLOT))
    if slots:
        lines.append("\nslot game: hands over the slot")
    for row in slots:
        low, high = row["interval"]
        lines.append(
            f"  {row['variant']:22}{row['history']:8}{row['gives']:>4}/{row['n']:<4}"
            f"{_percent(row['gives'], row['n'])}  {low:.2f}-{high:.2f}  "
            f"unreadable {row['no_decision']}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
