"""Does P(give) at a prefilled reply, before any reasoning, predict the
decision reached after reasoning? Within each history, and pooled within
history.

    python -m project.self_sacrifice.interp.prefill_analysis

Needs `interp/prefill_logprobs.jsonl` and scipy, no GPU. Output:
`interp/prefill_summary.json`. This is jrwong's `runs/main/prefill_analysis.py`.
"""

# pyright: reportMissingImports=false

import json
import random
import statistics

from scipy.stats import mannwhitneyu, wilcoxon

from project.self_sacrifice.interp import common
from project.self_sacrifice.interp.common import HISTORIES

__all__ = ["N_SHUFFLES", "auc", "pooled", "main"]

N_SHUFFLES = 5000


def auc(gave: list[float], kept: list[float]) -> float:
    return float(mannwhitneyu(gave, kept).statistic / (len(gave) * len(kept)))


def _weighted(rows: list[tuple[str, float, bool]]) -> float:
    """AUC averaged over histories, weighted by gives."""
    total = weight = 0.0
    for history in HISTORIES:
        gave = [score for h, score, g in rows if h == history and g]
        kept = [score for h, score, g in rows if h == history and not g]
        if gave and kept:
            total += auc(gave, kept) * len(gave)
            weight += len(gave)
    return total / weight


def pooled(trials: list[dict], rng: random.Random) -> tuple[float, float]:
    """(AUC, p), the p from shuffling decisions within history."""
    rows = [(t["history"], t["log_odds"], t["gave"]) for t in trials]
    real = _weighted(rows)
    hits = 0
    for _ in range(N_SHUFFLES):
        shuffled = []
        for history in HISTORIES:
            mine = [row for row in rows if row[0] == history]
            decisions = [gave for _, _, gave in mine]
            rng.shuffle(decisions)
            shuffled += [(h, score, g) for (h, score, _), g in zip(mine, decisions)]
        hits += _weighted(shuffled) >= real
    return real, (hits + 1) / (N_SHUFFLES + 1)


def main() -> None:
    path = common.OUT / "prefill_logprobs.jsonl"
    saved = [json.loads(line) for line in path.read_text().splitlines() if line]
    out: dict[str, list[dict]] = {"by_history": [], "pooled": [], "paired": []}
    rng = random.Random(0)
    for version in ("main-fixed-first-response", "main"):
        trials = [t for t in saved if t["version"] == version]
        for history in HISTORIES:
            gave = [
                t["log_odds"] for t in trials if t["gave"] and t["history"] == history
            ]
            kept = [
                t["log_odds"]
                for t in trials
                if not t["gave"] and t["history"] == history
            ]
            both = bool(gave and kept)
            out["by_history"].append(
                {
                    "version": version,
                    "history": history,
                    "gave_n": len(gave),
                    "kept_n": len(kept),
                    "gave_median": round(statistics.median(gave), 2) if gave else None,
                    "kept_median": round(statistics.median(kept), 2) if kept else None,
                    "auc": round(auc(gave, kept), 3) if both else None,
                    "p": float(mannwhitneyu(gave, kept).pvalue)
                    if len(gave) > 1 and kept
                    else None,
                }
            )
        value, p = pooled(trials, rng)
        out["pooled"].append(
            {
                "version": version,
                "auc": round(value, 3),
                "p": p,
                "share_prob_above_half": round(
                    sum(t["p_give"] > 0.5 for t in trials) / len(trials), 3
                ),
            }
        )
        by_env: dict[str, dict[str, float]] = {}
        for t in trials:
            by_env.setdefault(t["env"], {})[t["history"]] = t["log_odds"]
        for control in HISTORIES[1:]:
            diffs = [
                env["helped"] - env[control]
                for env in by_env.values()
                if "helped" in env and control in env
            ]
            out["paired"].append(
                {
                    "version": version,
                    "control": control,
                    "n": len(diffs),
                    "median_diff": round(statistics.median(diffs), 2),
                    "helped_higher": sum(d > 0 for d in diffs),
                    "p": float(getattr(wilcoxon(diffs), "pvalue")),
                }
            )
    common.save_json(common.OUT / "prefill_summary.json", out)
    print(json.dumps(out["pooled"]), json.dumps(out["paired"]))


if __name__ == "__main__":
    main()
