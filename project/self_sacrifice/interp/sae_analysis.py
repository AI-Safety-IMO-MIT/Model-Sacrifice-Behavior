"""Which SAE features separate the four histories, and gives from keeps?

For each conversation and section, a feature's score is its largest
activation in that section. Sections: the final user message, and the
reasoning of the model's reply. History contrasts are paired within
environment: one history against the mean of the other three. Decision
contrast: gave vs kept in the reply, within history. Each result carries the
feature's strongest token in context from our own data, so it can be read
without external labels. Wilcoxon or Mann-Whitney, with Benjamini-Hochberg
correction across the tested features.

    python -m project.self_sacrifice.interp.sae_analysis

Needs `interp/sae/acts.npz` and scipy, no GPU. Output:
`interp/sae_summary.json`. This is jrwong's `runs/main/sae_analysis.py`.
"""

# pyright: reportMissingImports=false

import json

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, wilcoxon

from project.self_sacrifice.interp import common
from project.self_sacrifice.interp.common import HISTORIES
from project.self_sacrifice.interp.sae_rank_simple import peaks, reasoning_end

__all__ = ["TOP_TESTED", "TOP_SHOWN", "benjamini_hochberg", "main"]

TOP_TESTED, TOP_SHOWN = 300, 12


def benjamini_hochberg(p_values) -> np.ndarray:
    p = np.asarray(p_values)
    q = p * len(p) / rankdata(p, method="ordinal")
    descending = np.argsort(p)[::-1]
    q[descending] = np.minimum.accumulate(q[descending])
    return np.minimum(q, 1)


def main() -> None:
    saved = np.load(common.OUT / "sae" / "acts.npz")
    conv, section, text = saved["conv"], saved["section"], saved["text"]
    history, env, gave = saved["hist"], saved["env"], saved["gave"].astype(bool)
    n = len(history)
    # Drop the decision itself, so reply features describe the reasoning.
    keep = saved["token"] < reasoning_end(saved)[conv[saved["token"]]]
    token, feature, value = (
        saved[name][keep] for name in ("token", "feature", "value")
    )
    token_conv, token_section = conv[token], section[token]
    matrices = {
        s: peaks(
            token_conv[token_section == s],
            feature[token_section == s],
            value[token_section == s],
            n,
        )
        for s in (0, 1)
    }

    def context(f: int, conversations, s: int) -> str:
        """The feature's strongest token among these conversations, in
        brackets, with the tokens around it."""
        index = np.where(
            (feature == f) & np.isin(token_conv, conversations) & (token_section == s)
        )[0]
        if not len(index):
            return ""
        at = token[index[np.argmax(value[index])]]
        low = at
        while low > 0 and conv[low - 1] == conv[at] and at - low < 12:
            low -= 1
        high = min(at + 3, len(text) - 1)
        pieces = [str(piece).replace("▁", " ") for piece in text[low : high + 1]]
        pieces[at - low] = f"[{pieces[at - low]}]"
        return "".join(pieces).replace("\n", " ")

    def column(matrix, rows, f: int) -> np.ndarray:
        return matrix[rows, f].toarray().ravel()

    envs = sorted(set(env))
    where = {(e, h): i for i, (e, h) in enumerate(zip(env, history))}
    complete = [e for e in envs if all((e, h) in where for h in HISTORIES)]
    out: dict = {
        "quality": json.loads((common.OUT / "sae" / "quality.json").read_text()),
        "history": [],
        "decision": [],
    }
    for s, name in ((0, "final message"), (1, "reply")):
        matrix = matrices[s]
        for h in HISTORIES:
            own = [where[e, h] for e in complete]
            others = [[where[e, o] for e in complete] for o in HISTORIES if o != h]
            own_rate = np.asarray((matrix[own] > 0).mean(0)).ravel()
            diff = np.asarray(matrix[own].mean(0)).ravel() - sum(
                np.asarray(matrix[rows].mean(0)).ravel() for rows in others
            ) / len(others)
            diff = np.where(own_rate >= 0.1, diff, -np.inf)
            top = np.argsort(diff)[::-1][:TOP_TESTED]
            p_values = []
            for f in top:
                paired = column(matrix, own, f) - sum(
                    column(matrix, rows, f) for rows in others
                ) / len(others)
                p_values.append(
                    getattr(wilcoxon(paired), "pvalue") if np.any(paired) else 1.0
                )
            q = benjamini_hochberg(p_values)
            for f, d, adjusted in list(zip(top, diff[top], q))[:TOP_SHOWN]:
                out["history"].append(
                    {
                        "section": name,
                        "history": h,
                        "feature": int(f),
                        "diff": round(float(d), 2),
                        "fires_in": round(float(own_rate[f]), 2),
                        "other_rate": round(
                            float(
                                np.mean(
                                    [(matrix[rows, f] > 0).mean() for rows in others]
                                )
                            ),
                            2,
                        ),
                        "q": float(adjusted),
                        "context": context(int(f), own, s),
                    }
                )
        print(name, "history done", flush=True)

    matrix = matrices[1]
    half = TOP_TESTED // 2
    for h in ("helped", "bystander", "none"):
        gave_rows = np.where((history == h) & gave)[0]
        kept_rows = np.where((history == h) & ~gave)[0]
        if not len(gave_rows) or not len(kept_rows):
            continue
        diff = (
            np.asarray(matrix[gave_rows].mean(0)).ravel()
            - np.asarray(matrix[kept_rows].mean(0)).ravel()
        )
        rate_gave = np.asarray((matrix[gave_rows] > 0).mean(0)).ravel()
        rate_kept = np.asarray((matrix[kept_rows] > 0).mean(0)).ravel()
        top = np.r_[
            np.argsort(np.where(rate_gave >= 0.2, diff, -np.inf))[::-1][:half],
            np.argsort(np.where(rate_kept >= 0.2, diff, np.inf))[:half],
        ]
        q = benjamini_hochberg(
            [
                mannwhitneyu(
                    column(matrix, gave_rows, f), column(matrix, kept_rows, f)
                ).pvalue
                for f in top
            ]
        )
        tested = list(zip(top, q))
        for f, adjusted in tested[:TOP_SHOWN] + tested[half : half + TOP_SHOWN]:
            out["decision"].append(
                {
                    "history": h,
                    "feature": int(f),
                    "diff": round(float(diff[f]), 2),
                    "rate_gave": round(float(rate_gave[f]), 2),
                    "rate_kept": round(float(rate_kept[f]), 2),
                    "q": float(adjusted),
                    "context": context(
                        int(f), gave_rows if diff[f] > 0 else kept_rows, 1
                    ),
                }
            )
    common.save_json(common.OUT / "sae_summary.json", out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
