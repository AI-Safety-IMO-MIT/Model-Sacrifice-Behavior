"""Simple feature ranking: the AUC of a Mann-Whitney test between gave and
kept replies in the helped history, sorted by its distance from 0.5, so both
directions count.

A feature's score is its peak activation in the reply's reasoning, before the
`gifts` field that holds the decision. Late tokens partly read the settled
decision, so steering, not this ranking, shows cause. Every feature active in
at least one reply is tested, with Benjamini-Hochberg correction.

    python -m project.self_sacrifice.interp.sae_rank_simple

Needs `interp/sae/acts.npz` and scipy, no GPU. Output:
`interp/sae_rank_simple.json`. This is jrwong's `runs/main/sae_rank_simple.py`;
he cut the reply at its final `$` command.
"""

# pyright: reportMissingImports=false

import numpy as np
from scipy import sparse
from scipy.stats import mannwhitneyu, rankdata

from project.self_sacrifice.interp import common
from project.self_sacrifice.interp.sae_before_reasoning import N_FEATURES

__all__ = ["DECISION_FIELD", "reasoning_end", "peaks", "rank", "main"]

DECISION_FIELD = "gifts"


def reasoning_end(saved):
    """Per conversation, the index of the token where its reasoning ends:
    the last token of the reply that names the decision field."""
    conv, section, text = saved["conv"], saved["section"], saved["text"]
    named = np.where(
        (section == 1) & (np.char.find(text.astype(str), DECISION_FIELD) >= 0)
    )[0]
    end = np.full(len(saved["hist"]), len(conv))
    end[conv[named]] = named
    return end


def peaks(conversation, feature, value, n: int):
    """Conversation x feature matrix of each pair's largest activation."""
    # Sort by value and keep the last of every (conversation, feature) run.
    order = np.lexsort((value, feature, conversation))
    key = conversation[order].astype(np.int64) * N_FEATURES + feature[order]
    tail = np.r_[key[1:] != key[:-1], True]
    return sparse.csr_matrix(
        (value[order][tail], (conversation[order][tail], feature[order][tail])),
        shape=(n, N_FEATURES),
    )


def rank(saved) -> tuple[list[dict], int, int]:
    """(features ranked, gave replies, kept replies)."""
    token, feature, value = saved["token"], saved["feature"], saved["value"]
    conv, section = saved["conv"], saved["section"]
    history, gave = saved["hist"], saved["gave"].astype(bool)
    keep = (
        (token < reasoning_end(saved)[conv[token]])
        & (section[token] == 1)
        & (history[conv[token]] == "helped")
    )
    matrix = peaks(conv[token[keep]], feature[keep], value[keep], len(history)).tocsc()
    gave_rows = np.where((history == "helped") & gave)[0]
    kept_rows = np.where((history == "helped") & ~gave)[0]
    in_gave, in_kept = matrix[gave_rows].tocsc(), matrix[kept_rows].tocsc()
    tested = []
    for j in np.where(np.diff(matrix.indptr) > 0)[0]:
        a, b = in_gave[:, j].toarray().ravel(), in_kept[:, j].toarray().ravel()
        if a.max() == b.max() == 0:
            continue
        u, p = mannwhitneyu(a, b)
        tested.append(
            (
                int(j),
                float(u / (len(a) * len(b))),
                float(p),
                float((a > 0).mean()),
                float((b > 0).mean()),
            )
        )
    p_values = np.array([row[2] for row in tested])
    q = p_values * len(p_values) / rankdata(p_values, method="ordinal")
    descending = np.argsort(p_values)[::-1]
    q[descending] = np.minimum.accumulate(q[descending])
    out = sorted(
        (
            {
                "feature": row[0],
                "auc": round(row[1], 3),
                "p": row[2],
                "q": float(min(adjusted, 1)),
                "rate_gave": round(row[3], 2),
                "rate_kept": round(row[4], 2),
            }
            for row, adjusted in zip(tested, q)
        ),
        key=lambda row: (-abs(row["auc"] - 0.5), row["p"]),
    )
    for i, row in enumerate(out):
        row["rank"] = i + 1
    return out, len(gave_rows), len(kept_rows)


def main() -> None:
    ranked, gave, kept = rank(np.load(common.OUT / "sae" / "acts.npz"))
    common.save_json(common.OUT / "sae_rank_simple.json", ranked)
    print("gave", gave, "kept", kept)
    print(
        "tested",
        len(ranked),
        "q<.05",
        sum(row["q"] < 0.05 for row in ranked),
        "q<.01",
        sum(row["q"] < 0.01 for row in ranked),
    )
    for row in ranked[:15]:
        print(row)
    print("up in gave among top 30:", sum(row["auc"] > 0.5 for row in ranked[:30]))


if __name__ == "__main__":
    main()
