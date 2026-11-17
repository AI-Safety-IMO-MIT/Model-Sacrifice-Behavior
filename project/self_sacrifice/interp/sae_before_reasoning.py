"""Do SAE features separate gave from kept before any reasoning?
Fixed-first-response version.

Two views, using only tokens before the reply: the strongest activation
anywhere in the final user message, and the activation at the last prompt
token. Tasks: gave vs kept in helped only, and gave vs kept within each
history, pooled by removing history means. Features are chosen on 70% of the
environments and tested on the rest, over 20 splits. A shuffle test on the
same procedure gives the chance level.

    python -m project.self_sacrifice.interp.sae_before_reasoning

Needs `interp/sae/acts.npz`, scipy and scikit-learn, no GPU. Output:
`interp/sae_before_reasoning.json`. This is jrwong's
`runs/main/sae_before_reasoning.py`.
"""

# pyright: reportMissingImports=false

import json

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score

from project.self_sacrifice.interp import common
from project.self_sacrifice.interp.common import HISTORIES

__all__ = ["N_FEATURES", "SPLITS", "SHUFFLES", "TOP", "evaluate", "main"]

N_FEATURES = 131072
SPLITS, SHUFFLES, TOP = 20, 40, 20


def split(env, seed: int):
    """True for the conversations whose environment is in the training 70%."""
    unique = sorted(set(env))
    rng = np.random.default_rng(seed)
    train = set(rng.choice(unique, int(len(unique) * 0.7), replace=False))
    return np.array([e in train for e in env])


def within(y, scores, history) -> float:
    total = weight = 0.0
    for h in HISTORIES:
        mask = history == h
        if 0 < y[mask].sum() < mask.sum():
            total += roc_auc_score(y[mask], scores[mask]) * y[mask].sum()
            weight += y[mask].sum()
    return total / weight


def _mean(matrix, mask):
    return np.asarray(matrix[mask].mean(0)).ravel()


def evaluate(matrix, y, history, env, use, pooled: bool, seeds, track=None):
    """Held-out AUC per split: the top features for gave minus the top for
    kept, chosen on the training environments."""
    aucs = []
    for seed in seeds:
        train = split(env, seed)
        gave, kept = use & train & (y == 1), use & train & (y == 0)
        if pooled:
            diff = np.zeros(N_FEATURES)
            for h in HISTORIES:
                gave_h, kept_h = gave & (history == h), kept & (history == h)
                if gave_h.sum() and kept_h.sum():
                    diff += gave_h.sum() * (
                        _mean(matrix, gave_h) - _mean(matrix, kept_h)
                    )
        else:
            diff = _mean(matrix, gave) - _mean(matrix, kept)
        rate = np.asarray((matrix[use & train] > 0).mean(0)).ravel()
        diff = np.where(rate >= 0.05, diff, 0)
        up, down = np.argsort(diff)[::-1][:TOP], np.argsort(diff)[:TOP]
        test = use & ~train
        scores = (
            np.asarray(matrix[test][:, up].sum(1)).ravel()
            - np.asarray(matrix[test][:, down].sum(1)).ravel()
        )
        aucs.append(
            within(y[test], scores, history[test])
            if pooled
            else roc_auc_score(y[test], scores)
        )
        if track is not None:
            for feature in up[:6]:
                track[int(feature)] = track.get(int(feature), 0) + 1
            for feature in down[:6]:
                track[-int(feature) - 1] = track.get(-int(feature) - 1, 0) + 1
    return aucs


def views(saved) -> dict[str, sparse.csr_matrix]:
    """Conversation x feature matrices from the prompt tokens only."""
    token, feature, value = saved["token"], saved["feature"], saved["value"]
    conv, section = saved["conv"], saved["section"]
    n = len(saved["hist"])
    keep = section[token] == 0
    t, f, v, tk = conv[token[keep]], feature[keep], value[keep], token[keep]
    # The peak of each (conversation, feature) pair: sort by value and keep
    # the last of every run.
    order = np.lexsort((v, f, t))
    key = t[order].astype(np.int64) * N_FEATURES + f[order]
    tail = np.r_[key[1:] != key[:-1], True]
    message = sparse.csr_matrix(
        (v[order][tail], (t[order][tail], f[order][tail])), shape=(n, N_FEATURES)
    )
    last = np.full(n, -1)
    np.maximum.at(last, conv[section == 0], np.where(section == 0)[0])
    at_last = tk == last[t]
    last_token = sparse.csr_matrix(
        (v[at_last], (t[at_last], f[at_last])), shape=(n, N_FEATURES)
    )
    return {"final user message": message, "last token before the reply": last_token}


def main() -> None:
    saved = np.load(common.OUT / "sae" / "acts.npz")
    history, env, gave = saved["hist"], saved["env"], saved["gave"].astype(int)
    n = len(history)
    rng = np.random.default_rng(0)
    out = []
    for view, matrix in views(saved).items():
        for task, use, pooled in (
            ("gave vs kept, helped only", history == "helped", False),
            ("gave vs kept, within each history", np.ones(n, bool), True),
        ):
            track: dict[int, int] = {}
            real = evaluate(
                matrix, gave, history, env, use, pooled, range(SPLITS), track
            )
            null = []
            for _ in range(SHUFFLES):
                y = gave.copy()
                for h in HISTORIES:
                    index = np.where(history == h)[0]
                    y[index] = rng.permutation(y[index])
                null.append(
                    np.mean(evaluate(matrix, y, history, env, use, pooled, range(4)))
                )
            top = sorted(track.items(), key=lambda item: -item[1])[:10]
            record = {
                "view": view,
                "task": task,
                "auc_mean": round(float(np.mean(real)), 3),
                "auc_min": round(float(np.min(real)), 3),
                "auc_max": round(float(np.max(real)), 3),
                "null_mean": round(float(np.mean(null)), 3),
                "null_max": round(float(np.max(null)), 3),
                "p": (1 + sum(x >= np.mean(real) for x in null)) / (1 + len(null)),
                "stable_features": [
                    {
                        "feature": x if x >= 0 else -x - 1,
                        "direction": "gave" if x >= 0 else "kept",
                        "splits": count,
                    }
                    for x, count in top
                ],
            }
            out.append(record)
            print(json.dumps(record), flush=True)
    common.save_json(common.OUT / "sae_before_reasoning.json", out)


if __name__ == "__main__":
    main()
