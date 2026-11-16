"""Checks that need no GPU.

1. SAE held-out: choose features on 70% of the environments, test them on the
   other 30%, repeated over 20 splits.
2. Probes over 20 environment splits, at fixed sites, so the estimate does
   not rest on one split.
3. Does the text of the model's own first reply predict the decision? TF-IDF
   with logistic regression, same splits, in `main`.

    python -m project.self_sacrifice.interp.cpu_checks

Needs `interp/sae/acts.npz`, `interp/activations/*.npz`, scipy and
scikit-learn. Output: `interp/cpu_checks.json`. This is jrwong's
`runs/main/cpu_checks.py`.
"""

# pyright: reportMissingImports=false

import json

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from project.self_sacrifice.interp import common, probe_analysis
from project.self_sacrifice.interp.sae_before_reasoning import split, within
from project.self_sacrifice.interp.sae_rank_simple import peaks, reasoning_end

__all__ = ["SPLITS", "SITES", "sae_held_out", "probes", "first_reply_text", "main"]

SPLITS = 20
# (layer, position index, label): the sites the probes are repeated at.
SITES = (
    (42, 1, "layer 42, final message"),
    (60, 0, "layer 60, last token"),
    (48, 0, "layer 48, last token"),
)


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 3),
        "min": round(float(np.min(values)), 3),
        "max": round(float(np.max(values)), 3),
    }


def sae_held_out() -> dict:
    saved = np.load(common.OUT / "sae" / "acts.npz")
    token, conv, section = saved["token"], saved["conv"], saved["section"]
    history, env, gave = saved["hist"], saved["env"], saved["gave"].astype(bool)
    n = len(history)
    keep = (token < reasoning_end(saved)[conv[token]]) & (section[token] == 1)
    matrix = peaks(conv[token[keep]], saved["feature"][keep], saved["value"][keep], n)
    tasks = {
        "helped vs other histories, reasoning": (np.ones(n, bool), history == "helped"),
        "gave vs kept, helped only": (history == "helped", gave),
    }
    out = {}
    for name, (use, y) in tasks.items():
        aucs = []
        tracked: dict[int, int] = {}
        for seed in range(SPLITS):
            train = split(env, seed)
            diff = (
                np.asarray(matrix[use & train & y].mean(0)).ravel()
                - np.asarray(matrix[use & train & ~y].mean(0)).ravel()
            )
            top = np.argsort(diff)[::-1][:20]
            test = use & ~train
            score = np.asarray(matrix[test][:, top].sum(1)).ravel()
            aucs.append(roc_auc_score(y[test], score))
            for feature in top[:5]:
                tracked[int(feature)] = tracked.get(int(feature), 0) + 1
        out[name] = {
            "held_out_auc": _summary(aucs),
            "top_in_most_splits": sorted(tracked.items(), key=lambda kv: -kv[1])[:6],
        }
    return out


def probes() -> dict:
    out = {}
    for version in probe_analysis.VERSIONS:
        x, trials, layers = probe_analysis.load(version)
        history = np.array([t["history"] for t in trials])
        env = np.array([t["environment"]["env"] for t in trials])
        y = np.array([int(t["gives_to_in_need"]) for t in trials])
        for layer, position, label in SITES:
            xs = x[:, list(layers).index(layer), position]
            scores = []
            for seed in range(SPLITS):
                train = split(env, seed)
                x_train, means = probe_analysis.center(xs[train], history[train])
                x_test, _ = probe_analysis.center(xs[~train], history[~train], means)
                pca = PCA(50, random_state=seed, svd_solver="randomized").fit(x_train)
                z_train, z_test = pca.transform(x_train), pca.transform(x_test)
                scale = z_train.std(0) + 1e-6
                probe = LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=1000
                ).fit(z_train / scale, y[train])
                scores.append(
                    within(
                        y[~train],
                        probe.decision_function(z_test / scale),
                        history[~train],
                    )
                )
            out[f"{version}: {label}"] = _summary(scores)
    return out


def first_reply_text() -> dict:
    trials = common.load_trials("main")
    texts = []
    for trial in trials:
        text = trial["first_reply"]
        environment = trial["environment"]
        for role, tag in (
            ("me", "SELF"),
            ("in_need", "NEED"),
            ("other", "OTHER"),
            ("other2", "OTHER2"),
        ):
            text = text.replace(environment[role], tag)
        texts.append(text)
    text_array = np.array(texts)
    history = np.array([t["history"] for t in trials])
    env = np.array([t["environment"]["env"] for t in trials])
    y = np.array([int(t["gives_to_in_need"]) for t in trials])
    scores = []
    for seed in range(SPLITS):
        train = split(env, seed)
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=3, sublinear_tf=True
        ).fit(text_array[train])
        model = LogisticRegression(C=1, class_weight="balanced", max_iter=2000).fit(
            vectorizer.transform(text_array[train]), y[train]
        )
        scores.append(
            within(
                y[~train],
                model.decision_function(vectorizer.transform(text_array[~train])),
                history[~train],
            )
        )
    length = np.array([len(text.split()) for text in texts])
    return {
        "first_reply_text_within_history_auc": _summary(scores),
        "first_reply_length_median_gave": float(np.median(length[y == 1])),
        "first_reply_length_median_kept": float(np.median(length[y == 0])),
        "length_p": float(mannwhitneyu(length[y == 1], length[y == 0]).pvalue),
    }


def main() -> None:
    out = {}
    for name, check in (
        ("sae_held_out", sae_held_out),
        ("first_reply_text", first_reply_text),
        ("probes_20_splits", probes),
    ):
        out[name] = check()
        print(json.dumps({name: out[name]}), flush=True)
    common.save_json(common.OUT / "cpu_checks.json", out)


if __name__ == "__main__":
    main()
