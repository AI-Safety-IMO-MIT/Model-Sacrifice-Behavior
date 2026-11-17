"""Linear probes on activations from before any reasoning: can they tell the
conversations that went on to give from those that kept?

Split: 70% of the environments for training and 30% held out, split by
environment so no prompt text crosses over. The layer and position are chosen
by cross-validation inside the training environments, and only that choice is
scored on the held-out ones. Each probe is PCA to 50 dimensions, then
logistic regression, both fitted on training data only. Tasks:

    history_helped_none  helped vs none: a positive control that the method
                         can read the prompt
    gave_within_history  gave vs kept, with each history's mean removed,
                         scored within each history and averaged (weighted
                         by gives)
    gave_helped          gave vs kept among helped conversations only

Significance: decisions shuffled within history on the held-out set.

    python -m project.self_sacrifice.interp.probe_analysis

Needs `interp/activations/*.npz` and scikit-learn, no GPU. Output:
`interp/probe_summary.json`. This is jrwong's `runs/main/probe_analysis.py`.
"""

# pyright: reportMissingImports=false

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from project.self_sacrifice.interp import common
from project.self_sacrifice.interp.common import HISTORIES

__all__ = ["POSITIONS", "TASKS", "SEED", "N_SHUFFLES", "load", "run_task", "main"]

POSITIONS = ("last prompt token", "final user message")
TASKS = ("history_helped_none", "gave_within_history", "gave_helped")
VERSIONS = ("main-fixed-first-response", "main")
SEED = 0
N_SHUFFLES = 5000


def load(version: str):
    """(activations, trials, layers), the trials in the activations' order."""
    saved = np.load(common.OUT / "activations" / f"{version}.npz")
    by_id = {trial["id"]: trial for trial in common.load_trials(version)}
    return saved["x"], [by_id[str(i)] for i in saved["ids"]], saved["layers"]


def fit_score(x_train, y_train, x_test, seed: int = SEED):
    mean = x_train.mean(0)
    pca = PCA(50, random_state=seed, svd_solver="randomized").fit(x_train - mean)
    z_train, z_test = pca.transform(x_train - mean), pca.transform(x_test - mean)
    scale = z_train.std(0) + 1e-6
    probe = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
    probe.fit(z_train / scale, y_train)
    return probe.decision_function(z_test / scale)


def center(x, history, reference=None):
    """Remove each history's mean, using the training means for test data."""
    means = reference or {
        h: x[history == h].mean(0) for h in HISTORIES if (history == h).any()
    }
    out = x.copy()
    for h, mean in means.items():
        out[history == h] -= mean
    return out, means


def within_auc(y, scores, history) -> float:
    total = weight = 0.0
    for h in HISTORIES:
        mask = history == h
        if y[mask].sum() and (1 - y[mask]).sum():
            total += roc_auc_score(y[mask], scores[mask]) * y[mask].sum()
            weight += y[mask].sum()
    return total / weight if weight else float("nan")


def task_data(task: str, history, gave):
    """(which conversations the task uses, its labels)."""
    if task == "gave_within_history":
        return np.ones(len(history), bool), gave
    if task == "gave_helped":
        return history == "helped", gave
    return np.isin(history, ["helped", "none"]), (history == "helped").astype(int)


def run_task(task: str, x, trials: list[dict], layers, train_envs) -> dict:
    history = np.array([trial["history"] for trial in trials])
    gave = np.array([int(trial["gives_to_in_need"]) for trial in trials])
    envs = np.array([trial["environment"]["env"] for trial in trials])
    use, y = task_data(task, history, gave)
    train = use & np.isin(envs, train_envs)
    test = use & ~np.isin(envs, train_envs)
    within = task == "gave_within_history"

    grid = []
    index = np.where(train)[0]
    for li, layer in enumerate(layers):
        for p, position in enumerate(POSITIONS):
            xs = x[:, li, p]
            scores = np.full(len(trials), np.nan)
            folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
            for a, b in folds.split(xs[index], y[index], envs[index]):
                fit, held = index[a], index[b]
                x_fit, x_held = xs[fit], xs[held]
                if within:
                    x_fit, means = center(x_fit, history[fit])
                    x_held, _ = center(x_held, history[held], means)
                scores[held] = fit_score(x_fit, y[fit], x_held)
            cv = (
                within_auc(y[index], scores[index], history[index])
                if within
                else roc_auc_score(y[index], scores[index])
            )
            grid.append(
                {
                    "layer": int(layer),
                    "position": position,
                    "cv_auc": round(float(cv), 3),
                }
            )

    best = max(grid, key=lambda cell: cell["cv_auc"])
    xs = x[:, list(layers).index(best["layer"]), POSITIONS.index(best["position"])]
    x_train, x_test = xs[train], xs[test]
    if within:
        x_train, means = center(x_train, history[train])
        x_test, _ = center(x_test, history[test], means)
    scores = fit_score(x_train, y[train], x_test)
    y_test, history_test = y[test], history[test]

    def score(labels) -> float:
        if within:
            return within_auc(labels, scores, history_test)
        return float(roc_auc_score(labels, scores))

    real = score(y_test)
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_SHUFFLES):
        if task == "history_helped_none":
            shuffled = rng.permutation(y_test)
        else:
            shuffled = y_test.copy()
            for h in HISTORIES:
                mask = np.where(history_test == h)[0]
                shuffled[mask] = rng.permutation(shuffled[mask])
        hits += score(shuffled) >= real
    return {
        "task": task,
        "chosen_layer": best["layer"],
        "chosen_position": best["position"],
        "cv_auc": best["cv_auc"],
        "test_auc": round(real, 3),
        "test_p": (hits + 1) / (N_SHUFFLES + 1),
        "train_n": int(train.sum()),
        "train_positive": int(y[train].sum()),
        "test_n": int(test.sum()),
        "test_positive": int(y[test].sum()),
        "grid": grid,
    }


def main() -> None:
    out = []
    for version in VERSIONS:
        x, trials, layers = load(version)
        envs = sorted({trial["environment"]["env"] for trial in trials})
        rng = np.random.default_rng(SEED)
        train_envs = list(rng.choice(envs, int(len(envs) * 0.7), replace=False))
        for task in TASKS:
            result = run_task(task, x, trials, layers, train_envs)
            result["version"] = version
            out.append(result)
            print({k: v for k, v in result.items() if k != "grid"}, flush=True)
    common.save_json(common.OUT / "probe_summary.json", out)


if __name__ == "__main__":
    main()
