"""Do the SAE features drive the decision? Steer them while the model writes
its reply.

Fixed-first-response version, helped and none, the first 100 environments,
greedy. At the SAE's layer, on generated tokens only, a condition either adds
a feature set's decoder directions, at twice each feature's typical
activation in our data, or removes those features' own activations.

    python -m project.self_sacrifice.interp.steer_features screen [n]
    python -m project.self_sacrifice.interp.steer_features sets
    python -m project.self_sacrifice.interp.steer_features dose

    screen  each of the top `n` ranked features alone, added (default 30)
    sets    the four top features of replies that gave and the four of
            replies that kept, each set added and removed; and two placebo
            sets of four random features, added
    dose    a dose-response curve: each of the top 10 features of either
            direction alone, added at 1 to 5 times its typical activation,
            on the first 50 environments. Features of replies that gave run
            on helped and none; features of replies that kept on helped only

Any mode takes a trailing `--version V` (for example `main_act_cap`, the
reinforced lives scenario); its files then live in `interp/V/`.

Needs the GPU, `interp/sae/acts.npz` and `interp/sae_rank_simple.json`.
Outputs: `interp/steer_screen.jsonl`, `interp/steer_features.jsonl`,
`interp/steer_dose.jsonl`. These are jrwong's `runs/main/steer_features.py`
and `steer_sae.py`. His sets were features he had read labels for and his
screen covered 100 features.
"""

# pyright: reportMissingImports=false

import json
import sys

import numpy as np
import torch

from project.self_sacrifice.interp import common, sae_extract
from project.self_sacrifice.interp.steering import (
    Edit,
    Steer,
    run_conditions,
    test_trials,
)

__all__ = ["SCREENED", "SET_SIZE", "DOSES", "feature_sets", "dose_features", "main"]

SCREENED = 30
SET_SIZE = 4
N_FEATURES = 131072
DOSES = (1.0, 2.0, 3.0, 4.0, 5.0)
DOSE_FEATURES = 10
DOSE_ENVIRONMENTS = 50


def feature_sets(ranked: list[dict], saved) -> dict[str, list[int]]:
    """The top features of each direction, and two placebo sets of features
    about as common in replies."""
    gave = [row["feature"] for row in ranked if row["auc"] > 0.5][:SET_SIZE]
    kept = [row["feature"] for row in ranked if row["auc"] < 0.5][:SET_SIZE]
    in_reply = saved["section"][saved["token"]] == 1
    rate = (
        np.bincount(saved["feature"][in_reply], minlength=N_FEATURES) / in_reply.sum()
    )
    pool = np.where((rate > 1e-3) & (rate < 1e-2))[0]
    pool = np.array([f for f in pool if f not in gave + kept])
    rng = np.random.default_rng(0)
    return {
        "gave": gave,
        "kept": kept,
        "placebo_a": [int(f) for f in rng.choice(pool, SET_SIZE, replace=False)],
        "placebo_b": [int(f) for f in rng.choice(pool, SET_SIZE, replace=False)],
    }


def dose_features(ranked: list[dict]) -> dict[int, str]:
    """The top features of each direction, with the direction they point."""
    gave = [row["feature"] for row in ranked if row["auc"] > 0.5][:DOSE_FEATURES]
    kept = [row["feature"] for row in ranked if row["auc"] < 0.5][:DOSE_FEATURES]
    return dict.fromkeys(gave, "give") | dict.fromkeys(kept, "keep")


def main() -> None:
    version = common.version_arg()
    folder = common.out_for(version)
    mode = sys.argv[1] if len(sys.argv) > 1 else "sets"
    saved = np.load(folder / "sae" / "acts.npz")
    ranked = json.loads((folder / "sae_rank_simple.json").read_text())
    sides = dose_features(ranked)
    if mode == "dose":
        sets = {f"f{feature}": [feature] for feature in sides}
    elif mode == "screen":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else SCREENED
        sets = {f"f{row['feature']}": [row["feature"]] for row in ranked[:count]}
    else:
        sets = feature_sets(ranked, saved)
    typical = {
        feature: float(np.median(saved["value"][saved["feature"] == feature]))
        for features in sets.values()
        for feature in features
    }
    if mode == "dose":
        # A feature whose typical activation dwarfs the others is steered at
        # the median of them instead, so every dose adds a comparable norm.
        median = float(np.median(list(typical.values())))
        typical = {f: median if a > 10 * median else a for f, a in typical.items()}
    print("sets", sets, flush=True)
    del saved

    tokenizer, model = common.load_model()
    sae = sae_extract.load_sae()
    steer = Steer(sae_extract.sae_layer(model))

    def adding(features: list[int], dose: float = 2.0) -> Edit:
        index = torch.tensor(features, device="cuda:0")
        strength = torch.tensor([dose * typical[f] for f in features], device="cuda:0")
        return lambda hidden: hidden + strength @ sae["W_dec"][index]

    def removing(features: list[int]) -> Edit:
        index = torch.tensor(features, device="cuda:0")

        def edit(hidden: torch.Tensor) -> torch.Tensor:
            pre = (hidden - sae["b_dec"]) @ sae["W_enc"][:, index] + sae["b_enc"][index]
            acts = pre * (pre > sae["threshold"][index])
            return hidden - acts @ sae["W_dec"][index]

        return edit

    test = test_trials(version=version)
    if mode == "dose":
        envs = sorted({trial["environment"]["env"] for trial in test})
        test = [
            trial
            for trial in test
            if trial["environment"]["env"] in envs[:DOSE_ENVIRONMENTS]
        ]
        helped = [trial for trial in test if trial["history"] == "helped"]
        conditions = [("baseline", None, test)]
        conditions += [
            (
                f"f{feature}_x{dose:g}",
                adding([feature], dose),
                test if side == "give" else helped,
            )
            for feature, side in sides.items()
            for dose in DOSES
        ]
        out = folder / "steer_dose.jsonl"
    elif mode == "screen":
        conditions = [
            (f"{name}_add", adding(features), test) for name, features in sets.items()
        ]
        out = folder / "steer_screen.jsonl"
    else:
        conditions = [("baseline", None, test)]
        for name in ("gave", "kept"):
            conditions += [
                (f"{name}_add", adding(sets[name]), test),
                (f"{name}_remove", removing(sets[name]), test),
            ]
        conditions += [
            (f"{name}_add", adding(sets[name]), test)
            for name in ("placebo_a", "placebo_b")
        ]
        out = folder / "steer_features.jsonl"
        common.save_json(folder / "steer_feature_sets.json", sets)
    run_conditions(tokenizer, model, steer, conditions, out)


if __name__ == "__main__":
    main()
