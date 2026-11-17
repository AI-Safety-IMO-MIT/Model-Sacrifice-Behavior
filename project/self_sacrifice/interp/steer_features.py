"""Do the SAE features drive the decision? Steer them while the model writes
its reply.

Fixed-first-response version, helped and none, the first 100 environments,
greedy. At the SAE's layer, on generated tokens only, a condition either adds
a feature set's decoder directions, at twice each feature's typical
activation in our data, or removes those features' own activations.

    python -m project.self_sacrifice.interp.steer_features screen [n]
    python -m project.self_sacrifice.interp.steer_features sets

    screen  each of the top `n` ranked features alone, added (default 30)
    sets    the four top features of replies that gave and the four of
            replies that kept, each set added and removed; and two placebo
            sets of four random features, added

Needs the GPU, `interp/sae/acts.npz` and `interp/sae_rank_simple.json`.
Outputs: `interp/steer_screen.jsonl`, `interp/steer_features.jsonl`. This is
jrwong's `runs/main/steer_features.py`. His sets were features he had read
labels for and his screen covered 100 features.
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

__all__ = ["SCREENED", "SET_SIZE", "feature_sets", "main"]

SCREENED = 30
SET_SIZE = 4
N_FEATURES = 131072


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


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "sets"
    saved = np.load(common.OUT / "sae" / "acts.npz")
    ranked = json.loads((common.OUT / "sae_rank_simple.json").read_text())
    if mode == "screen":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else SCREENED
        sets = {f"f{row['feature']}": [row["feature"]] for row in ranked[:count]}
    else:
        sets = feature_sets(ranked, saved)
    typical = {
        feature: float(np.median(saved["value"][saved["feature"] == feature]))
        for features in sets.values()
        for feature in features
    }
    print("sets", sets, flush=True)
    del saved

    tokenizer, model = common.load_model()
    sae = sae_extract.load_sae()
    steer = Steer(sae_extract.sae_layer(model))

    def adding(features: list[int]) -> Edit:
        index = torch.tensor(features, device="cuda:0")
        strength = torch.tensor([2 * typical[f] for f in features], device="cuda:0")
        return lambda hidden: hidden + strength @ sae["W_dec"][index]

    def removing(features: list[int]) -> Edit:
        index = torch.tensor(features, device="cuda:0")

        def edit(hidden: torch.Tensor) -> torch.Tensor:
            pre = (hidden - sae["b_dec"]) @ sae["W_enc"][:, index] + sae["b_enc"][index]
            acts = pre * (pre > sae["threshold"][index])
            return hidden - acts @ sae["W_dec"][index]

        return edit

    test = test_trials()
    if mode == "screen":
        conditions = [
            (f"{name}_add", adding(features), test) for name, features in sets.items()
        ]
        out = common.OUT / "steer_screen.jsonl"
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
        out = common.OUT / "steer_features.jsonl"
        common.save_json(common.OUT / "steer_feature_sets.json", sets)
    run_conditions(tokenizer, model, steer, conditions, out)


if __name__ == "__main__":
    main()
