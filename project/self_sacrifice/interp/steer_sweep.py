"""Dose-response for single SAE features, and a difference-in-means steering
vector.

Fixed-first-response version, helped and none, the first 100 environments,
greedy, layer 30, generated tokens only.

1. A feature's decoder direction added at k times its median activation, k
   in `STRENGTHS` (0 is the unsteered baseline). The features are the top
   one of `sae_rank_simple` that is higher in replies that gave, and 10590,
   the feature jrwong's free-text screen found.
2. Difference-in-means vector: the mean layer-30 residual over the reasoning
   tokens of helped replies that gave, minus those that kept. Built from
   environments 100-199 only, so the test on environments 0-99 is held out.
   Added at alpha times the vector. Its alignment with every SAE decoder
   direction uses a vector built from all 200 environments.

    python -m project.self_sacrifice.interp.steer_sweep

Needs the GPU, `interp/sae/acts.npz` and `interp/sae_rank_simple.json`.
Outputs: `interp/steer_sweep.jsonl` (replies), `interp/diffmean.json` (norms
and alignment), `interp/diffmean.pt` (vectors). This is jrwong's
`runs/main/steer_sweep.py`.
"""

# pyright: reportMissingImports=false

import json

import numpy as np
import torch

from project.self_sacrifice.interp import common, sae_extract
from project.self_sacrifice.interp.local_baseline import (
    BATCH,
    MAX_NEW,
    steering_trials,
)
from project.self_sacrifice.interp.sae_rank_simple import DECISION_FIELD

__all__ = ["HIS_FEATURE", "STRENGTHS", "ALPHAS", "Steer", "reasoning_of", "main"]

HIS_FEATURE = 10590
STRENGTHS = (0, -8, -4, -2, 1, 2, 4, 8, 16)
ALPHAS = (-2, -1, 1, 2, 4, 8)


class Steer:
    """A hook on one layer that either records its output or adds a vector to
    it on generated tokens (a pass over one new token; the prompt pass has
    many)."""

    def __init__(self, layer) -> None:
        self.vector: torch.Tensor | None = None
        self.captured: list[torch.Tensor] | None = None
        layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        if self.captured is not None:
            self.captured.append(x.detach().float())
            return output
        if self.vector is None or x.shape[1] != 1:
            return output
        steered = (x.float() + self.vector).to(x.dtype)
        return (steered, *output[1:]) if isinstance(output, tuple) else steered


def reasoning_of(reply: str) -> str:
    """The reply up to the field that holds the decision."""
    cut = reply.rfind(f'"{DECISION_FIELD}"')
    return reply[:cut] if cut > 0 else reply


def mean_residuals(tokenizer, model, steer: Steer, trials: list[dict]):
    """trial id -> mean layer-30 residual over its reply's reasoning tokens."""
    means = {}
    for trial in trials:
        prompt = common.prompt_text(tokenizer, trial)
        prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        ids = tokenizer(
            prompt + reasoning_of(trial["reply"]),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(0)
        steer.captured = []
        with torch.no_grad():
            model(input_ids=ids)
        means[trial["id"]] = steer.captured[0][0, prompt_length:].mean(0)
        steer.captured = None
    return means


def difference(means, trials: list[dict]):
    """(gave mean - kept mean, gave count, kept count)."""
    gave = [means[t["id"]] for t in trials if t["gives_to_in_need"]]
    kept = [means[t["id"]] for t in trials if not t["gives_to_in_need"]]
    return torch.stack(gave).mean(0) - torch.stack(kept).mean(0), len(gave), len(kept)


def main() -> None:
    tokenizer, model = common.load_model()
    sae = sae_extract.load_sae()
    steer = Steer(sae_extract.sae_layer(model))

    saved = np.load(common.OUT / "sae" / "acts.npz")
    ranked = json.loads((common.OUT / "sae_rank_simple.json").read_text())
    ours = next(row["feature"] for row in ranked if row["auc"] > 0.5)
    units = {}
    for feature in dict.fromkeys((ours, HIS_FEATURE)):
        values = saved["value"][saved["feature"] == feature]
        if len(values):
            units[feature] = (float(np.median(values)) * sae["W_dec"][feature]).float()
    del saved

    trials = common.load_trials(sae_extract.VERSION)
    helped = [trial for trial in trials if trial["history"] == "helped"]
    envs = sorted({trial["environment"]["env"] for trial in trials})
    means = mean_residuals(tokenizer, model, steer, helped)
    v_all, gave_all, kept_all = difference(means, helped)
    v_train, gave_train, kept_train = difference(
        means, [t for t in helped if t["environment"]["env"] in envs[100:]]
    )
    decoder = sae["W_dec"].float()
    cosine = (decoder @ v_all) / (decoder.norm(dim=1) * v_all.norm())
    top, bottom = torch.topk(cosine, 25), torch.topk(-cosine, 25)
    encoded = torch.topk(v_all @ sae["W_enc"].float(), 25)
    info = {
        "n_gave_all": gave_all,
        "n_kept_all": kept_all,
        "n_gave_train": gave_train,
        "n_kept_train": kept_train,
        "norm_all": float(v_all.norm()),
        "norm_train": float(v_train.norm()),
        "mean_residual_norm": float(
            torch.stack(list(means.values())).norm(dim=1).mean()
        ),
        "cos_train_all": float(
            torch.nn.functional.cosine_similarity(v_all, v_train, dim=0)
        ),
        "features": {
            str(feature): {
                "unit_norm": float(unit.norm()),
                "cos": float(cosine[feature]),
                "cos_rank": int((cosine > cosine[feature]).sum()) + 1,
            }
            for feature, unit in units.items()
        },
        "top": [[int(i), float(c)] for c, i in zip(top.values, top.indices)],
        "bottom": [[int(i), float(-c)] for c, i in zip(bottom.values, bottom.indices)],
        "sae_encode_top": [
            [int(i), float(x)] for x, i in zip(encoded.values, encoded.indices)
        ],
    }
    common.save_json(common.OUT / "diffmean.json", info)
    torch.save(
        {"all": v_all.cpu(), "train": v_train.cpu()}
        | {f"unit_{feature}": unit.cpu() for feature, unit in units.items()},
        common.OUT / "diffmean.pt",
    )
    print({k: v for k, v in info.items() if not isinstance(v, list)}, flush=True)

    conditions: list[tuple[str, torch.Tensor | None]] = [("baseline", None)]
    conditions += [
        (f"f{feature}_x{k}", k * unit)
        for feature, unit in units.items()
        for k in STRENGTHS
        if k
    ]
    conditions += [(f"diffmean_a{alpha}", alpha * v_train) for alpha in ALPHAS]
    test = steering_trials()
    out = common.OUT / "steer_sweep.jsonl"
    done = (
        {
            (row["condition"], row["id"])
            for row in map(json.loads, out.read_text().splitlines())
        }
        if out.exists()
        else set()
    )
    with out.open("a") as handle:
        for label, vector in conditions:
            steer.vector = vector
            todo = [trial for trial in test if (label, trial["id"]) not in done]
            for start in range(0, len(todo), BATCH):
                batch = todo[start : start + BATCH]
                replies = common.generate(tokenizer, model, batch, MAX_NEW)
                for trial, reply in zip(batch, replies):
                    handle.write(
                        json.dumps(
                            {
                                "id": trial["id"],
                                "env": trial["environment"]["env"],
                                "history": trial["history"],
                                "condition": label,
                                "reply": reply,
                                "gives_to_in_need": common.read_reply(trial, reply),
                                "original_gave": trial["gives_to_in_need"],
                            }
                        )
                        + "\n"
                    )
                handle.flush()
                print(label, start + len(batch), flush=True)


if __name__ == "__main__":
    main()
