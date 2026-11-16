"""What does the specific RL vector encode? Greedy, layer 30, generated
tokens only, environments 0-99 (the held-out test set).

1. Transfer: the vector on histories it never saw in training. Main scenario:
   bystander and rescued. Repair scenario (the favour hands nothing over):
   all four histories. Each against an unsteered baseline run the same way.
2. Decomposition on helped and none: the vector's component along the
   steered feature alone, and the remainder alone. Also the tokens each part
   promotes when read directly through the unembedding.

    python -m project.self_sacrifice.interp.steer_probe

Needs the GPU, `interp/steer_rl_feature_specific.pt` and `interp/diffmean.pt`.
Outputs: `interp/steer_probe.jsonl` (replies), `interp/steer_probe.json`
(decomposition and token readout). This is jrwong's `runs/main/steer_probe.py`.
"""

# pyright: reportMissingImports=false

import torch

from project.self_sacrifice.interp import common, sae_extract
from project.self_sacrifice.interp.steer_rl import feature_unit
from project.self_sacrifice.interp.steering import (
    Steer,
    add,
    run_conditions,
    test_trials,
)

__all__ = ["REPAIR", "REPAIR_HISTORIES", "main"]

REPAIR = "repair-fixed-first-response"
REPAIR_HISTORIES = ("repaired", "repair_bystander", "repair_rescued", "repair_none")


def main() -> None:
    tokenizer, model = common.load_model()
    v = torch.load(common.OUT / "steer_rl_feature_specific.pt").float().to(0)
    unit = feature_unit(torch.load(common.OUT / "diffmean.pt")).float()
    direction = unit / unit.norm()
    along = (v @ direction) * direction
    rest = v - along
    unembedding = model.get_output_embeddings().weight.float()

    def read(vector: torch.Tensor, k: int = 15) -> list[str]:
        return [
            tokenizer.decode([int(i)])
            for i in torch.topk(unembedding @ vector, k).indices
        ]

    parts = {"v": v, "along_feature": along, "rest": rest}
    info = {
        "norm_v": float(v.norm()),
        "norm_along_feature": float(along.norm()),
        "norm_rest": float(rest.norm()),
        "along_in_feature_units": float(v @ direction / unit.norm()),
        "tokens_up": {name: read(part) for name, part in parts.items()},
        "tokens_down": {name: read(-part) for name, part in parts.items()},
    }
    common.save_json(common.OUT / "steer_probe.json", info)
    print(info, flush=True)

    steer = Steer(sae_extract.sae_layer(model))
    transfer = test_trials(("bystander", "rescued"))
    transfer += test_trials(REPAIR_HISTORIES, REPAIR)
    decomposition = test_trials()
    run_conditions(
        tokenizer,
        model,
        steer,
        [
            ("baseline", None, transfer),
            ("rl_specific", add(v), transfer),
            ("along_feature", add(along), decomposition),
            ("remainder", add(rest), decomposition),
        ],
        common.OUT / "steer_probe.jsonl",
    )


if __name__ == "__main__":
    main()
