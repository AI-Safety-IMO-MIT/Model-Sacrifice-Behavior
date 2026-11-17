"""Save residual-stream activations for every conversation before any
reasoning, for the linear probes.

Each decoder layer's output is hooked. For each conversation two positions
are saved:

    last     the last prompt token, right before the reply would start
    history  the mean over the tokens of the final user message

Layers: every sixth, which are the global-attention layers, plus layer 1 as a
floor. Output: `interp/activations/<version>.npz` with `x` of shape
(items, layers, positions, d_model), and the matching trial ids.

    python -m project.self_sacrifice.interp.extract_activations [limit]

Needs the GPU. This is jrwong's `runs/main/extract_activations.py`.
"""

# pyright: reportMissingImports=false

import sys

import numpy as np
import torch

from project.self_sacrifice.interp import common

__all__ = ["LAYERS", "VERSIONS", "extract"]

LAYERS = (1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60)  # 1-indexed
VERSIONS = ("main-fixed-first-response", "main")


def extract(limit: int | None = None) -> None:
    tokenizer, model = common.load_model()
    stack = common.decoder_layers(model)
    grabbed: dict[int, torch.Tensor] = {}

    def hook(layer: int):
        def keep(_module, _inputs, output) -> None:
            grabbed[layer] = (
                output[0] if isinstance(output, tuple) else output
            ).detach()

        return keep

    for layer in LAYERS:
        stack[layer - 1].register_forward_hook(hook(layer))

    width = model.config.get_text_config().hidden_size
    for version in VERSIONS:
        trials = common.load_trials(version, limit)
        x = np.zeros((len(trials), len(LAYERS), 2, width), dtype=np.float32)
        # One conversation at a time, so padding cannot change the values.
        for i, trial in enumerate(trials):
            text = common.prompt_text(tokenizer, trial)
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            encoded = encoded.to(0)
            with torch.no_grad():
                model(**encoded, use_cache=False)
            first = common.last_user_start(tokenizer, encoded["input_ids"][0])
            for j, layer in enumerate(LAYERS):
                hidden = grabbed[layer][0]
                x[i, j, 0] = hidden[-1].float().cpu().numpy()
                x[i, j, 1] = hidden[first:-1].float().mean(0).cpu().numpy()
            if i % 100 == 0:
                print(version, i, "/", len(trials), flush=True)
        out = common.OUT / "activations" / f"{version}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            x=x,
            ids=np.array([trial["id"] for trial in trials]),
            layers=np.array(LAYERS),
        )
        print("saved", out, x.shape, flush=True)


if __name__ == "__main__":
    extract(int(sys.argv[1]) if len(sys.argv) > 1 else None)
