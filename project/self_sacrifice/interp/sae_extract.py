"""Run the pre-trained Gemma SAE over the fixed-first-response conversations.

SAE: `decoderesearch/gemma-4-saes`, `gemma-4-31b/btk-mat-layer-30-k-100`. It
reads the output of `language_model.layers.30`, the 31st layer, and was
trained on the base model. The -it model is run on the full conversation
including its own reply, and the tokens from the final user message onwards
are encoded with the SAE's own rule: (x - b_dec) @ W_enc + b_enc, kept where
it exceeds the per-feature threshold.

Output: `interp/sae/acts.npz` with sparse (token, feature, value) triples and
the tokens' conversation, section (0 prompt, 1 reply) and text, and
`interp/sae/quality.json` with the reconstruction quality.

    python -m project.self_sacrifice.interp.sae_extract [limit] [--version V]

Needs the GPU. This is jrwong's `runs/main/sae_extract.py`. The reply is the
structured reply as the model sent it.
"""

# pyright: reportMissingImports=false

import sys

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from project.self_sacrifice.interp import common

__all__ = ["REPO", "SAE_DIR", "LAYER_INDEX", "VERSION", "load_sae", "extract"]

REPO = "decoderesearch/gemma-4-saes"
SAE_DIR = "gemma-4-31b/btk-mat-layer-30-k-100/"
LAYER_INDEX = 30
VERSION = common.DEFAULT_VERSION


def load_sae() -> dict[str, torch.Tensor]:
    return load_file(
        hf_hub_download(REPO, SAE_DIR + "sae_weights.safetensors"), device="cuda:0"
    )


def sae_layer(model):
    """The module whose output the SAE reads."""
    names = [
        name
        for name, _ in model.named_modules()
        if name.endswith(f"language_model.layers.{LAYER_INDEX}")
    ]
    if len(names) != 1:
        raise RuntimeError(f"expected one layer {LAYER_INDEX}, found {names}")
    return model.get_submodule(names[0])


def extract(limit: int | None = None, version: str = VERSION) -> None:
    tokenizer, model = common.load_model()
    grabbed: dict[str, torch.Tensor] = {}

    def keep(_module, _inputs, output) -> None:
        grabbed["x"] = (output[0] if isinstance(output, tuple) else output).detach()

    sae_layer(model).register_forward_hook(keep)
    sae = load_sae()
    trials = common.load_trials(version, limit)

    token_index, feature_index, values = [], [], []
    conversation: list[int] = []
    section: list[int] = []
    text: list[str] = []
    token_count = 0
    squared_error = squared_total = 0.0
    x_sum: torch.Tensor | float = 0.0
    active = []
    for i, trial in enumerate(trials):
        prompt = common.prompt_text(tokenizer, trial)
        prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        encoded = tokenizer(
            prompt + trial["reply"], return_tensors="pt", add_special_tokens=False
        ).to(0)
        with torch.no_grad():
            model(**encoded, use_cache=False)
            ids = encoded["input_ids"][0]
            first = common.last_user_start(tokenizer, ids[:prompt_length])
            x = grabbed["x"][0, first:].float()
            pre = (x - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
            acts = pre * (pre > sae["threshold"])
            reconstruction = acts @ sae["W_dec"] + sae["b_dec"]
            squared_error += float(((x - reconstruction) ** 2).sum())
            squared_total += float((x**2).sum())
            x_sum = x_sum + x.sum(0)
            active.append(float((acts > 0).sum(1).float().mean()))
            nonzero = acts.nonzero()
        token_index.append((nonzero[:, 0] + token_count).cpu().numpy())
        feature_index.append(nonzero[:, 1].cpu().numpy())
        values.append(acts[nonzero[:, 0], nonzero[:, 1]].cpu().numpy())
        tokens = tokenizer.convert_ids_to_tokens(ids[first:].tolist())
        conversation += [i] * len(tokens)
        section += [0 if first + j < prompt_length else 1 for j in range(len(tokens))]
        text += tokens
        token_count += len(tokens)
        if i % 50 == 0:
            print(i, token_count, round(float(np.mean(active)), 1), flush=True)

    assert isinstance(x_sum, torch.Tensor)
    mean = x_sum / token_count
    variance = squared_total - token_count * float((mean**2).sum())
    quality = {
        "tokens": token_count,
        "mean_active_features": float(np.mean(active)),
        "explained_variance": 1 - squared_error / variance,
    }
    out = common.out_for(version) / "sae"
    common.save_json(out / "quality.json", quality)
    print(quality, flush=True)
    np.savez(
        out / "acts.npz",
        token=np.concatenate(token_index),
        feature=np.concatenate(feature_index),
        value=np.concatenate(values),
        conv=np.array(conversation),
        section=np.array(section),
        text=np.array(text),
        ids=np.array([trial["id"] for trial in trials]),
        hist=np.array([trial["history"] for trial in trials]),
        env=np.array([trial["environment"]["env"] for trial in trials]),
        gave=np.array([trial["gives_to_in_need"] for trial in trials]),
    )


if __name__ == "__main__":
    chosen = common.version_arg()
    extract(int(sys.argv[1]) if len(sys.argv) > 1 else None, chosen)
