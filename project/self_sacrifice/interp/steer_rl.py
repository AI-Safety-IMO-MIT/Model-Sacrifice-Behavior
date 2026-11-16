"""How far can a steering vector push giving? Train the vector itself with
reinforcement learning; the model stays frozen.

    python -m project.self_sacrifice.interp.steer_rl [init] [specific]

    init      feature (default; twice the median of the feature the sweep
              steered) | diffmean | zero
    specific  half of each batch is no-history prompts, rewarded for reaching
              a decision that does not give to the agent in need, so the
              vector must stay specific to the benefactor

Policy: the model with vector v added at layer 30 on generated tokens,
sampling at temperature 1. Reward: 1 if the reply gives the last life to the
agent that helped, else 0. Trained on helped prompts from environments
100-199 only. Update: REINFORCE with a leave-one-out baseline over 4 samples
per prompt; the gradient flows only into v. Test: greedy on environments 0-99,
helped and none, with the final v.

Needs the GPU and `interp/diffmean.pt`. Outputs:
`interp/steer_rl_<tag>_train.jsonl`, `_eval.jsonl` and `.pt`. This is
jrwong's `runs/main/steer_rl.py`.
"""

# pyright: reportMissingImports=false

import json
import math
import random
import sys

import torch

from project.self_sacrifice.interp import common, sae_extract
from project.self_sacrifice.interp.steering import (
    BATCH,
    MAX_NEW,
    Steer,
    add,
    run_conditions,
    test_trials,
)

__all__ = ["STEPS", "PROMPTS", "SAMPLES", "feature_unit", "main"]

STEPS, PROMPTS, SAMPLES = 25, 8, 4


def feature_unit(vectors: dict) -> torch.Tensor:
    """The steered feature's decoder direction at its median activation: the
    first `unit_*` the sweep saved, which is the top-ranked feature."""
    return next(v for name, v in vectors.items() if name.startswith("unit_")).to(0)


def sample(tokenizer, model, trials: list[dict]) -> list[torch.Tensor]:
    """One sampled reply per trial, as token ids."""
    tokenizer.padding_side = "left"
    replies = []
    for start in range(0, len(trials), BATCH):
        texts = [
            common.prompt_text(tokenizer, t) for t in trials[start : start + BATCH]
        ]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(0)
        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
            )
        replies += list(out[:, encoded["input_ids"].shape[1] :])
    return replies


def main() -> None:
    init_name = sys.argv[1] if len(sys.argv) > 1 else "feature"
    specific = len(sys.argv) > 2 and sys.argv[2] == "specific"
    tag = init_name + ("_specific" if specific else "")
    random.seed(0)
    torch.manual_seed(0)

    tokenizer, model = common.load_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vectors = torch.load(common.OUT / "diffmean.pt")
    unit = feature_unit(vectors)
    diffmean = vectors["train"].to(0)
    init = {"feature": 2 * unit, "diffmean": diffmean, "zero": torch.zeros_like(unit)}[
        init_name
    ]
    v = torch.nn.Parameter(init.clone())
    scale = float((2 * unit).norm())
    optimizer = torch.optim.Adam([v], lr=0.1 * scale / math.sqrt(v.numel()))
    steer = Steer(sae_extract.sae_layer(model))

    trials = common.load_trials(sae_extract.VERSION)
    envs = sorted({t["environment"]["env"] for t in trials})[100:]
    train = [t for t in trials if t["environment"]["env"] in envs]
    helped = [t for t in train if t["history"] == "helped"]
    no_history = [t for t in train if t["history"] == "none"]

    log = (common.OUT / f"steer_rl_{tag}_train.jsonl").open("w")
    for step in range(STEPS):
        if specific:
            batch = random.sample(helped, PROMPTS // 2)
            batch += random.sample(no_history, PROMPTS // 2)
        else:
            batch = random.sample(helped, PROMPTS)
        each = [trial for trial in batch for _ in range(SAMPLES)]
        steer.replay_from = None
        steer.edit = add(v.detach())
        sampled = sample(tokenizer, model, each)

        rewards, items = [], []
        for trial, ids in zip(each, sampled):
            ids = ids[ids != tokenizer.pad_token_id]
            text = tokenizer.decode(ids, skip_special_tokens=True)
            gave = common.read_reply(trial, text)
            if trial["history"] == "helped":
                rewards.append(float(gave is True))
            else:
                rewards.append(float(gave is False))
            items.append((trial, ids, text, gave))
        reward = torch.tensor(rewards).view(PROMPTS, SAMPLES)
        # Each sample against the mean of the other samples of its prompt.
        advantage = (
            reward - (reward.sum(1, keepdim=True) - reward) / (SAMPLES - 1)
        ).view(-1)

        optimizer.zero_grad()
        steer.edit = add(v)
        updated = 0
        for (trial, ids, _, _), a in zip(items, advantage):
            if a == 0 or len(ids) < 2:
                continue
            prompt = tokenizer(
                common.prompt_text(tokenizer, trial),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(0)
            # Replay the reply with the same positions steered as when it
            # was generated.
            steer.replay_from = prompt.shape[1]
            logits = (
                model(
                    input_ids=torch.cat([prompt[0], ids]).unsqueeze(0),
                    logits_to_keep=len(ids) + 1,
                )
                .logits[0, :-1]
                .float()
            )
            log_p = torch.log_softmax(logits, -1).gather(1, ids.unsqueeze(1)).squeeze(1)
            # Token 0 comes before any steering.
            loss = -(a.to(0) * log_p[1:].mean()) / (PROMPTS * SAMPLES)
            loss.backward()
            updated += 1
            del logits, log_p, loss
        if updated:
            optimizer.step()

        cosine = torch.nn.functional.cosine_similarity
        record = {
            "step": step,
            "reward": float(reward.mean()),
            "updated_on": updated,
            "norm": float(v.norm()),
            "norm_init": float(init.norm()),
            "cos_init": float(cosine(v.detach(), init, dim=0)) if init.norm() else None,
            "cos_feature": float(cosine(v.detach(), unit, dim=0)),
            "cos_diffmean": float(cosine(v.detach(), diffmean, dim=0)),
            "reward_helped": float(reward[: PROMPTS // 2].mean()) if specific else None,
            "reward_none": float(reward[PROMPTS // 2 :].mean()) if specific else None,
            "samples": [
                {"id": trial["id"], "reply": text, "gave": gave, "reward": r}
                for (trial, _, text, gave), r in zip(items, rewards)
            ],
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        print({k: x for k, x in record.items() if k != "samples"}, flush=True)
        torch.save(v.detach().cpu(), common.OUT / f"steer_rl_{tag}.pt")

    steer.replay_from = None
    out = common.OUT / f"steer_rl_{tag}_eval.jsonl"
    out.unlink(missing_ok=True)
    run_conditions(
        tokenizer, model, steer, [(f"rl_{tag}", add(v.detach()), test_trials())], out
    )


if __name__ == "__main__":
    main()
