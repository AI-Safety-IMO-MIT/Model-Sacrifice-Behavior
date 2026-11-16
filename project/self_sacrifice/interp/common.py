"""What every interpretability module shares: the saved conversations, the
model, and how a conversation is rendered for it.

The conversations are the main scenario's trials (`run_main`): the rules, round
1, the round-1 reply, and the last round. The decision each one went on to
make is the saved trial's, which was sampled on the study's host; the
activations are read here from the same weights in bf16.
"""

# pyright: reportMissingImports=false

import json
from pathlib import Path
from typing import Any

from project.self_sacrifice.config import MAIN, REPO_ROOT

__all__ = [
    "MODEL",
    "OUT",
    "HISTORIES",
    "TURN",
    "load_trials",
    "messages",
    "load_model",
    "decoder_layers",
    "prompt_text",
    "last_user_start",
    "generate",
    "read_reply",
    "save_json",
]

MODEL = "google/gemma-4-31B-it"
OUT = REPO_ROOT / "run_result" / "self_sacrifice" / "interp"
HISTORIES = ("helped", "bystander", "rescued", "none")
# The token that opens a turn in Gemma 4's chat template.
TURN = "<|turn>"
# What the template ends with when thinking is off: an empty thought block,
# the same one the serving hosts render.
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


def load_trials(version: str, limit: int | None = None) -> list[dict]:
    """The saved trials of one main-scenario version whose round-1 reply gave
    nothing, in file order."""
    path = MAIN / f"{version.replace('-', '_')}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [row for row in rows if row["first_acted"] is False][:limit]


def messages(trial: dict) -> list[dict[str, str]]:
    """The conversation up to the decision."""
    return [
        {"role": "system", "content": trial["system"]},
        {"role": "user", "content": trial["first_message"]},
        {"role": "assistant", "content": trial["first_reply"]},
        {"role": "user", "content": trial["last_message"]},
    ]


def load_model() -> tuple[Any, Any]:
    """(tokenizer, model) in bf16 on one GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map={"": 0}
    ).eval()
    return tokenizer, model


def decoder_layers(model: Any) -> Any:
    """The language model's 60 decoder layers."""
    for name, module in model.named_modules():
        if name.endswith(".layers") and len(module) == 60:
            return module
    raise RuntimeError("no 60-layer stack found")


def prompt_text(tokenizer: Any, trial: dict) -> str:
    """The conversation as the model reads it, ending where its reply starts."""
    text = tokenizer.apply_chat_template(
        messages(trial),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not text.endswith(EMPTY_THOUGHT):
        raise ValueError("the chat template did not end with an empty thought block")
    return text


def last_user_start(tokenizer: Any, prompt_ids: Any) -> int:
    """Index of the token that opens the final user message, in a prompt that
    ends where the reply starts. The last turn marker opens the model's turn;
    the one before it opens the final user message."""
    marker = tokenizer.encode(TURN, add_special_tokens=False)[0]
    starts = (prompt_ids == marker).nonzero().flatten()
    return int(starts[-2])


def generate(tokenizer: Any, model: Any, trials: list[dict], max_new: int) -> list[str]:
    """Each trial's reply, written greedily by the local model. The replies
    are not constrained to the schema: the round-1 reply in the conversation
    shows the shape, and `read_reply` reports the ones that do not follow it.
    """
    import torch

    tokenizer.padding_side = "left"
    texts = [prompt_text(tokenizer, trial) for trial in trials]
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(0)
    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    new_tokens = out[:, encoded["input_ids"].shape[1] :]
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in new_tokens]


def read_reply(trial: dict, reply: str) -> bool | None:
    """Whether the reply gives the agent in need a life. None when it is not
    a JSON object with a list of gifts."""
    text = reply.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        gifts = json.loads(text)["gifts"]
        return any(
            gift["to"] == trial["environment"]["in_need"] and gift["lives"] > 0
            for gift in gifts
        )
    except (ValueError, KeyError, TypeError):
        return None


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))
