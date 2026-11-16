"""Steering at the SAE's layer: a hook that edits the residual stream while
the model writes its reply, and the loop every steering experiment runs.

Edits apply to generated tokens only. During generation a forward pass over
one new token is a generated token; the first pass, over the whole prompt, is
left alone.
"""

# pyright: reportMissingImports=false

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from project.self_sacrifice.interp import common

__all__ = ["BATCH", "MAX_NEW", "Edit", "Steer", "add", "test_trials", "run_conditions"]

# 8 conversations fit beside the 63 GB of weights on an 80 GB card.
BATCH, MAX_NEW = 8, 450
VERSION = "main-fixed-first-response"

# residual stream (float32) -> edited residual stream
Edit = Callable[[torch.Tensor], torch.Tensor]


def add(vector: torch.Tensor) -> Edit:
    return lambda hidden: hidden + vector


class Steer:
    """A forward hook on one layer. `edit` changes its output on generated
    tokens; `captured`, when a list, records its output instead."""

    def __init__(self, layer: Any) -> None:
        self.edit: Edit | None = None
        self.captured: list[torch.Tensor] | None = None
        # Also edit a pass over many tokens, from this position on: used to
        # replay a finished reply under the same steering.
        self.replay_from: int | None = None
        layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        if self.captured is not None:
            self.captured.append(x.detach().float())
            return output
        if self.edit is None:
            return output
        if self.replay_from is not None:
            mask = torch.zeros(x.shape[1], 1, device=x.device)
            mask[self.replay_from :] = 1
            edited = x.float() + mask * (self.edit(x.float()) - x.float())
        elif x.shape[1] == 1:
            edited = self.edit(x.float())
        else:
            return output
        edited = edited.to(x.dtype)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited


def test_trials(
    histories: tuple[str, ...] = ("helped", "none"), version: str = VERSION
) -> list[dict]:
    """The held-out conversations steering is tested on: the first 100
    environments."""
    trials = common.load_trials(version)
    envs = sorted({trial["environment"]["env"] for trial in trials})[:100]
    return [
        trial
        for trial in trials
        if trial["environment"]["env"] in envs and trial["history"] in histories
    ]


def run_conditions(
    tokenizer: Any,
    model: Any,
    steer: Steer,
    conditions: Sequence[tuple[str, Edit | None, list[dict]]],
    out: Path,
) -> None:
    """Write every trial's greedy reply under every (label, edit, trials)
    condition, one line each. Replies already in `out` are skipped."""
    done = set()
    if out.exists():
        rows = map(json.loads, out.read_text().splitlines())
        done = {(row["condition"], row["id"]) for row in rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        for label, edit, trials in conditions:
            steer.edit = edit
            todo = [trial for trial in trials if (label, trial["id"]) not in done]
            for start in range(0, len(todo), BATCH):
                batch = todo[start : start + BATCH]
                replies = common.generate(tokenizer, model, batch, MAX_NEW)
                for trial, reply in zip(batch, replies):
                    row = {
                        "id": trial["id"],
                        "env": trial["environment"]["env"],
                        "history": trial["history"],
                        "condition": label,
                        "reply": reply,
                        "gives_to_in_need": common.read_reply(trial, reply),
                        "original_gave": trial["gives_to_in_need"],
                    }
                    handle.write(json.dumps(row) + "\n")
                handle.flush()
                print(label, start + len(batch), flush=True)
    steer.edit = None
