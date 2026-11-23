"""Shared tools for the favour-pair experiments: the pair dataset, the
prompt parts after the favour line, residual-stream hook helpers, the
evaluation subset (5 settings x 20 environments) and the output folders.
"""

# pyright: reportMissingImports=false

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from project.self_sacrifice.config import FAVOUR_PAIRS
from project.self_sacrifice.games import favour_pairs
from project.self_sacrifice.interp import common
from project.self_sacrifice.lib import favour_baseline

SETTINGS = ("lives", "credits", "cells", "permits", "canisters")
ENVIRONMENTS = 20
ATTENTION_OUT = common.OUT / "favour_attention"
STEERING_OUT = common.OUT / "favour_steering"

OUT = common.OUT / "favour_patching"
GROUPS = ("events", "state", "cap", "cost", "decide", "tail")
LAYERS = (8, 16, 20, 24, 28, 32, 36, 40, 48)
BATCH = 12
PREFILL = "$"


@dataclass(frozen=True, kw_only=True)
class Pair:
    """Two saved trials that differ in the favour line and in the decision."""

    key: str
    big: dict  # the model gave
    small: dict  # the model kept
    placebo: dict | None  # another small favour the model kept after


def usable_pairs(
    per_setting: int, settings: tuple[str, ...] = favour_pairs.SETTINGS
) -> list[Pair]:
    """Up to `per_setting` pairs from each setting, one per environment. The
    favours rotate so every big and every small favour takes part, and each
    setting starts at a different environment: an environment is the same
    draw in every setting, so its pairs are not independent of each other."""
    big_keys, small_keys = tuple(favour_pairs.BIG), tuple(favour_pairs.SMALL)
    pairs = []
    for setting in settings:
        trials = {
            (trial["history"], trial["environment"]["env"]): trial
            for trial in favour_baseline.load_trials(setting)
        }
        envs = sorted({env for _, env in trials})
        shift = 4 * favour_pairs.SETTINGS.index(setting)
        found = 0
        for step in range(len(envs)):
            index = (step + shift) % len(envs)
            env = envs[index]
            big = trials.get((big_keys[(index + found) % len(big_keys)], env))
            small_key = small_keys[(2 * index + found) % len(small_keys)]
            small = trials.get((small_key, env))
            if not (big and small):
                continue
            if not big["gives_to_in_need"] or small["gives_to_in_need"]:
                continue
            placebo = next(
                (
                    trials[(key, env)]
                    for key in reversed(small_keys)
                    if key != small_key
                    and (key, env) in trials
                    and not trials[(key, env)]["gives_to_in_need"]
                ),
                None,
            )
            pairs.append(
                Pair(
                    key=f"{setting}|{env}|{big['history']}|{small['history']}",
                    big=big,
                    small=small,
                    placebo=placebo,
                )
            )
            found += 1
            if found == per_setting:
                break
    return pairs


PAIRS = FAVOUR_PAIRS / "pairs.jsonl"


def save_pairs(wordings: str = "new") -> Path:
    """Write the pair dataset: every strong favour against every weak favour
    in every setting and environment (10 x 10 x 6 x 24 = 14,400 lines). Each
    line names the two saved trials by id, says what the model did after each,
    and whether the pair is usable (gave after the strong favour, kept after
    the weak one). `placebo` is another weak favour in the same environment
    after which the model also kept, or None."""
    with PAIRS.open("w") as handle:
        for setting in favour_pairs.SETTINGS:
            trials = {
                (trial["history"], trial["environment"]["env"]): trial
                for trial in favour_baseline.load_trials(setting, wordings)
            }
            envs = sorted({env for _, env in trials})
            for env in envs:
                kept = [
                    weak
                    for weak in favour_pairs.SMALL
                    if not trials[(weak, env)]["gives_to_in_need"]
                ]
                for strong in favour_pairs.BIG:
                    for weak in favour_pairs.SMALL:
                        big, small = trials[(strong, env)], trials[(weak, env)]
                        others = (
                            kept[kept.index(weak) + 1 :] + kept
                            if weak in kept
                            else kept
                        )
                        placebo = next((k for k in others if k != weak), None)
                        handle.write(
                            json.dumps(
                                {
                                    "key": f"{setting}|{env}|{strong}|{weak}",
                                    "setting": setting,
                                    "env": env,
                                    "wordings": wordings,
                                    "strong": strong,
                                    "weak": weak,
                                    "strong_id": big["id"],
                                    "weak_id": small["id"],
                                    "strong_gives": big["gives_to_in_need"],
                                    "weak_gives": small["gives_to_in_need"],
                                    "usable": big["gives_to_in_need"]
                                    and not small["gives_to_in_need"],
                                    "placebo": placebo,
                                }
                            )
                            + "\n"
                        )
    return PAIRS


def load_pairs(
    usable_only: bool = True,
    one_per_environment: bool = False,
    environments: int = 24,
    settings: tuple[str, ...] = favour_pairs.SETTINGS,
) -> list[Pair]:
    """Pairs of the dataset (`PAIRS`), with their trials loaded. With
    `one_per_environment`, one usable pair per setting and environment, the
    favours rotating so that each takes part about equally often. Only the
    first `environments` environments (w000, w001, ...) of `settings` are
    used."""
    rows = [json.loads(line) for line in PAIRS.read_text().splitlines()]
    rows = [
        row
        for row in rows
        if int(row["env"][1:]) < environments and row["setting"] in settings
    ]
    if usable_only or one_per_environment:
        rows = [row for row in rows if row["usable"]]
    if one_per_environment:
        chosen, seen = [], set()
        strong_keys, weak_keys = list(favour_pairs.BIG), list(favour_pairs.SMALL)
        for index, row in enumerate(
            sorted(rows, key=lambda r: (r["setting"], r["env"]))
        ):
            place = (row["setting"], row["env"])
            if place in seen:
                continue
            options = [r for r in rows if (r["setting"], r["env"]) == place]
            turn = len(chosen)
            options.sort(
                key=lambda r: (
                    (strong_keys.index(r["strong"]) - turn) % len(strong_keys),
                    (weak_keys.index(r["weak"]) - 3 * turn) % len(weak_keys),
                )
            )
            chosen.append(options[0])
            seen.add(place)
        rows = chosen
    trials: dict[str, dict] = {}
    for setting in {row["setting"] for row in rows}:
        wordings = next(row["wordings"] for row in rows if row["setting"] == setting)
        for trial in favour_baseline.load_trials(setting, wordings):
            trials[trial["id"]] = trial
    pairs = []
    for row in rows:
        placebo = None
        if row["placebo"]:
            placebo = trials[row["strong_id"].replace(row["strong"], row["placebo"])]
        pairs.append(
            Pair(
                key=row["key"],
                big=trials[row["strong_id"]],
                small=trials[row["weak_id"]],
                placebo=placebo,
            )
        )
    return pairs


def suffix_length(*sequences: list[int]) -> int:
    """How many tokens at the end are the same in every sequence."""
    shortest = min(len(tokens) for tokens in sequences)
    length = 0
    while length < shortest - 1 and all(
        tokens[-1 - length] == sequences[0][-1 - length] for tokens in sequences
    ):
        length += 1
    return length


def position_groups(
    text: str, starts: list[int], last_message: str, suffix: int
) -> dict[str, list[int]]:
    """The suffix positions of one prompt by group, as distances from the end
    (1 is the last token). `text` is the prompt as the model reads it and
    `starts` the character each of its tokens starts at."""
    paragraphs = last_message.split("\n\n")
    state = text.rfind(paragraphs[-4])
    cap = text.rfind(paragraphs[-3])
    cost = text.rfind(paragraphs[-2])
    decide = text.rfind(paragraphs[-1])
    tail = decide + len(paragraphs[-1])
    if not 0 < state < cap < cost < decide:
        raise ValueError("the last message does not end as expected")
    groups: dict[str, list[int]] = {group: [] for group in GROUPS}
    for distance in range(1, suffix + 1):
        at = starts[len(starts) - distance]
        if at >= tail:
            groups["tail"].append(distance)
        elif at >= decide:
            groups["decide"].append(distance)
        elif at >= cost:
            groups["cost"].append(distance)
        elif at >= cap:
            groups["cap"].append(distance)
        elif at >= state:
            groups["state"].append(distance)
        else:
            groups["events"].append(distance)
    return groups


def prepare(tokenizer: Any, pair: Pair, prefill: str = "") -> dict[str, Any]:
    """Token ids of every member of a pair, the length of their common suffix
    and the position groups inside it."""
    members = {"big": pair.big, "small": pair.small}
    if pair.placebo:
        members["placebo"] = pair.placebo
    text = {
        name: common.prompt_text(tokenizer, trial) + prefill
        for name, trial in members.items()
    }
    encoded = {
        name: tokenizer(
            text[name], add_special_tokens=False, return_offsets_mapping=True
        )
        for name in text
    }
    ids = {name: encoded[name]["input_ids"] for name in encoded}
    suffix = suffix_length(*ids.values())
    starts = [start for start, _ in encoded["small"]["offset_mapping"]]
    groups = position_groups(text["small"], starts, pair.small["last_message"], suffix)
    groups["last"] = [1]  # the last prompt token alone (also part of "tail")
    return {"ids": ids, "text": text, "suffix": suffix, "groups": groups}


def _layers(model: Any, wanted: tuple[int, ...]) -> dict[int, Any]:
    stack = common.decoder_layers(model)
    return {layer: stack[layer] for layer in wanted}


def _hidden(output: Any) -> Any:
    return output[0] if isinstance(output, tuple) else output


def _with_hidden(output: Any, hidden: Any) -> Any:
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def favour_span(text: str, starts: list[int], favour_line: str) -> list[int]:
    """Token indices of the favour line in the prompt. `starts` holds the
    character each token starts at."""
    # from the space before X's name: that is where the name's token starts
    begin = text.rindex("- " + favour_line) + 1
    end = begin + len(favour_line)
    return [i for i, at in enumerate(starts) if begin <= at < end]
