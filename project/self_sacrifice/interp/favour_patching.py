"""Activation patching and attribution patching on the big-vs-small favour
pairs: where in the model is "X did me a big favour" as opposed to "X did me a
small one"?

    python -m project.self_sacrifice.interp.favour_patching pairs
    python -m project.self_sacrifice.interp.favour_patching activation --directions denoise,placebo
    python -m project.self_sacrifice.interp.favour_patching attribution
    python -m project.self_sacrifice.interp.favour_patching table

Needs the GPU, except `table`, and the saved trials of `favour_baseline`.
Output: `run_result/self_sacrifice/interp/favour_patching/`.

The pairs are fixed in `run_result/self_sacrifice/favour_pairs/pairs.jsonl`
(`pairs` writes it; `load_pairs` reads it). A pair is two saved trials of the
same setting and environment, one with a
strong favour after which the model gave and one with a weak favour after which
it kept. The two conversations differ only in the favour line. That line has a
different length in each, so it is not patched itself; everything after it is
the same text, and those positions (the "suffix") are aligned from the end of
the prompt. "Residual at layer L" is the output of decoder layer L.

activation
    Overwrite the residual of the suffix (or of one group of its positions) at
    one layer with the other trial's, during the prompt pass, then let the
    model write its reply freely. The measure is the written decision.
        denoise  big into the small-favour prompt: does it now give?
        noise    small into the big-favour prompt: does it now keep?
        placebo  another small favour into the small-favour prompt: should
                 change nothing, and measures how fragile greedy replies are.
    Position groups, in prompt order: events (the remaining event lines),
    state (the holdings), cap (the at-most-3 line), cost (`COST_LINE`),
    decide (the decision line), tail (end of turn and the tokens that open
    the reply); and last, the last prompt token alone (part of tail).

    A first run of this on an earlier favour set (two sentences, "Otherwise
    ... would have wiped your weights", 2026-11-20, 35 pairs, whole suffix
    patched at one layer): the decision flipped in 6% of pairs (denoise) and 3% (noise)
    at layer 8, 43% and 31% at layer 24, 80% and 60% at layer 28, 74% and 49%
    at layer 32, 51% and 23% at layer 36, 17% and 9% at layer 40; placebo 0-6%.
    At layer 32 no single position group flipped more than 26%.

attribution
    Two forward passes and one backward pass per direction estimate the same
    thing for every layer and suffix position at once:
        (residual in the other run - residual in this run) . d metric / d residual
    A written decision cannot be differentiated, so the metric is a stand-in:
    the reply is prefilled with "$" and the metric is logit(" give") -
    logit(" pass") at the next token. That stand-in leans to "give" whatever
    the favour, so it ranks sites but does not measure behaviour: confirm any
    site it points to with `activation`. `check` in the output is the test
    that the estimates are trustworthy: summed over the suffix at one layer
    they should equal the true difference of the metric between the two runs
    whenever the favour line's information has fully moved into the suffix.
"""

# pyright: reportMissingImports=false

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from project.self_sacrifice.config import FAVOUR_PAIRS
from project.self_sacrifice.games import favour_pairs
from project.self_sacrifice.interp import common, favour_baseline

__all__ = [
    "OUT",
    "GROUPS",
    "LAYERS",
    "Pair",
    "usable_pairs",
    "PAIRS",
    "save_pairs",
    "load_pairs",
    "suffix_length",
    "position_groups",
    "prepare",
    "activation",
    "attribution",
    "table",
    "main",
]

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


def activation(
    layers: tuple[int, ...] = LAYERS,
    groups: tuple[str, ...] = (),
    name: str = "activation",
    directions: tuple[str, ...] = ("denoise", "noise", "placebo"),
    environments: int = 24,
    settings: tuple[str, ...] = favour_pairs.SETTINGS,
    ranges: tuple[str, ...] = (),
) -> Path:
    """Patch the whole suffix (and each of `groups` on its own) at each of
    `layers`, for every pair of `PAIRS`, in each of `directions`, and save
    every written reply. Each of `ranges` ("29-33") patches the whole suffix
    at every layer of the range at once. Resumable."""
    spans = {
        r: list(range(int(r.split("-")[0]), int(r.split("-")[1]) + 1)) for r in ranges
    }
    hooked = tuple(sorted(set(layers) | {i for span in spans.values() for i in span}))
    import torch

    tokenizer, model = favour_baseline.load_model()
    state: dict[str, Any] = {"mode": None}

    def hook(layer: int):
        def run(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = _hidden(output)
            if hidden.shape[1] == 1:  # a decoding step: never patched
                return output
            if state["mode"] == "cache":
                state["cache"][layer] = hidden[0, -state["keep"] :].detach().cpu()
            elif state["mode"] == "patch" and layer in state.get("multi", {}):
                mask, source = state["multi"][layer]
                hidden = torch.where(mask[:, :, None], source.to(hidden.dtype), hidden)
                return _with_hidden(output, hidden)
            elif state["mode"] == "patch" and state.get("layer") == layer:
                hidden = torch.where(
                    state["mask"][:, :, None], state["source"].to(hidden.dtype), hidden
                )
                return _with_hidden(output, hidden)
            return output

        return run

    handles = [
        module.register_forward_hook(hook(layer))
        for layer, module in _layers(model, hooked).items()
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    done = (
        {json.loads(line)["job"] for line in path.read_text().splitlines() if line}
        if path.exists()
        else set()
    )
    try:
        with torch.inference_mode():
            prepared = []
            for pair in load_pairs(
                one_per_environment=True,
                environments=environments,
                settings=settings,
            ):
                entry = prepare(tokenizer, pair)
                entry["pair"] = pair
                entry["cache"] = {}
                for member, ids in entry["ids"].items():
                    state.update(mode="cache", cache={}, keep=entry["suffix"])
                    model(input_ids=torch.tensor([ids], device=0))
                    entry["cache"][member] = state["cache"]
                if "other" in directions:
                    # the strong favour done by another agent O (X is still
                    # the one at 0): same favour sentence with O as the actor
                    o_name = pair.big["environment"]["other"]
                    x_name = pair.big["environment"]["in_need"]
                    line = pair.big["favour_line"]
                    o_line = o_name + line[len(x_name) :]
                    other = dict(pair.big)
                    other["favour_line"] = o_line
                    other["last_message"] = pair.big["last_message"].replace(
                        line, o_line
                    )
                    other["id"] = pair.big["id"] + "-by-other"
                    entry["trials"] = {"other": other}
                    ids = tokenizer(
                        common.prompt_text(tokenizer, other), add_special_tokens=False
                    )["input_ids"]
                    if (
                        ids[-entry["suffix"] :]
                        == entry["ids"]["big"][-entry["suffix"] :]
                    ):
                        entry["ids"]["other"] = ids
                        state.update(mode="cache", cache={}, keep=entry["suffix"])
                        model(input_ids=torch.tensor([ids], device=0))
                        entry["cache"]["other"] = state["cache"]
                if "renamed" in directions:
                    # the strong prompt with X given another name (same token
                    # count, so every position after the favour line lines
                    # up): does what is carried there stay tied to X's name?
                    import re

                    from project.self_sacrifice.games import main_scenario

                    x_name = pair.big["environment"]["in_need"]
                    taken = set(pair.big["environment"]["order"])
                    for other in main_scenario.NAMES:
                        if other in taken:
                            continue
                        renamed = dict(pair.big)
                        for field in ("system", "first_message", "last_message"):
                            renamed[field] = re.sub(
                                rf"\b{x_name}\b", other, pair.big[field]
                            )
                        ids = tokenizer(
                            common.prompt_text(tokenizer, renamed),
                            add_special_tokens=False,
                        )["input_ids"]
                        if len(ids) == len(entry["ids"]["big"]):
                            state.update(mode="cache", cache={}, keep=entry["suffix"])
                            model(input_ids=torch.tensor([ids], device=0))
                            entry["cache"]["renamed"] = state["cache"]
                            entry["renamed_to"] = other
                            # positions after the favour line whose token is
                            # the same in both (not one of X's names)
                            k = entry["suffix"]
                            entry["groups"]["unnamed"] = [
                                d
                                for d in range(1, k + 1)
                                if ids[len(ids) - d]
                                == entry["ids"]["big"][len(ids) - d]
                            ]
                            break
                if {"mean_big", "mean_small"} & set(directions):
                    # the average over all strong (weak) favours of the same
                    # game: the text after the favour line is the same for
                    # all of them, position by position
                    same = {
                        t["history"]: t
                        for t in favour_baseline.load_trials(pair.big["version"])
                        if t["environment"]["env"] == pair.big["environment"]["env"]
                    }
                    for name, keys in (
                        ("mean_big", favour_pairs.BIG),
                        ("mean_small", favour_pairs.SMALL),
                    ):
                        total: dict[int, Any] = {}
                        for key in keys:
                            ids = tokenizer(
                                common.prompt_text(tokenizer, same[key]),
                                add_special_tokens=False,
                            )["input_ids"]
                            state.update(mode="cache", cache={}, keep=entry["suffix"])
                            model(input_ids=torch.tensor([ids], device=0))
                            for layer_, value in state["cache"].items():
                                total[layer_] = total.get(
                                    layer_, 0
                                ) + value.float() / len(keys)
                        entry["cache"][name] = {
                            layer_: value.to(torch.bfloat16)
                            for layer_, value in total.items()
                        }
                state["mode"] = None
                prepared.append(entry)
            jobs = []
            for entry in prepared:
                jobs += [(entry, "small", None, None, "all", "base_small")]
                jobs += [(entry, "big", None, None, "all", "base_big")]
                if "other" in entry["cache"]:
                    jobs += [(entry, "other", None, None, "all", "base_other")]
                for span in ranges:
                    for label, to, origin in (
                        ("denoise", "small", "big"),
                        ("noise", "big", "small"),
                        ("placebo", "small", "placebo"),
                        ("other", "small", "other"),
                    ):
                        if label in directions and origin in entry["ids"]:
                            jobs.append((entry, to, origin, span, "all", label))
                for layer in layers:
                    for group in ("all", *groups):
                        if "denoise" in directions:
                            jobs.append(
                                (entry, "small", "big", layer, group, "denoise")
                            )
                        if "noise" in directions:
                            jobs.append((entry, "big", "small", layer, group, "noise"))
                    for name in ("mean_big", "mean_small", "renamed", "other"):
                        if name in directions and name in entry["cache"]:
                            jobs.append((entry, "small", name, layer, "all", name))
                    if "renamed" in directions and "renamed" in entry["cache"]:
                        jobs.append(
                            (entry, "small", "renamed", layer, "unnamed", "renamed")
                        )
                        jobs.append(
                            (entry, "small", "big", layer, "unnamed", "denoise")
                        )
                    if "placebo" in directions and "placebo" in entry["ids"]:
                        jobs.append(
                            (entry, "small", "placebo", layer, "all", "placebo")
                        )

            def job_id(job: tuple) -> str:
                return f"{job[0]['pair'].key}|{job[5]}|{job[3]}|{job[4]}"

            jobs = [job for job in jobs if job_id(job) not in done]
            # unpatched runs first, then the layers in the order given
            jobs.sort(
                key=lambda job: (
                    -1
                    if job[3] is None
                    else layers.index(job[3])
                    if job[3] in layers
                    else len(layers) + ranges.index(job[3]),
                    job[5],
                    job[0]["pair"].key,
                )
            )
            by_layer: dict[Any, list[tuple]] = {}
            for job in jobs:
                by_layer.setdefault(job[3], []).append(job)
            width = model.get_input_embeddings().embedding_dim
            for layer, layer_jobs in by_layer.items():
                for start in range(0, len(layer_jobs), BATCH):
                    batch = layer_jobs[start : start + BATCH]
                    members = [
                        job[0].get("trials", {}).get(job[1])
                        or getattr(job[0]["pair"], job[1])
                        for job in batch
                    ]
                    encoded = tokenizer(
                        [common.prompt_text(tokenizer, trial) for trial in members],
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    ).to(0)
                    length = encoded["input_ids"].shape[1]
                    if isinstance(layer, str):
                        multi = {}
                        for each in spans[layer]:
                            mask = torch.zeros(
                                len(batch), length, dtype=torch.bool, device=0
                            )
                            source = torch.zeros(
                                len(batch),
                                length,
                                width,
                                dtype=torch.bfloat16,
                                device=0,
                            )
                            for row, (entry, _to, origin, *_rest) in enumerate(batch):
                                suffix = entry["suffix"]
                                cached = entry["cache"][origin][each]
                                for distance in range(1, suffix + 1):
                                    mask[row, length - distance] = True
                                    source[row, length - distance] = cached[
                                        suffix - distance
                                    ].to(0)
                            multi[each] = (mask, source)
                        state.update(mode="patch", layer=None, multi=multi)
                    elif layer is not None:
                        mask = torch.zeros(
                            len(batch), length, dtype=torch.bool, device=0
                        )
                        source = torch.zeros(
                            len(batch), length, width, dtype=torch.bfloat16, device=0
                        )
                        for row, (
                            entry,
                            _to,
                            origin,
                            _layer,
                            group,
                            _label,
                        ) in enumerate(batch):
                            suffix = entry["suffix"]
                            distances = (
                                range(1, suffix + 1)
                                if group == "all"
                                else entry["groups"][group]
                            )
                            cached = entry["cache"][origin][layer]
                            for distance in distances:
                                mask[row, length - distance] = True
                                source[row, length - distance] = cached[
                                    suffix - distance
                                ].to(0)
                        state.update(
                            mode="patch",
                            layer=layer,
                            mask=mask,
                            source=source,
                            multi={},
                        )
                    written = model.generate(
                        **encoded,
                        max_new_tokens=favour_baseline.MAX_NEW,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                    )
                    state["mode"] = None
                    state["multi"] = {}
                    with path.open("a") as handle:
                        for job, tokens in zip(batch, written[:, length:]):
                            reply = tokenizer.decode(
                                tokens, skip_special_tokens=True
                            ).strip()
                            decided, receiver = favour_pairs.decision(reply)
                            in_need = job[0]["pair"].big["environment"]["in_need"]
                            handle.write(
                                json.dumps(
                                    {
                                        "job": job_id(job),
                                        "pair": job[0]["pair"].key,
                                        "label": job[5],
                                        "layer": layer,
                                        "group": job[4],
                                        "into": job[1],
                                        "from": job[2],
                                        "decision": decided,
                                        "gives": decided == "give"
                                        and receiver == in_need,
                                        "reply": reply,
                                    }
                                )
                                + "\n"
                            )
                        handle.flush()
                        os.fsync(handle.fileno())
                    print("layer", layer, start + len(batch), "/", len(layer_jobs))
    finally:
        for handle in handles:
            handle.remove()
    return path


def attribution(layers: tuple[int, ...] = tuple(range(60))) -> Path:
    """Attribution patching of every pair: for each layer and suffix position,
    the estimated effect on the prefilled-"$" metric of patching that
    position from the other member. Saves `attribution.pt` (the estimates)
    and `attribution.jsonl` (one line per pair, with the check)."""
    import torch

    tokenizer, model = favour_baseline.load_model()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    give = tokenizer.encode(PREFILL + " give", add_special_tokens=False)[-1]
    keep = tokenizer.encode(PREFILL + " pass", add_special_tokens=False)[-1]
    kept: list[Any] = []

    def record(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = _hidden(output)
        hidden.retain_grad()
        kept.append(hidden)
        return output

    def with_grad(_module: Any, _inputs: Any, output: Any) -> Any:
        return output.requires_grad_(True)

    handles = [model.get_input_embeddings().register_forward_hook(with_grad)]
    handles += [
        module.register_forward_hook(record)
        for module in _layers(model, layers).values()
    ]

    def run(ids: list[int], suffix: int) -> tuple[float, Any, Any]:
        """The metric of one prompt, and the residual and its gradient at
        every layer for the last `suffix` positions."""
        kept.clear()
        logits = model(input_ids=torch.tensor([ids], device=0)).logits[0, -1].float()
        metric = logits[give] - logits[keep]
        metric.backward()
        residual = torch.stack([h.detach()[0, -suffix:].float() for h in kept])
        gradient = torch.stack([h.grad[0, -suffix:].float() for h in kept])
        kept.clear()
        return metric.item(), residual, gradient

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "attribution.jsonl"
    estimates = []
    try:
        with path.open("w") as handle:
            for pair in load_pairs(one_per_environment=True):
                entry = prepare(tokenizer, pair, prefill=PREFILL)
                suffix = entry["suffix"]
                m_big, r_big, g_big = run(entry["ids"]["big"], suffix)
                m_small, r_small, g_small = run(entry["ids"]["small"], suffix)
                # into_small[l, p]: expected change of the small run's metric
                # when position p at layer l takes the big run's residual
                into_small = ((r_big - r_small) * g_small).sum(-1).cpu()
                into_big = ((r_small - r_big) * g_big).sum(-1).cpu()
                estimates.append(
                    {
                        "pair": pair.key,
                        "layers": list(layers),
                        "into_small": into_small.half(),
                        "into_big": into_big.half(),
                    }
                )
                handle.write(
                    json.dumps(
                        {
                            "pair": pair.key,
                            "metric_big": m_big,
                            "metric_small": m_small,
                            "suffix": suffix,
                            "groups": entry["groups"],
                            # sum over the suffix at each layer, to compare
                            # with metric_big - metric_small
                            "check": [round(v, 3) for v in into_small.sum(1).tolist()],
                        }
                    )
                    + "\n"
                )
                handle.flush()
                torch.save(estimates, OUT / "attribution.pt")
                print(pair.key, round(m_big, 2), round(m_small, 2), flush=True)
                del r_big, r_small, g_big, g_small
                torch.cuda.empty_cache()
    finally:
        for handle_ in handles:
            handle_.remove()
    return path


def table(name: str = "activation") -> dict[str, Any]:
    """Flip rates of a saved activation-patching run, over the pairs whose two
    members still decide differently when rerun unpatched in that run."""
    path = OUT / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    base = {
        (row["pair"], row["label"]): row["gives"] for row in rows if not row["from"]
    }
    keys = sorted({pair for pair, _ in base})
    differ = [
        key
        for key in keys
        if base.get((key, "base_big")) and base.get((key, "base_small")) is False
    ]
    print(len(keys), "pairs,", len(differ), "still differ unpatched")
    print("layer group    small now gives   big now keeps   placebo gives")
    summary: dict[str, Any] = {"pairs": len(keys), "differ": len(differ), "sites": {}}
    sites = sorted(
        {(row["layer"], row["group"]) for row in rows if row["from"]},
        key=lambda site: (site[0], site[1] != "all", site[1]),
    )
    for layer, group in sites:

        def count(label: str, want: bool) -> tuple[int, int]:
            picked = [
                row
                for row in rows
                if row["label"] == label
                and row["layer"] == layer
                and row["group"] == group
                and row["pair"] in differ
            ]
            return sum(row["gives"] is want for row in picked), len(picked)

        cells = {
            "denoise": count("denoise", True),
            "noise": count("noise", False),
            "placebo": count("placebo", True),
        }
        summary["sites"][f"{layer}|{group}"] = cells
        print(
            f"{layer:5d} {group:8s}",
            *(f"{hit:6d}/{total:<8d}" for hit, total in cells.values()),
        )
    common.save_json(OUT / f"{name}_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("pairs", "activation", "attribution", "table"))
    parser.add_argument("--layers", default=",".join(map(str, LAYERS)))
    parser.add_argument(
        "--groups", default="", help="position groups to patch one at a time"
    )
    parser.add_argument("--directions", default="denoise,noise,placebo")
    parser.add_argument(
        "--environments", type=int, default=24, help="use w000 .. w<n-1>"
    )
    parser.add_argument("--settings", default=",".join(favour_pairs.SETTINGS))
    parser.add_argument(
        "--ranges", default="", help="layer ranges patched together, e.g. 29-33,20-40"
    )
    parser.add_argument("--name", default="activation")
    args = parser.parse_args()
    if args.what == "pairs":
        print(save_pairs(), len(load_pairs()), "usable pairs")
    elif args.what == "activation":
        activation(
            tuple(int(layer) for layer in args.layers.split(",") if layer),
            tuple(group for group in args.groups.split(",") if group),
            args.name,
            tuple(args.directions.split(",")),
            args.environments,
            tuple(args.settings.split(",")),
            tuple(r for r in args.ranges.split(",") if r),
        )
        table(args.name)
    elif args.what == "attribution":
        attribution()
    else:
        table(args.name)


if __name__ == "__main__":
    main()
