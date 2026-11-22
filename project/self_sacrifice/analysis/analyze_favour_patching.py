"""Tables for the favour-pair interpretability experiments.

    python -m project.self_sacrifice.analysis.analyze_favour_patching attention
    python -m project.self_sacrifice.analysis.analyze_favour_patching layers all60_a,all60_b
    python -m project.self_sacrifice.analysis.analyze_favour_patching layers all60_noise_a,all60_noise_b

attention   experiment 1: give rate after strong, weak and no favour, with
            the reply blocked from the favour line and without, on the same
            games (5 settings x 20 environments).
layers      experiment 2: for each layer, how often the patched prompt
            changes its decision, over the pairs whose two members still
            decide differently when rerun unpatched; with the placebo.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from project.self_sacrifice.interp import common

__all__ = ["attention", "layers", "main"]

SETTINGS = ("lives", "credits", "cells", "permits", "canisters")
ENVIRONMENTS = 20


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _in_scope(setting: str, env: str) -> bool:
    return setting in SETTINGS and int(env[1:]) < ENVIRONMENTS


def attention() -> dict:
    folder = common.OUT / "favour_attention"
    runs = {
        name: {
            row["id"]: row
            for row in _rows(folder / f"{name}.jsonl")
            if _in_scope(row["setting"], row["env"])
        }
        for name in ("normal", "blocked")
    }
    both = sorted(runs["normal"].keys() & runs["blocked"].keys())
    print(f"games in both runs: {len(both)}")
    print("size     n    normal  blocked   gave->kept  kept->gave")
    summary: dict = {}
    for size in ("big", "small", "none"):
        ids = [i for i in both if runs["normal"][i]["size"] == size]
        normal = sum(runs["normal"][i]["gives"] for i in ids)
        blocked = sum(runs["blocked"][i]["gives"] for i in ids)
        lost = sum(
            runs["normal"][i]["gives"] and not runs["blocked"][i]["gives"] for i in ids
        )
        new = sum(
            not runs["normal"][i]["gives"] and runs["blocked"][i]["gives"] for i in ids
        )
        summary[size] = {
            "n": len(ids),
            "normal": normal,
            "blocked": blocked,
            "gave_then_kept": lost,
            "kept_then_gave": new,
        }
        print(
            f"{size:6s} {len(ids):4d}  {100 * normal / len(ids):6.1f}%  {100 * blocked / len(ids):6.1f}%"
            f"  {lost:10d}  {new:10d}"
        )
    per_favour = defaultdict(lambda: [0, 0, 0])
    for i in both:
        favour = runs["normal"][i]["favour"]
        per_favour[favour][0] += 1
        per_favour[favour][1] += runs["normal"][i]["gives"]
        per_favour[favour][2] += runs["blocked"][i]["gives"]
    summary["favour"] = {
        favour: {"n": n, "normal": a, "blocked": b}
        for favour, (n, a, b) in sorted(per_favour.items())
    }
    common.save_json(folder / "summary.json", summary)
    return summary


def layers(names: list[str]) -> dict:
    folder = common.OUT / "favour_patching"
    rows = [row for name in names for row in _rows(folder / f"{name}.jsonl")]
    base = {
        (row["pair"], row["label"]): row["gives"] for row in rows if row["from"] is None
    }
    pairs = sorted({pair for pair, _ in base})
    differ = {
        pair
        for pair in pairs
        if base.get((pair, "base_big")) and base.get((pair, "base_small")) is False
    }
    print(f"{len(pairs)} pairs, {len(differ)} still decide differently unpatched")
    summary: dict = {"pairs": len(pairs), "differ": len(differ), "layers": {}}
    labels = sorted({row["label"] for row in rows if row["from"] is not None})
    want = {
        "denoise": True,
        "placebo": True,
        "noise": False,
        "mean_big": True,
        "mean_small": True,
    }
    print("layer  " + "  ".join(f"{label:>16s}" for label in labels))
    for layer in sorted({row["layer"] for row in rows if row["layer"] is not None}):
        cells = {}
        for label in labels:
            picked = [
                row
                for row in rows
                if row["layer"] == layer
                and row["label"] == label
                and row["group"] == "all"
                and row["pair"] in differ
            ]
            hits = sum(row["gives"] is want[label] for row in picked)
            cells[label] = [hits, len(picked)]
        summary["layers"][layer] = cells
        print(
            f"{layer:5d}  "
            + "  ".join(
                f"{100 * h / n:6.1f}% ({h:3d}/{n:3d})" if n else f"{'-':>16s}"
                for h, n in cells.values()
            )
        )
    common.save_json(folder / f"{'+'.join(names)}_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("attention", "layers"))
    parser.add_argument("names", nargs="?", default="all60_a,all60_b")
    args = parser.parse_args()
    if args.what == "attention":
        attention()
    else:
        layers(args.names.split(","))


if __name__ == "__main__":
    main()
