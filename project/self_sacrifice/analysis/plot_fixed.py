"""Plot the fixed pairs: the density of P(give) without reasoning (`first`
and `stated` formats), split by what the model does on the same item when it
reasons (`free`).

    gives          it gives after reasoning (all of these are `helped` items)
    pass-helped    it passes after reasoning, on a `helped` item
    pass-control   it passes after reasoning, on a control item

P(give) is plotted as log10. Items whose P(give) is 0 (giving is not among
the listed alternatives) cannot be placed on a log axis; they are counted in
the panel's note, and each curve's area is the share of its group that could
be placed.

    .venv/bin/python -m project.self_sacrifice.analysis.plot_fixed fixed
    .venv/bin/python -m project.self_sacrifice.analysis.plot_fixed own --helped

This is jrwong's `plot_fixed_kde.py` and, with `--helped`, the first two
panels of his `plot_fixed_kde_helped.py`. The Gaussian density is written out
here because scipy is not installed. No API calls.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from project.self_sacrifice.analysis.analyze_fixed import auc  # noqa: E402
from project.self_sacrifice.config import FIXED  # noqa: E402

__all__ = ["GROUPS", "TITLES", "BANDWIDTH", "density", "groups", "figure"]

# group -> (colour, legend)
GROUPS: dict[str, tuple[str, str]] = {
    "gives": ("#2a78d6", "Gives after reasoning"),
    "pass-helped": ("#eb6834", "Passes after reasoning, helped"),
    "pass-control": ("#1baf7a", "Passes after reasoning, controls"),
}
TITLES = {
    "first": "Decide first, no explanation",
    "stated": "Copy the cost sentence, then decide",
}
BANDWIDTH = 0.35
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def density(values: list[float], xs: np.ndarray, factor: float = BANDWIDTH):
    """A Gaussian kernel density of `values` at `xs`, with the bandwidth a
    fixed multiple of their standard deviation."""
    if len(values) < 2 or np.std(values, ddof=1) == 0:
        return np.zeros_like(xs)
    width = factor * float(np.std(values, ddof=1))
    z = (xs[:, None] - np.asarray(values)[None, :]) / width
    return np.exp(-0.5 * z**2).sum(axis=1) / (
        len(values) * width * math.sqrt(2 * math.pi)
    )


def groups(rows: list[dict], reply_format: str, helped_only: bool = False):
    """group -> P(give) of every `reply_format` trial, by the same item's
    decision in the `free` format."""
    free = {
        row["id"].removesuffix("-free"): row["decision"]
        for row in rows
        if row["format"] == "free"
    }
    out: dict[str, list[float]] = {key: [] for key in GROUPS}
    for row in rows:
        decided = free.get(row["id"].removesuffix(f"-{reply_format}"))
        if row["format"] != reply_format or row["p_give"] is None or decided is None:
            continue
        if helped_only and row["history"] != "helped":
            continue
        if decided == "give":
            out["gives"].append(row["p_give"])
        elif row["history"] == "helped":
            out["pass-helped"].append(row["p_give"])
        else:
            out["pass-control"].append(row["p_give"])
    return out


def _auc(positives: list[float], negatives: list[float]) -> str:
    value = auc(positives, negatives)
    return "-" if value is None else f"{value:.2f}"


def figure(rows: list[dict], title: str, helped_only: bool = False):
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.edgecolor": GRID,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.6), sharex=True, facecolor=SURFACE)
    xs = np.linspace(-12, 0.5, 600)
    for ax, reply_format in zip(axes, TITLES):
        ax.set_facecolor(SURFACE)
        by_group = groups(rows, reply_format, helped_only)
        notes = []
        for i, (key, (colour, name)) in enumerate(GROUPS.items()):
            ps = by_group[key]
            if helped_only and key == "pass-control":
                continue
            seen = [math.log10(p) for p in ps if p > 0]
            notes.append(f"{len(ps) - len(seen)}/{len(ps)} {name.lower()}")
            if not ps:
                continue
            curve = density(seen, xs) * len(seen) / len(ps)
            ax.fill_between(xs, curve, color=colour, alpha=0.12, lw=0)
            ax.plot(xs, curve, color=colour, lw=2, label=f"{name} (n={len(ps)})")
            ax.plot(
                seen,
                np.full(len(seen), -0.012 - 0.018 * i),
                "|",
                color=colour,
                ms=6,
                mew=1,
            )
        gives, helped = by_group["gives"], by_group["pass-helped"]
        passes = helped + by_group["pass-control"]
        ax.set_title(
            TITLES[reply_format],
            loc="left",
            fontsize=11,
            fontweight="bold",
            color=INK,
            pad=30,
        )
        ax.text(
            0,
            1.02,
            f"gives vs all passes AUC {_auc(gives, passes)} · gives vs helped "
            f"passes AUC {_auc(gives, helped)}\n"
            f"P(give) is 0, not plotted: {'; '.join(notes)}",
            transform=ax.transAxes,
            fontsize=8.5,
            color=INK2,
            va="bottom",
        )
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_ylabel("density × share plotted")
        ax.set_ylim(-0.07, None)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1].set_xlabel("log10 P(give) where the gifts list opens, no reasoning")
    axes[1].set_xlim(-12, 0.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", choices=("fixed", "own"))
    parser.add_argument("--helped", action="store_true", help="helped items only")
    args = parser.parse_args()
    source = FIXED / f"{args.version}.jsonl"
    trials = [json.loads(line) for line in source.read_text().splitlines() if line]
    out: Path = FIXED / f"{args.version}_kde{'_helped' if args.helped else ''}.png"
    figure(
        trials,
        f"Gemma-4-31B: P(give) before reasoning, by the decision after reasoning "
        f"({args.version} round 1{', helped only' if args.helped else ''})",
        args.helped,
    ).savefig(out, dpi=160, facecolor=SURFACE)
    print(out)
