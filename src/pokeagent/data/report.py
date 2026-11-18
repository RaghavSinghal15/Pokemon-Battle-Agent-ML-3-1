"""Figures for reports/data_report.md, drawn from reports/results/data_summary.json.

    python -m pokeagent.data.report

The summary is written by `python -m pokeagent.data.build`; this script only reads it, so the
figures can be redrawn without the data (the summary holds counts, not battles).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt  # noqa: E402

from pokeagent.config import REPORTS_DIR  # noqa: E402

SUMMARY_PATH = REPORTS_DIR / "results" / "data_summary.json"
FIGURES_DIR = REPORTS_DIR / "figures"

# Colours: one blue for the data, grey for what a rule removes, dark ink for text and markers.
BLUE = "#2a78d6"
GREY = "#c3c2b7"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def rating_figure(summary: dict, path: Path) -> None:
    """Histogram of player ratings in usable games, with the chosen cut-off."""
    hist = summary["ratings_before_rating_filter"]["player_rating_histogram_50"]
    cutoff = summary["filtering"]["min_rating"]
    lefts = [int(k) for k in hist]
    counts = list(hist.values())
    # Colour a 50-point bin blue if most of it is at or above the cut-off.
    colours = [BLUE if left + 25 >= cutoff else GREY for left in lefts]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(lefts, counts, width=46, align="edge", color=colours)
    ax.axvline(cutoff, color=INK, linewidth=1.2, linestyle="--")
    top = max(counts)
    ax.text(
        cutoff - 15, top * 0.97, f"cut-off {cutoff:,}", color=INK, fontsize=9, va="top", ha="right"
    )
    ax.text(1010, top * 1.06, "below the cut-off", color=MUTED, fontsize=9, va="top")
    ax.text(2230, top * 0.75, "kept, if the other\nplayer is too", color=INK, fontsize=9, va="top")
    ax.set_ylim(0, top * 1.1)
    ax.set_xlabel("Player rating (50-point bins)", color=MUTED, fontsize=9)
    ax.set_ylabel("Players", color=MUTED, fontsize=9)
    ax.set_title(
        "Player ratings in usable games: two groups, casual (~1,000-1,200) and strong (~1,900+)",
        loc="left",
        fontsize=10,
        color=INK,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def funnel_figure(summary: dict, path: Path) -> None:
    """Battles left after each filtering rule."""
    steps = summary["filtering"]["counts"]
    labels = [name.replace("after ", "").replace("keeping ", "") for name, _ in steps]
    counts = [n for _, n in steps]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    y = list(range(len(steps)))[::-1]  # first step at the top
    ax.barh(y, counts, height=0.6, color=BLUE)
    for yi, n in zip(y, counts, strict=True):
        ax.text(n + max(counts) * 0.01, yi, f"{n:,}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.set_xlabel("Battles", color=MUTED, fontsize=9)
    ax.set_title("Battles left after each filtering rule", loc="left", fontsize=10, color=INK)
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def turns_figure(summary: dict, path: Path) -> None:
    """Game length of the kept battles (the last bin holds every game of 100+ turns)."""
    hist = summary["kept"]["turns_histogram_5"]
    lefts = [int(k) for k in hist]
    counts = list(hist.values())
    median = summary["kept"]["turns"]["p50"]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(lefts, counts, width=4.6, align="edge", color=BLUE)
    ax.axvline(median, color=INK, linewidth=1.2, linestyle="--")
    ax.text(median + 1, max(counts) * 0.97, f"median {median:.0f}", color=INK, fontsize=9, va="top")
    ticks = list(range(0, 101, 10))
    ax.set_xticks(ticks, [str(t) if t < 100 else "100+" for t in ticks])
    ax.set_xlabel("Turns played (5-turn bins)", color=MUTED, fontsize=9)
    ax.set_ylabel("Battles", color=MUTED, fontsize=9)
    ax.set_title("How long the kept battles last", loc="left", fontsize=10, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(summary_path: Path = SUMMARY_PATH, out_dir: Path = FIGURES_DIR) -> list[Path]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "data_ratings.png",
        out_dir / "data_filtering.png",
        out_dir / "data_turns.png",
    ]
    rating_figure(summary, paths[0])
    funnel_figure(summary, paths[1])
    turns_figure(summary, paths[2])
    return paths


if __name__ == "__main__":
    for written in make_figures():
        print(f"Wrote {written}")
