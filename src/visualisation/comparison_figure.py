"""
src/visualisation/comparison_figure.py

Draws the README figure from the results of experiments/model_comparison (E3).
Nothing is computed here: every point and interval comes from results.json.

What this script does:
    1. Left panel: macro-F1 of each model, per subset and identity bin
    2. Right panel: LoRA minus each baseline, with a line at zero
    3. Saves one PNG per theme, so the README can follow the reader's light or
       dark mode

Why points and not a line across the bins?
    Each bin holds a different set of classes, so macro-F1 is not strictly
    comparable between rows. A line would suggest a trend the data cannot show.
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.config import cli_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Top to bottom. None leaves a gap between the summaries and the identity bins.
ROWS = [
    ("all", "All test sequences"),
    ("relative_free", "No training relative"),
    None,
    ("no_hit", "No hit"),
    ("<30", "Below 30%"),
    ("30-40", "30 to 40%"),
    ("40-50", "40 to 50%"),
    ("50-70", "50 to 70%"),
    (">=70", "70% and above"),
]
MODELS = {
    "lora": ("ESM-2 + LoRA", "o"),
    "probe": ("Linear probe", "s"),
    "neighbour": ("Nearest neighbour", "D"),
}
DIFFERENCES = {
    "lora_minus_probe": ("LoRA minus probe", "probe", "s"),
    "lora_minus_neighbour": ("LoRA minus neighbour", "neighbour", "D"),
}
# Checked for colour-blind separation on both surfaces
THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e", "grid": "#e1e0d9",
        "axis": "#c3c2b7", "lora": "#2a78d6", "probe": "#eb6834", "neighbour": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "secondary": "#c3c2b7", "grid": "#2c2c2a",
        "axis": "#383835", "lora": "#3987e5", "probe": "#d95926", "neighbour": "#199e70",
    },
}


def row_label(name: str, view: dict) -> str:
    return f"{name}\n{view['n_sequences']:,} sequences, {view['n_classes']} classes"


def draw_interval(ax, y, score, color, marker, surface, label=None) -> None:
    """A point with its 95% interval; the surface-coloured edge separates close marks."""
    value = score["macro_f1"] if "macro_f1" in score else score["difference"]
    ax.errorbar(
        value, y,
        xerr=[[value - score["ci_low"]], [score["ci_high"] - value]],
        fmt=marker, color=color, markersize=7, markeredgecolor=surface, markeredgewidth=1,
        elinewidth=1.5, capsize=0, label=label, zorder=3,
    )


def style_axis(ax, theme: dict, positions: list[int]) -> None:
    ax.set_facecolor(theme["surface"])
    ax.set_ylim(max(positions) + 0.6, min(positions) - 0.6)
    ax.grid(axis="x", color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=theme["secondary"], labelsize=9, length=0)
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])


def plot(results: dict, path: Path, theme_name: str) -> None:
    theme = THEMES[theme_name]
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    views = results["views"]
    rows = [
        (i, *row) for i, row in enumerate(ROWS)
        if row is not None and views[row[0]].get("n_sequences")
    ]
    positions = [row[0] for row in rows]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 6.6), sharey=True, gridspec_kw={"wspace": 0.06},
        facecolor=theme["surface"],
    )
    for ax in (left, right):
        style_axis(ax, theme, positions)

    offsets = dict(zip(MODELS, [-0.24, 0.0, 0.24]))
    for model, (label, marker) in MODELS.items():
        for n, (y, key, _) in enumerate(rows):
            draw_interval(
                left, y + offsets[model], views[key]["models"][model], theme[model], marker,
                theme["surface"], label=label if n == 0 else None,
            )

    offsets = dict(zip(DIFFERENCES, [-0.14, 0.14]))
    for key, (label, baseline, marker) in DIFFERENCES.items():
        for n, (y, view, _) in enumerate(rows):
            draw_interval(
                right, y + offsets[key], views[view][key]["macro_f1"], theme[baseline], marker,
                theme["surface"], label=label if n == 0 else None,
            )

    left.set_yticks(positions, [row_label(name, views[key]) for _, key, name in rows])
    left.tick_params(axis="y", labelcolor=theme["ink"], labelsize=9.5)
    left.set_xlim(-0.03, 1.03)
    left.set_xlabel("Test macro-F1", color=theme["secondary"], fontsize=10)

    right.axvline(0, color=theme["secondary"], linewidth=1.2, zorder=1)
    right.set_xlim(-0.3, 0.5)
    right.set_xlabel(
        "Paired difference in macro-F1  (right of zero: LoRA better)",
        color=theme["secondary"], fontsize=10,
    )
    # The identity bins sit under their own heading, in the gap between the groups
    gap = ROWS.index(None)
    left.annotate(
        "Identity of the best training hit", xy=(0, gap + 0.15),
        xycoords=left.get_yaxis_transform(), xytext=(-8, 0), textcoords="offset points",
        ha="right", va="center", fontsize=9.5, fontweight="bold", color=theme["ink"],
    )

    for ax in (left, right):
        legend = ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0),
            ncol=len(ax.get_legend_handles_labels()[1]),
            frameon=False, fontsize=9.5, handletextpad=0.3, columnspacing=1.2,
        )
        for text in legend.get_texts():
            text.set_color(theme["ink"])

    fig.suptitle(
        "ESM-2 + LoRA against both baselines", x=0.012, y=0.985, ha="left",
        fontsize=14, fontweight="bold", color=theme["ink"],
    )
    fig.text(
        0.012, 0.925,
        f"Homology-aware test split, {views['all']['n_clusters']:,} clusters. Horizontal lines "
        "are 95% stratified cluster-bootstrap intervals. Single training run.",
        ha="left", fontsize=9.5, color=theme["secondary"],
    )
    fig.subplots_adjust(left=0.2, right=0.985, top=0.83, bottom=0.09)
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def run(cfg: DictConfig) -> None:
    output_dir = Path(cfg.comparison.output_dir)
    results_path = output_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run: "
            "uv run python experiments/model_comparison/run_comparison.py"
        )
    results = json.loads(results_path.read_text())
    for theme_name in THEMES:
        path = output_dir / f"model_comparison_{theme_name}.png"
        plot(results, path, theme_name)
        logger.info(f"Figure saved to {path}")


if __name__ == "__main__":
    run(cli_config("Draw the README figure from the model comparison results"))
