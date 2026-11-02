"""
src/visualisation/reliability_figure.py

Draws the README reliability diagram from src/calibration/temperature.py's
results. Nothing is computed here.

How to read it:
    Each point is one confidence bin. On the diagonal, a confidence of 80% is
    right 80% of the time; below it, the model is overconfident.
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.config import cli_config  # noqa: E402
from src.visualisation.comparison_figure import THEMES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

PANELS = [("all", "All test sequences"), ("relative_free", "No training relative")]
# Colour slots follow the theme's first two series
STAGES = {"before": ("Before scaling", "probe", "s"), "after": ("After scaling", "lora", "o")}


def ece_label(name: str, score: dict) -> str:
    return f"{name}: ECE {score['ece']:.3f} ({score['ci_low']:.3f} to {score['ci_high']:.3f})"


def plot(results: dict, path: Path, theme_name: str, min_bin_sequences: int) -> None:
    theme = THEMES[theme_name]
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5.6), sharey=True, gridspec_kw={"wspace": 0.06},
        facecolor=theme["surface"],
    )
    for ax, (key, title) in zip(axes, PANELS):
        view = results["views"][key]
        ax.set_facecolor(theme["surface"])
        ax.plot(
            [0, 1], [0, 1], color=theme["secondary"], linewidth=1.2, zorder=1,
            label="Perfect calibration",
        )
        for stage, (name, color, marker) in STAGES.items():
            bins = [b for b in view[stage]["bins"] if b["n_sequences"] >= min_bin_sequences]
            ax.plot(
                [b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins],
                color=theme[color], marker=marker, markersize=7, linewidth=2,
                markeredgecolor=theme["surface"], markeredgewidth=1, zorder=3,
                label=ece_label(name, view[stage]),
            )
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(color=theme["grid"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=theme["secondary"], labelsize=9, length=0)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        for side in ["bottom", "left"]:
            ax.spines[side].set_color(theme["axis"])
        ax.set_xlabel("Mean confidence in the bin", color=theme["secondary"], fontsize=10)
        ax.set_title(
            f"{title}\n{view['n_sequences']:,} sequences, {view['n_clusters']:,} clusters, "
            f"{view['n_classes']} classes",
            color=theme["ink"], fontsize=10, loc="left",
        )
        legend = ax.legend(loc="lower right", frameon=False, fontsize=9, handletextpad=0.5)
        for text in legend.get_texts():
            text.set_color(theme["ink"])
    axes[0].set_ylabel("Accuracy in the bin", color=theme["secondary"], fontsize=10)

    fig.suptitle(
        f"Calibration of ESM-2 + LoRA, temperature {results['temperature']:.2f}",
        x=0.012, y=0.98, ha="left", fontsize=14, fontweight="bold", color=theme["ink"],
    )
    fig.text(
        0.012, 0.905,
        "Homology-aware test split. Temperature fitted on validation. Below the diagonal, the "
        "model is overconfident. Intervals are 95% cluster bootstraps.",
        ha="left", fontsize=9.5, color=theme["secondary"],
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.79, bottom=0.1)
    fig.savefig(path, dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def run(cfg: DictConfig) -> None:
    output_dir = Path(cfg.calibration.output_dir)
    results_path = output_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} not found. Run: uv run python -m src.calibration.temperature"
        )
    results = json.loads(results_path.read_text())
    for theme_name in THEMES:
        path = output_dir / f"reliability_{theme_name}.png"
        plot(results, path, theme_name, cfg.calibration.min_bin_sequences)
        logger.info(f"Figure saved to {path}")


if __name__ == "__main__":
    run(cli_config("Draw the reliability diagram from the calibration results"))
