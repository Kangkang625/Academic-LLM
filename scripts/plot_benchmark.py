"""Generate the grouped benchmark chart from the recorded evaluation results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def main():
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    datasets = ["数据问答", "多跳问答", "论文问答"]
    series = [
        ("我们的模型（训练后）", [51.00, 64.29, 82.69], "#5368f5"),
        ("Qwen（训练前）", [39.00, 57.14, 59.62], "#9daaf5"),
        ("GPT-6.1-sol", [31.00, 64.29, 75.00], "#c4c4c4"),
        ("DeepSeek Flash", [33.00, 71.43, 73.08], "#e0e0e0"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.2), facecolor="white")
    positions = np.arange(len(datasets))
    width = 0.16
    for index, (label, values, color) in enumerate(series):
        bars = ax.bar(positions + (index - 1.5) * width, values, width * 0.91,
                      color=color, label=label, zorder=3)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0,
                    f"{value:.2f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if index == 0 else "normal", color="#242424")

    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(datasets) - 0.5)
    ax.set_xticks(positions, datasets, fontsize=11)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([])
    ax.grid(axis="y", color="#ececec", linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, pad=8)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#bdbdbd")
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.035), ncol=4,
                       frameon=False, fontsize=10, handlelength=1.5, columnspacing=2)
    legend.get_texts()[0].set_color("#5368f5")
    legend.get_texts()[0].set_fontweight("bold")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.85, bottom=0.14)
    destination = Path(__file__).resolve().parents[1] / "assets" / "benchmark.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
