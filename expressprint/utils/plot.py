from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_layer_ax_vit_sub(
    ax, mean, model_name, colors=["royalblue", "mediumorchid", "hotpink", "darkorange", "forestgreen", "white"]
):
    model_title = {
        "dinov2_reg": "DINOv2-reg",
        "dinov2": "DINOv2",
        "mae": "MAE",
        "open_clip": "Open CLIP",
        "openai_clip": "OpenAI CLIP",
        "vit_orig": "ViT",
        "samvit": "SAM-ViT",
    }

    x_axis = np.arange(mean.shape[-1]) + 1

    for i in range(mean.shape[0]):
        ax.plot(
            x_axis,
            mean[i],
            label=f"Top {i+1}" if i < 5 else "median",
            color=colors[i],
            linestyle="-",
            marker="o" if i < 5 else "v",
            markerfacecolor="none",
            markersize=5,
        )

    ax.set_title(model_title.get(model_name, model_name), fontsize=18, fontweight="bold", color="white")
    ax.set_ylabel("Magnitudes", fontsize=18, color="white")
    ax.set_xlabel("Blocks", fontsize=18, labelpad=4.0, color="white")

    num_layers = mean.shape[1]
    xtick_label = [1, num_layers // 4, num_layers // 2, num_layers * 3 // 4, num_layers]
    ax.set_xticks(xtick_label)
    ax.set_xticklabels(xtick_label, fontsize=16, color="white")
    ax.set_yticklabels(ax.get_yticks(), fontsize=16, color="white")

    ax.tick_params(axis="x", which="major", pad=2.0, colors="white")
    ax.tick_params(axis="y", which="major", pad=0.4, colors="white")

    ax.grid(axis="x", color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", color="gray", linestyle="--", alpha=0.3)

    ax.set_facecolor("#323A48")
    ax.legend(facecolor="#323A48", edgecolor="white", labelcolor="white", fontsize=12, loc="upper left")


def plot_layer_ax_vit(mean, model_name, plot_path: Optional[str] = None):
    fig = plt.figure(figsize=(8, 6))
    fig.set_facecolor("#323A48")
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.0)

    ax = fig.add_subplot(1, 1, 1)
    plot_layer_ax_vit_sub(ax, mean, model_name)

    if plot_path:
        plot_path = Path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, bbox_inches="tight", dpi=200, facecolor=fig.get_facecolor())
    else:
        plt.show()
