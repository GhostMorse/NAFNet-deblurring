"""Visualisation helpers for qualitative inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def _to_numpy(img: torch.Tensor):
    """``(C, H, W)`` tensor in ``[0, 1]`` -> ``(H, W, C)`` numpy array."""
    return img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def save_triplet(blur_triplet: torch.Tensor, sharp_target: torch.Tensor, prediction: Optional[torch.Tensor], path: str | Path) -> None:
    """Save a row of [t-1, t, t+1, target, (prediction)] images to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = [(blur_triplet[0], "blur t-1"), (blur_triplet[1], "blur t"), (blur_triplet[2], "blur t+1"), (sharp_target, "sharp target")]
    if prediction is not None:
        images.append((prediction, "prediction"))

    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    for ax, (img, title) in zip(axes, images):
        ax.imshow(_to_numpy(img))
        ax.set_title(title)
        ax.set_axis_off()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_difference_map(prediction: torch.Tensor, target: torch.Tensor, path: str | Path) -> None:
    """Save a heatmap of the per-pixel absolute error between prediction and target."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diff = (prediction.detach().cpu() - target.detach().cpu()).abs().mean(0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(diff.numpy(), cmap="inferno")
    ax.set_title("absolute error")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
