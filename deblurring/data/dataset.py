"""GoPro video-deblurring dataset.

Each sample is a triplet of consecutive blurry frames ``(t-1, t, t+1)`` and the
sharp ground-truth frame at time ``t``. The GoPro layout is::

    <root>/<video>/blur_gamma/*.png
    <root>/<video>/sharp/*.png

with blurry and sharp frames sharing filenames.
"""

from __future__ import annotations

import os
from glob import glob
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GoProDataset(Dataset):
    """Triplets of blurry frames with the sharp centre frame as target.

    :param data_path: Root directory containing the per-video folders.
    :param stride: Temporal gap between the frames of a triplet. ``stride=1``
        uses ``(t-1, t, t+1)``; ``stride=2`` uses ``(t-2, t, t+2)``.
    :param transform: Optional Albumentations transform built with
        ``additional_targets`` for the extra frames (see
        :func:`deblurring.data.transforms.build_train_transform`).
    """

    def __init__(self, data_path: str = os.path.join("GoPro", "train"), stride: int = 1, transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.transform = transform
        self.blur_triplets: list[list[str]] = []
        self.sharp_targets: list[str] = []

        for video in sorted(os.listdir(data_path)):
            video_dir = os.path.join(data_path, video)
            sharp_paths = sorted(glob(os.path.join(video_dir, "sharp", "*.png")))
            blur_paths = sorted(glob(os.path.join(video_dir, "blur_gamma", "*.png")))
            assert len(sharp_paths) == len(blur_paths), f"frame count mismatch in {video_dir}"

            for i in range(len(blur_paths) - 2 * stride):
                self.blur_triplets.append([blur_paths[i], blur_paths[i + stride], blur_paths[i + 2 * stride]])
                self.sharp_targets.append(sharp_paths[i + stride])

    def __len__(self) -> int:
        return len(self.blur_triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        blur_tm1 = self._load(self.blur_triplets[idx][0])
        blur_t = self._load(self.blur_triplets[idx][1])
        blur_tp1 = self._load(self.blur_triplets[idx][2])
        sharp_t = self._load(self.sharp_targets[idx])

        if self.transform is not None:
            out = self.transform(image=blur_t, image1=blur_tm1, image2=blur_tp1, sharp=sharp_t)
            blur_t, blur_tm1, blur_tp1, sharp_t = out["image"], out["image1"], out["image2"], out["sharp"]

        return {
            "blur_triplet": torch.stack([self._to_tensor(blur_tm1), self._to_tensor(blur_t), self._to_tensor(blur_tp1)]),
            "sharp_target": self._to_tensor(sharp_t),
        }

    @staticmethod
    def _load(path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"))

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)
