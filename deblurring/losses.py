"""Loss functions for image restoration.

A simple L1 loss is a strong baseline. The composite :class:`DeblurLoss`
combines a Charbonnier (smooth-L1-like) term with a frequency-domain term and
an edge term, which together tend to produce sharper, less over-smoothed
results than L1 or L2 alone.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier loss: ``sqrt((pred - target)^2 + eps^2)``, a robust L1 surrogate."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))


class FFTLoss(nn.Module):
    """L1 distance between the 2-D Fourier transforms of prediction and target."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="backward")
        target_fft = torch.fft.rfft2(target, norm="backward")
        return torch.mean(torch.abs(pred_fft - target_fft))


class EdgeLoss(nn.Module):
    """L1 distance between Sobel edge maps of prediction and target."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight_x", sobel_x.repeat(3, 1, 1, 1))
        self.register_buffer("weight_y", sobel_y.repeat(3, 1, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = F.conv2d(pred, self.weight_x, padding=1, groups=3)
        pred_dy = F.conv2d(pred, self.weight_y, padding=1, groups=3)
        target_dx = F.conv2d(target, self.weight_x, padding=1, groups=3)
        target_dy = F.conv2d(target, self.weight_y, padding=1, groups=3)
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class DeblurLoss(nn.Module):
    """Weighted sum of Charbonnier, FFT and edge losses.

    ``forward`` returns ``(total, charbonnier, fft, edge)`` so the individual
    terms can be logged.
    """

    def __init__(self, w_charb: float = 1.0, w_fft: float = 0.1, w_edge: float = 0.05) -> None:
        super().__init__()
        self.charb = CharbonnierLoss()
        self.fft = FFTLoss()
        self.edge = EdgeLoss()
        self.w_charb, self.w_fft, self.w_edge = w_charb, w_fft, w_edge

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_c = self.charb(pred, target)
        loss_f = self.fft(pred, target)
        loss_e = self.edge(pred, target)
        total = self.w_charb * loss_c + self.w_fft * loss_f + self.w_edge * loss_e
        return total, loss_c, loss_f, loss_e
