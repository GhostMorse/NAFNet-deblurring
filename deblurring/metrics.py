"""Image-quality metrics (PSNR and SSIM) for tensors in ``[0, 1]``."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Mean PSNR (dB) over a batch of ``(B, C, H, W)`` images."""
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-10)
    return (10.0 * torch.log10(max_val**2 / mse)).mean().item()


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)
    kernel_2d = (g @ g.t()).unsqueeze(0).unsqueeze(0)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


@torch.no_grad()
def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11, max_val: float = 1.0) -> float:
    """Mean SSIM over a batch, using an 11x11 Gaussian window (per channel)."""
    channels = pred.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, pred.device, pred.dtype)
    pad = window_size // 2

    mu_p = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_t = F.conv2d(target, window, padding=pad, groups=channels)
    mu_p2, mu_t2, mu_pt = mu_p**2, mu_t**2, mu_p * mu_t

    sigma_p2 = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_p2
    sigma_t2 = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_t2
    sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pt

    c1, c2 = (0.01 * max_val) ** 2, (0.03 * max_val) ** 2
    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / ((mu_p2 + mu_t2 + c1) * (sigma_p2 + sigma_t2 + c2))
    return ssim_map.mean().item()
