"""Tests for the PSNR / SSIM metrics."""

import torch

from deblurring.metrics import psnr, ssim


def test_psnr_identical_is_high():
    x = torch.rand(2, 3, 32, 32)
    assert psnr(x, x) > 60.0


def test_psnr_decreases_with_noise():
    x = torch.rand(2, 3, 32, 32)
    noisy = (x + 0.1 * torch.randn_like(x)).clamp(0, 1)
    assert psnr(x, noisy) < psnr(x, x)


def test_ssim_identical_is_one():
    x = torch.rand(2, 3, 64, 64)
    assert ssim(x, x) > 0.99


def test_ssim_in_unit_range():
    x, y = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64)
    val = ssim(x, y)
    assert -1.0 <= val <= 1.0
