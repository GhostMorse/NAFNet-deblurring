"""Output-shape tests for the baseline and improved models."""

import torch

from deblurring.models import AlignedNAFNet, UNet


def test_unet_output_shape():
    model = UNet()
    x = torch.rand(1, 3, 3, 128, 128)  # (B, T, C, H, W)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 128, 128)


def test_unet_handles_non_multiple_resolution():
    model = UNet()
    x = torch.rand(1, 3, 3, 130, 70)  # not a multiple of 16 -> internal padding
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 130, 70)


def test_aligned_nafnet_output_shape():
    model = AlignedNAFNet()
    x = torch.rand(1, 3, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 64, 64)
    assert torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0  # output is clamped to [0, 1]
