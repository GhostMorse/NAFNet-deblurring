"""Tests for the loss functions."""

import torch

from deblurring.losses import CharbonnierLoss, DeblurLoss, EdgeLoss, FFTLoss


def test_charbonnier_small_when_equal():
    x = torch.rand(2, 3, 16, 16)
    assert CharbonnierLoss(eps=1e-3)(x, x).item() < 1e-2


def test_fft_and_edge_zero_when_equal():
    x = torch.rand(2, 3, 16, 16)
    assert FFTLoss()(x, x).item() < 1e-4
    assert EdgeLoss()(x, x).item() < 1e-6


def test_deblur_loss_returns_components():
    pred, target = torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32)
    total, c, f, e = DeblurLoss(w_charb=300.0, w_fft=1.0, w_edge=50.0)(pred, target)
    for t in (total, c, f, e):
        assert t.ndim == 0 and torch.isfinite(t)
    assert total.item() > 0
