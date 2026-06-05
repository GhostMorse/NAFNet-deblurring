"""Tests for the NAFNet building blocks (ported from the notebook's checks)."""

import torch

from deblurring.modules import NAFNetBlock, SCA, SimpleGate


def test_simple_gate_shape_and_values():
    sg = SimpleGate()
    assert sg(torch.rand(16, 64, 32, 32)).shape == (16, 32, 32, 32)
    assert torch.allclose(sg(torch.ones(2, 4, 8, 8)), torch.ones(2, 2, 8, 8))
    mixed = torch.cat([torch.ones(1, 2, 4, 4) * 3, torch.ones(1, 2, 4, 4) * 5], dim=1)
    assert torch.allclose(sg(mixed), torch.ones(1, 2, 4, 4) * 15)


def test_sca_preserves_shape_and_grad():
    out = SCA(64)(torch.rand(4, 64, 32, 32))
    assert out.shape == (4, 64, 32, 32)
    small = SCA(8)(torch.rand(2, 8, 16, 16))
    assert small.shape == (2, 8, 16, 16)
    assert small.requires_grad


def test_nafnet_block_shape_and_finite():
    block = NAFNetBlock(64)
    assert block(torch.rand(2, 64, 32, 32)).shape == (2, 64, 32, 32)

    zeros_out = NAFNetBlock(16)(torch.zeros(1, 16, 8, 8))
    assert zeros_out.shape == (1, 16, 8, 8)
    assert torch.isfinite(zeros_out).all()
    assert sum(p.numel() for p in block.parameters()) > 0


def test_nafnet_block_starts_as_identity():
    # beta and gamma are zero-initialised, so an untrained block is identity.
    block = NAFNetBlock(16)
    x = torch.rand(1, 16, 8, 8)
    assert torch.allclose(block(x), x, atol=1e-6)
