"""Core NAFNet layers: channel-wise LayerNorm, SimpleGate and channel attention."""

from __future__ import annotations

import torch
import torch.nn as nn


class _LayerNormFunction(torch.autograd.Function):
    """Channel-wise LayerNorm with an explicit (memory-efficient) backward."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, C, _, _ = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        return weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        _, C, _, _ = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of an ``(N, C, H, W)`` tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Split channels in half and multiply: ``x1 * x2`` (a non-linearity-free gate)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    """Simplified Channel Attention (NAFNet): global average pool -> 1x1 conv -> scale."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class ImprovedSCA(nn.Module):
    """SE-style channel attention: pool -> 1x1 -> ReLU -> 1x1 -> sigmoid -> scale.

    A small bottleneck (``reduction``) and a sigmoid gate make the channel
    weights data-dependent in ``[0, 1]``; used by the improved model.
    """

    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid_channels = max(in_channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.sigmoid(self.conv2(self.relu(self.conv1(self.pool(x)))))
        return x * w
