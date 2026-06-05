"""Multi-frame alignment and fusion modules for the improved model."""

from __future__ import annotations

import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops


class AlignmentDCNv2(nn.Module):
    """Align a neighbour frame's features to the reference using deformable conv.

    Offsets and modulation masks are predicted from the concatenated
    neighbour/reference features, then a single modulated deformable
    convolution (DCNv2) warps the neighbour features toward the reference.
    """

    def __init__(self, channels: int, groups: int = 4) -> None:
        super().__init__()
        self.groups = groups
        out_channels = 3 * 9 * groups  # (offset_x, offset_y, mask) * k*k * groups
        self.offset_mask_conv = nn.Conv2d(channels * 2, out_channels, kernel_size=3, padding=1)
        self.dcn_weight = nn.Parameter(torch.Tensor(channels, channels // groups, 3, 3))
        nn.init.kaiming_uniform_(self.dcn_weight, a=math.sqrt(5))

    def forward(self, feat_nbr: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        out = self.offset_mask_conv(torch.cat([feat_nbr, feat_t], dim=1))
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return ops.deform_conv2d(
            input=feat_nbr, offset=offset, weight=self.dcn_weight, bias=None, padding=(1, 1), mask=mask
        )


class TemporalAttentionFusion(nn.Module):
    """Fuse three aligned frame features with per-pixel softmax attention weights."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, tm1: torch.Tensor, t: torch.Tensor, tp1: torch.Tensor) -> torch.Tensor:
        attn = F.softmax(self.attention_net(torch.cat([tm1, t, tp1], dim=1)), dim=1)
        w_tm1, w_t, w_tp1 = attn[:, 0:1], attn[:, 1:2], attn[:, 2:3]
        return tm1 * w_tm1 + t * w_t + tp1 * w_tp1


class DynamicGlobalFilter(nn.Module):
    """A learnable global filter in the frequency domain (residual).

    Multiplies the rFFT of the feature map by a learnable complex weight and
    transforms back, giving each channel a global receptive field cheaply.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(dim, 1, 1, 2, dtype=torch.float32) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm="ortho")
        x_fft = x_fft * torch.view_as_complex(self.complex_weight)
        return torch.fft.irfft2(x_fft, s=(H, W), norm="ortho") + x
