"""NAFNet block (Non-linear Activation Free)."""

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from .layers import LayerNorm2d, SCA, SimpleGate


class NAFNetBlock(nn.Module):
    """A NAFNet block: a spatial-mixing branch and a channel-mixing (FFN) branch.

    Spatial: ``LayerNorm -> 1x1 -> depthwise 3x3 -> SimpleGate -> SCA -> 1x1``.
    Channel: ``LayerNorm -> 1x1 -> SimpleGate -> 1x1``.
    Each branch is residual, scaled by a learnable per-channel factor
    (``beta`` / ``gamma``) initialised to zero so the block starts as identity
    (LayerScale), which stabilises deep training.

    :param c: Number of channels (preserved by the block).
    :param DW_Expand: Channel-expansion factor of the spatial branch.
    :param FFN_Expand: Channel-expansion factor of the FFN branch.
    :param sca_cls: Channel-attention module to use (``SCA`` or ``ImprovedSCA``).
    """

    def __init__(self, c: int, DW_Expand: int = 2, FFN_Expand: int = 2, sca_cls: Type[nn.Module] = SCA) -> None:
        super().__init__()
        dw_channel = c * DW_Expand
        ffn_channel = FFN_Expand * c

        self.norm1 = LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, dw_channel, kernel_size=1)
        self.dwconv = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel)
        self.sg = SimpleGate()
        self.sca = sca_cls(dw_channel // 2)
        self.pw2 = nn.Conv2d(dw_channel // 2, c, kernel_size=1)

        self.norm2 = LayerNorm2d(c)
        self.pw3 = nn.Conv2d(c, ffn_channel, kernel_size=1)
        self.sg2 = SimpleGate()
        self.pw4 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.pw1(x)
        x = self.dwconv(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.pw2(x)
        x = inp + x * self.beta

        y = self.norm2(x)
        y = self.pw3(y)
        y = self.sg2(y)
        y = self.pw4(y)
        return x + y * self.gamma
