"""Baseline NAFNet U-Net for video deblurring.

The three input frames are concatenated along the channel axis (the naive
multi-frame aggregation) and fed through a NAFNet-style encoder/decoder. The
network predicts a residual that is added to the centre blurry frame.
"""

from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.nafnet_block import NAFNetBlock


class UNet(nn.Module):
    """NAFNet U-Net.

    :param block: Callable ``block(channels) -> nn.Module`` that preserves shape.
    :param in_channel: Input channels (``3 frames x 3 = 9``).
    :param out_channel: Output channels (``3``).
    :param width: Channel count at the top level.
    :param middle_blk_num: Number of blocks in the bottleneck.
    :param enc_blk_nums / dec_blk_nums: Blocks per encoder / decoder stage.
    """

    def __init__(
        self, block: Callable[[int], nn.Module] = NAFNetBlock, in_channel: int = 9, out_channel: int = 3,
        width: int = 16, middle_blk_num: int = 1, enc_blk_nums: List[int] = [1, 1, 1, 28], dec_blk_nums: List[int] = [1, 1, 1, 1],
    ) -> None:
        super().__init__()
        self.intro = nn.Conv2d(in_channel, width, kernel_size=3, padding=1)
        self.ending = nn.Conv2d(width, out_channel, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[block(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[block(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.ConvTranspose2d(chan, chan // 2, 2, 2))
            chan //= 2
            self.decoders.append(nn.Sequential(*[block(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """:param inp: ``(B, T, C, H, W)`` blurry triplet -> ``(B, C, H, W)`` sharp centre."""
        B, T, C, H, W = inp.shape
        inp = inp.reshape(B, T * C, H, W)

        inp = self.pad(inp)
        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up in zip(self.decoders, self.ups):
            x = up(x)
            x = x + encs.pop()
            x = decoder(x)

        x = self.ending(x)
        x = x + inp[:, 3:6]  # residual w.r.t. the centre (t) frame
        return x[:, :, :H, :W]
