"""Improved video-deblurring model with explicit multi-frame alignment.

Where the baseline simply concatenates the three frames, this model extracts
per-frame features, aligns the neighbouring frames to the centre with
deformable convolution, fuses them with per-pixel temporal attention, and then
runs a NAFNet U-Net (with a frequency-domain global filter in the bottleneck).
This directly targets the baseline's main weakness: it cannot compensate for
motion between frames.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.alignment import AlignmentDCNv2, DynamicGlobalFilter, TemporalAttentionFusion
from ..modules.layers import ImprovedSCA
from ..modules.nafnet_block import NAFNetBlock


class AlignedNAFNet(nn.Module):
    """Alignment + temporal-attention fusion + NAFNet U-Net.

    :param width: Top-level channel count.
    :param block: Block factory; defaults to a NAFNet block with ``ImprovedSCA``.
    """

    def __init__(
        self, in_channel: int = 3, out_channel: int = 3, width: int = 24,
        enc_blk_nums: List[int] = [1, 1, 1, 28], dec_blk_nums: List[int] = [1, 1, 1, 1],
        block: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        if block is None:
            block = partial(NAFNetBlock, sca_cls=ImprovedSCA)

        self.feat_extract = nn.Conv2d(in_channel, width, kernel_size=3, padding=1)
        self.align_module = AlignmentDCNv2(channels=width, groups=4)
        self.temporal_fusion = TemporalAttentionFusion(channels=width)

        self.intro = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.ending = nn.Conv2d(width, out_channel, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[block(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(block(chan), DynamicGlobalFilter(dim=chan), block(chan))

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
        _, _, _, H, W = inp.shape
        frame_tm1, frame_t, frame_tp1 = inp[:, 0], inp[:, 1], inp[:, 2]

        feat_tm1 = self.feat_extract(frame_tm1)
        feat_t = self.feat_extract(frame_t)
        feat_tp1 = self.feat_extract(frame_tp1)

        aligned_tm1 = self.align_module(feat_tm1, feat_t)
        aligned_tp1 = self.align_module(feat_tp1, feat_t)
        fused_feat = self.temporal_fusion(aligned_tm1, feat_t, aligned_tp1)

        x = self.pad(fused_feat)
        x = self.intro(x)

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
        return torch.clamp(x[:, :, :H, :W] + frame_t, 0.0, 1.0)
