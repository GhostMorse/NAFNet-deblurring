"""Reproducibility, device, and lightweight model-cost utilities."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> Tuple[float, float]:
    """Approximate multiply-accumulate cost and parameter count of ``model``.

    Counts MACs for conv and linear layers via forward hooks (this covers the
    dominant cost; element-wise ops, attention and deformable-conv sampling are
    not counted, so the figure is a lower bound). ``input_shape`` is the input
    tensor shape *without* the batch dimension, e.g. ``(3, 3, 720, 1280)`` for a
    triplet of RGB frames.

    :returns: ``(gflops, params_in_millions)``.
    """
    macs = 0

    def conv_hook(module, inputs, output):
        nonlocal macs
        out_elems = output.numel()  # N * C_out * H_out * W_out
        kernel = module.kernel_size[0] * module.kernel_size[1]
        macs += out_elems * (module.in_channels // module.groups) * kernel

    def linear_hook(module, inputs, output):
        nonlocal macs
        macs += output.numel() * module.in_features

    handles = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    model(torch.zeros(1, *input_shape, device=device))
    if was_training:
        model.train()
    for h in handles:
        h.remove()

    gflops = 2 * macs / 1e9  # 1 MAC = 2 FLOPs
    params_m = count_parameters(model) / 1e6
    return gflops, params_m
