"""Training and evaluation loops."""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import psnr as compute_psnr
from .metrics import ssim as compute_ssim


def train_epoch(model, loader, optimizer, scheduler, device, criterion, epoch: int) -> float:
    """One training epoch. ``criterion`` may return a tensor or a tuple whose
    first element is the total loss (e.g. :class:`~deblurring.losses.DeblurLoss`)."""
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        blur = batch["blur_triplet"].to(device)
        sharp = batch["sharp_target"].to(device)

        optimizer.zero_grad()
        output = model(blur)
        loss = criterion(output, sharp)
        if isinstance(loss, tuple):
            loss = loss[0]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
    return total_loss / len(loader)


def train(
    model, train_loader: DataLoader, optimizer, scheduler, device, criterion,
    num_epochs: int = 100, save_path: Optional[str] = None, save_epoch: int = 10,
) -> None:
    """Full training loop with periodic checkpointing."""
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, device, criterion, epoch)
        lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}, Loss: {loss:.4f}, LR: {lr:.2e}")
        if save_path and epoch % save_epoch == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, os.path.join(save_path, f"epoch_{epoch}.pth"))


@torch.no_grad()
def evaluate(model, dataloader: DataLoader) -> Tuple[float, float]:
    """Mean PSNR and SSIM over a dataloader."""
    device = next(model.parameters()).device
    model.eval()
    total_psnr = total_ssim = 0.0
    for batch in tqdm(dataloader, desc="eval", leave=False):
        blur = batch["blur_triplet"].to(device)
        sharp = batch["sharp_target"].to(device)
        pred = model(blur).clamp(0, 1)
        total_psnr += compute_psnr(pred, sharp)
        total_ssim += compute_ssim(pred, sharp)
    return total_psnr / len(dataloader), total_ssim / len(dataloader)


@torch.no_grad()
def evaluate_with_transform(model, dataloader: DataLoader, triplet_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
    """Mean PSNR after transforming each input triplet with ``triplet_fn``.

    Useful for frame-aggregation ablations (e.g. feeding three copies of the
    centre frame, or reversing the temporal order).
    """
    device = next(model.parameters()).device
    model.eval()
    total = 0.0
    for batch in tqdm(dataloader, desc="eval", leave=False):
        blur = batch["blur_triplet"].to(device)
        sharp = batch["sharp_target"].to(device)
        pred = model(triplet_fn(blur)).clamp(0, 1)
        total += compute_psnr(pred, sharp)
    return total / len(dataloader)
