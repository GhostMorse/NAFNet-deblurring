"""Train a deblurring model on the GoPro dataset.

Expected layout (pass the parent via ``--data-root``)::

    GoPro/train/<video>/{blur_gamma,sharp}/*.png
    GoPro/test/<video>/{blur_gamma,sharp}/*.png

Examples
--------
    # baseline NAFNet U-Net (L1 loss)
    python train.py --model baseline --data-root GoPro --epochs 100

    # improved aligned model (Charbonnier + FFT + edge loss)
    python train.py --model improved --data-root GoPro --epochs 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deblurring.data import GoProDataset, build_train_transform
from deblurring.engine import train
from deblurring.losses import DeblurLoss
from deblurring.models import AlignedNAFNet, UNet
from deblurring.utils import count_parameters, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GoPro video-deblurring model.")
    parser.add_argument("--model", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--data-root", type=Path, default=Path("GoPro"))
    parser.add_argument("--save-path", type=Path, default=Path("weights"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-epoch", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    print(f"Device: {device}")

    dataset = GoProDataset(
        data_path=str(args.data_root / "train"), transform=build_train_transform(args.crop_size)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    print(f"Training samples: {len(dataset)}")

    if args.model == "baseline":
        model = UNet().to(device)
        criterion: nn.Module = nn.L1Loss()
    else:
        model = AlignedNAFNet().to(device)
        criterion = DeblurLoss(w_charb=300.0, w_fft=1.0, w_edge=50.0).to(device)
    print(f"Model: {args.model} | trainable params: {count_parameters(model) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(loader) * args.epochs)

    train(
        model, loader, optimizer, scheduler, device, criterion,
        num_epochs=args.epochs, save_path=str(args.save_path), save_epoch=args.save_epoch,
    )

    args.save_path.mkdir(parents=True, exist_ok=True)
    final_path = args.save_path / f"{args.model}_final.pth"
    torch.save({"model_state_dict": model.state_dict()}, final_path)
    print(f"Saved final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
