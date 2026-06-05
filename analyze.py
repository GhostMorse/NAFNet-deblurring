"""Analysis utilities: frame-aggregation ablations and a cost/resolution table.

    python analyze.py --model baseline --checkpoint weights/baseline_final.pth --data-root GoPro
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deblurring.data import GoProDataset
from deblurring.engine import evaluate, evaluate_with_transform
from deblurring.models import AlignedNAFNet, UNet
from deblurring.utils import get_device, measure_flops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablations and cost analysis for a deblurring model.")
    parser.add_argument("--model", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("GoPro"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(prefer_cuda=not args.cpu)
    model = (UNet() if args.model == "baseline" else AlignedNAFNet()).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    test_ds = GoProDataset(data_path=str(args.data_root / "test"), transform=None)
    loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # Lower bound: how good is the blurry centre frame already?
    center_psnr = evaluate_with_transform(model, loader, lambda b: torch.stack([b[:, 1]] * 3, dim=1))
    full_psnr, full_ssim = evaluate(model, loader)
    print(f"Model PSNR/SSIM (real triplet):      {full_psnr:.2f} dB / {full_ssim:.4f}")
    print(f"PSNR with 3 copies of centre frame:  {center_psnr:.2f} dB")

    # Frame-aggregation ablations.
    reverse_psnr = evaluate_with_transform(model, loader, lambda b: b[:, [2, 1, 0]])
    print(f"PSNR with reversed frame order:      {reverse_psnr:.2f} dB")

    stride2_ds = GoProDataset(data_path=str(args.data_root / "test"), stride=2, transform=None)
    stride2_loader = DataLoader(stride2_ds, batch_size=4, shuffle=False, num_workers=args.num_workers, drop_last=False)
    stride2_psnr = evaluate_with_transform(model, stride2_loader, lambda b: b)
    print(f"PSNR with stride-2 frames:           {stride2_psnr:.2f} dB")

    # Cost vs resolution.
    print("\nCost vs resolution:")
    for h, w in [(180, 320), (360, 640), (720, 1280)]:
        gflops, params_m = measure_flops(model, (3, 3, h, w))
        print(f"  {h}x{w}: {gflops:.1f} GFLOPs, {params_m:.2f}M params")


if __name__ == "__main__":
    main()
