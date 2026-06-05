"""Evaluate a trained deblurring model, or deblur a folder of frames.

Evaluate on a GoPro-format split (reports PSNR/SSIM, optionally saves examples):
    python deblur.py --model baseline --checkpoint weights/baseline_final.pth \
        --data-root GoPro --save-vis vis --num-vis 3

Deblur a directory of consecutive blurry frames (writes sharp centre frames):
    python deblur.py --model improved --checkpoint weights/improved_final.pth \
        --frames-dir my_blurry_frames --output-dir my_deblurred
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from deblurring.data import GoProDataset
from deblurring.engine import evaluate
from deblurring.models import AlignedNAFNet, UNet
from deblurring.utils import get_device
from deblurring.viz import save_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate / run a deblurring model.")
    parser.add_argument("--model", choices=["baseline", "improved"], default="baseline")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("GoPro"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Folder of consecutive blurry PNG frames to deblur.")
    parser.add_argument("--output-dir", type=Path, default=Path("deblurred"))
    parser.add_argument("--save-vis", type=Path, default=None, help="Save qualitative comparison images here.")
    parser.add_argument("--num-vis", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def build_model(name: str, checkpoint: Path, device: torch.device):
    model = (UNet() if name == "baseline" else AlignedNAFNet()).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0)


@torch.no_grad()
def deblur_frames(model, frames_dir: Path, output_dir: Path, device: torch.device) -> None:
    paths = sorted(glob(str(frames_dir / "*.png")) + glob(str(frames_dir / "*.jpg")))
    if len(paths) < 3:
        raise SystemExit("Need at least 3 consecutive frames.")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [_to_tensor(np.array(Image.open(p).convert("RGB"))) for p in paths]
    for i in range(1, len(frames) - 1):
        triplet = torch.stack([frames[i - 1], frames[i], frames[i + 1]]).unsqueeze(0).to(device)
        pred = model(triplet).clamp(0, 1)[0].cpu()
        out = (pred.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        out_path = output_dir / f"{Path(paths[i]).stem}_deblurred.png"
        Image.fromarray(out).save(out_path)
        print(f"{Path(paths[i]).name} -> {out_path.name}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(prefer_cuda=not args.cpu)
    model = build_model(args.model, args.checkpoint, device)

    if args.frames_dir is not None:
        deblur_frames(model, args.frames_dir, args.output_dir, device)
        return

    dataset = GoProDataset(data_path=str(args.data_root / args.split), transform=None)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=args.num_workers, drop_last=False)
    mean_psnr, mean_ssim = evaluate(model, loader)
    print(f"{args.split}: PSNR = {mean_psnr:.2f} dB, SSIM = {mean_ssim:.4f}")

    if args.save_vis is not None:
        for k in range(min(args.num_vis, len(dataset))):
            item = dataset[k * (len(dataset) // max(args.num_vis, 1)) % len(dataset)]
            pred = model(item["blur_triplet"].unsqueeze(0).to(device)).clamp(0, 1)[0].cpu()
            save_triplet(item["blur_triplet"], item["sharp_target"], pred, args.save_vis / f"example_{k}.png")
        print(f"Saved {args.num_vis} comparison image(s) to {args.save_vis}")


if __name__ == "__main__":
    main()
