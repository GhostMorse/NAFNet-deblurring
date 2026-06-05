# Video Deblurring on GoPro (NAFNet)

Multi-frame video deblurring in PyTorch. Given three consecutive blurry frames
`(t-1, t, t+1)`, the model reconstructs the sharp centre frame. Two models are
provided: a **NAFNet** ([Simple Baselines for Image Restoration](https://arxiv.org/abs/2204.04676))
U-Net baseline that concatenates the frames, and an **improved model** that
explicitly *aligns* neighbouring frames with deformable convolution before
fusing them. Trained and evaluated on the [GoPro](https://seungjunnah.github.io/Datasets/gopro)
dataset.

## Overview

Motion blur appears when the camera or scene moves during exposure. Single-image
deblurring has to hallucinate lost detail; video deblurring can instead borrow
information from neighbouring frames. The networks here predict a residual that
is added to the blurry centre frame, so they only need to learn the *correction*
rather than the whole image. Quality is measured with **PSNR** and **SSIM**.

## Models

### Baseline — NAFNet U-Net

The three frames are concatenated along the channel axis (`3 x 3 = 9` input
channels) and passed through a NAFNet encoder/decoder. NAFNet is "non-linear
activation free": instead of ReLU/GELU it uses **SimpleGate** (split channels in
half and multiply) and **Simplified Channel Attention** (global pool + `1x1`
conv). Each **NAFNet block** has a spatial-mixing branch
(`LayerNorm -> 1x1 -> depthwise 3x3 -> SimpleGate -> SCA -> 1x1`) and a
channel-mixing branch (`LayerNorm -> 1x1 -> SimpleGate -> 1x1`), both residual
and scaled by zero-initialised LayerScale factors so training starts stable.

### Improved — aligned multi-frame model

Channel concatenation has no way to compensate for motion between frames. The
improved model addresses this directly:

1. **Per-frame feature extraction** with a shared conv.
2. **Deformable-convolution alignment (DCNv2)** warps the `t-1` and `t+1`
   features onto the `t` frame, compensating for motion.
3. **Temporal-attention fusion** combines the aligned features with per-pixel
   softmax weights.
4. A **NAFNet U-Net** with a **dynamic global filter** (a learnable
   frequency-domain filter) in the bottleneck for a cheap global receptive
   field.

It is trained with a composite **Charbonnier + FFT + edge** loss, which tends to
preserve high-frequency detail better than L1 alone.

## Project structure

```
video-deblurring-nafnet/
|- deblurring/
|  |- data/
|  |  |- dataset.py        # GoProDataset (frame triplets)
|  |  \- transforms.py     # joint Albumentations augmentations
|  |- modules/
|  |  |- layers.py         # LayerNorm2d, SimpleGate, SCA, ImprovedSCA
|  |  |- nafnet_block.py   # NAFNetBlock
|  |  \- alignment.py      # AlignmentDCNv2, TemporalAttentionFusion, DynamicGlobalFilter
|  |- models/
|  |  |- unet.py           # UNet (baseline)
|  |  \- my_model.py       # AlignedNAFNet (improved)
|  |- losses.py            # Charbonnier / FFT / edge / DeblurLoss
|  |- metrics.py           # PSNR, SSIM
|  |- engine.py            # train / evaluate / ablation eval
|  |- viz.py               # qualitative comparison images
|  \- utils.py             # seeding, device, FLOP/param counting
|- train.py                # training entry point
|- deblur.py               # evaluation + inference entry point
|- analyze.py              # frame-aggregation ablations + cost/resolution table
\- tests/                  # pytest suite
```

## Installation

```bash
git clone https://github.com/<your-username>/video-deblurring-nafnet.git
cd video-deblurring-nafnet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is recommended for training; evaluation and the tests run on CPU.

## Data

Download the GoPro dataset and arrange it as:

```
GoPro/
|- train/<video>/{blur_gamma,sharp}/*.png
\- test/<video>/{blur_gamma,sharp}/*.png
```

GoPro has 2103 train and 1111 test frame pairs (1280x720) across 22 videos.
Blurry and sharp frames share filenames; the dataset builds `(t-1, t, t+1)`
triplets with the sharp `t` frame as the target.

## Usage

### Training

```bash
# baseline NAFNet U-Net (L1 loss)
python train.py --model baseline --data-root GoPro --epochs 100

# improved aligned model (Charbonnier + FFT + edge loss)
python train.py --model improved --data-root GoPro --epochs 50
```

Checkpoints are written to `weights/`. Augmentation is a random `256x256` crop
with flips and 90-degree rotations applied jointly to all four images.

### Evaluation and inference

```bash
# PSNR / SSIM on the test split (+ optional comparison images)
python deblur.py --model baseline --checkpoint weights/baseline_final.pth \
    --data-root GoPro --save-vis vis --num-vis 3

# deblur a folder of consecutive blurry frames
python deblur.py --model improved --checkpoint weights/improved_final.pth \
    --frames-dir my_frames --output-dir my_deblurred
```

### Analysis

```bash
python analyze.py --model baseline --checkpoint weights/baseline_final.pth --data-root GoPro
```

This runs the frame-aggregation ablations and prints a GFLOPs/parameter table at
several resolutions.

## Implementation notes

- **Residual learning:** both models predict a correction added to the blurry
  centre frame; the improved model clamps the result to `[0, 1]`.
- **LayerNorm2d** is a channel-wise LayerNorm with an explicit, memory-efficient
  backward.
- **Padding:** inputs are padded to a multiple of `2^depth` inside the network
  and cropped back, so any resolution works.
- **Alignment** uses `torchvision.ops.deform_conv2d` (modulated / DCNv2).

## Analysis & findings

The baseline clears the standard GoPro bar (PSNR >= 27 dB on the test split).
Ablating *how* the frames are aggregated is the most informative experiment:

- **Three copies of the centre frame** (no neighbours) -> PSNR drops. Without a
  temporal context window the model cannot recover structure that is lost to
  blur in the centre frame.
- **Stride-2 frames** `(t-2, t, t+2)` -> PSNR drops. Frames further away are
  less similar to the target, so the borrowed information is less useful.
- **Reversed frame order** `(t+1, t, t-1)` -> PSNR essentially unchanged.
  Temporal *order* does not matter; what matters is having similar neighbouring
  frames from which to reconstruct structure.

Together these reveal the core limitation of naive channel concatenation: it has
no explicit motion compensation and treats the frames order-agnostically.
Qualitatively, the baseline struggles most on regions with vegetation and other
fine, intricate structure under strong inter-frame motion. On the cost side,
PSNR rises with resolution, and inference time is dominated by overhead at low
resolution and by compute at full resolution.

Because the dataset is heterogeneous, the dataset-mean PSNR/SSIM is a weak
summary; per-example inspection is more informative for understanding failure
modes (hence the visualisation tools in `deblur.py` / `viz.py`).

**Improvement.** The aligned model adds a deformable-convolution alignment
module so the network can explicitly compensate for the inter-frame motion that
the baseline cannot handle, plus temporal-attention fusion and a frequency-domain
global filter, trained with the composite Charbonnier/FFT/edge loss.

## Tests

```bash
pytest
```

Covers the blocks (SimpleGate, SCA, NAFNet block — including its identity
initialisation), both models' output shapes (and internal padding), the loss
terms, and the PSNR/SSIM metrics. All tests run on CPU.

## License

Released under the [MIT License](LICENSE).
