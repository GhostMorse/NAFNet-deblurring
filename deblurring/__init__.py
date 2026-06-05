"""Video deblurring on the GoPro dataset with a NAFNet U-Net.

Public API
----------
- :class:`~deblurring.models.unet.UNet` (baseline)
- :class:`~deblurring.models.my_model.AlignedNAFNet` (improved)
- :class:`~deblurring.data.dataset.GoProDataset`
- losses in :mod:`deblurring.losses`, metrics in :mod:`deblurring.metrics`
- training / evaluation in :mod:`deblurring.engine`
"""

from .data.dataset import GoProDataset
from .engine import evaluate, train
from .losses import DeblurLoss
from .metrics import psnr, ssim
from .models.my_model import AlignedNAFNet
from .models.unet import UNet

__version__ = "0.1.0"

__all__ = ["UNet", "AlignedNAFNet", "GoProDataset", "DeblurLoss", "psnr", "ssim", "train", "evaluate"]
