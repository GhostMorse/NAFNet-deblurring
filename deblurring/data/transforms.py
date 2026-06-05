"""Training-time augmentations for frame triplets.

Albumentations' ``additional_targets`` keeps the three blurry frames and the
sharp target geometrically in sync (the same crop / flip / rotation is applied
to all of them).
"""

from __future__ import annotations


def build_train_transform(crop_size: int = 256):
    """Random crop + flips + 90-degree rotations, applied jointly to all frames."""
    import albumentations as A

    return A.Compose(
        [
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ],
        additional_targets={"image1": "image", "image2": "image", "sharp": "image"},
    )
