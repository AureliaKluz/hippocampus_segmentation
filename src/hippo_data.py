import json
from pathlib import Path

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ResizeWithPadOrCropd,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "Task04_Hippocampus"
SPLIT_FILE = PROJECT_ROOT / "configs" / "split.json"
IMAGE_SIZE = (48, 64, 48)


def get_files(split_name):
    with open(SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)
    return [
        {
            "image": str(DATA_ROOT / "imagesTr" / f"{case}.nii.gz"),
            "label": str(DATA_ROOT / "labelsTr" / f"{case}.nii.gz"),
            "case": case,
        }
        for case in split[split_name]
    ]


def get_transforms(augment=False):
    steps = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ResizeWithPadOrCropd(
            keys=["image", "label"], spatial_size=IMAGE_SIZE, mode="constant"
        ),
    ]
    if augment:
        steps += [
            RandAffined(
                keys=["image", "label"],
                prob=0.5,
                rotate_range=(0.17, 0.17, 0.17),
                translate_range=(3, 3, 3),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        ]
    return Compose(steps)


def get_dataset(split_name, cache=False, augment=False):
    files = get_files(split_name)
    transforms = get_transforms(augment=augment)
    if cache:
        return CacheDataset(
            data=files, transform=transforms, cache_rate=1.0, num_workers=0
        )
    return Dataset(data=files, transform=transforms)


def get_loader(split_name, batch_size=2, shuffle=False, cache=False,
               augment=False):
    ds = get_dataset(split_name, cache=cache, augment=augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0)