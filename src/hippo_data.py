import json
from pathlib import Path

from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
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


def get_transforms():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
        ResizeWithPadOrCropd(
            keys=["image", "label"], spatial_size=IMAGE_SIZE, mode="constant"
        ),
    ])


def get_dataset(split_name, cache=False):
    files = get_files(split_name)
    transforms = get_transforms()
    if cache:
        return CacheDataset(
            data=files, transform=transforms, cache_rate=1.0, num_workers=0
        )
    return Dataset(data=files, transform=transforms)


def get_loader(split_name, batch_size=2, shuffle=False, cache=False):
    ds = get_dataset(split_name, cache=cache)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)