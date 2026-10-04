import numpy as np
import pandas as pd
import matplotlib
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    ResizeWithPadOrCropd,
)
from monai.utils import set_determinism

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hippo_data import (
    IMAGE_SIZE,
    PROJECT_ROOT,
    get_dataset,
    get_files,
    get_loader,
    get_transforms,
)

set_determinism(seed=42)

loader = get_loader("train", batch_size=4, shuffle=False)
batch = next(iter(loader))
img, lbl = batch["image"], batch["label"]
print("Kształt obrazu:", tuple(img.shape), "oczekiwany: (4, 1, *%s)" % (IMAGE_SIZE,))
print("Kształt maski:", tuple(lbl.shape))
print("Typ obrazu / maski:", img.dtype, lbl.dtype)
print("Wartości w masce:", np.unique(lbl.numpy()))
for i in range(img.shape[0]):
    x = img[i, 0].numpy()
    print(f"  obraz {i} ({batch['case'][i]}): średnia={x.mean():.3f}, std={x.std():.3f}")

df = pd.read_csv(PROJECT_ROOT / "results" / "dataset_summary.csv").set_index("case")
bad = 0
for item in get_dataset("test"):
    n = int((item["label"] > 0).sum())
    expected = int(df.loc[item["case"], "vol_total_mm3"])
    if n != expected:
        bad += 1
        print("NIEZGODNE:", item["case"], n, expected)
print("\nPrzypadki testowe z niezgodną objętością:", bad)

hdr = pd.read_csv(PROJECT_ROOT / "results" / "header_info.csv")
hdr["group"] = hdr["dtype"]
hdr.loc[hdr["high_scale"], "group"] = "float32_high"

train_files = {f["case"]: f for f in get_files("train")}
chosen = []
for group in ["uint8", "float32", "float32_high"]:
    case = next(c for c in hdr.loc[hdr["group"] == group, "case"]
                if c in train_files)
    chosen.append((group, train_files[case]))

raw_tf = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ResizeWithPadOrCropd(
        keys=["image", "label"], spatial_size=IMAGE_SIZE, mode="constant"
    ),
])
full_tf = get_transforms()

fig, axes = plt.subplots(3, 3, figsize=(11, 14))
for row, (group, f) in enumerate(chosen):
    raw = raw_tf(f)
    proc = full_tf(f)
    x_raw = raw["image"][0].numpy()
    x = proc["image"][0].numpy()
    m = proc["label"][0].numpy()
    cz = int(np.argwhere(m > 0)[:, 2].mean())

    axes[row, 0].imshow(x_raw[:, :, cz].T, cmap="gray", origin="lower")
    axes[row, 0].set_title(
        f"{group} ({f['case']})\nprzed z-score, zakres "
        f"{x_raw.min():.0f}-{x_raw.max():.0f}", fontsize=9)

    axes[row, 1].imshow(x[:, :, cz].T, cmap="gray", origin="lower")
    axes[row, 1].set_title(
        f"po z-score, zakres {x.min():.1f}-{x.max():.1f}", fontsize=9)

    axes[row, 2].imshow(x[:, :, cz].T, cmap="gray", origin="lower")
    axes[row, 2].imshow(
        np.ma.masked_where(m[:, :, cz].T == 0, m[:, :, cz].T),
        cmap="autumn", alpha=0.5, origin="lower", vmin=1, vmax=2)
    axes[row, 2].set_title("po z-score, z maską", fontsize=9)

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
out_file = PROJECT_ROOT / "results" / "dataloader_check.png"
plt.savefig(out_file, dpi=150)
plt.close()
print("Zapisano:", out_file)