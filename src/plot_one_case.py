from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
root = project_root / "data" / "Task04_Hippocampus"

images = sorted(p for p in (root / "imagesTr").glob("*.nii.gz")
                if not p.name.startswith("._"))
img_path = images[0]
lbl_path = root / "labelsTr" / img_path.name

img = nib.load(img_path).get_fdata()
lbl = nib.load(lbl_path).get_fdata().astype(int)

coords = np.argwhere(lbl > 0)
cx, cy, cz = coords.mean(axis=0).astype(int)

views = [
    ("oś 0", img[cx, :, :], lbl[cx, :, :]),
    ("oś 1", img[:, cy, :], lbl[:, cy, :]),
    ("oś 2", img[:, :, cz], lbl[:, :, cz]),
]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, (title, im, mk) in enumerate(views):
    axes[0, i].imshow(im.T, cmap="gray", origin="lower")
    axes[0, i].set_title(f"{title} - obraz")
    axes[1, i].imshow(im.T, cmap="gray", origin="lower")
    overlay = np.ma.masked_where(mk.T == 0, mk.T)
    axes[1, i].imshow(overlay, cmap="autumn", alpha=0.5, origin="lower",
                      vmin=1, vmax=2)
    axes[1, i].set_title(f"{title} - z maską")
    for ax in axes[:, i]:
        ax.axis("off")

fig.suptitle(img_path.name)
plt.tight_layout()

out_dir = project_root / "results"
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "case_001_slices.png"
plt.savefig(out_file, dpi=150)
print("Zapisano:", out_file)