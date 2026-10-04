from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
root = project_root / "data" / "Task04_Hippocampus"
out_dir = project_root / "results"
out_dir.mkdir(exist_ok=True)

images = sorted(p for p in (root/"imagesTr").glob("*.nii.gz")
                if not p.name.startswith("._"))
print("Liczba obrazów: ", len(images))

rows = []
for img_path in images:
    lbl_path = root/"labelsTr"/img_path.name
    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)

    img = img_nii.get_fdata()
    lbl = lbl_nii.get_fdata().astype(int)
    spacing = img_nii.header.get_zooms()
    voxel_mm3 = float(np.prod(spacing))

    n1 = int((lbl==1).sum())
    n2 = int((lbl==2).sum())
    rows.append({
        "case": img_path.name.replace(".nii.gz", ""),
        "shape": "x".join(map(str, img.shape)),
        "shape_match_label": img.shape == lbl.shape,
        "spacing": "x".join(f"{s:.2f}" for s in spacing),
        "orientation": "".join(nib.aff2axcodes(img_nii.affine)),
        "int_min": float(img.min()),
        "int_max": float(img.max()),
        "int_mean": float(img.mean()),
        "labels": ",".join(map(str, np.unique(lbl))),
        "vol_anterior_mm3": n1*voxel_mm3,
        "vol_posterior_mm3": n2*voxel_mm3,
        "vol_total_mm3": (n1+n2)*voxel_mm3
    })

df = pd.DataFrame(rows)
df["ratio_ant_post"] = df["vol_anterior_mm3"]/df["vol_posterior_mm3"]

def zscore(s):
    return (s-s.mean())/s.std()

df["z_total"] = zscore(df["vol_total_mm3"])
df["z_ratio"] = zscore(df["ratio_ant_post"])
df["outlier"] = (df["z_total"].abs()>3) | (df["z_ratio"].abs()>3)

out_file = out_dir / "dataset_summary.csv"
df.to_csv(out_file, index=False)
print("Zapisano: ", out_file)

print("\nRozmiary obrazów (10 najczęstszych): ")
print(df["shape"].value_counts().head(10))
print("\nLiczba różnych rozmiarów: ", df["shape"].nunique())
print("Spacing: ", df["spacing"].value_counts().to_dict())
print("Orientacje: ", df["orientation"].value_counts().to_dict())
print("Etykiety: ", df["labels"].value_counts().to_dict())
print("Maska i obraz tego samego rozmiaru we wszystkich: ",
      bool(df["shape_match_label"].all()))

print("\nStatystyki objętości [mm3]: ")
print(df[["vol_anterior_mm3", "vol_posterior_mm3",
          "vol_total_mm3"]].describe().round(1))

print("\nPrzypadki odstające: ")
cols = ["case", "vol_anterior_mm3", "vol_posterior_mm3",
        "vol_total_mm3", "ratio_ant_post"]
print(df.loc[df["outlier"], cols].round(2) if df["outlier"].any() else "brak")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, title in zip(
    axes,
    ["vol_anterior_mm3", "vol_posterior_mm3", "vol_total_mm3"],
    ["Anterior", "Posterior", "Total"]
):
    ax.hist(df[col], bins=30)
    ax.set_title(title)
    ax.set_xlabel("objętość [mm3]")
    ax.set_ylabel("liczba przypadków")
plt.tight_layout()
plt.savefig(out_dir/"volume_distribution.png", dpi=150)
plt.close()
print("Zapisano wykres: ", out_dir / "volume_distribution.png")
