from pathlib import Path
import nibabel as nib
import numpy as np

project_root = Path(__file__).resolve().parent.parent
root = project_root / "data" / "Task04_Hippocampus"
print("Szukam danych w:", root)
print("Folder istnieje:", root.exists())

images = sorted(p for p in (root / "imagesTr").glob("*.nii.gz")
                if not p.name.startswith("._"))
print("Liczba obrazów:", len(images))

img_path = images[0]
lbl_path = root / "labelsTr" / img_path.name
print("Obraz:", img_path.name)

img = nib.load(img_path)
lbl = nib.load(lbl_path)

img_data = img.get_fdata()
lbl_data = lbl.get_fdata().astype(int)

spacing = img.header.get_zooms()
print("Shape obrazu:", img_data.shape)
print("Spacing (mm):", spacing)
print("Zakres intensywności:", img_data.min(), "-", img_data.max())
print("Wartości w masce:", np.unique(lbl_data))

voxel_mm3 = float(np.prod(spacing))
for label in np.unique(lbl_data):
    if label == 0:
        continue
    n = int((lbl_data == label).sum())
    print(f"Etykieta {label}: {n} voxeli, {n * voxel_mm3:.1f} mm3")