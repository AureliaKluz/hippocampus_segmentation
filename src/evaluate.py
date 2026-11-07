import argparse
import json
import matplotlib
import numpy as np
import pandas as pd
import torch
from monai.networks.nets import UNet
from scipy import ndimage
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hippo_data import PROJECT_ROOT, get_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True, help="nazwa uruchomienia")
parser.add_argument("--split", default="val", choices=["val", "test"])
parser.add_argument("--checkpoint", default="best.pt")
parser.add_argument("--final", action="store_true",
                    help="wymagane dla zbioru testowego")
args = parser.parse_args()

if args.split == "test" and not args.final:
    raise SystemExit("Zbiór testowy służy tylko do ewaluacji końcowej. "
                     "Dodaj --final")

run_dir = PROJECT_ROOT / "results" / "runs" / args.name
with open(run_dir / "config.json", encoding="utf-8") as f:
    cfg = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=tuple(cfg["channels"]),
    strides=(2,) * (len(cfg["channels"]) - 1),
    num_res_units=cfg["res_units"],
).to(device)
model.load_state_dict(
    torch.load(run_dir / args.checkpoint, map_location=device))
model.eval()

def dice(a, b):
    denom = a.sum() + b.sum()
    return np.nan if denom == 0 else 2.0 * (a & b).sum() / denom

def surface(mask):
    return mask & ~ndimage.binary_erosion(mask)

def hausdorff(a, b):
    if a.sum() == 0 or b.sum() == 0:
        return np.nan, np.nan
    sa, sb = surface(a), surface(b)
    d_to_b = ndimage.distance_transform_edt(~sb)
    d_to_a = ndimage.distance_transform_edt(~sa)
    ab = d_to_b[sa]
    ba = d_to_a[sb]
    hd = max(ab.max(), ba.max())
    hd95 = max(np.percentile(ab, 95), np.percentile(ba, 95))
    return hd, hd95

rows = []
with torch.no_grad():
    for item in get_dataset(args.split):
        x = item["image"].as_tensor().unsqueeze(0).to(device)
        pred = model(x).argmax(dim=1)[0].cpu().numpy()
        ref = item["label"].as_tensor()[0].numpy().astype(int)

        row = {"case": item["case"]}
        for name, p, r in [
            ("ant", pred == 1, ref == 1),
            ("post", pred == 2, ref == 2),
            ("whole", pred > 0, ref > 0),
        ]:
            hd, hd95 = hausdorff(p, r)
            row[f"dice_{name}"] = dice(p, r)
            row[f"hd_{name}"] = hd
            row[f"hd95_{name}"] = hd95
            row[f"vol_pred_{name}"] = float(p.sum())
            row[f"vol_ref_{name}"] = float(r.sum())
        rows.append(row)

df = pd.DataFrame(rows)

hdr = pd.read_csv(PROJECT_ROOT / "results" / "header_info.csv")
hdr["group"] = hdr["dtype"]
hdr.loc[hdr["high_scale"], "group"] = "float32_high"
df = df.merge(hdr[["case", "group"]], on="case", how="left")

for name in ["ant", "post", "whole"]:
    df[f"err_{name}"] = df[f"vol_pred_{name}"] - df[f"vol_ref_{name}"]
    df[f"relerr_{name}"] = 100 * df[f"err_{name}"] / df[f"vol_ref_{name}"]
df["abs_relerr_whole"] = df["relerr_whole"].abs()

out_csv = run_dir / f"eval_{args.split}.csv"
df.to_csv(out_csv, index=False)

def ms(col):
    return f"{df[col].mean():.3f} ± {df[col].std():.3f}"

print(f"\nUruchomienie: {args.name}, zbiór: {args.split}, "
      f"przypadków: {len(df)}")
for name, label in [("ant", "Anterior"), ("post", "Posterior"),
                    ("whole", "Razem")]:
    r = pearsonr(df[f"vol_pred_{name}"], df[f"vol_ref_{name}"])[0]
    print(f"\n{label}:")
    print(f"  Dice  {ms(f'dice_{name}')}")
    print(f"  HD    {ms(f'hd_{name}')} mm")
    print(f"  HD95  {ms(f'hd95_{name}')} mm")
    print(f"  Objętość: MAE {df[f'err_{name}'].abs().mean():.1f} mm3, "
          f"średni błąd bezwzględny "
          f"{df[f'relerr_{name}'].abs().mean():.2f}%, "
          f"bias {df[f'err_{name}'].mean():.1f} mm3, r = {r:.3f}")

print("\nŚrednie według grupy skali intensywności:")
cols = ["dice_ant", "dice_post", "dice_whole", "hd95_whole",
        "abs_relerr_whole"]
print(df.groupby("group")[cols].mean().round(3))
print("Liczba przypadków:", df["group"].value_counts().to_dict())

odd = df[df["case"].isin(["hippocampus_114", "hippocampus_217"])]
if len(odd):
    print("\nPrzypadki odstające pod względem proporcji:")
    print(odd[["case", "dice_ant", "dice_post", "dice_whole",
               "relerr_whole"]].round(3))

mean_v = (df["vol_pred_whole"] + df["vol_ref_whole"]) / 2
diff = df["vol_pred_whole"] - df["vol_ref_whole"]
bias, sd = diff.mean(), diff.std()
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(mean_v, diff, s=18)
ax.axhline(bias, color="k")
ax.axhline(bias + 1.96 * sd, color="gray", linestyle="--")
ax.axhline(bias - 1.96 * sd, color="gray", linestyle="--")
ax.set_xlabel("średnia objętość [mm3]")
ax.set_ylabel("różnica (model − referencja) [mm3]")
ax.set_title(f"Bland–Altman, objętość całkowita, {args.split}")
plt.tight_layout()
out_png = run_dir / f"eval_{args.split}_bland_altman.png"
plt.savefig(out_png, dpi=150)
plt.close()
print("\nZapisano:", out_csv)
print("Zapisano:", out_png)