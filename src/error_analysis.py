import argparse
import json

import matplotlib
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hippo_data import PROJECT_ROOT, get_dataset, get_files  # noqa: E402
from models import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="aug")
parser.add_argument("--split", default="val", choices=["val", "test"])
parser.add_argument("--final", action="store_true",
                    help="wymagane dla zbioru testowego")
parser.add_argument("--n", type=int, default=3)
parser.add_argument("--extra", nargs="*", default=[],
                    help="dodatkowe przypadki do narysowania")
args = parser.parse_args()

if args.split == "test" and not args.final:
    raise SystemExit("Zbiór testowy: dodaj --final.")
extra = args.extra
if args.split == "test" and not extra:
    extra = ["hippocampus_217"]

run_dir = PROJECT_ROOT / "results" / "runs" / args.name
with open(run_dir / "config.json", encoding="utf-8") as f:
    cfg = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(cfg).to(device)
model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
model.eval()

ev = pd.read_csv(run_dir / f"eval_{args.split}.csv").sort_values("dice_whole")
files = get_files(args.split)
index = {f["case"]: i for i, f in enumerate(files)}
ds = get_dataset(args.split)


def predict(case):
    item = ds[index[case]]
    image = item["image"].as_tensor()
    with torch.no_grad():
        pred = model(image.unsqueeze(0).to(device)).argmax(dim=1)[0]
    ref = item["label"].as_tensor()[0].numpy().astype(int)
    return image[0].numpy(), ref, pred.cpu().numpy().astype(int)

conf = np.zeros((3, 3))
for case in ev["case"]:
    _, ref, pred = predict(case)
    conf += np.bincount((ref * 3 + pred).ravel(), minlength=9).reshape(3, 3)

names = ["tło", "Anterior", "Posterior"]
print(f"Zbiór: {args.split}, model: {args.name}")
print("\nMacierz pomyłek (wiersze: referencja, kolumny: model), voxele:")
print(pd.DataFrame(conf.astype(int), index=names, columns=names))
print("\nUdział w wierszu:")
print(pd.DataFrame(conf / conf.sum(axis=1, keepdims=True),
                   index=names, columns=names).round(3))

errors = {
    "tło uznane za hipokamp": conf[0, 1] + conf[0, 2],
    "hipokamp uznany za tło": conf[1, 0] + conf[2, 0],
    "pomylenie Anterior z Posterior": conf[1, 2] + conf[2, 1],
}
total = sum(errors.values())
print("\nStruktura błędów:")
for k, v in errors.items():
    print(f"  {k}: {int(v)} voxeli ({100 * v / total:.1f}%)")

cols = ["case", "group", "dice_ant", "dice_post", "dice_whole",
        "hd95_whole", "relerr_whole"]
print("\nNajgorsze przypadki:")
print(ev.head(args.n)[cols].round(3).to_string(index=False))
print("\nNajlepsze przypadki:")
print(ev.tail(args.n)[cols].round(3).to_string(index=False))

vol_cols = ["case", "dice_whole", "vol_ref_ant", "vol_pred_ant",
            "vol_ref_post", "vol_pred_post"]
extra = [c for c in extra if c in set(ev["case"])]
if extra:
    print("\nWskazane przypadki:")
    sub = ev[ev["case"].isin(extra)][vol_cols].copy()
    sub["ratio_ref"] = sub["vol_ref_ant"] / sub["vol_ref_post"]
    sub["ratio_pred"] = sub["vol_pred_ant"] / sub["vol_pred_post"]
    print(sub.round(3).to_string(index=False))

selected = [(c, "najgorszy") for c in ev["case"].head(args.n)]
selected += [(c, "najlepszy") for c in ev["case"].tail(args.n)]
selected += [(c, "wskazany") for c in extra
             if c not in {s[0] for s in selected}]
dice_of = ev.set_index("case")["dice_whole"]
cmap_err = ListedColormap(["red", "blue", "gold"])

fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.2 * len(selected)))
for row, (case, label) in enumerate(selected):
    x, ref, pred = predict(case)
    cz = int(np.argwhere(ref > 0)[:, 2].mean())
    img = x[:, :, cz].T
    r = ref[:, :, cz].T
    p = pred[:, :, cz].T

    err = np.zeros_like(r)
    err[(r == 0) & (p > 0)] = 1
    err[(r > 0) & (p == 0)] = 2
    err[(r > 0) & (p > 0) & (r != p)] = 3

    for col, (mask, cmap, vmax) in enumerate(
        [(r, "autumn", 2), (p, "autumn", 2), (err, cmap_err, 3)]
    ):
        ax = axes[row, col]
        ax.imshow(img, cmap="gray", origin="lower")
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap=cmap,
                  alpha=0.55, origin="lower", vmin=1, vmax=vmax)
        ax.axis("off")
    axes[row, 0].set_title(f"{label}: {case}\nDice całość {dice_of[case]:.3f}",
                           fontsize=9)
    if row == 0:
        axes[row, 1].set_title("model", fontsize=9)
        axes[row, 2].set_title(
            "błędy: czerwony - tło uznane za hipokamp,\n"
            "niebieski - hipokamp uznany za tło,\n"
            "żółty - pomyłka Anterior i Posterior", fontsize=8)

plt.tight_layout()
out_file = run_dir / f"error_analysis_{args.split}.png"
plt.savefig(out_file, dpi=150)
plt.close()
print("\nZapisano:", out_file)