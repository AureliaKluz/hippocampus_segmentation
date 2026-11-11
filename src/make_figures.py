import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
runs_dir = project_root / "results" / "runs"
out_dir = project_root / "results" / "final"
out_dir.mkdir(parents=True, exist_ok=True)

groups = {
    "baseline": ["baseline", "baseline_s43", "baseline_s44"],
    "aug": ["aug", "aug_s43", "aug_s44"],
    "unetr": ["unetr", "unetr_s43", "unetr_s44"],
}
labels = {
    "baseline": "3D U-Net bez augmentacji",
    "aug": "3D U-Net z augmentacjami",
    "unetr": "UNETR z augmentacjami",
}
group_names = {
    "uint8": "uint8",
    "float32": "float32",
    "float32_high": "float32, wysoka skala",
}
group_order = ["uint8", "float32", "float32_high"]
final_run = "aug"

def comma(text):
    return text.replace(".", ",")

def write_tex(path, header, rows, col_format):
    lines = ["\\begin{tabular}{" + col_format + "}", "\\hline",
             " & ".join(header) + " \\\\ \\hline"]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    text = "\n".join(lines)
    text = text.replace("±", "$\\pm$").replace("%", "\\%")
    text = text.replace("mm³", "mm$^3$")
    path.write_text(text, encoding="utf-8")


def run_metrics(ev):
    return {
        "dice_ant": ev["dice_ant"].mean(),
        "dice_post": ev["dice_post"].mean(),
        "dice_whole": ev["dice_whole"].mean(),
        "hd_whole": ev["hd_whole"].mean(),
        "hd95_whole": ev["hd95_whole"].mean(),
        "relerr": ev["abs_relerr_whole"].mean(),
        "bias": ev["err_whole"].mean(),
        "r": pearsonr(ev["vol_pred_whole"], ev["vol_ref_whole"])[0],
    }

rows_def = [
    ("Dice Anterior", "dice_ant", 3),
    ("Dice Posterior", "dice_post", 3),
    ("Dice całość", "dice_whole", 3),
    ("HD całość [mm]", "hd_whole", 2),
    ("HD95 całość [mm]", "hd95_whole", 2),
    ("Błąd względny objętości [%]", "relerr", 2),
    ("Obciążenie objętości [mm³]", "bias", 0),
    ("Korelacja objętości r", "r", 3),
]

# walidacja i test
for split in ["val", "test"]:
    evs = {m: [pd.read_csv(runs_dir / r / f"eval_{split}.csv") for r in runs]
           for m, runs in groups.items()}

    metrics = {m: pd.DataFrame([run_metrics(e) for e in evs[m]])
               for m in groups}
    header = ["Metryka"] + [labels[m] for m in groups]
    table = []
    for name, key, d in rows_def:
        row = [name]
        for m in groups:
            mean, std = metrics[m][key].mean(), metrics[m][key].std()
            row.append(f"{mean:.{d}f} ± {std:.{d}f}")
        table.append(row)
    pd.DataFrame(table, columns=header).to_csv(
        out_dir / f"table_main_{split}.csv", index=False, encoding="utf-8-sig")
    write_tex(out_dir / f"table_main_{split}.tex", header,
              [[comma(c) for c in r] for r in table], "lccc")

    gh = ["Model", "Grupa", "n", "Dice całość", "HD95 [mm]",
          "Błąd obj. [%]"]
    grows = []
    for m in groups:
        allc = pd.concat(evs[m])
        for g in group_order:
            sub = allc[allc["group"] == g]
            grows.append([
                labels[m], group_names[g], sub["case"].nunique(),
                f"{sub['dice_whole'].mean():.3f}",
                f"{sub['hd95_whole'].mean():.2f}",
                f"{sub['abs_relerr_whole'].mean():.2f}",
            ])
    pd.DataFrame(grows, columns=gh).to_csv(
        out_dir / f"table_groups_{split}.csv", index=False,
        encoding="utf-8-sig")
    write_tex(out_dir / f"table_groups_{split}.tex", gh,
              [[comma(str(c)) for c in r] for r in grows], "llcccc")

# średnia z trzech ziaren i zakres min-max
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for model, runs in groups.items():
    logs = [pd.read_csv(runs_dir / r / "log.csv") for r in runs]
    epochs = logs[0]["epoch"]
    for ax, col in zip(axes, ["train_loss", "val_dice_mean"]):
        arr = np.vstack([lg[col].values for lg in logs])
        ax.plot(epochs, arr.mean(axis=0), label=labels[model])
        ax.fill_between(epochs, arr.min(axis=0), arr.max(axis=0), alpha=0.2)
axes[0].set_yscale("log")
axes[0].set_title("Strata treningowa")
axes[0].set_ylabel("Strata (skala logarytmiczna)")
axes[1].set_title("Dice na zbiorze walidacyjnym (średnia obu części)")
axes[1].set_ylabel("Dice")
axes[1].set_ylim(0.80, 0.90)
for ax in axes:
    ax.set_xlabel("epoka")
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.savefig(out_dir / "fig_learning_curves.png", dpi=200)
plt.close()

# wolumetria na zbiorze testowym
ev = pd.read_csv(runs_dir / final_run / "eval_test.csv")
ref, pred = ev["vol_ref_whole"], ev["vol_pred_whole"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
lim = [min(ref.min(), pred.min()) - 100, max(ref.max(), pred.max()) + 100]
axes[0].scatter(ref, pred, s=18)
axes[0].plot(lim, lim, color="k", linewidth=1)
axes[0].set_xlabel("Objętość referencyjna [mm³]")
axes[0].set_ylabel("Objętość z modelu [mm³]")
axes[0].set_title(f"Objętość całkowita, r = {pearsonr(ref, pred)[0]:.3f}")
mean_v, diff = (ref + pred) / 2, pred - ref
bias, sd = diff.mean(), diff.std()
axes[1].scatter(mean_v, diff, s=18)
axes[1].axhline(bias, color="k")
axes[1].axhline(bias + 1.96 * sd, color="gray", linestyle="--")
axes[1].axhline(bias - 1.96 * sd, color="gray", linestyle="--")
axes[1].set_xlabel("Średnia objętość [mm³]")
axes[1].set_ylabel("Różnica: model − referencja [mm³]")
axes[1].set_title(f"Bland–Altman (obciążenie {bias:+.0f} mm³)")
plt.tight_layout()
plt.savefig(out_dir / "fig_volume_test.png", dpi=200)
plt.close()

# Dice (model finalny, test)
fig, ax = plt.subplots(figsize=(6, 4.3))
data = [ev.loc[ev["group"] == g, "dice_whole"].values for g in group_order]
ax.boxplot(data, showfliers=False)
for i, d in enumerate(data, start=1):
    ax.scatter(np.random.default_rng(0).normal(i, 0.05, len(d)), d,
               s=14, alpha=0.6)
ax.set_xticks(range(1, len(group_order) + 1))
ax.set_xticklabels([f"{group_names[g]}\n(n={len(d)})"
                    for g, d in zip(group_order, data)])
ax.set_ylabel("Dice całości")
ax.set_title("Zbiór testowy, 3D U-Net z augmentacjami")
plt.tight_layout()
plt.savefig(out_dir / "fig_dice_by_group_test.png", dpi=200)
plt.close()

# przykłady segmentacji z analizy błędów
src = runs_dir / final_run / "error_analysis_test.png"
if src.exists():
    shutil.copy(src, out_dir / "fig_error_analysis_test.png")

print("Zapisano pliki w:", out_dir)
for p in sorted(out_dir.iterdir()):
    print(" ", p.name)