from pathlib import Path
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
run = "baseline"
log = pd.read_csv(project_root / "results" / "runs" / run / "log.csv")
best_epoch = int(log.loc[log["val_dice_mean"].idxmax(), "epoch"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(log["epoch"], log["train_loss"])
axes[0].set_xlabel("epoka")
axes[0].set_ylabel("strata treningowa")
axes[0].set_title("Strata treningowa")

axes[1].plot(log["epoch"], log["val_dice_anterior"], label="Anterior")
axes[1].plot(log["epoch"], log["val_dice_posterior"], label="Posterior")
axes[1].plot(log["epoch"], log["val_dice_mean"], label="średnia", color="k")
axes[1].axvline(best_epoch, color="gray", linestyle="--",
                label=f"najlepsza epoka ({best_epoch})")
axes[1].set_ylim(0.5, 0.95)
axes[1].set_xlabel("epoka")
axes[1].set_ylabel("Dice (walidacja)")
axes[1].set_title("Dice na zbiorze walidacyjnym")
axes[1].legend()

plt.tight_layout()
out_file = project_root / "results" / f"{run}_curves.png"
plt.savefig(out_file, dpi=150)
plt.close()
print("Najlepsza epoka:", best_epoch)
print("Zapisano:", out_file)