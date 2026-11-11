import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="val", choices=["val", "test"])
args = parser.parse_args()

project_root = Path(__file__).resolve().parent.parent
runs_dir = project_root / "results" / "runs"

groups = {
    "baseline": ["baseline", "baseline_s43", "baseline_s44"],
    "aug": ["aug", "aug_s43", "aug_s44"],
    "unetr": ["unetr", "unetr_s43", "unetr_s44"],
}

rows = []
frames = []
for model, runs in groups.items():
    for run in runs:
        ev = pd.read_csv(runs_dir / run / f"eval_{args.split}.csv")
        ev["model"] = model
        ev["run"] = run
        frames.append(ev)
        rows.append({
            "model": model,
            "run": run,
            "dice_ant": ev["dice_ant"].mean(),
            "dice_post": ev["dice_post"].mean(),
            "dice_whole": ev["dice_whole"].mean(),
            "hd_whole": ev["hd_whole"].mean(),
            "hd95_whole": ev["hd95_whole"].mean(),
            "relerr_whole_pct": ev["abs_relerr_whole"].mean(),
            "bias_whole_mm3": ev["err_whole"].mean(),
            "r_whole": pearsonr(ev["vol_pred_whole"],
                                ev["vol_ref_whole"])[0],
        })

per_run = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print(f"Zbiór: {args.split}")
print("Wyniki dla każdego uruchomienia:")
print(per_run.round(3).to_string(index=False))

metrics = ["dice_ant", "dice_post", "dice_whole", "hd_whole", "hd95_whole",
           "relerr_whole_pct", "bias_whole_mm3", "r_whole"]
print("\nŚrednia ± odchylenie standardowe z trzech ziaren:")
summary = {}
for model in groups:
    sub = per_run[per_run["model"] == model]
    summary[model] = {
        m: f"{sub[m].mean():.3f} ± {sub[m].std():.3f}" for m in metrics
    }
print(pd.DataFrame(summary).to_string())

print("\nRóżnica Dice całości dla każdego ziarna:")
pairs = [("aug", "baseline"), ("unetr", "aug")]
for a_model, b_model in pairs:
    for a, b in zip(groups[a_model], groups[b_model]):
        da = per_run.loc[per_run["run"] == a, "dice_whole"].iloc[0]
        db = per_run.loc[per_run["run"] == b, "dice_whole"].iloc[0]
        print(f"  {a} minus {b}: {da - db:+.4f}")

all_cases = pd.concat(frames)
by_group = (all_cases.groupby(["model", "group"])
            [["dice_whole", "hd95_whole", "abs_relerr_whole"]]
            .mean().round(3))
print("\nŚrednie według grupy skali intensywności "
      "(po wszystkich ziarnach i przypadkach):")
print(by_group)
print("\nLiczba przypadków w grupach (jedno uruchomienie):")
print(frames[0]["group"].value_counts().to_dict())

out_file = project_root / "results" / f"compare_seeds_{args.split}.csv"
per_run.round(4).to_csv(out_file, index=False)
print("\nZapisano:", out_file)