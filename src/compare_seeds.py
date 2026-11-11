from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

project_root = Path(__file__).resolve().parent.parent
runs_dir = project_root / "results" / "runs"

groups = {
    "baseline": ["baseline", "baseline_s43", "baseline_s44"],
    "aug": ["aug", "aug_s43", "aug_s44"],
    "unetr": ["unetr", "unetr_s43", "unetr_s44"],
}

rows = []
for model, runs in groups.items():
    for run in runs:
        ev = pd.read_csv(runs_dir / run / "eval_val.csv")
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

base = per_run[per_run["model"] == "baseline"].reset_index(drop=True)
aug = per_run[per_run["model"] == "aug"].reset_index(drop=True)
print("\nRóżnica Dice całości dla każdego ziarna:")
for b, a in zip(base["run"], aug["run"]):
    d = (aug.loc[aug["run"] == a, "dice_whole"].iloc[0]
         - base.loc[base["run"] == b, "dice_whole"].iloc[0])
    print(f"  {a} vs {b}: {d:+.4f}")

per_run.round(4).to_csv(project_root / "results" / "compare_seeds_val.csv",
                        index=False)
print("\nZapisano: results/compare_seeds_val.csv")
