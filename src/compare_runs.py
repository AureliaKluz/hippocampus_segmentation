import json
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
runs = ["baseline", "aug", "aug_wide"]
runs_dir = project_root / "results" / "runs"

base = pd.read_csv(runs_dir / "baseline" / "eval_val.csv").set_index("case")

rows = []
for name in runs:
    run_dir = runs_dir / name
    log = pd.read_csv(run_dir / "log.csv")
    ev = pd.read_csv(run_dir / "eval_val.csv")
    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    best = log.loc[log["val_dice_mean"].idxmax()]
    rows.append({
        "run": name,
        "channels": "-".join(map(str, cfg["channels"])),
        "augment": cfg.get("augment", False),
        "best_epoch": int(best["epoch"]),
        "dice_ant": ev["dice_ant"].mean(),
        "dice_post": ev["dice_post"].mean(),
        "dice_whole": ev["dice_whole"].mean(),
        "hd95_whole": ev["hd95_whole"].mean(),
        "abs_relerr_whole_%": ev["abs_relerr_whole"].mean(),
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
df.round(4).to_csv(project_root / "results" / "compare_runs_val.csv",
                   index=False)

print("\nRóżnica Dice względem baseline, dla każdego przypadku:")
for name in runs[1:]:
    ev = pd.read_csv(runs_dir / name / "eval_val.csv").set_index("case")
    d = ev["dice_whole"] - base["dice_whole"]
    print(f"{name}: średnia różnica {d.mean():+.4f} (odch. {d.std():.4f}), "
          f"lepiej w {(d > 0).sum()} z {len(d)} przypadków")