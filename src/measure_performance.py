import json
import platform
import statistics
import time

import pandas as pd
import torch

from hippo_data import IMAGE_SIZE, PROJECT_ROOT
from models import build_model

models = {
    "3D U-Net bez augmentacji": ["baseline", "baseline_s43", "baseline_s44"],
    "3D U-Net z augmentacjami": ["aug", "aug_s43", "aug_s44"],
    "UNETR z augmentacjami": ["unetr", "unetr_s43", "unetr_s44"],
}
runs_dir = PROJECT_ROOT / "results" / "runs"
x = torch.randn(1, 1, *IMAGE_SIZE)

print("Procesor:", platform.processor())
print("Liczba wątków PyTorch:", torch.get_num_threads())

rows = []
for label, runs in models.items():
    run_dir = runs_dir / runs[0]
    with open(run_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    model = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    size_mb = (run_dir / "best.pt").stat().st_size / 1e6

    with torch.no_grad():
        for _ in range(3):
            model(x)
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model(x)
            times.append(time.perf_counter() - t0)

    logs = [pd.read_csv(runs_dir / r / "log.csv") for r in runs]
    epoch_s = statistics.median(pd.concat(logs)["seconds"])
    total_min = statistics.mean(lg["seconds"].sum() for lg in logs) / 60

    rows.append({
        "model": label,
        "parametry": params,
        "plik_modelu_MB": round(size_mb, 1),
        "epoka_GPU_s": round(epoch_s, 1),
        "trening_150_epok_min": round(total_min, 1),
        "wnioskowanie_CPU_ms": round(1000 * statistics.median(times), 0),
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
print()
print(df.to_string(index=False))

out_dir = PROJECT_ROOT / "results" / "final"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "performance.csv", index=False, encoding="utf-8-sig")
print("\nZapisano:", out_dir / "performance.csv")