import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
HIGH_SCALE_TRESHOLD = 10000

project_root = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_root / "results" / "dataset_summary.csv")
hdr = pd.read_csv(project_root / "results"/ "header_info.csv")
df = df.merge(hdr[["case", "dtype"]], on="case")

df["group"]=df["dtype"]
df.loc[df["int_max"]> HIGH_SCALE_TRESHOLD, "group"] = "float32_high"

groups = df.set_index("case")["group"]
cases=df["case"].tolist()

train_cases, temp_cases = train_test_split(
    cases, test_size=0.30, random_state=SEED, stratify=groups.loc[cases]
)
val_cases, test_cases = train_test_split(
    temp_cases, test_size=0.50, random_state=SEED,
    stratify=groups.loc[temp_cases]
)

assert not set(train_cases) & set(val_cases)
assert not set(train_cases) & set(test_cases)
assert not set(val_cases) & set(test_cases)
assert len(train_cases) + len(val_cases) + len(test_cases) == len(cases)

split = {
    "seed": SEED,
    "stratified_by": "dtype (uint8/float32) and int_max > 10000",
    "train": sorted(train_cases),
    "val": sorted(val_cases),
    "test": sorted(test_cases),
}

out_dir = project_root / "configs"
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "split.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(split, f, indent=2)
print("Zapisano:", out_file)

df["split"] = "train"
df.loc[df["case"].isin(split["val"]), "split"] = "val"
df.loc[df["case"].isin(split["test"]), "split"] = "test"

print("\nLiczba przypadków:", df["split"].value_counts().to_dict())
print("\nGrupy w podziałach:")
print(pd.crosstab(df["group"], df["split"]))
print("\nObjętość całkowita [mm3] w podziałach:")
print(df.groupby("split")["vol_total_mm3"].agg(["mean", "std"]).round(1))
print("\nPrzypadki odstające trafiły do:")
print(df.loc[df["outlier"], ["case", "split"]])
