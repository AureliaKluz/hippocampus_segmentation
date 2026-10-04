from pathlib import Path
import nibabel as nib
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
root = project_root / "data"/"Task04_Hippocampus"/"imagesTr"
df = pd.read_csv(project_root / "results"/"dataset_summary.csv")

rows = []
for case, int_max in zip(df["case"], df["int_max"]):
    h = nib.load(root/f"{case}.nii.gz").header
    rows.append({
        "case": case,
        "high_scale": int_max>10000,
        "dtype": str(h.get_data_dtype()),
        "slope": str(h["scl_slope"]),
        "inter": str(h["scl_inter"]),
        "descrip": bytes(h["descrip"]).split(b"\x00")[0].decode(errors="replace"),
        "qform_code": str(h["qform_code"]),
        "sform_code": str(h["sform_code"])
    })

res = pd.DataFrame(rows)
for col in ["dtype", "slope", "inter", "descrip", "qform_code", "sform_code"]:
    print(f"\n{col}:")
    print(pd.crosstab(res[col], res["high_scale"]))

import numpy as np

res.to_csv(project_root / "results" / "header_info.csv", index=False)

merged = df.merge(res[["case", "dtype"]], on="case")
merged["group"] = merged["dtype"]
merged.loc[merged["int_max"] > 10000, "group"] = "float32_high"

print("\nint_max w grupach:")
print(merged.groupby("group")["int_max"].describe().round(0))
print("\nint_mean w grupach:")
print(merged.groupby("group")["int_mean"].describe().round(0))