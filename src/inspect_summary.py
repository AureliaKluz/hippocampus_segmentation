from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
df = pd.read_csv(project_root/"results"/"dataset_summary.csv")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

shape = df["shape"].str.split("x", expand=True).astype(int)
shape.columns = ["x", "y", "z"]
print("Rozmiary obrazów - min / max w każdej osi: ")
print(shape.agg(["min", "max"]))

print("\nIntensywności: ")
print(df[["int_min", "int_max", "int_mean"]].describe().round(1))

print("\nPrzypadki odstające: ")
cols = ["case", "vol_anterior_mm3", "vol_posterior_mm3", "vol_total_mm3",
        "ratio_ant_post", "z_total", "z_ratio"]
print(df.loc[df["outlier"], cols].round(2))

print("\nStosunek Anterior/Posterior w całym zbiorze: ")
print(df["ratio_ant_post"].describe().round(2))

big = df[df["int_max"] > 10000].sort_values("int_max", ascending=False)
print("\nLiczba przypadków z int_max > 10000:", len(big))
print(big[["case", "int_min", "int_max", "int_mean"]].head(10).round(1))