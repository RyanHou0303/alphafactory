from pathlib import Path
import pandas as pd
from factor_engine.factor_cleaning import(
    clean_factors,get_clean_factor_names
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_factor_names(path:Path)->list[str]:
    with open(path,"r") as f:
        factor_names = [line.strip() for line in f.readlines()]

    factor_names = [name for name in factor_names if name]

    return factor_names

in_path = PROJECT_ROOT / "data"/"factors"/"factor_factory.parquet"
factor_name_path = PROJECT_ROOT / "data"/"factors"/"factor_factory_names.txt"
out_path = PROJECT_ROOT / "data"/"factors"/"factor_factory_clean.parquet"
clean_factor_names_path = PROJECT_ROOT / "data"/"factors"/"factor_factory_clean_names.txt"

panel = pd.read_parquet(in_path)
factor_names = load_factor_names(factor_name_path)

print("Loaded factor factory panel:")
print(panel.shape)
print("\nRaw factor count:")
print(len(factor_names))
panel = clean_factors(
    panel,
    factor_names,
    0.01,
    0.99,
)
clean_factor_names = get_clean_factor_names(factor_names)
print("\nClean factor count:")
print(len(clean_factor_names))
print("\nNaN ratio of clean factors:")
print(panel[clean_factor_names].isna().mean().sort_values(ascending=False))
print("\nClean factor sample:")
print(panel[["date", "stock"] + clean_factor_names[:10]].head(30))

panel.to_parquet(out_path, index=False)
with open(clean_factor_names_path, "w") as f:
    for name in clean_factor_names:
        f.write(name + "\n")

print(f"\nSaved clean factor factory panel to {out_path}")

print(f"Saved clean factor names to {clean_factor_names_path}")

