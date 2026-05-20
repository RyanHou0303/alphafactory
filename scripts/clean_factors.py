from pathlib import Path

import pandas as pd

from factor_engine.factor_cleaning import (
    clean_factors,
    get_clean_factor_names,
)
from factor_engine.factors.basic_factors import get_basic_factor_names


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # ============================================================
    # Paths
    # ============================================================

    in_path = PROJECT_ROOT / "data" / "factors" / "basic_factors.parquet"

    out_dir = PROJECT_ROOT / "data" / "factors"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "clean_factors.parquet"

    # ============================================================
    # Load factor data
    # ============================================================

    panel = pd.read_parquet(in_path)

    print("Loaded factor panel:")
    print(panel.shape)

    # ============================================================
    # Clean factors
    # ============================================================

    factor_names = get_basic_factor_names()
    print(factor_names)
    panel = clean_factors(

        panel,

        factor_names,

        0.01,

        0.99,

    )

    clean_factor_names = get_clean_factor_names(factor_names)



    # ============================================================
    # Diagnostics
    # ============================================================

    print("\nRaw factor names:")
    print(factor_names)

    print("\nClean factor names:")
    print(clean_factor_names)

    print("\nNaN ratio of clean factors:")
    print(panel[clean_factor_names].isna().mean().sort_values(ascending=False))

    print("\nClean factor sample:")
    print(panel[["date", "stock"] + clean_factor_names].head(30))

    # ============================================================
    # Save
    # ============================================================

    panel.to_parquet(out_path, index=False)

    print(f"\nSaved clean factors to {out_path}")


if __name__ == "__main__":
    main()