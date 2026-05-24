from pathlib import Path

import pandas as pd

from factor_engine.factors.factor_factory import add_all_factor_families


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    in_path = PROJECT_ROOT / "data" / "processed" / "us_equity_daily_clean.parquet"

    out_dir = PROJECT_ROOT / "data" / "factors"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "factor_factory.parquet"
    factor_names_path = out_dir / "factor_factory_names.txt"

    panel = pd.read_parquet(in_path)

    print("Loaded clean daily panel:")
    print(panel.shape)

    panel, factor_names = add_all_factor_families(panel)

    print("\nGenerated factor count:")
    print(len(factor_names))

    print("\nGenerated factor names:")
    for name in factor_names:
        print(name)

    print("\nNaN ratio:")
    print(panel[factor_names].isna().mean().sort_values(ascending=False))

    panel.to_parquet(out_path, index=False)

    with open(factor_names_path, "w") as f:
        for name in factor_names:
            f.write(name + "\n")

    print(f"\nSaved factor factory panel to {out_path}")
    print(f"Saved factor names to {factor_names_path}")


if __name__ == "__main__":
    main()