from pathlib import Path

import pandas as pd

from factor_engine.factor_eval import evaluate_ic
from factor_engine.factor_cleaning import get_clean_factor_names
from factor_engine.factors.basic_factors import get_basic_factor_names


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    in_path = PROJECT_ROOT / "data" / "factors" / "clean_factors.parquet"

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "factor_ic_summary.csv"

    panel = pd.read_parquet(in_path)

    raw_factor_names = get_basic_factor_names()
    clean_factor_names = get_clean_factor_names(raw_factor_names)

    summary_1d = evaluate_ic(
        panel=panel,
        factor_names=clean_factor_names,
        return_name="future_return_1d",
    )

    print("\n1D Factor IC Summary:")
    print(summary_1d)

    summary_1d.to_csv(out_path, index=False)

    print(f"\nSaved IC summary to {out_path}")


if __name__ == "__main__":
    main()