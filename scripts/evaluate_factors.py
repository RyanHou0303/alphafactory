from pathlib import Path
import pandas as pd
from factor_engine.factor_eval import calc_daily_ic
from factor_engine.factors.basic_factors import get_basic_factor_names()


PROJECTROOT = Path(__file__).resolve().parents[1]

def main():
    in_path = PROJECTROOT / "data" / "factors"/"clean_factors.parquet"

    out_dir = PROJECTROOT/"results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir/"factor_ic_summary.csv"

    panel = pd.read_parquet(in_path)

    raw_factors_name = get_basic_factor_names()

    clean_factor_names = get_clean_factor_name(raw_factor_names)

    summary = evaluate_factors_ic(
        panel,
        factor_names = clean_factor_names,
        return_names = "future_return_1d"
    )
    print("\n Factor IC summary:")
    print(summary)

    summary.to_csv(out_path, index=False)
if __name__ == "__main__":
    main()