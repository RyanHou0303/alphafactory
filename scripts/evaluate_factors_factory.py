from pathlib import Path

import pandas as pd

from factor_engine.factor_eval import evaluate_ic


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_factor_names(path: Path) -> list[str]:
    with open(path, "r") as f:
        factor_names = [line.strip() for line in f.readlines()]

    return [name for name in factor_names if name]


def main() -> None:
    in_path = PROJECT_ROOT / "data" / "factors" / "factor_factory_clean.parquet"
    factor_names_path = PROJECT_ROOT / "data" / "factors" / "factor_factory_clean_names.txt"

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "factor_factory_ic_summary.csv"

    panel = pd.read_parquet(in_path)

    clean_factor_names = load_factor_names(factor_names_path)

    summary_1d = evaluate_ic(
        panel=panel,
        factor_names=clean_factor_names,
        return_name="future_return_1d",
    )

    print("\n1D Factor Factory IC Summary:")
    print(summary_1d)
    print(panel.columns)

    summary_1d.to_csv(out_path, index=False)

    print(f"\nSaved IC summary to {out_path}")


if __name__ == "__main__":
    main()