import numpy as np
import pandas as pd
from pathlib import Path

out_dir = Path("../../data/factors")
out_dir.mkdir(parents=True,exist_ok=True)
out_path= out_dir/"basic_factors.parquet"


from factor_engine.operators import (
    pct_change,
    ts_std,
    ts_mean,
    ts_corr,
)


def add_basic_factors(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()

    # IMPORTANT:
    # Time-series operators need stock-date sorting.
    panel = panel.sort_values(["stock", "date"]).reset_index(drop=True)

    # ============================================================
    # Momentum factors
    # ============================================================

    panel["mom_5d"] = pct_change(panel, "close", 5)
    panel["mom_20d"] = pct_change(panel, "close", 20)
    panel["mom_60d"] = pct_change(panel, "close", 60)

    # ============================================================
    # Reversal factors
    # ============================================================

    panel["rev_5d"] = -panel["mom_5d"]
    panel["rev_20d"] = -panel["mom_20d"]

    # ============================================================
    # Volatility factors
    # ============================================================

    panel["vol_20d"] = ts_std(panel, "return_1d", 20)
    panel["vol_60d"] = ts_std(panel, "return_1d", 60)

    # ============================================================
    # Volume / liquidity factors
    # ============================================================

    panel["volume_mom_20d"] = (
        panel["volume"] / ts_mean(panel, "volume", 20).replace(0, np.nan) - 1.0
    )

    panel["dollar_volume_20d"] = ts_mean(panel, "dollar_volume", 20)

    panel["amihud_raw"] = (
        panel["return_1d"].abs()
        / panel["dollar_volume"].replace(0, np.nan)
    )

    panel["amihud_illiq_20d"] = ts_mean(panel, "amihud_raw", 20)

    # ============================================================
    # Price-volume factor
    # ============================================================

    panel["price_volume_corr_20d"] = ts_corr(panel, "close", "volume", 20)

    # Final output sort: better for later cross-sectional analysis
    panel = panel.sort_values(["date", "stock"]).reset_index(drop=True)

    return panel


def get_basic_factor_names() -> list[str]:
    return [
        "mom_5d",
        "mom_20d",
        "mom_60d",
        "rev_5d",
        "rev_20d",
        "vol_20d",
        "vol_60d",
        "volume_mom_20d",
        "dollar_volume_20d",
        "amihud_illiq_20d",
        "price_volume_corr_20d",
    ]


if __name__ == "__main__":
    panel = pd.read_parquet("../../data/processed/us_equity_daily_clean.parquet")

    panel = add_basic_factors(panel)

    factor_names = get_basic_factor_names()

    print(panel.columns)
    print(panel[["date", "stock"] + factor_names].head(30))

    print("\nNaN ratio:")
    print(panel[factor_names].isna().mean().sort_values(ascending=False))

    panel.to_parquet(out_path, index=False)

    print(f"\nSaved to {out_path}")