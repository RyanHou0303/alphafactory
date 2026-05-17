from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow
import fastparquet

#======================================================
#config
##======================================================

raw_path = Path("../data/raw/us_equity_daily_raw.parquet")
out_dir = Path("../data/processed")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir/"us_equity_daily_clean.parquet"

raw = pd.read_parquet(raw_path)

print("Raw shape:", raw.shape)
print("Raw columns example:", raw.columns[:10])

#======================================================
#Convert wide MultiIndex to long panel
#======================================================

print(raw.columns)
if isinstance(raw.columns, pd.MultiIndex):
    panel=(
        raw.stack(level=1)
        .reset_index().rename(columns={
            "Date":"date",
            "index":"index",
            "Ticker":"stock",
            "Open":"open",
            "High":"high",
            "Low":"low",
            "Close":"close",
            "Volume":"volume",
    })
    )
else:
    raise ValueError("Expected MultiIndex column from yfinance")

panel.columns = [str(c).lower() for c in panel.columns]
print(panel.head())

# ============================================================
# Basic type cleaning
# ============================================================

panel["date"]= pd.to_datetime(panel["date"])
panel["stock"]=panel["stock"].astype(str)

numeric_cols = ["open", "high", "low", "close", "volume"]

for col in numeric_cols:
    panel[col] = pd.to_numeric(panel[col], errors="coerce")

# ============================================================
# Sort data
# ============================================================
panel = panel.sort_values(["stock","date"]).reset_index(drop=True)

# ============================================================
# Remove invalid rows
# ============================================================

price_cols = ["open", "high", "low", "close"]
for col in price_cols:
    panel.loc[panel[col]<0,col] = np.nan
panel.loc[panel["volume"]<0,"volume"] = np.nan

bad_high_low = panel["high"]<panel["low"]
bad_high = (panel["high"]<panel["open"])|(panel["high"]<panel["close"])
bad_low = (panel["low"]>panel["open"]) | (panel["low"]>panel["close"])
bad_ohlc = bad_high_low | bad_high| bad_low

print("bad_ohlc rows", bad_ohlc.sum())
panel.loc[bad_ohlc, price_cols] = np.nan

missing_summary = (
    panel.groupby("stock")[numeric_cols].apply(lambda x: x.isna().mean()).reset_index()
)

print("\nMissing ratio by stock:")
print(missing_summary)
missing = panel.groupby("stock")["close"]

close_missing = panel.groupby("stock")["close"].apply(lambda x: x.isna().mean())

valid_stocks = close_missing[close_missing<=0.05].index
panel = panel[panel["stock"].isin(valid_stocks)].copy()
print("Number of valid stocks:", len(valid_stocks))
print("Panel shape after dropping bad stocks:", panel.shape)



# ============================================================
#Forward Fill Nan
# ============================================================

panel[price_cols] = panel.groupby("stock")[price_cols].ffill()

panel["volume"]=panel["volume"].fillna(0)

# ============================================================
#Basic variable
# ============================================================
panel["return_1d"]=(
    panel.groupby("stock")["close"].pct_change()
)

panel["loc_return_1d"]=(
    panel.groupby("stock")["close"].transform(lambda x: np.log(x/x.shift(1)))
)
panel["dollar_volume"]=panel["close"]*panel["volume"]

print(panel)


# ============================================================
# Future returns for factor testing
# ============================================================
panel["future_return_1d"]=(panel.groupby("stock")["close"].pct_change()
                           .groupby(panel["stock"]).shift(-1))
panel["future_return_5d"]=(panel.groupby("stock")["close"].pct_change(5)
                           .groupby(panel["stock"]).shift(-5))

panel["future_return_20d"] = (panel.groupby("stock")["close"].pct_change(20)
                           .groupby(panel["stock"]).shift(-20))

# ============================================================
# Final clean up
# ============================================================

panel = panel.sort_values(["date","stock"]).reset_index(drop=True)
print("\nClean panel:")
print(panel.head())

print("\nFinal columns:")
print(panel.columns.tolist())
print("\nFinal shape:", panel.shape)

panel.to_parquet(out_path)
print(f"\nSaving to {out_path}")