from pathlib import Path
import pandas as pd
from factor_engine.operators import(delay,delta,pct_change,ts_mean,ts_rank,ts_corr,cs_rank,cs_zscore,ts_std)

data_path = Path("../data/processed/us_equity_daily_clean.parquet")
panel = pd.read_parquet(data_path)

panel=panel.sort_values(["stock","date"]).reset_index(drop=True)


panel["close_delay_5"]=delay(panel,"close",5)
panel["close_delay_10"]=delay(panel,"close",10)
panel["return_5d_test"]=pct_change(panel,"close",5)
panel["ma_20"]=ts_mean(panel,"close",20)
panel["vol_20"]=ts_std(panel,"close",20)
panel["close_ts_rank_20"]=ts_rank(panel,"close",20)
panel["price_volume_corr_20"] = ts_corr(panel, "close", "volume", 20)
print(panel)

panel["return_5d_rank"]=cs_rank(panel,"return_5d_test")
panel["return_5d_zscore"]=cs_zscore(panel,"return_5d_test")
print(panel.head(30))

print("\n NaN ratio:")
print(
    panel[
        ["close_delay_5",
         "close_delay_10",
         "return_5d_test",
         "ma_20",
         "vol_20",
         "close_ts_rank_20",
         "price_volume_corr_20",
         "return_5d_rank",
         "return_5d_zscore",
         ]
        ].isna().mean()
)

print("\n Cross section check:")
check = (
    panel.dropna(subset=["return_5d_zscore"])
    .groupby("date")["return_5d_zscore"]
    .agg(["mean","std"])
)
print(check.head())
print(check.describe())