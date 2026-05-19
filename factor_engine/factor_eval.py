import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr


PROJECTROOT = Path(__file__).resolve().parents[1]
in_path = PROJECTROOT / "data" / "factors" / "clean_factors.parquet"
panel = pd.read_parquet(in_path)
print(panel.columns)
print(panel)
def calc_daily_ic(
        panel: pd.DataFrame,
        factor_name: str,
        return_name: str = "future_return_1d",
        method:str = "pearson"
)->pd.DataFrame:
    """
    calculate daily cross-section IC
    IC_t = corr(factor_i,t, future_return_i,t)
    methods:
    pearson->normal IC
    spearman->rank IC
    """
    def corr_one_day(panel: pd.DataFrame=panel, factor_name: str=factor_name, return_name: str=return_name) -> pd.DataFrame:
        temp = panel[[factor_name,return_name]].dropna()
        if len(temp)<5:
            return np.nan

        return temp[factor_name].corr(temp[return_name],method=method)
    ic = panel.groupby("date").apply(corr_one_day)

    ic.name =f"{factor_name}_{return_name}_{method}_ic"

    return ic


print(calc_daily_ic(panel, "mom_5d").name,calc_daily_ic(panel, "mom_5d").mean())
def evaluate_ic(panel: pd.DataFrame,factor_names: list[str],return_name: str = "future_return_1d")->pd.DataFrame:
    rows = []
    panel = panel.sort_values(["date","stock"]).reset_index(drop=True)
    for factor_name in factor_names:
        pearson_ic = calc_daily_ic(panel,factor_name,return_name=return_name,method = "pearson")
        rank_ic = calc_daily_ic(panel,factor_name,return_name=return_name,method = "spearman")
        rows.append({pearson_ic.name:pearson_ic.mean()})
        rows.append({rank_ic.name:rank_ic.mean()})
    merged_data = {k: v for d in rows for k,v in d.items()}
    return pd.DataFrame.from_dict(merged_data,orient='index',columns=['IC_value'])


print(evaluate_ic(panel,["mom_5d","mom_20d",'rev_5d', 'rev_5d', 'rev_20d', 'vol_20d', 'vol_60d', 'volume_mom_20d',
       'dollar_volume_20d', 'amihud_raw', 'amihud_illiq_20d',
       'price_volume_corr_20d', 'mom_5d_clean', 'mom_20d_clean',
       'mom_60d_clean', 'rev_5d_clean', 'rev_20d_clean', 'vol_20d_clean',
       'vol_60d_clean', 'volume_mom_20d_clean', 'dollar_volume_20d_clean',
       'amihud_illiq_20d_clean', 'price_volume_corr_20d_clean'],"future_return_1d"))





