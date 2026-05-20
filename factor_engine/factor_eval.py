import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr


PROJECTROOT = Path(__file__).resolve().parents[1]
in_path = PROJECTROOT / "data" / "factors" / "clean_factors.parquet"
panel = pd.read_parquet(in_path)
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

def summarize_ic(
        ic: pd.DataFrame,
)->dict:
    ic = ic.dropna()
    if len(ic) == 0:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "icir": np.nan,
            "positive_ic_ratio": np.nan,
            "n_days": 0,
        }

    mean_ic = ic.mean()
    std_ic = ic.std()
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": mean_ic / std_ic if std_ic != 0 else np.nan,
        "positive_ic_ratio": (ic > 0).mean(),
        "n_days": len(ic),
    }


print(calc_daily_ic(panel, "mom_5d").name,calc_daily_ic(panel, "mom_5d").mean())
def evaluate_ic(panel: pd.DataFrame,factor_names: list[str],return_name: str = "future_return_1d")->pd.DataFrame:
    rows = []
    panel = panel.sort_values(["date","stock"]).reset_index(drop=True)
    for factor_name in factor_names:
        pearson_ic = calc_daily_ic(panel,factor_name,return_name=return_name,method = "pearson")
        rank_ic = calc_daily_ic(panel,factor_name,return_name=return_name,method = "spearman")

        pearson_ic_summary = summarize_ic(pearson_ic)
        rank_ic_summary = summarize_ic(rank_ic)

        rows.append(
            {
                "factor": factor_name,
                "return": return_name,
                "pearson_mean_ic":pearson_ic_summary["mean_ic"],
                "pearson_std_ic":pearson_ic_summary["std_ic"],
                "spearman_rank_ic": rank_ic_summary["mean_ic"],
                "spearman_std_ic": rank_ic_summary["std_ic"],
                "n_days" : pearson_ic_summary["n_days"],
            }
        )
    result = pd.DataFrame(rows)
    result = result.sort_values("spearman_rank_ic",ascending=False).reset_index(drop=True)
    pd.set_option('display.max_columns', None)
    return result


print(evaluate_ic(panel,[ 'mom_5d_clean', 'mom_20d_clean',
       'mom_60d_clean', 'rev_5d_clean', 'rev_20d_clean', 'vol_20d_clean',
       'vol_60d_clean', 'volume_mom_20d_clean', 'dollar_volume_20d_clean',
       'amihud_illiq_20d_clean', 'price_volume_corr_20d_clean'],"future_return_1d"))





