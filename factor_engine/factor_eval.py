import numpy as np
import pandas as pd

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
    def fetch(panel: pd.DataFrame, factor_name: str, return_name: str=return_name) -> pd.DataFrame:
        temp = panel[[factor_name,return_name]].dropna()
        if len(temp)<5:
            return np.nan

        return temp[factor_name].corr(temp[return_name],method=method)
    ic = panel.groupby("date").apply(fetch)

    ic.name =f"{factor_name}_{return_name}_{method}_ic"

    return ic

