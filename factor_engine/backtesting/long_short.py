

import pandas as pd
import numpy as np
factor_data = pd.read_parquet("/Users/bohanhou/PycharmProjects/alphafactory/data/factors/factor_factory_clean.parquet")
from factor_engine.portfolio.signal_combination import *
from factor_engine.factor_eval import *
factor_data["combined_alpha"] = build_combined_alpha(

    panel=factor_data,

    selected_factors=selected,

    weighting="rank_icir",   # or "equal"

)

def weights(alpha:pd.Series,top_n:int=3,bottom_n:int=3)->pd.Series:
    weights = pd.Series(0.0,index=alpha.index)

    x = alpha.replace([np.inf,-np.inf],np.nan).dropna()

    long_idx = x.nlargest(top_n).index
    short_idx = x.nsmallest(bottom_n).index

    weights.loc[long_idx]=0.5/top_n
    weights.loc[short_idx]=0.5/bottom_n

    return weights


def alpha_to_long_short_weights(
        panel: pd.DataFrame,
        factor_names: list[str],
        return_col: str = "future_return_5d",
        alpha_col:str="combined_alpha",
        ic_window: int = 252,
        top_n: int = 3,
        bottom_n: int = 3,
        frequency: int =1,
)->pd.DataFrame:
    """convert combined alpha to long short weights
    for each date:
    1: demean alpha, 2: normalize by sum(abs(alpha))
    long top 5, short bottom 5 according to alpha
    different balance frequency, 1, 5, 21, 63 ,etc
    """
    panel = panel.copy()
    panel["_row_id"]=np.arange(len(panel))
    panel = panel.sort_values(["date","stock"]).reset_index(drop=True)

    unique_dates=(
        panel["date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    rebalance_dates = set(unique_dates.iloc[::frequency])
    selected_records = []

    for i in range(ic_window,len(unique_dates),rebalance_dates):
        trade_date = unique_dates.iloc[i]
        hist_dates = unique_dates.iloc[i-ic_window:i]

        window_data = panel[panel["date"].isin(hist_dates)]
        one_day = panel[panel["date"]==trade_date]

        """calculate IC in a period of time, then"""
        ic_summary = calc_ic_summary_in_window(
            window_data=window_data,
            factor_names=factor_names,
            return_col=return_col,
        )
        """
        example ic_summary output:

                                 factor  mean_rank_ic  ...  directon  n_days
        0   price_volume_corr_20d_clean     -0.016902  ...      -1.0    2742
        1        amihud_illiq_20d_clean      0.015960  ...       1.0    2741
        2                 vol_60d_clean      0.025198  ...       1.0    2701
        3                 vol_20d_clean      0.020046  ...       1.0    2741
        4                 mom_20d_clean     -0.014007  ...      -1.0    2741
        5                 rev_20d_clean      0.014007  ...       1.0    2741
        6                  mom_5d_clean     -0.013158  ...      -1.0    2756
        7                  rev_5d_clean      0.013158  ...       1.0    2756
        8       dollar_volume_20d_clean     -0.009994  ...      -1.0    2742
        9                 mom_60d_clean      0.008859  ...       1.0    2701
        10         volume_mom_20d_clean      0.001420  ...       1.0    2742
        
        """


        """select top factor based on the ic summary for past obeservation windows"""
        selected_factor = ic_summary.head(top_n).copy()


        alpha = build_combined_alpha(one_day,selected_factors=selected_factor,weighting="rank_icir")





    weights = weights_one_day(panel)






    weights_wide = (
        panel.pivot(index="date",columns="stock", values="raw_weight").sort_index()
    )

    weights_wide = weights_wide.ffill().fillna(0.0)
    weights_long = (
        weights_wide.stack().rename("target_weight").reset_index()
    )
    panel = panel.merge(
        weights_long,
        on=["date","stock"],
        how = "left",
    )

    panel= panel.sort_values("_row_id")
    return panel["target_weight"].fillna(0.0).reset_index(drop=True)





    return panel.groupby("date")[alpha_col].transform(weights_one_day)


def calculate_portfolio_returns(
        panel:pd.DataFrame,
        weight_col:str,
        return_col: str = "future_return_1d",
        cost_bps: float= 5,

)->pd.DataFrame:
    """Calculate daily long short portfolio return
    portfolio_return_t = sum_i weights_i,t*future-return_i,t
    transaction cost: turnover_t* costbps/10000"""

    panel = panel.copy()

    panel = panel.sort_values(["date","stock"]).reset_index(drop=True)

    temp = panel[["date","stock",weight_col,return_col]].copy()
    temp = temp.dropna(subset=[weight_col,return_col])

    temp["weighted_return"] = temp[weight_col]*temp[return_col]

    daily = (
        temp
        .groupby("date")["weighted_return"].sum().to_frame("gross_return")
    )

    weights_wide = (
        temp.pivot(index ="date",columns="stock",values=weight_col).fillna(0.0)
    )

    turnover = weights_wide.diff().abs().sum(axis=1)
    if len(turnover)>0:
        turnover.iloc[0]=weights_wide.iloc[0].abs().sum()

    daily["turnover"]=turnover.reindex(daily.index).fillna(0.0)

    daily["cost"]=daily["turnover"]*cost_bps/10000.0

    daily["net_return"]=daily["gross_return"]-daily["cost"]
    daily["gross_equity"]=(1+daily["gross_return"]).cumprod()
    daily["net_equity"]=(1.0+daily["net_return"]).cumprod()

    return daily.reset_index()

def summarize_backtest(daily_returns: pd.DataFrame) -> dict:

    """

    Summarize long-short backtest performance.

    """

    ret = daily_returns["net_return"].dropna()

    annual_return = ret.mean() * 252

    annual_volatility = ret.std() * np.sqrt(252)

    sharpe = (

        annual_return / annual_volatility

        if annual_volatility != 0

        else np.nan

    )

    equity = daily_returns["net_equity"]

    running_max = equity.cummax()

    drawdown = equity / running_max - 1.0

    summary = {

        "annual_return": annual_return,

        "annual_volatility": annual_volatility,

        "sharpe": sharpe,

        "max_drawdown": drawdown.min(),

        "average_daily_turnover": daily_returns["turnover"].mean(),

        "total_return": daily_returns["net_equity"].iloc[-1] - 1.0,

    }

    return summary

factor_data["target_weights"]= alpha_to_long_short_weights(factor_data)
returns = calculate_portfolio_returns(factor_data,weight_col="target_weights")
summary = summarize_backtest(returns)
print(returns)
print(pd.Series(summary))



