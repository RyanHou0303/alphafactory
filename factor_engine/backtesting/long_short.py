

import pandas as pd
import numpy as np
factor_data = pd.read_parquet("/Users/bohanhou/PycharmProjects/alphafactory/data/factors/factor_factory_clean.parquet")
from factor_engine.portfolio.signal_combination import *
factor_data["combined_alpha"] = build_combined_alpha(

    panel=factor_data,

    selected_factors=selected,

    weighting="rank_icir",   # or "equal"

)
def alpha_to_long_short_weights(
        panel: pd.DataFrame,
        alpha_col:str="combined_alpha",
        top_n: int = 5,
        bottom_n: int = 5,
)->pd.DataFrame:
    """convert combined alpha to long short weights
    for each date:
    1: demean alpha, 2: normalize by sum(abs(alpha))
    long top 5, short bottom 5 according to alpha
    """
    def weights_one_day(x:pd.DataFrame)->pd.Series:
        weights = pd.Series(0.0,index=x.index)
        x = x.replace([np.inf,-np.inf],np.nan).dropna()

        if len(x)<top_n+ bottom_n:
            return weights

        long_idx = x.nlargest(top_n).index
        short_idx = x.nsmallest(bottom_n).index

        weights.loc[long_idx]=0.5/top_n
        weights.loc[short_idx]=-0.5/bottom_n

        return weights

    return panel.groupby("date")[alpha_col].transform(weights_one_day)


def calculate_portfolio_returns(
        panel:pd.DataFrame,
        weight_col:str,
        return_col: str = "future_return_1d",
        cost_bps: float= 5.0
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
print(summary)



