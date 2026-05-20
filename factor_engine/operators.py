"""
This file provides some basic operators help to calculate the factors
"""

import numpy as np
import pandas as pd
from pandas import DataFrame


def delay(panel: pd.DataFrame, column: str, n: int) -> pd.Series:
    """
    Time-series lag by stock.

    Example:
        delay(panel, "close", 5)
        means close price 5 trading days ago for each stock
    """
    return panel.groupby("stock")[column].shift(n)


def delta(panel: pd.DataFrame, column: str, n: int) -> pd.Series:
    """
    Difference between current value and value n days ago
    """
    return panel[column] - delay(panel, column, n)


def pct_change(panel: pd.DataFrame, column: str, n: int) -> pd.Series:
    """
    pct change over n trading days
    """
    lagged = delay(panel, column, n)
    return panel[column] / lagged - 1


def log_return(panel: pd.DataFrame, column: str, n: int) -> pd.Series:
    """
    log return over n trading days
    """
    lagged = delay(panel, column, n)
    return np.log(panel[column] / lagged)


def ts_mean(
    panel: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """
    calculte time series mean over trading window
    :param panel:
    :param column:
    :param window:
    :param min_periods:
    :return:
    """
    if min_periods is None:
        min_periods = window

    return (
        panel
        .groupby("stock")[column]
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )

def ts_std(panel: pd.DataFrame, column: str, window: int, min_periods: int|None = None) -> pd.Series:
    """calculate time series standard deviation over trading window"""
    if min_periods is None:
        min_periods = window
    return panel.groupby("stock")[column].rolling(window=window,min_periods=min_periods).std().reset_index(level=0,drop=True)

def ts_sum(
    panel: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """
    rolling sum by stock
    """
    if min_periods is None:
        min_periods = window

    return (
        panel
        .groupby("stock")[column]
        .rolling(window=window, min_periods=min_periods)
        .sum()
        .reset_index(level=0, drop=True)
    )


def ts_rank(
    panel: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """
    It returns the percentile rank of the latest value inside the rolling window
    """
    if min_periods is None:
        min_periods = window

    def rank_last(x: pd.Series) -> float:
        return x.rank(pct=True).iloc[-1]

    return (
        panel
        .groupby("stock")[column]
        .rolling(window=window, min_periods=min_periods)
        .apply(rank_last, raw=False)
        .reset_index(level=0, drop=True)
    )


def ts_corr(
    panel: pd.DataFrame,
    column_x: str,
    column_y: str,
    window: int,
    min_periods: int | None = None,
) -> DataFrame:
    """
    rolling correlation between two columns
    """
    if min_periods is None:
        min_periods = window

    return (
        panel
        .groupby("stock")
        .apply(
            lambda g: g[column_x]
            .rolling(window=window, min_periods=min_periods)
            .corr(g[column_y])
        )
        .reset_index(level=0, drop=True)
    )


def ts_cov(
    panel: pd.DataFrame,
    column_x: str,
    column_y: str,
    window: int,
    min_periods: int | None = None,
) -> DataFrame:
    if min_periods is None:
        min_periods = window

    return (
        panel
        .groupby("stock")
        .apply(
            lambda g: g[column_x]
            .rolling(window=window, min_periods=min_periods)
            .cov(g[column_y])
        )
        .reset_index(level=0, drop=True)
    )


def cs_zscore(
    panel: pd.DataFrame,
    column: str,
) -> pd.Series:

    """providing the cross-section z-score in certain day"""
    grouped = panel.groupby("date")[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return (panel[column] - mean) / std.replace(0, np.nan)


def cs_rank(panel: pd.DataFrame, column: str) -> DataFrame:

    """out the rank of each item in cross-section"""
    return panel.groupby("date")[column].rank(pct=True)


def winsorize(
    panel: pd.DataFrame,
    column: str,
    lower: float,
    upper: float,
) -> pd.Series:

    """winsorized the upper and lower outlier"""
    def clip_one_day(x: pd.Series) -> pd.Series:
        lo = x.quantile(lower)
        hi = x.quantile(upper)
        return x.clip(lower=lo, upper=hi)

    return panel.groupby("date")[column].transform(clip_one_day)


def scale_to_unit_abs_sum(panel: pd.DataFrame, column: str) -> pd.Series:
    denom = panel.groupby("date")[column].transform(lambda x: x.abs().sum())
    return panel[column] / denom.replace(0, np.nan)