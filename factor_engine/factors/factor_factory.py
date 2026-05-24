"""
This file provides some groups of factors based on different market assumption and ecnomic intuition
"""


"""column in "clean factor.parquet" the Index(['date', 'stock', 'close', 'high', 'low', 'open', 'volume', 'return_1d',
       'loc_return_1d', 'dollar_volume', 'future_return_1d',
       'future_return_5d', 'future_return_20d', 'mom_5d', 'mom_20d', 'mom_60d',
       'rev_5d', 'rev_20d', 'vol_20d', 'vol_60d', 'volume_mom_20d',
       'dollar_volume_20d', 'amihud_raw', 'amihud_illiq_20d',
       'price_volume_corr_20d', 'mom_5d_clean', 'mom_20d_clean',
       'mom_60d_clean', 'rev_5d_clean', 'rev_20d_clean', 'vol_20d_clean',
       'vol_60d_clean', 'volume_mom_20d_clean', 'dollar_volume_20d_clean',
       'amihud_illiq_20d_clean', 'price_volume_corr_20d_clean'],
      dtype='str')"""
import numpy as np
import pandas as pd

from factor_engine.operators import (
    pct_change,
    ts_mean,
    ts_std,
    ts_corr,
)


def add_momentum_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    '''
    medium-tem and short-term price moment factors
    :param panel:
    '''
    factor_names = []

    for n in [3, 5, 10, 20, 60, 120]:
        name = f"mom_{n}d"
        panel[name] = pct_change(panel, "close", n)
        factor_names.append(name)

    return panel, factor_names


def add_reversal_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
        Short-term reversal factors.
        Negative past return means stronger reversal signal.
    """
    factor_names = []

    for n in [1, 3, 5, 10, 20]:
        mom_name = f"mom_{n}d"

        if mom_name not in panel.columns:
            panel[mom_name] = pct_change(panel, "close", n)

        name = f"rev_{n}d"
        panel[name] = -panel[mom_name]
        factor_names.append(name)

    return panel, factor_names


def add_volatility_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    realized volaitlity factors based on daily returns
    """
    factor_names = []

    for n in [5, 10, 20, 60]:
        name = f"vol_{n}d"
        panel[name] = ts_std(panel, "return_1d", n)
        factor_names.append(name)

    return panel, factor_names


def add_volume_shock_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    abnormal volume/ volume shock factors
    """
    factor_names = []

    for n in [5, 10, 20, 60]:
        mean_name = f"volume_mean_{n}d"
        ratio_name = f"volume_ratio_{n}d"

        panel[mean_name] = ts_mean(panel, "volume", n)
        panel[ratio_name] = panel["volume"] / panel[mean_name].replace(0, np.nan) - 1.0

        factor_names.append(ratio_name)

    return panel, factor_names


def add_liquidity_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """abs(return)/dollar_volume"""
    factor_names = []

    panel["amihud_raw"] = (
        panel["return_1d"].abs() / panel["dollar_volume"].replace(0, np.nan)
    )

    for n in [5, 10, 20, 60]:
        name = f"amihud_illiq_{n}d"
        panel[name] = ts_mean(panel, "amihud_raw", n)
        factor_names.append(name)

    panel["amihud_change_20_60d"] = (
        panel["amihud_illiq_20d"] / panel["amihud_illiq_60d"].replace(0, np.nan) - 1
    )

    panel["amihud_change_5_20d"] = (
        panel["amihud_illiq_5d"] / panel["amihud_illiq_20d"].replace(0, np.nan) - 1
    )

    panel["amihud_change_5_10d"] = (
        panel["amihud_illiq_5d"] / panel["amihud_illiq_10d"].replace(0, np.nan) - 1
    )

    factor_names.extend([
        "amihud_change_20_60d",
        "amihud_change_5_20d",
        "amihud_change_5_10d",
    ])

    for n in [5, 10, 20, 60]:
        name = f"dollar_volume_{n}d"
        panel[name] = ts_mean(panel, "dollar_volume", n)
        factor_names.append(name)

    return panel, factor_names


def add_price_volume_interaction_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Price-volume interaction factors.
    Return-volume interaction factors
    """
    factor_names = []

    for n in [5, 10, 20, 60]:
        name = f"ret_volume_corr_{n}d"
        panel[name] = ts_corr(panel, "return_1d", "volume", n)
        factor_names.append(name)

    for n in [5, 10, 20, 60]:
        name = f"price_volume_corr_{n}d"
        panel[name] = ts_corr(panel, "close", "volume", n)
        factor_names.append(name)

    return panel, factor_names


def add_risk_adjusted_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Risk-adjusted momentum and reversal factors.
    Idea: same return signal, but scaled by realized volatility.
    """
    factor_names = []

    for n in [5, 10, 20, 60]:
        mom_name = f"mom_{n}d"
        rev_name = f"rev_{n}d"
        vol_name = f"vol_{n}d"

        if mom_name not in panel.columns:
            panel[mom_name] = pct_change(panel, "close", n)

        if rev_name not in panel.columns:
            panel[rev_name] = -panel[mom_name]

        if vol_name not in panel.columns:
            panel[vol_name] = ts_std(panel, "return_1d", n)

        risk_mom_name = f"risk_adj_mom_{n}d"
        risk_rev_name = f"risk_adj_rev_{n}d"

        panel[risk_mom_name] = panel[mom_name] / panel[vol_name].replace(0, np.nan)
        panel[risk_rev_name] = panel[rev_name] / panel[vol_name].replace(0, np.nan)

        factor_names.extend([risk_mom_name, risk_rev_name])

    return panel, factor_names


def add_range_gap_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """intraday range and overnight gap factors"""
    factor_names = []

    panel["high_low_range"] = panel["high"] / panel["low"].replace(0, np.nan) - 1.0
    panel["close_to_high"] = panel["close"] / panel["high"].replace(0, np.nan) - 1.0
    panel["close_to_low"] = panel["close"] / panel["low"].replace(0, np.nan) - 1.0

    prev_close = panel.groupby("stock")["close"].shift(1)

    panel["overnight_return"] = panel["open"] / prev_close.replace(0, np.nan) - 1.0
    panel["intraday_return"] = panel["close"] / panel["open"].replace(0, np.nan) - 1.0
    panel["gap_reversal"] = -panel["overnight_return"]

    factor_names.extend(
        [
            "high_low_range",
            "close_to_high",
            "close_to_low",
            "overnight_return",
            "intraday_return",
            "gap_reversal",
        ]
    )

    return panel, factor_names


def add_high_volume_reversal_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """interaction between short-term reversal and abnormal volume"""

    factor_names = []

    if "volume_ratio_20d" not in panel.columns:
        mean_volume = ts_mean(panel, "volume", 20)
        panel["volume_ratio_20d"] = panel["volume"] / mean_volume.replace(0, np.nan) - 1.0

    for n in [1, 3, 5, 10]:
        mom_name = f"mom_{n}d"

        if mom_name not in panel.columns:
            panel[mom_name] = pct_change(panel, "close", n)

        name = f"high_volume_rev_{n}d"
        panel[name] = -panel[mom_name] * panel["volume_ratio_20d"]
        factor_names.append(name)

    return panel, factor_names


def add_all_factor_families(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add all candidate factor families.
    Returns:
        panel with candidate factors
        list of factor names generated by the factory

    """
    # Only copy once here. Family functions will add columns to this copied panel.
    panel = panel.copy()
    panel = panel.sort_values(["stock", "date"]).reset_index(drop=True)

    all_factor_names = []

    family_functions = [
        add_momentum_family,
        add_reversal_family,
        add_volatility_family,
        add_volume_shock_family,
        add_liquidity_family,
        add_price_volume_interaction_family,
        add_risk_adjusted_family,
        add_range_gap_family,
        add_high_volume_reversal_family,
    ]

    for func in family_functions:
        panel, factor_names = func(panel)
        all_factor_names.extend(factor_names)

    # remove duplicated factor names while keeping order

    all_factor_names = list(dict.fromkeys(all_factor_names))

    missing_cols = [col for col in all_factor_names if col not in panel.columns]

    if missing_cols:
        raise ValueError(f"Generated factor names missing from panel: {missing_cols}")

    panel = panel.sort_values(["date", "stock"]).reset_index(drop=True)

    return panel, all_factor_names