"""
This file provides some groups of factors based on different market assumption and ecnomic intuition
"""

import numpy as np
import pandas as pd

from factor_engine.operators import(pct_change,ts_mean,ts_std,ts_corr)
from factor_engine.operators import pct_change

print(pct_change)
def add_momentum_family(panel: pd.DataFrame)->tuple[pd.DataFrame,list[str]]:
    '''
    medium-tem and short-term price moment factors
    :param panel:
    '''
    panel = panel.copy()
    factor_names=[]
    for n in [3,5,10,20,60,120]:
        name = f"mom_{n}d"
        panel[name] = pct_change(panel,"close",n)
        factor_names.append(name)
    return panel,factor_names

def add_reversal_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
        Short-term reversal factors.
        Negative past return means stronger reversal signal.
    """
    panel = panel.copy()
    factor_names = []
    for n in [1, 3, 5, 10, 20]:
        mom_name = f"mom_{n}d"

        if mom_name not in panel.columns:
            panel[mom_name]=pct_change(panel,"close",n)
        name = f"rev_{n}d"
        panel[name] = -panel[mom_name]
        factor_names.append(name)
    return panel,factor_names

def add_volatility_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    realized volaitlity factors based on daily returns
    """
    panel = panel.copy()
    factor_names =[]

    for n in [5,10,20,60]:
        name = f"vol_{n}d"
        panel[name]=ts_std(panel,"return_1d",n)
        factor_names.append(name)

    return panel, factor_names

def add_volume_shock_family(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    abnormal volume/ volume shock factors
    """
    panel = panel.copy()
    factor_names = []
    for n in [5,10,20,60]:
        mean_name = f"volume_mean_{n}d"
        ratio_name = f"volume_ratio_{n}d"
        panel[mean_name]=ts_mean(panel,"volume",n)
        panel[ratio_name]= panel["volume"]/panel[mean_name].replace(0,np.nan)-1.0
        factor_names.append(ratio_name)
    return panel,factor_names
