import numpy as np
import pandas as pd

from factor_engine.operators import winsorize,cs_zscore

def clean_factor(
        panel:pd.DataFrame,
        factor_name:str,
        lower:float=0.01,
        upper:float=0.99,
)->pd.Series:
     """
     Clean one factor by
      1. winsorizing cross-sectionally by date
      2. z-scoring cross-sectionally by date
    """

     temp = panel[["date","stock",factor_name]].copy()
     winsorized_name = factor_name+"_winsorized"

     temp[winsorized_name] = winsorize(
         temp,
         factor_name,
         lower=lower,
         upper=upper,
     )

     clean_name = factor_name+"_z_score_clean"
     temp[clean_name] = cs_zscore(
         temp,
         winsorized_name,
     )
     return temp[clean_name]

def clean_factors(
        panel:pd.DataFrame,
        factor_names:list[str],
        lower:float=0.01,
        upper:float=0.99,
)->pd.DataFrame:

    """
    clean multiple factors
    for each factor, clean it by winsorizing and z-scoring
    """

    panel = panel.copy()
    panel = panel.sort_values(["date","stock"]).reset_index(drop=True)

    for factor_name in factor_names:
        clean_name = factor_name+"_clean"

        panel[clean_name] = clean_factor(panel,factor_name,lower=lower,upper=upper)

    return panel





def get_clean_factor_names(factor_names: list[str]) -> list[str]:
    """
    Convert raw factor names to clean factor names.
    """
    return [factor_name + "_clean" for factor_name in factor_names]




