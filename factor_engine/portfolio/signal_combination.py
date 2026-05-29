import numpy as np
import pandas as pd
ic_summary=pd.read_csv("/Users/bohanhou/PycharmProjects/alphafactory/results/factor_factory_ic_summary.csv")
factor_data = pd.read_parquet("/Users/bohanhou/PycharmProjects/alphafactory/data/factors/factor_factory_clean.parquet")
print(ic_summary)

pd.set_option('display.max_columns', None)
def select_factors_by_ic(
        ic_summary:pd.DataFrame,
        top_n: int =10,
        min_abs_icir:float = 0.0
)->pd.DataFrame:
    """
    select factor base on absolute rank of ICIR(top n)
    """
    summary = ic_summary.copy()
    summary["rank_icir"]=(
        summary["spearman_rank_ic"]/summary["spearman_std_ic"].replace(0,np.nan)
    )
    summary["abs_rank_icir"]=summary["rank_icir"].abs()

    summary["direction"]=np.where(summary["rank_icir"]>=0,1.0,-1.0)

    selected = (
        summary[summary["abs_rank_icir"]>=min_abs_icir].sort_values("abs_rank_icir",ascending=False).head(top_n).reset_index(drop=True)
    )
    return selected

selected = select_factors_by_ic(ic_summary)
print(selected)

def build_combined_alpha(
        panel:pd.DataFrame,
        selected_factors:pd.DataFrame,
        weighting: str="equal"
):
    """
    building combined alpha based on different weightings(linear combination of some factors columns)

    weighting:
        "equal" -> equal weighted factors
        "rank_icir" -> weights proportional to abs(rank_icir)
    """
    factor_names = selected_factors["factor"].tolist()

    #create some pd.series: directions and weights
    direction = selected_factors.set_index("factor")["direction"]
    print(f"direction:\n{direction}")


    if weighting =="equal":
        weights = pd.Series(1.0,index=factor_names)
        weights = weights/weights.abs().sum()

    elif weighting =="rank_icir":
        weights = selected_factors.set_index("factor")["abs_rank_icir"]
        weights = weights/weights.abs().sum()

    else:

        raise ValueError(f"Unknown weighting method{weighting}")



    alpha = pd.Series(0.0,index = panel.index)

    for factor in factor_names:
        alpha+=weights[factor]*direction[factor]*panel[factor]
    return alpha



factor_data["combined_alpha"] = build_combined_alpha(

    panel=factor_data,

    selected_factors=selected,

    weighting="rank_icir",   # or "equal"

)
print(factor_data["combined_alpha"])




