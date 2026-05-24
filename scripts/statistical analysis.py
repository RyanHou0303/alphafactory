from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

in_path = PROJECT_ROOT / "data"/"processed"/"us_equity_daily_clean.parquet"
panel = pd.read_parquet(in_path)
pd.set_option('display.max_columns', None)
print(panel.head(),panel.columns)

df = panel[panel["stock"] == "AAPL"].copy().reset_index(drop=True)
print(df)


def push_response(ticker:str,tau:int, n_bins= 50):
    """
    perform push and response test for the instrument, help to identify the trending following or mean reversion
    property, normally, a positive slope around zero dot means trend-following(positive push corresponding positive response)
    """

    df = panel[panel["stock"]==ticker].copy().reset_index(drop=True)
    if tau<=0:
        raise ValueError["tau must be a positive integer"]

    if len(df["close"])<=2*tau:
        raise ValueError("Price series is too short fo the choose tau")
    p = df["close"]
    push= p-p.shift(tau)
    response = p.shift(-tau)-p

    PR= pd.DataFrame({
        "push":push,
        "response":response,
    }).dropna()

    PR["bin"] = pd.qcut(PR["push"],q=n_bins,duplicates="drop")
    grouped =PR.groupby("bin").agg({"push": "mean","response":"mean"}).dropna()
    print(grouped)
    return grouped

def variance_ratio(ticker:str,q:int):
    """perfromance Andrew Lo variance ratio test(simple version) to identify if it is more trend following or mean reversion compared
     to random walk"""
    df = panel[panel["stock"] == ticker].copy().reset_index(drop=True)
    logp = np.log(df["close"])
    r1 = logp.diff().dropna()
    rq = logp.diff(q).dropna()

    vr = np.var(rq,ddof=1)/(q*np.var(r1,ddof=1))

    return vr
q_list = np.arange(1,100,2)
vr_list=[]

for q in q_list:
    vr=variance_ratio("AAPL",q)
    vr_list.append(vr)
    print(f"q={q:>3}, VR = {vr:.4f}")

plt.figure(figsize=(8,5))
plt.plot(q_list, vr_list, marker='o')
plt.axhline(1.0, linestyle='--')
plt.xlabel("q")
plt.ylabel("Variance Ratio")
plt.title("Variance Ratio vs q")
plt.grid(True)
plt.show()




tau_list = [i for i in range(1,100)]
"""
for i in tau_list:
        push_response_result = push_response("AAPL",tau=i)
        plt.figure(figsize=(7,5))
        plt.plot(push_response_result["push"],push_response_result["response"])
        plt.axhline(0,linewidth=1)
        plt.axvline(0,linewidth=1)
        plt.xlabel("push")
        plt.ylabel("average response")
        plt.title(f"tau = {i}")
        plt.tight_layout()
        plt.show()
"""




