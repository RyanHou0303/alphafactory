from pathlib import Path
import yfinance as yf
import pandas as pd

tickers = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "AVGO", "AMD", "ADBE", "CRM",

    # Semis / software / old tech
    "ORCL", "QCOM", "TXN", "INTC", "IBM",
    "NOW", "INTU", "AMAT", "LRCX", "MU",

    # Financials
    "JPM", "BAC", "GS", "MS", "C",
    # Consumer / industrial
    "WMT", "COST", "HD", "MCD", "NKE",

    # Healthcare
    "LLY", "UNH", "JNJ", "PFE", "MRK",

]

start_date = "2015-01-01"
end_date="2026-01-01"

out_dir = Path("../data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir/"us_equity_daily_raw.parquet"

raw = yf.download(tickers=tickers,start=start_date,end=end_date,auto_adjust=True)
print("Raw shape: ",raw.shape)
print(raw.head())

raw.to_parquet(out_path)
print(f"Parquet saved to {out_path}")