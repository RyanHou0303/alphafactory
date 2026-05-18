from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_path = PROJECT_ROOT / "data" / "processed" / "us_equity_daily_clean.parquet"
print(PROJECT_ROOT)
s1=pd.Series(99,index=['a','b','c','d'],name="MyNumber",dtype='int').rank(pct=True)
print(s1)

panel= pd.read_parquet(data_path)
print(panel.columns)
print(panel)

