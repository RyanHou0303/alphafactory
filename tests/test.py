from token import PERCENT

import pandas as pd
s1=pd.Series(99,index=['a','b','c','d'],name="MyNumber",dtype='int').rank(pct=True)
print(s1)

