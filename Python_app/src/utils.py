
from typing import List
import numpy as np
import pandas as pd

def sanitize_columns(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [sep.join(map(str, lvl)) for lvl in df.columns.values]
    else:
        df.columns = [str(c) for c in df.columns]
    return df

def pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

def to_weights_equal(n: int) -> np.ndarray:
    return np.ones(n)/n if n>0 else np.array([])
