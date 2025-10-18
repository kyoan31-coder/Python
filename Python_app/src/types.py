
from dataclasses import dataclass
import pandas as pd

@dataclass
class BacktestResult:
    equity: pd.Series
    periodic_returns: pd.Series
    weights: pd.DataFrame
    positions: pd.DataFrame
    prices: pd.DataFrame
    trades_log: pd.DataFrame
