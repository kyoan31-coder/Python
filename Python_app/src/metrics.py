
from typing import Tuple
import numpy as np
import pandas as pd

def annualized_return(series: pd.Series, periods_per_year: int = 252) -> float:
    s = series.dropna()
    if s.empty: return float("nan")
    total = s.iloc[-1] / s.iloc[0] - 1.0
    years = len(s) / periods_per_year
    if years <= 0: return float("nan")
    return (1 + total) ** (1 / years) - 1

def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty: return float("nan")
    return float(r.std(ddof=0) * (periods_per_year ** 0.5))

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty: return float("nan")
    rf_p = (1 + rf) ** (1/periods_per_year) - 1
    excess = r - rf_p
    vol = excess.std(ddof=0)
    if vol == 0 or pd.isna(vol): return float("nan")
    return float((excess.mean() / vol) * (periods_per_year ** 0.5))

def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty: return float("nan")
    rf_p = (1 + rf) ** (1/periods_per_year) - 1
    excess = r - rf_p
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or pd.isna(dd): return float("nan")
    return float((excess.mean() / dd) * (periods_per_year ** 0.5))

def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Series]:
    e = equity.dropna()
    if e.empty: return float("nan"), pd.Series(dtype=float, name="Drawdown")
    roll_max = e.cummax()
    dd = e / roll_max - 1.0
    return float(dd.min()), dd.rename("Drawdown")

def hist_var(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna()
    if r.empty: return float("nan")
    alpha = 1 - level
    return float(np.quantile(r, alpha))

def beta_alpha(returns: pd.Series, bench: pd.Series, rf: float = 0.0, periods_per_year: int = 252):
    r = returns.dropna()
    b = bench.dropna()
    idx = r.index.intersection(b.index)
    if len(idx) < 10: return float("nan"), float("nan")
    r = r.loc[idx]
    b = b.loc[idx]
    rf_p = (1 + rf) ** (1/periods_per_year) - 1
    r_ex = r - rf_p
    b_ex = b - rf_p
    cov = float(np.cov(r_ex, b_ex)[0,1])
    varb = float(np.var(b_ex))
    if varb == 0: return float("nan"), float("nan")
    beta = cov / varb
    alpha = (r_ex.mean() - beta*b_ex.mean()) * periods_per_year
    return float(beta), float(alpha)
