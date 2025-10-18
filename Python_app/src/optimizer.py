
from typing import Dict
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def markowitz_max_sharpe(returns_df: pd.DataFrame, rf: float=0.0, short_allowed: bool=False) -> Dict[str, float]:
    r = returns_df.copy()
    mu = r.mean().values
    cov = np.cov(r.values.T)
    n = r.shape[1]
    if n==0: return {}
    def obj(w):
        w = np.array(w)
        port_mu = mu.dot(w)
        port_sigma = float(np.sqrt(w.T.dot(cov).dot(w)))
        if port_sigma == 0: return 1e6
        rf_p = (1+rf)**(1/252) - 1
        sharpe = (port_mu - rf_p) / port_sigma
        return -sharpe
    bounds = (None, None) if short_allowed else (0.0, 1.0)
    bnds = [bounds]*n
    cons = [{'type':'eq','fun': lambda w: float(np.sum(w)-1.0)}]
    x0 = np.ones(n)/n
    if not HAS_SCIPY:
        best_s, best_w = -1e9, x0
        rng = np.random.default_rng(0)
        for _ in range(10000):
            w = rng.random(n); w /= w.sum()
            s = -obj(w)
            if s>best_s: best_s, best_w = s, w
        return {k: float(v) for k,v in zip(r.columns, best_w)}
    res = minimize(obj, x0, bounds=bnds, constraints=cons, method='SLSQP', options={'maxiter':500})
    w_opt = x0 if not res.success else res.x
    return {k: float(v) for k,v in zip(r.columns, w_opt)}
