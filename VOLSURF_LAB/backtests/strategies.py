import numpy as np
import pandas as pd
from datetime import timedelta
from models.black_scholes import bs_price
from models.greeks import delta as bs_delta

def delta_hedged_straddle(histo: pd.DataFrame, S0: float, r: float, sigma: float, tenor_days: int = 30):
    """
    Backtest simple : à chaque jour t, on achète 1 call + 1 put ATM (K=S_t),
    on delta-hedge quotidiennement (delta du call + delta du put), et on tient jusqu'à expiry (tenor_days).
    Pricing mark-to-model (BS) avec volatilité tenue fixe (=sigma) pour l'exemple.
    """
    histo = histo.copy()
    histo['S'] = histo['Close']
    histo = histo.dropna(subset=['S'])
    histo = histo.sort_index()

    pnl_series = []
    positions = []
    for start_date in histo.index[:-tenor_days]:
        end_date = start_date + timedelta(days=tenor_days)
        if end_date not in histo.index: 
            continue
        S_start = histo.loc[start_date,'S']
        K = S_start
        T0 = tenor_days/252.0
        c0 = bs_price(S_start,K,r,sigma,T0,call=True)
        p0 = bs_price(S_start,K,r,sigma,T0,call=False)
        delta0 = bs_delta(S_start,K,r,sigma,T0,True) + bs_delta(S_start,K,r,sigma,T0,False)
        cash = -(c0+p0)
        shares = -delta0
        idx = histo.loc[start_date:end_date].index
        prev = start_date
        for t in idx[1:]:
            dt = (t - prev).days/252.0
            T0 -= dt
            S_t = histo.loc[t,'S']
            c_t = bs_price(S_t,K,r,sigma,max(T0,1e-6),True)
            p_t = bs_price(S_t,K,r,sigma,max(T0,1e-6),False)
            dS = S_t - histo.loc[prev,'S']
            cash += shares * dS
            d_call = bs_delta(S_t,K,r,sigma,max(T0,1e-6),True)
            d_put  = bs_delta(S_t,K,r,sigma,max(T0,1e-6),False)
            new_shares = -(d_call + d_put)
            shares = new_shares
            prev = t
        S_T = histo.loc[end_date,'S']
        payoff = max(S_T-K,0)+max(K-S_T,0)
        final = cash + payoff
        pnl_series.append((end_date, final))
        positions.append((start_date, K))
    df = pd.DataFrame(pnl_series, columns=['date','pnl']).set_index('date')
    return df
