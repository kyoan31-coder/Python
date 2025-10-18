import pandas as pd
from data.market_data import get_underlying_history
from backtests.strategies import delta_hedged_straddle

def run_straddle_backtest(ticker="SPY", sigma=0.20, tenor_days=30, period="2y"):
    histo = get_underlying_history(ticker, period=period, interval="1d")
    pnl = delta_hedged_straddle(histo, histo['Close'].iloc[0], r=0.02, sigma=sigma, tenor_days=tenor_days)
    stats = {
        "mean_pnl": pnl['pnl'].mean(),
        "std_pnl": pnl['pnl'].std(),
        "hit_ratio": (pnl['pnl']>0).mean(),
        "n_trades": len(pnl),
        "sum_pnl": pnl['pnl'].sum()
    }
    return pnl, stats
