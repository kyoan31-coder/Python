from backtests.engine import run_straddle_backtest

if __name__ == "__main__":
    pnl, stats = run_straddle_backtest(ticker="SPY", sigma=0.22, tenor_days=30, period="2y")
    print("Backtest stats:", stats)
    print(pnl.head())
