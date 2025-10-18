
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from .types import BacktestResult
from .utils import sanitize_columns, to_weights_equal

class Strategy:
    name: str = "Base"
    def run(self, prices: pd.DataFrame, capital: float) -> BacktestResult:
        raise NotImplementedError

class BuyAndHoldStrategy(Strategy):
    name = "Buy & Hold"
    def run(self, prices: pd.DataFrame, capital: float) -> BacktestResult:
        prices = sanitize_columns(prices.dropna(how="all").ffill().dropna(axis=1, how="all"))
        tickers = list(prices.columns); n=len(tickers)
        if n==0: raise ValueError("No valid price columns.")
        w = to_weights_equal(n)
        first = prices.iloc[0]
        shares = (capital*w)/first.replace(0,np.nan)
        shares = shares.fillna(0.0)
        equity = (prices*shares).sum(axis=1)
        rets = equity.pct_change().fillna(0.0)
        weights = (prices.mul(shares,axis=1)).div(equity,axis=0).fillna(0.0)
        trades = pd.DataFrame({
            "Date":[prices.index[0]]*n,"Ticker":tickers,"Action":["BUY"]*n,
            "Shares":shares.values,"Price":first.values,"CashFlow":-(shares.values*first.values)
        }).set_index("Date")
        positions = pd.DataFrame([shares.values], index=[prices.index[0]], columns=tickers)
        return BacktestResult(equity, rets, weights, positions, prices, trades)

class MACrossoverStrategy(Strategy):
    name = "MA Crossover"
    def __init__(self, fast: int=50, slow: int=200):
        self.fast = int(fast); self.slow = int(slow)
    def run(self, prices: pd.DataFrame, capital: float) -> BacktestResult:
        prices = sanitize_columns(prices.dropna(how="all").ffill().dropna(axis=1, how="all"))
        tickers = list(prices.columns); n=len(tickers)
        if n==0: raise ValueError("No valid price columns.")
        sma_fast = prices.rolling(self.fast).mean()
        sma_slow = prices.rolling(self.slow).mean()
        signal = (sma_fast > sma_slow).astype(float)
        weights_daily = signal.div(signal.sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0)
        rets_assets = prices.pct_change().fillna(0.0)
        port_rets = (weights_daily.shift(1).fillna(0.0) * rets_assets).sum(axis=1)
        equity = (1+port_rets).cumprod() * capital
        last_w = weights_daily.iloc[-1] if not weights_daily.empty else pd.Series(0,index=tickers)
        last_px = prices.iloc[-1]
        positions = pd.DataFrame([ (last_w*equity.iloc[-1]/last_px).fillna(0.0).values ], index=[prices.index[-1]], columns=tickers)
        trades = pd.DataFrame(columns=["Date","Ticker","Action","Shares","Price","CashFlow"]).set_index("Date")
        return BacktestResult(equity, port_rets, weights_daily, positions, prices, trades)

class VolTargetStrategy(Strategy):
    name = "Volatility Target (EQ)"
    def __init__(self, target_vol_ann: float=0.10, lookback: int=20, cap_leverage: float=3.0):
        self.target_vol_ann = float(target_vol_ann)
        self.lookback = int(lookback)
        self.cap_leverage = float(cap_leverage)
    def run(self, prices: pd.DataFrame, capital: float) -> BacktestResult:
        prices = sanitize_columns(prices.dropna(how="all").ffill().dropna(axis=1, how="all"))
        n = prices.shape[1]
        if n==0: raise ValueError("No valid price columns.")
        w_eq = to_weights_equal(n)
        rets_assets = prices.pct_change().fillna(0.0)
        port_ret_eq = rets_assets.dot(w_eq)
        daily_realized = port_ret_eq.rolling(self.lookback).std(ddof=0)
        vol_ann = daily_realized*(252 ** 0.5)
        lever = (self.target_vol_ann / vol_ann).clip(upper=self.cap_leverage).fillna(0.0)
        port_rets = port_ret_eq * lever.shift(1).fillna(0.0)
        equity = (1+port_rets).cumprod()*capital
        weights_daily = pd.DataFrame([w_eq]*len(prices), index=prices.index, columns=prices.columns).mul(lever, axis=0)
        last_px = prices.iloc[-1]
        last_w = weights_daily.iloc[-1].fillna(0.0)
        positions = pd.DataFrame([ (last_w*equity.iloc[-1]/last_px).fillna(0.0).values ], index=[prices.index[-1]], columns=prices.columns)
        trades = pd.DataFrame(columns=["Date","Ticker","Action","Shares","Price","CashFlow"]).set_index("Date")
        return BacktestResult(equity, port_rets, weights_daily, positions, prices, trades)
