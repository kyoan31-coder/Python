import yfinance as yf
import pandas as pd

def get_underlying_history(ticker: str, period="2y", interval="1d") -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period, interval=interval)
