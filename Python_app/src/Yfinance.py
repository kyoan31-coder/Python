
from datetime import date, timedelta
from typing import List, Optional
import pandas as pd
import streamlit as st
from .utils import sanitize_columns

# Optional deps
try:
    import yfinance as yf  # type: ignore
    HAS_YF = True
except Exception:
    HAS_YF = False

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "^GSPC"]

@st.cache_data(show_spinner=False)
def fetch_prices_yf(tickers: List[str], start: date, end: date) -> pd.DataFrame:
    if not HAS_YF: raise RuntimeError("yfinance non installé.")
    data = yf.download(tickers=tickers, start=start, end=end + timedelta(days=1),
                       progress=False, auto_adjust=True, threads=True)
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data.copy()
    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return sanitize_columns(prices)

def load_from_csv(upload) -> pd.DataFrame:
    df = pd.read_csv(upload)
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return sanitize_columns(df.dropna(how="all"))
