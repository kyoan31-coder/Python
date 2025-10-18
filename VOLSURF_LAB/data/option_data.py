import yfinance as yf
import pandas as pd

def list_expirations(ticker: str) -> list[str]:
    return yf.Ticker(ticker).options

def fetch_option_chain(ticker: str, expiration: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    opt = tk.option_chain(expiration)
    calls = opt.calls.assign(option_type="C")
    puts  = opt.puts.assign(option_type="P")
    df = pd.concat([calls, puts], ignore_index=True)
    keep = ["contractSymbol","lastTradeDate","strike","lastPrice","bid","ask",
            "change","percentChange","volume","openInterest","impliedVolatility","inTheMoney","option_type"]
    return df[keep].copy()
