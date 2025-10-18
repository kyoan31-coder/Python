import argparse
from datetime import datetime
import pandas as pd
import yfinance as yf

from data.option_data import list_expirations, fetch_option_chain
from data.market_data import get_underlying_history
from surface.vol_surface import build_surface, evaluate_surface
from utils.plotting import surface_3d
from config import settings

def cmd_fetch_options(ticker: str):
    exps = list_expirations(ticker)
    print(f"Found {len(exps)} expirations for {ticker}. Example: {exps[:5]}")
    if exps:
        exp = exps[0]
        df = fetch_option_chain(ticker, exp)
        df['expiration'] = exp
        print(df.head().to_string(index=False))

def cmd_fit_surface(ticker: str, exp: str | None):
    now = datetime.utcnow()
    px = yf.Ticker(ticker).history(period="5d")['Close'].iloc[-1]
    if exp is None:
        exp = list_expirations(ticker)[0]
    df = fetch_option_chain(ticker, exp)
    df['expiration'] = exp
    surface = build_surface(now, px, r=0.02, option_df=df)
    print(surface)
    import numpy as np
    ks = np.linspace(0.7*px, 1.3*px, 25)
    Ts = np.linspace(0.05, 0.8, 15)
    Z = evaluate_surface(surface, ks, Ts, px, r=0.02)
    fig = surface_3d(Ts, ks, Z, title=f"{ticker} IV Surface (SVI fit slices)")
    fig.show()

def main():
    parser = argparse.ArgumentParser(prog="volsurflab")
    sub = parser.add_subparsers(dest="cmd")

    p1 = sub.add_parser("fetch-options")
    p1.add_argument("--ticker", default=settings.default_ticker)

    p2 = sub.add_parser("fit-surface")
    p2.add_argument("--ticker", default=settings.default_ticker)
    p2.add_argument("--exp", default=None)

    args = parser.parse_args()
    if args.cmd == "fetch-options":
        cmd_fetch_options(args.ticker)
    elif args.cmd == "fit-surface":
        cmd_fit_surface(args.ticker, args.exp)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
