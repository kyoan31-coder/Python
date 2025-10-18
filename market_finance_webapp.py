# market_finance_webapp.py
# Streamlit WebApp: Portfolio Strategy Simulator (Finance de Marché)
# Author: <Your Name>
# Date: 2025-10-18
#
# Features:
# - User inputs (tickers, dates, capital, strategy params) via UI widgets
# - Fetch price data with yfinance or accept CSV upload
# - Implement simple strategies: Buy & Hold, Periodic Rebalance, Dollar-Cost Averaging (DCA)
# - Compute KPIs (CAGR, Volatility, Sharpe, Max Drawdown)
# - Interactive charts (equity curve, drawdown, weights) with Plotly
# - Export results to CSV + lightweight "run log" CSV
#
# To run locally:
#   1) pip install streamlit yfinance pandas numpy plotly
#   2) streamlit run market_finance_webapp.py
#
# CSV upload format (alternative to yfinance):
#   - Wide format with a 'Date' column (YYYY-MM-DD) and one column per ticker (closing prices).
#   - Example head:
#       Date, AAPL, MSFT, ^GSPC
#       2024-01-02, 185.64, 375.78, 4757.0
#
# Notes:
# - This app is designed for educational use only. Not investment advice.

import io
import os
import json  # <- IMPORTANT: required for run log writing
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Optional dependency (yfinance). The app still works with CSV upload. ---
try:
    import yfinance as yf  # type: ignore
    HAS_YF = True
except Exception:
    HAS_YF = False


# --------------------------- Utilities & Metrics ---------------------------

def sanitize_columns(df: pd.DataFrame, sep: str = "_") -> pd.DataFrame:
    """Ensure all column labels are flat strings (avoid MultiIndex tuples)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [sep.join(map(str, lvl)) for lvl in df.columns.values]
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def annualized_return(series: pd.Series, periods_per_year: int = 252) -> float:
    """Compound annual growth rate from daily equity curve."""
    series = series.dropna()
    if series.empty:
        return np.nan
    total_return = series.iloc[-1] / series.iloc[0] - 1.0
    # Try to infer years from index spacing, fallback to count/periods_per_year
    try:
        n_days = series.index.to_series().diff().dt.days.fillna(0).sum()
        years = max(n_days / 365.25, len(series) / periods_per_year)
    except Exception:
        years = len(series) / periods_per_year
    if years <= 0:
        return np.nan
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return returns.std(ddof=0) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Rf supplied as annual rate (e.g., 0.02)."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    rf_per_period = (1 + rf) ** (1 / periods_per_year) - 1
    excess = returns - rf_per_period
    vol = excess.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(periods_per_year)


def max_drawdown(equity: pd.Series) -> Tuple[float, pd.Series]:
    """Return (max_dd, drawdown_series)."""
    equity = equity.dropna()
    if equity.empty:
        return np.nan, pd.Series(dtype=float)
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min(), dd


def to_weights_equal(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    return np.ones(n) / n


# --------------------------- Data Loading ---------------------------

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMZN", "^GSPC"]

@st.cache_data(show_spinner=False)
def fetch_prices_yf(tickers: List[str], start: date, end: date) -> pd.DataFrame:
    """Fetch adjusted close prices using yfinance."""
    if not HAS_YF:
        raise RuntimeError("yfinance not installed. Use CSV upload.")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end + timedelta(days=1),
        progress=False,
        auto_adjust=True,
        threads=True,
    )
    if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data.copy()
    prices = prices.dropna(how="all")
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = sanitize_columns(prices)
    return prices


def load_from_csv(upload: io.BytesIO) -> pd.DataFrame:
    """Read a user CSV with Date + columns (tickers)."""
    df = pd.read_csv(upload)
    if "Date" not in df.columns:
        raise ValueError("CSV must contain a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    # Coerce numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(how="all")
    df = sanitize_columns(df)
    return df


# --------------------------- Strategies ---------------------------

@dataclass
class BacktestResult:
    equity: pd.Series
    periodic_returns: pd.Series
    weights: pd.DataFrame  # per date, per asset
    positions: pd.DataFrame # number of shares held per asset
    prices: pd.DataFrame
    trades_log: pd.DataFrame


def backtest_buy_and_hold(prices: pd.DataFrame, capital: float) -> BacktestResult:
    prices = prices.dropna(how="all").ffill().dropna(axis=1, how="all")
    prices = sanitize_columns(prices)
    tickers = list(prices.columns)
    n = len(tickers)
    if n == 0:
        raise ValueError("No valid price columns.")
    w = to_weights_equal(n)
    first = prices.iloc[0]
    # buy initial shares
    shares = (capital * w) / first.replace(0, np.nan)
    shares = shares.fillna(0.0)
    equity = (prices * shares).sum(axis=1)
    rets = equity.pct_change().fillna(0.0)

    weights = (prices.mul(shares, axis=1)).div(equity, axis=0).fillna(0.0)
    trades = pd.DataFrame({
        "Date": [prices.index[0]] * n,
        "Ticker": tickers,
        "Action": ["BUY"] * n,
        "Shares": shares.values,
        "Price": first.values,
        "CashFlow": -shares.values * first.values,
    })
    trades.set_index("Date", inplace=True)
    positions = pd.DataFrame([shares.values], index=[prices.index[0]], columns=tickers)
    return BacktestResult(equity, rets, weights, positions, prices, trades)


def backtest_rebalance(prices: pd.DataFrame, capital: float, freq: str = "M") -> BacktestResult:
    """Periodic equal-weight rebalancing at period end (M, Q, Y)."""
    prices = prices.dropna(how="all").ffill().dropna(axis=1, how="all")
    prices = sanitize_columns(prices)
    tickers = list(prices.columns)
    n = len(tickers)
    if n == 0:
        raise ValueError("No valid price columns.")
    w_target = to_weights_equal(n)

    # Resample to rebal dates
    rebal_dates = prices.resample(freq).last().index
    equity_list = []
    trades = []
    shares = pd.Series(0.0, index=tickers)
    cash = capital
    positions_df = []

    for i, dt in enumerate(prices.index):
        px = prices.loc[dt]
        port_val = (shares * px).sum() + cash
        equity_list.append((dt, port_val))

        if dt in rebal_dates or i == 0:
            target_val = port_val * w_target
            target_shares = target_val / px.replace(0, np.nan)
            target_shares = target_shares.fillna(0.0)
            delta = target_shares - shares
            cashflow = (delta * px).sum()

            if abs(cashflow) > 1e-9:
                action = np.where(delta.values > 0, "BUY", "SELL")
                trades.append(pd.DataFrame({
                    "Date": [dt] * n,
                    "Ticker": tickers,
                    "Action": action,
                    "Shares": delta.values,
                    "Price": px.values,
                    "CashFlow": -(delta.values * px.values),
                }))

            cash -= cashflow
            shares = target_shares

        positions_df.append(pd.DataFrame([shares.values], index=[dt], columns=tickers))

    equity = pd.Series([v for _, v in equity_list], index=[d for d, _ in equity_list], name="Equity")
    rets = equity.pct_change().fillna(0.0)
    # weights through time (using current shares)
    port_values = prices.mul(shares, axis=1)
    weights = port_values.div(port_values.sum(axis=1), axis=0).fillna(0.0)

    trades_log = pd.concat(trades) if len(trades) else pd.DataFrame(columns=["Date","Ticker","Action","Shares","Price","CashFlow"]).set_index("Date")
    positions = pd.concat(positions_df)
    return BacktestResult(equity, rets, weights, positions, prices, trades_log)


def backtest_dca(prices: pd.DataFrame, initial_capital: float, periodic_contribution: float, freq: str = "M") -> BacktestResult:
    """Dollar-Cost Averaging: invest a fixed amount at each period close, equal-weight split."""
    prices = prices.dropna(how="all").ffill().dropna(axis=1, how="all")
    prices = sanitize_columns(prices)
    tickers = list(prices.columns)
    n = len(tickers)
    if n == 0:
        raise ValueError("No valid price columns.")
    w = to_weights_equal(n)

    invest_dates = prices.resample(freq).last().index
    shares = pd.Series(0.0, index=tickers)
    cash = initial_capital
    equity_list = []
    trades = []
    positions_df = []

    for dt in prices.index:
        px = prices.loc[dt]

        if dt in invest_dates:
            budget = periodic_contribution + cash
            alloc = budget * w
            buy_shares = alloc / px.replace(0, np.nan)
            buy_shares = buy_shares.fillna(0.0)

            cf = (buy_shares * px).sum()
            trades.append(pd.DataFrame({
                "Date": [dt]*n,
                "Ticker": tickers,
                "Action": ["BUY"]*n,
                "Shares": buy_shares.values,
                "Price": px.values,
                "CashFlow": - (buy_shares.values * px.values),
            }))
            shares += buy_shares
            cash = budget - cf

        port_val = (shares * px).sum() + cash
        equity_list.append((dt, port_val))
        positions_df.append(pd.DataFrame([shares.values], index=[dt], columns=tickers))

    equity = pd.Series([v for _, v in equity_list], index=[d for d, _ in equity_list], name="Equity")
    rets = equity.pct_change().fillna(0.0)
    port_values = prices.mul(shares, axis=1)
    weights = port_values.div(port_values.sum(axis=1), axis=0).fillna(0.0)
    trades_log = pd.concat(trades) if len(trades) else pd.DataFrame(columns=["Date","Ticker","Action","Shares","Price","CashFlow"]).set_index("Date")
    positions = pd.concat(positions_df)
    return BacktestResult(equity, rets, weights, positions, prices, trades_log)


# --------------------------- Streamlit App ---------------------------

st.set_page_config(page_title="Portfolio Strategy Simulator", page_icon="💹", layout="wide")

st.title("💹 Portfolio Strategy Simulator — Finance de Marché")
st.caption("Une WebApp Streamlit pour tester des stratégies simples (Buy&Hold, Rebalance, DCA). **Éducatif uniquement**.")

with st.sidebar:
    st.header("⚙️ Paramètres")
    mode = st.radio("Source des données", ["Télécharger via yfinance", "Uploader un CSV"])

    # Tickerset
    tickers_default = DEFAULT_TICKERS.copy()
    custom = st.text_input("Ajouter des tickers séparés par des virgules (optionnel)", "")
    if custom.strip():
        extras = [t.strip().upper() for t in custom.split(",") if t.strip()]
        tickers_default = list(dict.fromkeys(tickers_default + extras))

    tickers = st.multiselect("Sélectionnez des tickers", options=tickers_default, default=["AAPL","MSFT","NVDA"])

    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Date de début", date.today() - timedelta(days=365*3))
    with col_dates[1]:
        end_date = st.date_input("Date de fin", date.today())

    st.divider()
    st.subheader("Stratégie")
    strategy = st.selectbox("Choix de la stratégie", ["Buy & Hold", "Rebalance périodique", "Dollar-Cost Averaging (DCA)"])

    initial_capital = st.number_input("Capital initial (€)", min_value=100.0, value=10000.0, step=100.0)

    rebal_freq = st.selectbox("Fréquence de rebalancement / DCA", ["M", "Q", "Y"], help="M = Mensuel, Q = Trimestriel, Y = Annuel")

    dca_amount = st.number_input("Contribution fixe par période (DCA)", min_value=0.0, value=500.0, step=50.0)

    rf_rate = st.number_input("Taux sans risque annuel (Sharpe)", min_value=0.0, value=0.02, step=0.005, format="%.3f")

    uploaded_file = None
    if mode == "Uploader un CSV":
        uploaded_file = st.file_uploader("Charger un CSV (Date + colonnes de prix)", type=["csv"])

    run = st.button("▶️ Lancer la simulation", use_container_width=True)

# --- Data retrieval ---
prices: Optional[pd.DataFrame] = None
if run:
    try:
        if mode == "Télécharger via yfinance":
            if not HAS_YF:
                st.error("yfinance n'est pas installé dans cet environnement. Installez-le (`pip install yfinance`) ou utilisez un CSV.")
            elif not tickers:
                st.warning("Veuillez sélectionner au moins un ticker.")
            else:
                prices = fetch_prices_yf(tickers, start_date, end_date)
        else:
            if uploaded_file is None:
                st.warning("Veuillez uploader un fichier CSV.")
            else:
                prices = load_from_csv(uploaded_file)
                # Filter columns to tickers if user selected any present
                available_cols = [c for c in tickers if c in prices.columns]
                if available_cols:
                    prices = prices[available_cols]
                prices = sanitize_columns(prices)

        if prices is not None and not prices.empty:
            prices = sanitize_columns(prices)
            st.success("Données chargées: " + ", ".join(map(str, list(prices.columns))))
            st.dataframe(prices.tail())

            # Run chosen strategy
            if strategy == "Buy & Hold":
                result = backtest_buy_and_hold(prices, capital=initial_capital)
            elif strategy == "Rebalance périodique":
                result = backtest_rebalance(prices, capital=initial_capital, freq=rebal_freq)
            else:
                result = backtest_dca(prices, initial_capital=initial_capital, periodic_contribution=dca_amount, freq=rebal_freq)

            equity = result.equity.rename("Equity")
            rets = result.periodic_returns.rename("Returns")

            # KPIs
            cagr = annualized_return(equity)
            vol = annualized_vol(rets)
            sharpe = sharpe_ratio(rets, rf=rf_rate)
            mdd, dd_series = max_drawdown(equity)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CAGR", f"{cagr*100:,.2f}%")
            c2.metric("Volatilité (ann.)", f"{vol*100:,.2f}%")
            c3.metric("Sharpe", f"{sharpe:,.2f}")
            c4.metric("Max Drawdown", f"{mdd*100:,.2f}%")

            # Charts
            st.subheader("Évolution du portefeuille")
            fig_eq = px.line(equity.reset_index(), x=equity.index.name or "index", y="Equity", labels={"x":"Date","Equity":"Valeur (€)"})
            fig_eq.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig_eq, use_container_width=True)

            st.subheader("Drawdown (retraits)")
            dd_df = dd_series.rename("Drawdown").reset_index()
            fig_dd = px.area(dd_df, x=dd_df.columns[0], y="Drawdown")
            fig_dd.update_layout(height=250, hovermode="x unified")
            st.plotly_chart(fig_dd, use_container_width=True)

            st.subheader("Poids par actif (fin de période)")
            # Approximate end weights using last positions row if available
            if not result.positions.empty:
                last_positions = result.positions.iloc[-1]
                last_prices = result.prices.iloc[-1]
                vals = last_positions * last_prices
                if vals.sum() != 0 and vals.notna().any():
                    w_end = vals / vals.sum()
                    fig_pie = px.pie(values=w_end.values, names=w_end.index, hole=0.35)
                    st.plotly_chart(fig_pie, use_container_width=True)

            with st.expander("📄 Détails: rendements périodiques & poids"):
                st.dataframe(pd.DataFrame({"Returns": rets}).tail(10))
                st.dataframe(result.weights.tail())

            # Export buttons
            out = pd.DataFrame({
                "Equity": equity,
                "Returns": rets,
            })
            csv_equity = out.to_csv(index=True).encode("utf-8")
            st.download_button("⬇️ Télécharger Equity & Returns (CSV)", csv_equity, file_name="equity_returns.csv", mime="text/csv")

            # Save a lightweight run log
            os.makedirs("results", exist_ok=True)
            run_log = {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "tickers": list(prices.columns),
                "start": str(prices.index.min().date()),
                "end": str(prices.index.max().date()),
                "strategy": strategy,
                "rebal_freq": rebal_freq,
                "dca_amount": dca_amount,
                "initial_capital": initial_capital,
                "metrics": {"CAGR": float(cagr) if pd.notna(cagr) else None,
                            "Vol": float(vol) if pd.notna(vol) else None,
                            "Sharpe": float(sharpe) if pd.notna(sharpe) else None,
                            "MaxDD": float(mdd) if pd.notna(mdd) else None},
            }
            with open(os.path.join("results", "runs_log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(run_log) + "\n")
            st.caption("📝 Exécution consignée dans results/runs_log.jsonl")

            # Show trades if available
            if not result.trades_log.empty:
                st.subheader("Journal des transactions (extrait)")
                st.dataframe(result.trades_log.tail(20))

                trade_csv = result.trades_log.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Télécharger le journal des transactions (CSV)", trade_csv, file_name="trades_log.csv", mime="text/csv")

        elif run:
            st.warning("Aucune donnée valide n'a été chargée. Vérifiez vos paramètres ou le CSV.")
    except Exception as e:
        st.error(f"Erreur: {e}")


# --------------------------- Footer ---------------------------
st.divider()
st.markdown("""
**Conseils pour votre vidéo (2–4 min)**  
- 🎯 *Objectif*: expliquer à quoi sert l'app (simuler des stratégies de portefeuille).  
- 🧩 *Structure du code*: un seul fichier avec fonctions (chargement, stratégies, métriques).  
- 🖱️ *Démo*: choisir des tickers, lancer, commenter les KPI et graphiques.  
- 🧠 *Compréhension*: expliquer les calculs (CAGR, Sharpe, drawdown) sans entrer dans chaque ligne de code.  
""")
