
# Streamlit launcher for the modular Portfolio Strategy Simulator (v2)
import os, json
from datetime import date, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.Yfinance import DEFAULT_TICKERS, fetch_prices_yf, load_from_csv
from src.utils import sanitize_columns, pct_returns
from src.metrics import annualized_return, annualized_vol, sharpe_ratio, sortino_ratio, max_drawdown, hist_var, beta_alpha
from src.strategies import BuyAndHoldStrategy, MACrossoverStrategy, VolTargetStrategy
from src.optimizer import markowitz_max_sharpe

st.set_page_config(page_title="Portfolio Strategy Simulator v2 (Modular)", page_icon="💹", layout="wide")
st.title("💹 Portfolio Strategy Simulator — v2 (Modulaire)")
st.caption("Stratégies élargies + KPIs avancés + Optimisation Markowitz. **Éducatif uniquement.**")

with st.sidebar:
    st.header("⚙️ Paramètres")
    mode = st.radio("Source des données", ["Télécharger via yfinance", "Uploader un CSV"])
    custom = st.text_input("Ajouter des tickers séparés par des virgules (optionnel)", "")
    tickers_default = DEFAULT_TICKERS.copy()
    if custom.strip():
        extras = [t.strip().upper() for t in custom.split(",") if t.strip()]
        tickers_default = list(dict.fromkeys(tickers_default + extras))
    tickers = st.multiselect("Sélectionnez des tickers", options=tickers_default, default=["AAPL","MSFT","NVDA"])
    col_dates = st.columns(2)
    with col_dates[0]:
        start_date = st.date_input("Date de début", date.today() - timedelta(days=365*5))
    with col_dates[1]:
        end_date = st.date_input("Date de fin", date.today())
    rf_rate = st.number_input("Taux sans risque annuel", min_value=0.0, value=0.02, step=0.005, format="%.3f")
    benchmark = st.selectbox("Benchmark (beta/alpha)",
                             options=["^GSPC","AAPL","MSFT"] + [t for t in tickers if t not in ["^GSPC","AAPL","MSFT"]],
                             index=0)
    st.divider()
    st.subheader("Stratégie")
    strategy_name = st.selectbox("Choix de la stratégie", ["Buy & Hold", "MA Crossover", "Volatility Target (EQ)"])
    initial_capital = st.number_input("Capital initial (€)", min_value=100.0, value=10000.0, step=100.0)
    fast = st.number_input("MA rapide (MA Crossover)", min_value=5, value=50, step=5)
    slow = st.number_input("MA lente (MA Crossover)", min_value=20, value=200, step=10)
    tgt_vol = st.number_input("Vol cible (Vol Target)", min_value=0.02, value=0.10, step=0.01, format="%.2f")
    lookback = st.number_input("Lookback Vol (jours)", min_value=5, value=20, step=1)
    st.divider()
    config_name = st.text_input("Nom de config (save/load)", "demo_config")
    btn_save = st.button("💾 Sauver config")
    btn_load = st.button("📂 Charger config")
    run = st.button("▶️ Lancer", use_container_width=True)

prices = None

if run:
    try:
        if mode == "Télécharger via yfinance":
            if not tickers:
                st.warning("Sélectionnez au moins un ticker.")
            else:
                prices = fetch_prices_yf(tickers, start_date, end_date)
        else:
            uploaded = st.file_uploader("Charger un CSV (Date + colonnes)", type=["csv"])
            if uploaded is None:
                st.warning("Veuillez uploader un CSV.")
            else:
                prices = load_from_csv(uploaded)
                cols = [c for c in tickers if c in prices.columns]
                if cols: prices = prices[cols]
                prices = sanitize_columns(prices)

        if prices is None or prices.empty:
            st.stop()

        # Save/Load config
        if btn_save:
            cfg = {
                "tickers": tickers,
                "start": str(start_date),
                "end": str(end_date),
                "rf_rate": rf_rate,
                "benchmark": benchmark,
                "strategy": strategy_name,
                "initial_capital": initial_capital,
                "fast": int(fast),
                "slow": int(slow),
                "tgt_vol": tgt_vol,
                "lookback": int(lookback),
            }
            with open(f"config_{config_name}.json","w",encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            st.success(f"Config '{config_name}' sauvegardée.")
        if btn_load:
            try:
                with open(f"config_{config_name}.json","r",encoding="utf-8") as f:
                    cfg = json.load(f)
                st.success(f"Config '{config_name}' chargée."); st.json(cfg)
            except Exception as e:
                st.error(f"Impossible de charger la config: {e}")

        # Tabs
        tabs = st.tabs(["📈 Résultats", "🧐 Analyse", "🧮 Optimisation"])

        # Strategy selection
        if strategy_name == "Buy & Hold":
            strategy = BuyAndHoldStrategy()
        elif strategy_name == "MA Crossover":
            strategy = MACrossoverStrategy(fast=int(fast), slow=int(slow))
        else:
            strategy = VolTargetStrategy(target_vol_ann=tgt_vol, lookback=int(lookback))

        res = strategy.run(prices, capital=initial_capital)

        equity = res.equity.rename("Equity")
        rets = res.periodic_returns.rename("Returns")

        # KPIs
        cagr = annualized_return(equity)
        vol = annualized_vol(rets)
        sharpe = sharpe_ratio(rets, rf=rf_rate)
        sortino = sortino_ratio(rets, rf=rf_rate)
        mdd, dd_series = max_drawdown(equity)
        var95 = hist_var(rets, 0.95)

        # Benchmark
        bench_series = None
        if benchmark in res.prices.columns:
            bench_series = res.prices[benchmark].pct_change().fillna(0.0)
        else:
            try:
                bench_px = fetch_prices_yf([benchmark], res.prices.index.min().date(), res.prices.index.max().date())
                bench_series = bench_px.iloc[:,0].pct_change().fillna(0.0)
            except Exception:
                bench_series = None
        beta, alpha = (np.nan, np.nan)
        if bench_series is not None:
            beta, alpha = beta_alpha(rets, bench_series, rf=rf_rate)

        with tabs[0]:
            st.success("Données chargées: " + ", ".join(map(str, list(res.prices.columns))))
            st.dataframe(res.prices.tail())
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("CAGR", f"{cagr*100:,.2f}%")
            c2.metric("Vol ann.", f"{vol*100:,.2f}%")
            c3.metric("Sharpe", f"{sharpe:,.2f}")
            c4.metric("Sortino", f"{sortino:,.2f}")
            c5.metric("Max DD", f"{mdd*100:,.2f}%")
            c6.metric("VaR 95% (quotid.)", f"{var95*100:,.2f}%")
            if not np.isnan(beta):
                st.caption(f"β vs {benchmark}: **{beta:.2f}**, α (ann.): **{alpha*100:.2f}%**")
            fig_eq = px.line(equity.reset_index(), x=equity.index.name or "index", y="Equity",
                             labels={"x":"Date","Equity":"Valeur (€)"})
            fig_eq.update_layout(height=380, hovermode="x unified")
            st.plotly_chart(fig_eq, use_container_width=True)
            dd_df = dd_series.rename("Drawdown").reset_index()
            fig_dd = px.area(dd_df, x=dd_df.columns[0], y="Drawdown")
            fig_dd.update_layout(height=220, hovermode="x unified")
            st.plotly_chart(fig_dd, use_container_width=True)
            out = pd.DataFrame({"Equity": equity, "Returns": rets})
            st.download_button("⬇️ Equity & Returns (CSV)", out.to_csv().encode("utf-8"),
                               file_name="equity_returns_v2.csv", mime="text/csv")

        with tabs[1]:
            st.subheader("Corrélations (rendements)")
            rets_assets = pct_returns(res.prices)
            corr = rets_assets.corr()
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto")
            st.plotly_chart(fig_corr, use_container_width=True)
            st.subheader("Distribution des rendements du portefeuille")
            st.plotly_chart(px.histogram(rets, nbins=60), use_container_width=True)

        with tabs[2]:
            st.subheader("Optimisation de portefeuille — Max Sharpe (Markowitz)")
            if res.prices.shape[1] >= 2:
                returns_df = pct_returns(res.prices).iloc[1:]
                weights_opt = markowitz_max_sharpe(returns_df, rf=rf_rate, short_allowed=False)
                if weights_opt:
                    w_ser = pd.Series(weights_opt)
                    st.write("Poids optimisés:")
                    st.dataframe(w_ser.to_frame("Weight").T if len(w_ser)>8 else w_ser)
                    st.plotly_chart(px.bar(w_ser, title="Poids optimisés (Max Sharpe)"), use_container_width=True)
                else:
                    st.info("Optimisation indisponible.")
            else:
                st.info("Sélectionnez ≥2 tickers.")

        # Run log
        os.makedirs("results", exist_ok=True)
        run_log = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "tickers": list(res.prices.columns),
            "start": str(res.prices.index.min().date()),
            "end": str(res.prices.index.max().date()),
            "strategy": strategy_name,
            "initial_capital": initial_capital,
            "metrics": {"CAGR": float(cagr) if pd.notna(cagr) else None,
                        "Vol": float(vol) if pd.notna(vol) else None,
                        "Sharpe": float(sharpe) if pd.notna(sharpe) else None,
                        "Sortino": float(sortino) if pd.notna(sortino) else None,
                        "MaxDD": float(mdd) if pd.notna(mdd) else None,
                        "VaR95": float(var95) if pd.notna(var95) else None,
                        "Beta": float(beta) if not np.isnan(beta) else None,
                        "Alpha_ann": float(alpha) if not np.isnan(alpha) else None},
        }
        with open(os.path.join("results", "runs_log_v2.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(run_log) + "\n")

    except Exception as e:
        st.error(f"Erreur: {e}")

st.divider()
st.markdown("""
**Guide rapide (v2 modulaire)**  
- Fichiers séparés : `src/` (données, stratégies, métriques, optimisateur) + `app.py` (lanceur).  
- Onglets: Résultats / Analyse / Optimisation.  
- KPIs: Sortino, VaR(95%), β/α vs benchmark.  
- Stratégies: MA crossover, Volatility targeting + Buy&Hold.  
- Optimisation Markowitz (max Sharpe).  
""")
