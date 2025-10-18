
# Portfolio Strategy Simulator — v2 (Modulaire)

WebApp Streamlit pour simuler des stratégies de portefeuille et afficher des KPIs avancés.

## Structure

```
portfolio_simulator_v2_modular/
├─ app.py                # Lanceur Streamlit
├─ src/
│  ├─ __init__.py
│  ├─ data.py           # Chargement des données (yfinance, CSV)
│  ├─ utils.py          # Aides (sanitise, pct_returns)
│  ├─ types.py          # Dataclasses (BacktestResult)
│  ├─ metrics.py        # KPIs (Sharpe, Sortino, Drawdown, VaR, beta/alpha)
│  ├─ strategies.py     # Stratégies (Buy&Hold, MA Crossover, Vol Target)
│  └─ optimizer.py      # Markowitz max Sharpe
└─ requirements.txt
```

## Installation & Lancement

1. Créez un environnement Python 3.10+ et installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

2. Lancez la webapp :
   ```bash
   streamlit run app.py
   ```

## Fonctionnalités
- **Données** : yfinance (ajusté) ou CSV avec colonne `Date`, colonnes = tickers.
- **Stratégies** : Buy&Hold, moving-average crossover, volatility targeting (equal-weight).
- **KPIs** : CAGR, volatilité annualisée, Sharpe, Sortino, max drawdown, VaR 95%, beta/alpha vs benchmark.
- **Optimisation** : Markowitz max Sharpe (fallback random si SciPy absent).
- **Export** : CSV Equity & Returns, sauvegarde / chargement de configurations JSON.

> Usage éducatif uniquement. Aucune garantie de performance.
