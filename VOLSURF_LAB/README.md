# VOLSURF-LAB — Equity Derivatives Vol Surface & Backtests

**But** : Construire une surface de volatilité (IV) à partir d'options (via Yahoo Finance),
fitter des slices **SVI**, calculer grecs, visualiser la surface, et lancer des mini backtests delta-hedgés.

## Features
- Download options (ex: `SPY`) + expirations.
- Calcul IV / grecs (Black-Scholes).
- Fit **SVI** par maturité (Nonlinear LS).
- Build surface (strike, maturity) -> IV/total variance.
- Visualisation 3D / slices / smile.
- Backtests:
  - Straddle ATM delta-hedgé (mark-to-model).
  - Covered call & put-write (simple PnL model-based).
  - Approx réplication variance swap (Carr-Madan proxies).

## Installation (recommandé)
```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                                 # installe volsurflab en editable
```

> Alternative : définir `PYTHONPATH=src` (voir `.env` fourni).

## Lancement
### CLI
```bash
python -m src.volsurflab.cli fetch-options --ticker SPY
python -m src.volsurflab.cli fit-surface --ticker SPY --exp 2025-11-21
```

### Streamlit App
```bash
streamlit run src/volsurflab/app/app.py
```

## Notes
- YFinance fournit la surface du **jour** (pas d'historique de chaines). Les backtests sont **mark-to-model**.
- Ce repo est un **sandbox EQD** : architecture propre, tests basiques, et code lisible.
