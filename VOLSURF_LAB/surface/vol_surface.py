import numpy as np
import pandas as pd
from datetime import datetime
from utils.dates import year_fraction
from models.svi import fit_svi_slice, svi_total_variance

def build_surface(now: datetime, S: float, r: float, option_df: pd.DataFrame):
    rows = []
    for exp, df_exp in option_df.groupby('expiration'):
        T = year_fraction(now, pd.to_datetime(exp).to_pydatetime())
        if T <= 0: 
            continue
        df_exp = df_exp.dropna(subset=['impliedVolatility','strike'])
        if len(df_exp) < 5:
            continue
        F = S * np.exp(r*T)
        w = (df_exp['impliedVolatility'].values**2) * T
        strikes = df_exp['strike'].values
        try:
            params = fit_svi_slice(strikes, F, w)
        except Exception:
            continue
        rows.append({"expiration":exp, "T":T, "F":F, **params})
    return pd.DataFrame(rows)

def evaluate_surface(surface_params: pd.DataFrame, strikes: np.ndarray, maturities: np.ndarray, S: float, r: float):
    Z = np.zeros((len(maturities), len(strikes)))
    for i,T in enumerate(maturities):
        F = S*np.exp(r*T)
        k = np.log(strikes/F)
        row = surface_params.iloc[(surface_params['T']-T).abs().argmin()]
        w = svi_total_variance(k, row.a, row.b, row.rho, row.m, row.sigma)
        iv = np.sqrt(np.maximum(w/T, 1e-10))
        Z[i,:] = iv
    return Z
