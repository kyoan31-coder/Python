import numpy as np
from models.svi import fit_svi_slice, svi_total_variance

def test_svi_fit_runs():
    strikes = np.array([80,90,100,110,120], dtype=float)
    F = 100.0
    w = 0.04*np.ones_like(strikes)  # flat 20% vol, T=1
    params = fit_svi_slice(strikes, F, w)
    assert "a" in params and "b" in params
