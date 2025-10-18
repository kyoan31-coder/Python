import numpy as np
from scipy.optimize import least_squares

def svi_total_variance(k, a, b, rho, m, sigma):
    return a + b*( rho*(k-m) + np.sqrt((k-m)**2 + sigma**2) )

def fit_svi_slice(strikes, fwd, total_var):
    k = np.log(np.array(strikes)/fwd)
    w = np.array(total_var)
    guess = np.array([0.01, 0.1, 0.0, 0.0, 0.1])  # a,b,rho,m,sigma
    def resid(params):
        a,b,rho,m,s = params
        return svi_total_variance(k,a,b,rho,m,s) - w
    bounds = (
        np.array([1e-6, 1e-6, -0.999, -2.0, 1e-6]),
        np.array([ 1.0 ,  5.0 ,  0.999,  2.0,  5.0 ])
    )
    res = least_squares(resid, guess, bounds=bounds, max_nfev=20000)
    params = res.x
    return {"a":params[0], "b":params[1], "rho":params[2], "m":params[3], "sigma":params[4]}
