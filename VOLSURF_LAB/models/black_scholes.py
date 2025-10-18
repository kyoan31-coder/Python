import numpy as np
from scipy.stats import norm

def d1(S, K, r, sigma, T, q=0.0):
    return (np.log(S/K) + (r - q + 0.5*sigma*sigma)*T) / (sigma*np.sqrt(T))

def d2(S, K, r, sigma, T, q=0.0):
    return d1(S,K,r,sigma,T,q) - sigma*np.sqrt(T)

def bs_price(S, K, r, sigma, T, call=True, q=0.0):
    if T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if call else max(0.0, K - S)
        return intrinsic
    D1, D2 = d1(S,K,r,sigma,T,q), d2(S,K,r,sigma,T,q)
    if call:
        return np.exp(-q*T)*S*norm.cdf(D1) - np.exp(-r*T)*K*norm.cdf(D2)
    else:
        return np.exp(-r*T)*K*norm.cdf(-D2) - np.exp(-q*T)*S*norm.cdf(-D1)
