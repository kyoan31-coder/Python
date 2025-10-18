import numpy as np
from scipy.stats import norm
from .black_scholes import d1, d2

def delta(S,K,r,sigma,T,call=True,q=0.0):
    D1 = d1(S,K,r,sigma,T,q)
    if call:
        return np.exp(-q*T)*norm.cdf(D1)
    return -np.exp(-q*T)*norm.cdf(-D1)

def vega(S,K,r,sigma,T,q=0.0):
    if T<=0 or sigma<=0: return 0.0
    return np.exp(-q*T)*S*np.sqrt(T)*norm.pdf(d1(S,K,r,sigma,T,q))
