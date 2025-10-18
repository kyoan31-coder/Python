import streamlit as st
import yfinance as yf
import numpy as np
from datetime import datetime
from data.option_data import list_expirations, fetch_option_chain
from surface.vol_surface import build_surface, evaluate_surface
from utils.plotting import surface_3d

st.set_page_config(page_title="VOLSURF-LAB", layout="wide")
st.title("VOLSURF-LAB — Vol Surface & SVI")

ticker = st.sidebar.text_input("Ticker", value="SPY")
r = st.sidebar.number_input("Risk-free rate", value=0.02, step=0.005)
exp_choice = st.sidebar.selectbox("Expiration", options=list_expirations(ticker))

with st.spinner("Fetching data..."):
    price = yf.Ticker(ticker).history(period="5d")['Close'].iloc[-1]
    chain = fetch_option_chain(ticker, exp_choice)
    chain['expiration'] = exp_choice

now = datetime.utcnow()
surface = build_surface(now, price, r, chain)
st.subheader("SVI Slice Parameters")
st.dataframe(surface)

kmin, kmax = 0.7*price, 1.3*price
ks = np.linspace(kmin, kmax, 40)
Ts = np.linspace(0.03, 1.0, 25)
Z = evaluate_surface(surface, ks, Ts, price, r)

st.subheader("Vol Surface (IV)")
fig = surface_3d(Ts, ks, Z, f"{ticker} IV Surface")
st.plotly_chart(fig, use_container_width=True)
