# Earnings Trading Dashboard — IV Crush Analysis

A desktop Python app (Tkinter + IB API) to analyze **implied volatility (IV) crush** around earnings for US equities. It fetches historical **stock prices, VIX**, and (if permissions allow) **option implied volatility** from Interactive Brokers, then prices an **ATM straddle** pre/post earnings via Black–Scholes, showing the effect of IV crush on option prices, P/L, and Greeks.

> **Disclaimer**: For education/research only. Not investment advice. Use paper trading.

---

## Features
- **IB connection UI** (host/port, connect/disconnect)
- **Earnings analysis inputs**: Ticker, earnings date (YYYY‑MM‑DD), and Days to Expiry
- **Live metrics**: Stock price, VIX level, current IV
- **IV crush panel**: Pre‑ and post‑earnings IV and **IV Crush %**
- **ATM straddle pricing** (Black–Scholes): call/put/straddle pre vs post
- **P/L summary**: Long vs short straddle P/L (dollar + %)
- **Greeks**: Delta & Vega pre/post and their changes
- **Charts**:
  - Stock price around earnings with IV overlay (dual axis)
  - VIX around earnings (or a bar chart of pre/post option prices if VIX missing)
- **Robust logging** area with timestamps

---

## Architecture
- **GUI**: `tkinter`, `ttk`, `matplotlib` canvas
- **Broker API**: `ibapi` (`EClient`/`EWrapper`) with background thread
- **Data**: IB historical bars for TRADES, VIX, and OPTION_IMPLIED_VOLATILITY
- **Pricing**: Black–Scholes call/put + Greeks (Delta, Vega) via `scipy.stats`

---

## Requirements
- **Python**: 3.10+
- **Packages**:
  ```bash
  pip install ibapi pandas numpy matplotlib scipy
  ```
  `tkinter` ships with most standard Python installers.

- **Interactive Brokers**: TWS or IB Gateway running locally, with **Market Data** permissions for the symbols analyzed and (ideally) **Options**/IV data entitlements.

---

## IB/TWS Setup
1. Launch **TWS** (Paper Trading recommended) or **IB Gateway**.
2. Enable API access: *Global Configuration → API → Settings*:
   - ☑ Enable ActiveX and Socket Clients
   - Trusted IPs: `127.0.0.1`
   - Read‑Only API is fine for this app
3. Note your port:
   - TWS Paper default: **7497** (matches app default)
   - IB Gateway Paper default: **4002**

> If you connect multiple apps, use **different client IDs** to avoid collisions.

---

## Run the App
```bash
python app.py
```

- In the **Connection** box, keep `Host=127.0.0.1`, set `Port` to your TWS/Gateway port, click **Connect**.
- Once connected, enter **Ticker** (e.g., `NVDA`), **Earnings Date** (e.g., `2025-08-27`), **Days to Expiry** (e.g., `30`), then click **Analyze IV Crush**.

---

## What the App Does
1. Pulls ~3 weeks of **daily stock bars** around the earnings date.
2. Pulls **VIX** (CBOE, `IND`, exchange `CBOE`) for context.
3. Attempts **OPTION_IMPLIED_VOLATILITY** history for the equity. If unavailable, it **estimates IV** from VIX:  
   `pre_iv ≈ 1.5×VIX, post_iv ≈ 1.2×VIX` (rough heuristic).
4. Picks **pre‑earnings** = last trading day ≤ earnings date; **post‑earnings** = first trading day > earnings date.
5. Prices an **ATM straddle** at the **pre‑earnings close** strike, then re‑prices post‑earnings at the **same strike** using the post‑earnings spot and IV.
6. Computes **IV Crush %**, option/straddle deltas, vegas, and **P/L** for long/short straddle.
7. Renders charts and a detailed log.

---

## Notes on Volatility Scaling
- IB’s `OPTION_IMPLIED_VOLATILITY` series may arrive as daily values. In this codebase, IV values are treated as **annualized decimals** in downstream pricing.
- There is an intentionally conservative switch where the √252 annualization is currently **disabled** in the IV pipeline (i.e., factor set to `1`). If your IV series is truly *daily* and not annualized, you can re‑enable annualization (`× sqrt(252)`) in the `analyze_iv_crush()` block where `implied_vol` is assigned.

---

## Model Assumptions
- **Black–Scholes** with constant IV over the (short) horizon
- **Risk‑free rate** fixed at **5%** (editable: `self.risk_free_rate`)
- **ATM strike** fixed at **pre‑earnings close**
- **Time to expiry** = user input (days/365)
- **RTH** only (useRTH=1) for historical bars

---

## Interpreting Results
- **IV Crush %**: `(pre_iv - post_iv) / pre_iv`
- **P/L Long Straddle**: positive if total post price > pre price (gap move and/or IV up), negative when IV collapses and move is insufficient
- **P/L Short Straddle**: the opposite sign
- **Delta**: magnitude near 0 for balanced straddles; change indicates skew introduced by the move
- **Vega**: drops after earnings; negative ΔVega harms long vol, benefits short vol

---

## Troubleshooting
- **“Not connected to Interactive Brokers”**: Ensure TWS/Gateway is running; correct host/port; API enabled.
- **No data returned**:
  - Check market data **entitlements** for the symbol and for **Options/IV**.
  - Try changing the `whatToShow` or increase `durationStr` if the window is too narrow.
- **IV data not available**: The app will log *“Implied volatility data not available – will estimate from VIX”* and proceed with a heuristic.
- **Error 2176 (fractional shares)**: The app filters that warning; safe to ignore for this workflow.
- **Server version issues**: Wait a few seconds after connect; confirm in the log you see a valid server version.
- **Permission errors**: If you see `No market data permissions`, contact IB or test with a different, permissioned symbol.

---

## Configuration Tips
- Change the default **risk‑free rate** in `EarningsTradingDashboard.__init__`.
- Toggle IV scaling in `analyze_iv_crush()` where `implied_vol` is computed.
- Modify the **analysis window** (currently ±10 days fetched, ±5 days plotted) in `create_visualizations()`.

---

## Roadmap Ideas
- Pull **actual options chain** (strikes/expiries) and price the **true ATM** per date
- Add **skew/smile** views and **term structure** panels
- Compute **realized move vs implied move** and backtest pre‑earnings straddles
- Export **CSV** of metrics & plots
- Add **thetas/gammas** and a **PnL surface** vs move & IV change

