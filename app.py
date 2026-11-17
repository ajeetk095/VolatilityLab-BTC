"""VolatilityLab-BTC: live market data, risk diagnostics, scenario projections and risk-based guidance."""
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from arch import arch_model
from scipy.stats import kurtosis, skew
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller, kpss

st.set_page_config(page_title="VolatilityLab-BTC", layout="wide")
st.title("VolatilityLab-BTC")
st.caption("Live prices, volatility diagnostics, scenario projections and risk-based guidance.")
st.warning("Educational tool only. This is not financial advice. Projections are model scenarios, not predictions, "
           "and crypto assets can lose most or all of their value. Speak to a licensed adviser before investing.")

# ---------------------------------------------------------------- sidebar
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", "BTC-USD").upper().strip()
years = st.sidebar.slider("History (years)", 2, 8, 5)
profile = st.sidebar.selectbox("Investor profile", ["Conservative", "Balanced", "Aggressive"], index=1)
amount = st.sidebar.number_input("Portfolio size (USD)", min_value=1000, max_value=10_000_000, value=10_000, step=1000)
horizon = st.sidebar.slider("Projection horizon (days)", 7, 180, 30)
live_on = st.sidebar.checkbox("Auto-refresh live price", value=True)
refresh = st.sidebar.select_slider("Refresh every (seconds)", options=[15, 30, 60, 120], value=30)

ANN = 365 if ticker.endswith("-USD") else 252
TARGET_VOL, MAX_W = {"Conservative": (0.10, 0.10), "Balanced": (0.20, 0.25), "Aggressive": (0.35, 0.50)}[profile]


# ---------------------------------------------------------------- data
def _close(df):
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    c = c.dropna()
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c


@st.cache_data(ttl=300, show_spinner="Fetching market data...")
def load_daily(t, yrs):
    return _close(yf.download(t, period=f"{yrs}y", interval="1d", progress=False, auto_adjust=True))


@st.cache_data(ttl=20)
def load_intraday(t):
    return _close(yf.download(t, period="1d", interval="5m", progress=False, auto_adjust=True))


try:
    close = load_daily(ticker, years)
except Exception as err:
    st.error(f"Could not download data for {ticker}: {err}")
    st.stop()
if len(close) < 250:
    st.warning("Not enough history. Check the ticker or increase the history length.")
    st.stop()

r = close.pct_change().dropna()
np.random.seed(42)
sim = pd.Series(np.random.normal(r.mean(), r.std(), len(r)), index=r.index)
sim_price = pd.Series(np.exp(np.cumsum(sim)), index=sim.index)


@st.cache_data(ttl=300, show_spinner="Fitting volatility model and simulating scenarios...")
def fit_and_simulate(ret_pct, h, ann, sims=2000):
    np.random.seed(1)
    res = arch_model(ret_pct, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    f = res.forecast(horizon=h, method="simulation", simulations=sims, reindex=False)
    var = f.variance.values[-1]
    return {"paths": f.simulations.values[-1], "sigma1": float(np.sqrt(var[0])),
            "ann_vol": float(np.sqrt(var.mean() * ann) / 100),
            "z": (ret_pct / res.conditional_volatility).dropna().values,
            "nu": float(res.params.get("nu", np.nan))}


tab_live, tab_diag, tab_proj, tab_guide, tab_future = st.tabs(
    ["Live market", "Diagnostics", "Risk and projections", "Guidance", "Future scope"])

# ---------------------------------------------------------------- live
with tab_live:
    def live_panel():
        try:
            intr = load_intraday(ticker)
            last, first = float(intr.iloc[-1]), float(intr.iloc[0])
        except Exception:
            intr, last, first = None, float(close.iloc[-1]), float(close.iloc[-2])
            st.info("Intraday feed unavailable; showing the latest daily close.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last price", f"{last:,.2f}", f"{(last / first - 1):+.2%}")
        c2.metric("Day high", f"{float(intr.max()):,.2f}" if intr is not None else "n/a")
        c3.metric("Day low", f"{float(intr.min()):,.2f}" if intr is not None else "n/a")
        c4.metric("Last updated (UTC)", datetime.now(timezone.utc).strftime("%H:%M:%S"))
        if intr is not None:
            st.line_chart(intr, height=260)
        st.caption("Data from Yahoo Finance via yfinance; may be delayed. " + ("Auto-refreshing." if live_on else "Auto-refresh is off."))

    if hasattr(st, "fragment"):
        live_panel = st.fragment(run_every=refresh if live_on else None)(live_panel)
    live_panel()
    st.subheader("Daily history")
    st.line_chart(close, height=240)

# ---------------------------------------------------------------- diagnostics
with tab_diag:
    def kp(x):
        return kpss(x, regression="c", nlags="auto")[1]
    stats = pd.DataFrame({
        "Real": [skew(r), kurtosis(r), het_arch(r)[1], adfuller(close.loc[r.index])[1], kp(close.loc[r.index]), adfuller(r)[1], kp(r)],
        "Simulated normal": [skew(sim), kurtosis(sim), het_arch(sim)[1], adfuller(sim_price)[1], kp(sim_price), adfuller(sim)[1], kp(sim)]},
        index=["Skewness", "Excess kurtosis", "ARCH-LM p-value", "Price ADF p", "Price KPSS p", "Returns ADF p", "Returns KPSS p"])
    st.dataframe(stats.round(4))
    st.caption("ADF null: unit root. KPSS null: stationary (p-values are table-bounded, 0.01 to 0.10). Excess kurtosis: normal = 0.")
    window = st.slider("Outlier window (days)", 10, 60, 20)
    med = r.rolling(window).median()
    mad = (r - med).abs().rolling(window).median() * 1.4826
    flag = ((r - med).abs() > 3 * mad).fillna(False)
    a, b = st.columns(2)
    with a:
        fig, ax = plt.subplots()
        ax.hist(r, bins=50, alpha=0.6, label="Real"); ax.hist(sim, bins=50, alpha=0.6, label="Simulated")
        ax.set(title="Return distributions", xlabel="Daily return", ylabel="Frequency"); ax.legend()
        st.pyplot(fig)
    with b:
        fig, ax = plt.subplots()
        ax.scatter(r.index, r, c=np.where(flag, "red", "blue"), s=6, alpha=0.5)
        ax.set(title=f"Outliers (3 x MAD): {int(flag.sum())} flagged", xlabel="Date", ylabel="Daily return")
        st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(r.rolling(window).std() * np.sqrt(ANN), label="Standard-deviation volatility")
    ax.plot(mad * np.sqrt(ANN), label="MAD volatility")
    ax.set(title="Annualised volatility", xlabel="Date", ylabel="Volatility"); ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------------- model
try:
    M = fit_and_simulate(r * 100, horizon, ANN)
except Exception as err:
    M = None
    model_error = str(err)

with tab_proj:
    if M is None:
        st.error(f"Model could not be fitted: {model_error}")
    else:
        last = float(close.iloc[-1])
        cum = np.cumprod(1 + M["paths"] / 100, axis=1)
        q = np.percentile(cum, [5, 25, 50, 75, 95], axis=0) * last
        z = M["z"]; q05, q01 = np.percentile(z, [5, 1])
        var95, var99 = -q05 * M["sigma1"] / 100, -q01 * M["sigma1"] / 100
        es99 = -z[z <= q01].mean() * M["sigma1"] / 100
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecast volatility (annualised)", f"{M['ann_vol']:.0%}")
        c2.metric("1-day 95% VaR", f"{var95:.1%}")
        c3.metric("1-day 99% VaR", f"{var99:.1%}")
        c4.metric("1-day 99% expected shortfall", f"{es99:.1%}")
        st.caption(f"Skewed-volatility model (GJR-GARCH(1,1), Student-t errors, degrees of freedom {M['nu']:.1f}); "
                   "VaR uses the model's own standardised residuals, so fat tails are included.")
        days = np.arange(1, horizon + 1)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(days, q[0], q[4], alpha=0.2, label="5th to 95th percentile")
        ax.fill_between(days, q[1], q[3], alpha=0.35, label="25th to 75th percentile")
        ax.plot(days, q[2], label="Median scenario"); ax.axhline(last, ls="--", color="grey", label="Current price")
        ax.set(title=f"Scenario fan chart, next {horizon} days (no drift assumed)", xlabel="Days ahead", ylabel="Price"); ax.legend()
        st.pyplot(fig)
        end = cum[:, -1]
        st.dataframe(pd.DataFrame({
            "Scenario": ["Bad (5th pct)", "Below median (25th)", "Median", "Above median (75th)", "Good (95th)"],
            "Price": q[:, -1].round(2), "Change": (q[:, -1] / last - 1).round(3),
            f"Value of ${amount:,.0f} position": (amount * q[:, -1] / last).round(0)}), hide_index=True)
        st.write(f"Model probability of finishing below today's price: **{(end < 1).mean():.0%}**. "
                 "These ranges reflect volatility only; they assume no trend and cannot foresee news or regime shifts.")

# ---------------------------------------------------------------- guidance
with tab_guide:
    if M is None:
        st.error("Guidance needs the volatility model.")
    else:
        ma50, ma200 = close.rolling(50).mean().iloc[-1], close.rolling(200).mean().iloc[-1]
        px = float(close.iloc[-1])
        trend = 1 if px > ma50 > ma200 else (-1 if px < ma200 else 0)
        rv = (r.rolling(30).std() * np.sqrt(ANN)).dropna()
        pct = float((rv < rv.iloc[-1]).mean())
        regime = 1 if pct < 0.33 else (-1 if pct < 0.9 and pct > 0.66 else (-2 if pct >= 0.9 else 0))
        dd = px / float(close.iloc[-365:].max()) - 1
        score = trend + regime
        stance, mult = ("Constructive", 1.0) if score >= 2 else (("Neutral", 0.75) if score >= 0 else ("Defensive", 0.4))
        w = min(MAX_W, TARGET_VOL / M["ann_vol"]) * mult
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk stance", stance)
        c2.metric("Volatility regime", ["Extreme", "High", "Normal", "Low"][{-2: 0, -1: 1, 0: 2, 1: 3}[regime]], f"{pct:.0%} percentile")
        c3.metric("Trend", {1: "Up", 0: "Mixed", -1: "Down"}[trend], f"{dd:.0%} from 1-year high")
        st.subheader(f"Suggested maximum exposure ({profile})")
        st.write(f"A volatility-targeting rule (aim for about {TARGET_VOL:.0%} portfolio volatility, cap {MAX_W:.0%}) gives a maximum of "
                 f"**{w:.1%} of the portfolio, about ${w * amount:,.0f}**, in {ticker}.")
        st.markdown(f"""
**Why:** forecast volatility is {M['ann_vol']:.0%} a year, and the 1-day 99% loss estimate is {var99:.1%}. Higher volatility means a smaller position for the same risk.

**Practical ideas to discuss with an adviser**
- Build the position in 4 equal tranches over several weeks instead of all at once.
- Treat the scenario 5th-percentile price over your horizon as a stress level; decide in advance whether you could hold through it.
- Rebalance when the position drifts above the cap, and re-check this page when the volatility regime changes.
- Keep the position small relative to diversified holdings, and consider hedging with options if you trade them.
- Never invest money you cannot afford to lose.
""")
        st.caption("Stance rule: trend (price vs 50 and 200-day averages) plus volatility regime. It is a transparent risk rule, not a buy or sell signal.")

# ---------------------------------------------------------------- future scope
with tab_future:
    st.markdown("""
**Where this project can go next**
1. **Backtest the risk model:** Kupiec and Christoffersen tests for VaR, and compare GARCH, EGARCH and GJR-GARCH out of sample.
2. **Intraday volatility:** realized volatility from 5-minute data for sharper short-term forecasts.
3. **Regime switching:** Markov-switching models to detect calm versus crisis states.
4. **Multi-asset portfolios:** Bitcoin with Ethereum, equities and gold; cointegration (Johansen) and correlation risk.
5. **Options analytics:** implied-volatility smile, skew-aware pricing and hedge ratios.
6. **More data:** on-chain metrics, funding rates and news sentiment as model inputs.
7. **Alerts and accounts:** email or Telegram alerts when volatility or VaR crosses a limit, and saved portfolios.
8. **Production hardening:** a paid data feed, an API layer, automated tests and Docker deployment.
""")
