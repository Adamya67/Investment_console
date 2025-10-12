# Public Investment Terminal (Streamlit)
# ------------------------------------
# Quick-start Streamlit app to publish a public-facing investment dashboard.
# Features:
# - Ticker search with live price/percent change (15-minute delayed for most equities)
# - Watchlist persisted in session
# - Chart with SMA(20/50) + RSI(14)
# - Basic fundamentals snapshot (market cap, P/E, beta) when available
# - External resource launcher (WSJ, Barchart, TradingView) for deeper analysis
# - Clear disclaimers
#
# How to run locally:
#   1) pip install streamlit yfinance pandas numpy plotly requests
#   2) streamlit run app.py
#
# Deploy options:
#   - Streamlit Community Cloud (free)
#   - Hugging Face Spaces (Gradio/Streamlit)
#   - Render, Fly.io, or any Docker host

import time
import math
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Public Investment Terminal", page_icon="📈", layout="wide")

# ---------------------------
# Utility helpers
# ---------------------------
@st.cache_data(show_spinner=False, ttl=300)
def fetch_price_summary(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info  # faster, limited fields
        current = info.get("last_price")
        prev_close = info.get("previous_close")
        currency = info.get("currency")
        exchange = info.get("exchange")
        if current is None:
            # fallback via history
            hist = tk.history(period="2d", interval="1d")
            if len(hist) > 0:
                current = float(hist.Close.iloc[-1])
                prev_close = float(hist.Close.iloc[-2]) if len(hist) > 1 else None
        change = (current - prev_close) if (current is not None and prev_close is not None) else None
        pct = (change / prev_close * 100.0) if (change is not None and prev_close not in (None, 0)) else None
        return {
            "current": current,
            "prev_close": prev_close,
            "change": change,
            "pct": pct,
            "currency": currency,
            "exchange": exchange,
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(show_spinner=False, ttl=300)
def fetch_history(ticker: str, period: str = "1y", interval: str = "1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        return df
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=600)
def fundamentals_snapshot(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        fields = {
            "Market Cap": info.get("marketCap"),
            "Trailing P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "PEG Ratio": info.get("pegRatio"),
            "Beta (5Y Monthly)": info.get("beta"),
            "EPS (TTM)": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "52W High": info.get("fiftyTwoWeekHigh"),
            "52W Low": info.get("fiftyTwoWeekLow"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
        }
        return fields
    except Exception:
        return {}


def compute_indicators(df: pd.DataFrame):
    out = df.copy()
    out["SMA20"] = out["Close"].rolling(20).mean()
    out["SMA50"] = out["Close"].rolling(50).mean()
    # RSI(14)
    delta = out["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    out["RSI14"] = 100 - (100 / (1 + rs))
    return out

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📊 Investment Terminal")

def init_state():
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = ["AAPL", "NVDA", "MSFT", "AMZN", "META"]
    if "period" not in st.session_state:
        st.session_state.period = "1y"
    if "interval" not in st.session_state:
        st.session_state.interval = "1d"

init_state()

with st.sidebar:
    ticker = st.text_input("Enter ticker (e.g., AAPL, NVDA, SPY)", value="AAPL").upper().strip()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➕ Add to Watchlist"):
            if ticker and ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker)
    with col_b:
        if st.button("➖ Remove"):
            if ticker in st.session_state.watchlist:
                st.session_state.watchlist.remove(ticker)

    st.divider()
    st.caption("Chart range")
    st.session_state.period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=3)
    st.session_state.interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

    st.divider()
    st.caption("Quick links")
    if ticker:
        st.markdown(f"- [WSJ](https://www.wsj.com/market-data/quotes/{ticker})")
        st.markdown(f"- [Barchart](https://www.barchart.com/stocks/quotes/{ticker}/overview)")
        st.markdown(f"- [TradingView](https://www.tradingview.com/symbols/{ticker})")

# ---------------------------
# Header
# ---------------------------
st.title("📈 Public Investment Terminal")
st.caption("Educational purposes only. Not investment advice. Data is delayed and may be inaccurate.")

# ---------------------------
# Main: Price Strip + Watchlist Table
# ---------------------------
col1, col2 = st.columns([2, 1])

with col1:
    if ticker:
        quote = fetch_price_summary(ticker)
        if quote.get("error"):
            st.error(f"Error fetching price: {quote['error']}")
        else:
            current = quote.get("current")
            pct = quote.get("pct")
            change = quote.get("change")
            currency = quote.get("currency") or ""
            exchange = quote.get("exchange") or ""
            delta_str = f"{change:+.2f} ({pct:+.2f}%)" if (change is not None and pct is not None) else ""
            st.subheader(f"{ticker} — {current:.2f} {currency}  {delta_str}")
            if exchange:
                st.caption(f"Exchange: {exchange} • Source: Yahoo Finance (delayed)")

with col2:
    # Watchlist mini-table with live quotes
    if st.session_state.watchlist:
        data = []
        for tk in st.session_state.watchlist:
            q = fetch_price_summary(tk)
            if q.get("current") is not None:
                data.append({
                    "Ticker": tk,
                    "Price": round(q["current"], 2),
                    "Change %": round(q["pct"], 2) if q.get("pct") is not None else None,
                })
        if data:
            df_watch = pd.DataFrame(data).set_index("Ticker")
            st.dataframe(df_watch, use_container_width=True)

st.divider()

# ---------------------------
# Chart
# ---------------------------
if ticker:
    hist = fetch_history(ticker, period=st.session_state.period, interval=st.session_state.interval)
    if hist is None or hist.empty:
        st.warning("No historical data available for this combination.")
    else:
        enriched = compute_indicators(hist)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=enriched.index, open=enriched["Open"], high=enriched["High"], low=enriched["Low"], close=enriched["Close"], name="Price"))
        fig.add_trace(go.Scatter(x=enriched.index, y=enriched["SMA20"], mode="lines", name="SMA20"))
        fig.add_trace(go.Scatter(x=enriched.index, y=enriched["SMA50"], mode="lines", name="SMA50"))
        fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📉 RSI(14)"):
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=enriched.index, y=enriched["RSI14"], mode="lines", name="RSI14"))
            rsi_fig.add_hrect(y0=30, y1=70, fillcolor="LightGray", opacity=0.3, line_width=0)
            rsi_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 100])
            st.plotly_chart(rsi_fig, use_container_width=True)

# ---------------------------
# Fundamentals
# ---------------------------
with st.expander("🧾 Fundamentals snapshot"):
    if ticker:
        snap = fundamentals_snapshot(ticker)
        if snap:
            # Pretty formatting
            def fmt(v):
                if isinstance(v, (int, float)):
                    # human readable large numbers
                    if abs(v) >= 1e12:
                        return f"{v/1e12:.2f}T"
                    if abs(v) >= 1e9:
                        return f"{v/1e9:.2f}B"
                    if abs(v) >= 1e6:
                        return f"{v/1e6:.2f}M"
                    return f"{v:.2f}"
                return v
            df = pd.DataFrame({"Metric": list(snap.keys()), "Value": [fmt(v) for v in snap.values()]})
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("Fundamentals not available for this ticker.")

# ---------------------------
# Footer / Disclaimers
# ---------------------------
st.divider()
st.markdown(
    """
**Disclaimers**  
- For **educational purposes only**. This is **not** financial advice.  
- Market data may be **delayed** or inaccurate. Verify before trading.  
- By using this app, you agree to the **Yahoo Finance / yfinance** terms of use in your environment.  
- If you plan to make this app public at scale, consider a commercial market-data API (Polygon, Finnhub, Alpha Vantage, Twelve Data) and add API key management, caching, and rate-limit protections.
    """
)
