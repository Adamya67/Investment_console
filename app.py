# Public Investment Terminal (Streamlit)
# ------------------------------------
# Now a three-page app:
# 1) Investment Terminal (charts, SMA20/50, RSI, fundamentals, watchlist)
# 2) Trading Command Center (quick links, grouped resources, notes, custom links)
# 3) Research Feed (compose weekly reports, archive posts, export/download, tweet links)
#
# How to run locally:
#   1) pip install streamlit yfinance pandas numpy plotly requests pyarrow
#   2) streamlit run app.py
#
# Deploy options:
#   - Streamlit Community Cloud (free)
#   - Hugging Face Spaces (Streamlit)
#   - Render, Fly.io, or any Docker host

import time
import math
from datetime import datetime
from urllib.parse import quote_plus
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Public Investment Terminal", page_icon="📈", layout="wide")

# ======================================================
# Global helpers/state
# ======================================================
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
    except Exception:
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

# ------------------------------------------------------
# Session State Init
# ------------------------------------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "MSFT", "AMZN", "META"]
if "period" not in st.session_state:
    st.session_state.period = "1y"
if "interval" not in st.session_state:
    st.session_state.interval = "1d"
# Command Center state
if "notes_plan" not in st.session_state:
    st.session_state.notes_plan = ""
if "notes_eod" not in st.session_state:
    st.session_state.notes_eod = ""
if "custom_links" not in st.session_state:
    st.session_state.custom_links = []  # list of {Name, URL}
if "expand_state" not in st.session_state:
    st.session_state.expand_state = {}  # group -> bool
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
# Research Feed state
if "reports" not in st.session_state:
    st.session_state.reports = []  # list of dicts

# ======================================================
# PAGES
# ======================================================
PAGE = st.sidebar.radio(
    "Select Page",
    ["Investment Terminal", "Trading Command Center", "Research Feed"],
    index=0,
)

# ======================================================
# PAGE 1: Investment Terminal
# ======================================================

def render_terminal():
    st.title("📈 Public Investment Terminal")
    st.caption("Educational purposes only. Not investment advice. Data is delayed and may be inaccurate.")

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
            st.link_button("WSJ", f"https://www.wsj.com/market-data/quotes/{ticker}")
            st.link_button("Barchart", f"https://www.barchart.com/stocks/quotes/{ticker}/overview")
            st.link_button("TradingView", f"https://www.tradingview.com/symbols/{ticker}")

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

    with st.expander("🧾 Fundamentals snapshot"):
        if ticker:
            snap = fundamentals_snapshot(ticker)
            if snap:
                def fmt(v):
                    if isinstance(v, (int, float)):
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

# ======================================================
# PAGE 2: Trading Command Center
# ======================================================

def _apply_theme():
    # lightweight dark/light toggle via CSS
    if st.session_state.dark_mode:
        bg = "#0f1420"; panel = "#161d2e"; text = "#e6ecff"; accent = "#5b8cff"
    else:
        bg = "#f7f9fc"; panel = "#ffffff"; text = "#0f1420"; accent = "#335eea"
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        .center-panel {{ background:{panel}; padding:1rem; border-radius:16px; border:1px solid rgba(255,255,255,0.08); }}
        .pill {{ margin:0.25rem; padding:0.5rem 0.8rem; border-radius:999px; border:1px solid rgba(255,255,255,0.15); display:inline-block; text-decoration:none; }}
        .pill:hover {{ background:{accent}22; }}
        .muted {{ opacity:0.7; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_command_center():
    _apply_theme()

    # Header controls
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Toggle Theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    with c2:
        if st.button("Save Notes"):
            st.success("Notes saved locally in session.")
    with c3:
        if st.button("Clear Notes"):
            st.session_state.notes_plan = ""
            st.session_state.notes_eod = ""
            st.success("Notes cleared.")

    st.title("Trading Command Center")
    st.caption("One-click hub for news, scanners, calendars, sentiment & your notes. (Links open in a new tab.)")

    # Quick Shortcuts
    st.subheader("Quick Shortcuts")
    qcols = st.columns(5)
    quick = [
        ("Simulator", "https://www.investopedia.com/simulator/"),
        ("Barchart Most Active", "https://www.barchart.com/stocks/most-active/daily"),
        ("Premarket Movers", "https://www.barrons.com/market-data/stocks/movers/pre-market"),
        ("Earnings Calendar", "https://www.nasdaq.com/market-activity/earnings"),
        ("TradingView", "https://www.tradingview.com/"),
        ("Economic Calendar", "https://www.investing.com/economic-calendar/"),
        ("StockTwits", "https://stocktwits.com/"),
        ("Unusual Whales", "https://unusualwhales.com/flow"),
    ]
    for i, (label, url) in enumerate(quick):
        with qcols[i % 5]:
            st.link_button(label, url)

    st.divider()

    # Link Groups
    st.subheader("All Links")
    search = st.text_input("Search tools… e.g., earnings, biotech, AI, defense, options flow")

    groups = {
        "Real-Time News": [
            ("WSJ Markets", "https://www.wsj.com/news/markets"),
            ("Bloomberg Markets", "https://www.bloomberg.com/markets"),
            ("Reuters Markets", "https://www.reuters.com/markets/"),
            ("CNBC Markets", "https://www.cnbc.com/markets/"),
        ],
        "Momentum Scanners": [
            ("Barchart Momentum", "https://www.barchart.com/stocks/signals/top-bottom"),
            ("Finviz Screener", "https://finviz.com/screener.ashx"),
            ("TradingView Top Gainers", "https://www.tradingview.com/markets/stocks-usa/market-movers-gainers/"),
            ("Yahoo Most Active", "https://finance.yahoo.com/most-active"),
        ],
        "Calendars": [
            ("Earnings Calendar (Nasdaq)", "https://www.nasdaq.com/market-activity/earnings"),
            ("Economic Calendar (Investing)", "https://www.investing.com/economic-calendar/"),
        ],
        "Analysis & Fundamentals": [
            ("TradingView", "https://www.tradingview.com/"),
            ("Barchart Overview", "https://www.barchart.com/stocks/overview"),
            ("GuruFocus Screener", "https://www.gurufocus.com/screener/"),
        ],
        "Sentiment & Options Flow": [
            ("Unusual Whales Flow", "https://unusualwhales.com/flow"),
            ("QuiverQuant", "https://www.quiverquant.com/"),
            ("StockTwits Trending", "https://stocktwits.com/rankings/trending"),
        ],
        "Sector Hubs": [
            ("ETFDB Sector ETFs", "https://etfdb.com/etfs/sector/"),
            ("SPDR Sector Dashboard", "https://www.ssga.com/us/en/individual/etfs/insights/sector-spdr-dashboard"),
            ("Finviz Groups", "https://finviz.com/groups.ashx?g=sector&v=210"),
            ("Seeking Alpha Sectors", "https://seekingalpha.com/markets/sectors"),
            ("Morningstar Sectors", "https://www.morningstar.com/sectors"),
        ],
    }

    # Expand/Collapse all controls
    a, b = st.columns(2)
    with a:
        if st.button("Expand All"):
            for k in groups.keys():
                st.session_state.expand_state[k] = True
    with b:
        if st.button("Collapse All"):
            for k in groups.keys():
                st.session_state.expand_state[k] = False

    # Render groups
    for group, links in groups.items():
        if search:
            filtered = [(n, u) for (n, u) in links if search.lower() in n.lower() or search.lower() in u.lower()]
        else:
            filtered = links
        count = len(filtered)
        expander = st.expander(f"{group}  —  {count} link{'s' if count != 1 else ''}", expanded=st.session_state.expand_state.get(group, False))
        with expander:
            if count == 0:
                st.caption("No matches.")
            for name, url in filtered:
                st.link_button(name, url)

    st.divider()

    # Notes & Review
    cA, cB = st.columns(2)
    with cA:
        st.subheader("Daily Plan")
        st.session_state.notes_plan = st.text_area("Lock in", value=st.session_state.notes_plan, height=160)
    with cB:
        st.subheader("End-of-day Review")
        st.session_state.notes_eod = st.text_area("What worked? What didn’t? Lessons for tomorrow…", value=st.session_state.notes_eod, height=160)

    st.divider()

    # Custom Links
    st.subheader("Custom Links")
    name = st.text_input("Name (e.g., My Trade Log)")
    url = st.text_input("URL (https://…)")
    add_col, _ = st.columns([1, 3])
    with add_col:
        if st.button("Add Link") and name and url:
            st.session_state.custom_links.append({"Name": name, "URL": url})
            st.success(f"Added {name}.")

    if st.session_state.custom_links:
        df = pd.DataFrame(st.session_state.custom_links)
        st.dataframe(df, use_container_width=True)

# ======================================================
# PAGE 3: Research Feed (Weekly Reports)
# ======================================================

def _md_for_post(post: dict) -> str:
    lines = [
        f"# {post['title']}",
        f"**Date:** {post['created_at']}  ",
        f"**Tickers:** {post['tickers']}  ",
        f"**Sentiment:** {post['sentiment']}  ",
        f"**Timeframe:** {post['timeframe']}  ",
        "",
        post['body'],
    ]
    return "
".join(lines)


def render_research_feed():
    st.title("📰 Research Feed — Weekly Reports")
    st.caption("Post your trade notes and findings. Export to Markdown or compose a tweet link. Not investment advice.")

    st.subheader("Compose New Post")
    c1, c2 = st.columns([2, 1])
    with c1:
        title = st.text_input("Title", placeholder="Weekly Report: Momentum plays and catalysts")
        body = st.text_area("Body", height=220, placeholder="Summary, thesis, entries/exits, risk, catalysts, lessons…")
    with c2:
        tickers = st.text_input("Tickers (comma-separated)", placeholder="AAPL, NVDA, SPY")
        sentiment = st.selectbox("Sentiment", ["Bullish", "Neutral", "Bearish"], index=0)
        timeframe = st.selectbox("Timeframe", ["Weekly", "Monthly", "Intraday", "Swing"], index=0)

    colA, colB, colC = st.columns([1,1,1])
    with colA:
        if st.button("Post"):
            if not title or not body:
                st.error("Please enter a title and body.")
            else:
                st.session_state.reports.insert(0, {
                    "title": title.strip(),
                    "body": body.strip(),
                    "tickers": tickers.upper().replace(" ", ""),
                    "sentiment": sentiment,
                    "timeframe": timeframe,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                })
                st.success("Posted.")
    with colB:
        if st.button("Clear Draft"):
            st.experimental_rerun()
    with colC:
        if st.button("Export All (Markdown)"):
            if st.session_state.reports:
                md = "

---

".join(_md_for_post(p) for p in st.session_state.reports)
                st.download_button("Download feed.md", data=md, file_name="research_feed.md")
            else:
                st.info("No posts yet.")

    st.divider()

    st.subheader("Your Posts")
    if not st.session_state.reports:
        st.info("No posts yet. Write your first weekly report above.")
    else:
        for idx, post in enumerate(st.session_state.reports):
            with st.container(border=True):
                st.markdown(f"### {post['title']}")
                st.caption(f"{post['created_at']} • {post['tickers']} • {post['sentiment']} • {post['timeframe']}")
                st.write(post['body'])

                # Buttons per post
                cc1, cc2, cc3 = st.columns([1,1,1])
                with cc1:
                    md = _md_for_post(post)
                    st.download_button("Download .md", data=md, file_name=f"{post['title'].replace(' ', '_')}.md", key=f"dl_{idx}")
                with cc2:
                    tweet_text = f"{post['title']} — {post['tickers']} — {post['sentiment']}
" + (post['body'][:220] + ("…" if len(post['body'])>220 else ""))
                    url = f"https://twitter.com/intent/tweet?text={quote_plus(tweet_text)}"
                    st.link_button("Compose Tweet", url, key=f"tw_{idx}")
                with cc3:
                    if st.button("Delete", key=f"del_{idx}"):
                        st.session_state.reports.pop(idx)
                        st.experimental_rerun()

# ======================================================
# Router
# ======================================================
if PAGE == "Investment Terminal":
    render_terminal()
elif PAGE == "Trading Command Center":
    render_command_center()
else:
    render_research_feed()
