# Public Investment Terminal (Streamlit)
# ------------------------------------
# Three-page app:
# 1) Investment Terminal (charts, SMA20/50, RSI, fundamentals, watchlist)
# 2) Trading Command Center (quick links, grouped resources, notes, custom links)
# 3) Research Feed (compose weekly reports, archive posts, export, tweet links)
#
# Run locally:
#   pip install streamlit yfinance pandas numpy plotly requests pyarrow
#   streamlit run app.py

from datetime import datetime
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
import os

st.set_page_config(page_title="Public Investment Terminal", page_icon="📈", layout="wide")

# ==========================
# Helpers
# ==========================
@st.cache_data(show_spinner=False, ttl=300)
def fetch_price_summary(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        current = info.get("last_price")
        prev_close = info.get("previous_close")
        currency = info.get("currency")
        exchange = info.get("exchange")
        if current is None:
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
    delta = out["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    out["RSI14"] = 100 - (100 / (1 + rs))
    return out

# ==========================
# Session state
# ==========================
ss = st.session_state
if "watchlist" not in ss:
    ss.watchlist = ["AAPL", "NVDA", "MSFT", "AMZN", "META"]
if "period" not in ss:
    ss.period = "1y"
if "interval" not in ss:
    ss.interval = "1d"
if "notes_plan" not in ss:
    ss.notes_plan = ""
if "notes_eod" not in ss:
    ss.notes_eod = ""
if "custom_links" not in ss:
    ss.custom_links = []
if "expand_state" not in ss:
    ss.expand_state = {}
if "dark_mode" not in ss:
    ss.dark_mode = True
if "reports" not in ss:
    ss.reports = []
if "is_author" not in ss:
    ss.is_author = False

# ==========================
# Page selector
# ==========================
PAGE = st.sidebar.radio(
    "Select Page",
    ["Investment Terminal", "Trading Command Center", "Research Feed"],
    index=0,
)

# ==========================
# Page 1: Investment Terminal
# ==========================

def render_terminal():
    st.title("Public Investment Terminal")
    st.caption("Educational purposes only. Not investment advice. Data is delayed and may be inaccurate.")

    with st.sidebar:
        ticker = st.text_input("Enter ticker (e.g., AAPL, NVDA, SPY)", value="AAPL").upper().strip()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Add to Watchlist"):
                if ticker and ticker not in ss.watchlist:
                    ss.watchlist.append(ticker)
        with c2:
            if st.button("Remove"):
                if ticker in ss.watchlist:
                    ss.watchlist.remove(ticker)
        st.divider()
        st.caption("Chart range")
        ss.period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=3)
        ss.interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
        st.divider()
        st.caption("Quick links")
        if ticker:
            st.markdown(f"- [WSJ](https://www.wsj.com/market-data/quotes/{ticker})")
            st.markdown(f"- [Barchart](https://www.barchart.com/stocks/quotes/{ticker}/overview)")
            st.markdown(f"- [TradingView](https://www.tradingview.com/symbols/{ticker})")

    cL, cR = st.columns([2, 1])
    with cL:
        if ticker:
            quote = fetch_price_summary(ticker)
            if quote.get("error"):
                st.error(f"Error fetching price: {quote['error']}")
            else:
                cur = quote.get("current")
                pct = quote.get("pct")
                chg = quote.get("change")
                curr = quote.get("currency") or ""
                exch = quote.get("exchange") or ""
                delta = f"{chg:+.2f} ({pct:+.2f}%)" if (chg is not None and pct is not None) else ""
                st.subheader(f"{ticker} — {cur:.2f} {curr}  {delta}")
                if exch:
                    st.caption(f"Exchange: {exch} • Source: Yahoo Finance (delayed)")
    with cR:
        if ss.watchlist:
            rows = []
            for tk in ss.watchlist:
                q = fetch_price_summary(tk)
                if q.get("current") is not None:
                    rows.append({"Ticker": tk, "Price": round(q["current"], 2), "Change %": round(q["pct"], 2) if q.get("pct") is not None else None})
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

    st.divider()

    if ticker:
        hist = fetch_history(ticker, period=ss.period, interval=ss.interval)
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
            with st.expander("RSI (14)"):
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=enriched.index, y=enriched["RSI14"], mode="lines", name="RSI14"))
                rsi_fig.add_hrect(y0=30, y1=70, fillcolor="LightGray", opacity=0.3, line_width=0)
                rsi_fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 100])
                st.plotly_chart(rsi_fig, use_container_width=True)

    with st.expander("Fundamentals snapshot"):
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
                st.dataframe(pd.DataFrame({"Metric": list(snap.keys()), "Value": [fmt(v) for v in snap.values()]}), hide_index=True, use_container_width=True)
            else:
                st.info("Fundamentals not available for this ticker.")

    st.divider()
    st.markdown("""
**Disclaimers**
- For **educational purposes only**. This is **not** financial advice.
- Market data may be **delayed** or inaccurate. Verify before trading.
- By using this app, you agree to the Yahoo Finance / yfinance terms of use in your environment.
- For public scale, consider a commercial market-data API and add key management, caching, and rate-limit protections.
""")

# ==========================
# Page 2: Trading Command Center
# ==========================

def _apply_theme():
    if ss.dark_mode:
        bg = "#0f1420"; panel = "#161d2e"; text = "#e6ecff"; accent = "#5b8cff"
    else:
        bg = "#f7f9fc"; panel = "#ffffff"; text = "#0f1420"; accent = "#335eea"
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        .center-panel {{ background:{panel}; padding:1rem; border-radius:16px; border:1px solid rgba(0,0,0,0.08); }}
        .pill {{ margin:0.25rem; padding:0.5rem 0.8rem; border-radius:999px; border:1px solid rgba(0,0,0,0.15); display:inline-block; text-decoration:none; }}
        .pill:hover {{ background:{accent}22; }}
        .muted {{ opacity:0.7; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_command_center():
    _apply_theme()
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Toggle Theme"):
            ss.dark_mode = not ss.dark_mode
            st.rerun()
    with c2:
        if st.button("Save Notes"):
            st.success("Notes saved (session only).")
    with c3:
        if st.button("Clear Notes"):
            ss.notes_plan = ""
            ss.notes_eod = ""
            st.success("Notes cleared.")

    st.title("Trading Command Center")
    st.caption("One-click hub for news, scanners, calendars, sentiment & your notes. Links open in a new tab.")

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
            st.markdown(f"[{label}]({url})")

    st.divider()

    st.subheader("All Links")
    search = st.text_input("Search tools (e.g., earnings, biotech, AI, defense, options flow)")

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

    a, b = st.columns(2)
    with a:
        if st.button("Expand All"):
            for k in groups.keys():
                ss.expand_state[k] = True
    with b:
        if st.button("Collapse All"):
            for k in groups.keys():
                ss.expand_state[k] = False

    for group, links in groups.items():
        if search:
            filtered = [(n, u) for (n, u) in links if search.lower() in n.lower() or search.lower() in u.lower()]
        else:
            filtered = links
        count = len(filtered)
        exp = st.expander(f"{group} — {count} link{'s' if count != 1 else ''}", expanded=ss.expand_state.get(group, False))
        with exp:
            if count == 0:
                st.caption("No matches.")
            for name, url in filtered:
                st.markdown(f"[{name}]({url})")

    st.divider()

    cA, cB = st.columns(2)
    with cA:
        st.subheader("Daily Plan")
        ss.notes_plan = st.text_area("Lock in", value=ss.notes_plan, height=160)
    with cB:
        st.subheader("End-of-day Review")
        ss.notes_eod = st.text_area("What worked? What didn't? Lessons for tomorrow...", value=ss.notes_eod, height=160)

    st.divider()
    st.subheader("Custom Links")
    name = st.text_input("Name (e.g., My Trade Log)")
    url = st.text_input("URL (https://...)")
    add_col, _ = st.columns([1, 3])
    with add_col:
        if st.button("Add Link") and name and url:
            ss.custom_links.append({"Name": name, "URL": url})
            st.success(f"Added {name}.")
    if ss.custom_links:
        st.dataframe(pd.DataFrame(ss.custom_links), use_container_width=True)

# ==========================
# Page 3: Research Feed
# ==========================

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
    return "\n".join(lines)


def render_research_feed():
    st.title("Research Feed — Weekly Reports")
    st.caption("Post your trade notes and findings. Export to Markdown or compose a tweet link. Not investment advice.")

    # ---- Author auth gate ----
    author_secret = st.secrets.get("AUTHOR_KEY") or os.getenv("AUTHOR_KEY")
    if not ss.is_author:
        with st.expander("Author sign-in (only you can post)", expanded=True):
            entered = st.text_input("Enter author key", type="password")
            col_ok, col_hint = st.columns([1,3])
            with col_ok:
                if st.button("Sign in"):
                    if author_secret and entered == str(author_secret):
                        ss.is_author = True
                        st.success("Signed in as author. You can post now.")
                        st.experimental_rerun()
                    else:
                        st.error("Invalid key. Viewing is allowed, posting is restricted.")
            with col_hint:
                st.caption("Set AUTHOR_KEY in Streamlit Secrets (Cloud: app → Settings → Secrets) or as an environment variable locally.")

    # ---- Compose (visible only to author) ----
    if ss.is_author:
        st.subheader("Compose New Post")
        c1, c2 = st.columns([2, 1])
        with c1:
            title = st.text_input("Title", placeholder="Weekly Report: Momentum plays and catalysts")
            body = st.text_area("Body", height=220, placeholder="Summary, thesis, entries/exits, risk, catalysts, lessons...")
        with c2:
            tickers = st.text_input("Tickers (comma-separated)", placeholder="AAPL, NVDA, SPY")
            sentiment = st.selectbox("Sentiment", ["Bullish", "Neutral", "Bearish"], index=0)
            timeframe = st.selectbox("Timeframe", ["Weekly", "Monthly", "Intraday", "Swing"], index=0)

        cA, cB, cC, cD = st.columns(4)
        with cA:
            if st.button("Post"):
                if not title or not body:
                    st.error("Please enter a title and body.")
                else:
                    ss.reports.insert(0, {
                        "title": title.strip(),
                        "body": body.strip(),
                        "tickers": tickers.upper().replace(" ", ""),
                        "sentiment": sentiment,
                        "timeframe": timeframe,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    })
                    st.success("Posted.")
        with cC:
            if st.button("Export All (Markdown)"):
                if ss.reports:
                    md = "\n\n---\n\n".join(_md_for_post(p) for p in ss.reports)
                    st.download_button(
                        "Download feed.md",
                        data=md,
                        file_name="research_feed.md"
                    )
                else:
                    st.info("No posts yet.")

with cD:
    if st.button("Sign out"):
        ss.is_author = False
        st.rerun()
    else:
        st.info("Viewing mode: you can read and download posts, but only the author can publish.")

    st.divider()
    st.subheader("Your Posts")
    if not ss.reports:
        st.info("No posts yet. Author can write the first weekly report above.")
    else:
        for idx, post in enumerate(ss.reports):
            with st.container():
                st.markdown(f"### {post['title']}")
                st.caption(f"{post['created_at']} • {post['tickers']} • {post['sentiment']} • {post['timeframe']}")
                st.write(post['body'])
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    md = _md_for_post(post)
                    st.download_button("Download .md", data=md, file_name=f"{post['title'].replace(' ', '_')}.md", key=f"dl_{idx}")
                with cc2:
                    tweet_text = f"{post['title']} — {post['tickers']} — {post['sentiment']}
" + (post['body'][:220] + ("..." if len(post['body'])>220 else ""))
                    url = f"https://twitter.com/intent/tweet?text={quote_plus(tweet_text)}"
                    st.markdown(f"[Compose Tweet]({url})")
                with cc3:
                    if ss.is_author and st.button("Delete", key=f"del_{idx}"):
                        ss.reports.pop(idx)
                        st.rerun()

# ==========================
# Router
# ==========================
if PAGE == "Investment Terminal":
    render_terminal()
elif PAGE == "Trading Command Center":
    render_command_center()
else:
    render_research_feed()
