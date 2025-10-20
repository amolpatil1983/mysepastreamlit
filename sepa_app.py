import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Minervini RS Screener", layout="wide")
st.title("📊 Minervini-Style Relative Strength (RS) Screener – NSE Stocks")

st.write("""
Upload a CSV file containing NSE stock symbols (e.g. `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, one per line).  
The app will:
- Fetch 1-year daily price data from Yahoo Finance (in one batch for speed)
- Compute % change over 1M, 3M, 6M, and 12M  
- Combine them to form **Minervini’s RS score**
- Rank all stocks by RS percentile (0–100)
- Display **Top 30% (RS > 70)** stocks and plot them
""")

uploaded_file = st.file_uploader("📂 Upload CSV with stock symbols", type=["csv"])

if uploaded_file is not None:
    symbols = pd.read_csv(uploaded_file, header=None)[0].dropna().unique().tolist()
    symbols = [s+".NS" for s in symbols]
    st.write(f"✅ Loaded {len(symbols)} stock symbols")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    st.info("Fetching 1-year data from Yahoo Finance (this may take a few seconds)...")
    try:
        data = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            group_by='ticker',
            progress=False,
            threads=True
        )
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        st.stop()

    # Extract adjusted close prices
    prices = {}
    failed = []

    for sym in symbols:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                adj_close = data[sym]["Adj Close"].dropna()
            else:
                adj_close = data["Adj Close"].dropna()
            if not adj_close.empty:
                prices[sym] = adj_close
            else:
                failed.append(sym)
        except Exception:
            failed.append(sym)

    if failed:
        st.warning(f"⚠️ Could not fetch data for {len(failed)} symbols: {', '.join(failed[:10])} {'...' if len(failed) > 10 else ''}")

    if not prices:
        st.error("No valid stock data retrieved. Please check your symbols or try again later.")
        st.stop()

    df = pd.DataFrame(prices).dropna(axis=1, how="any")

    def pct_change(period_days):
        """Return % change over a given lookback period."""
        if len(df) > period_days:
            return (df.iloc[-1] / df.iloc[-period_days] - 1) * 100
        else:
            return pd.Series(np.nan, index=df.columns)

    # Compute performance metrics
    perf_1m = pct_change(21)
    perf_3m = pct_change(63)
    perf_6m = pct_change(126)
    perf_12m = pct_change(len(df) - 1)

    # Minervini composite RS score
    rs_score = (perf_1m + perf_3m + perf_6m + perf_12m) / 4.0
    rs_rank = rs_score.rank(pct=True) * 100  # percentile rank (0–100)

    result = pd.DataFrame({
        "Symbol": rs_rank.index,
        "1M_%": perf_1m.values,
        "3M_%": perf_3m.values,
        "6M_%": perf_6m.values,
        "12M_%": perf_12m.values,
        "RS_Score": rs_score.values,
        "RS_Rank": rs_rank.values
    }).sort_values(by="RS_Rank", ascending=False)

    top_stocks = result[result["RS_Rank"] > 70]
    top10 = result.head(10)

    st.subheader("🏆 Top 30% Stocks by Minervini RS Rank")
    st.dataframe(top_stocks.reset_index(drop=True), use_container_width=True)

    # --- Visualization ---
    st.subheader("📈 RS Rank vs 1-Year Return (Top 10 Annotated)")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(result["RS_Rank"], result["12M_%"], alpha=0.6, color="gray", label="All Stocks")
    ax.scatter(top_stocks["RS_Rank"], top_stocks["12M_%"], color="tab:blue", label="Top RS Stocks (>70)")
    ax.axvline(70, color="red", linestyle="--", alpha=0.8, label="RS = 70 cutoff")

    for _, row in top10.iterrows():
        ax.text(row["RS_Rank"], row["12M_%"], row["Symbol"], fontsize=8, alpha=0.9)

    ax.set_xlabel("RS Rank (Percentile)")
    ax.set_ylabel("1-Year % Return")
    ax.set_title("Minervini RS Rank vs 1-Year Return (NSE Stocks)")
    ax.legend()
    st.pyplot(fig)

    # --- Download ---
    st.download_button(
        label="💾 Download Top RS Stocks (CSV)",
        data=top_stocks.to_csv(index=False).encode("utf-8"),
        file_name="top_minervini_rs_stocks.csv",
        mime="text/csv",
    )

else:
    st.info("Please upload a CSV file to begin.")
