import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

st.title("📊 Minervini-Style Relative Strength (RS) Calculator for NSE Stocks")

st.write("""
Upload a CSV file with NSE stock symbols (e.g. `RELIANCE.NS`, `TCS.NS`, one per line).  
The app will:
- Fetch 1-year daily price data from Yahoo Finance  
- Compute % change over 1 M, 3 M, 6 M and 12 M periods  
- Combine them equally to form Minervini’s RS score  
- Rank all stocks and show those in the **top 30 % (RS > 70)**  
""")

uploaded_file = st.file_uploader("📂 Upload CSV with stock symbols", type=["csv"])

if uploaded_file is not None:
    symbols = pd.read_csv(uploaded_file, header=None)[0].dropna().unique().tolist()
    st.write(f"✅ Loaded {len(symbols)} stock symbols")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    progress_bar = st.progress(0)
    prices = {}

    for i, sym in enumerate(symbols):
        try:
            data = yf.download(sym, start=start_date, end=end_date, progress=False)
            if not data.empty:
                prices[sym] = data["Adj Close"]
        except Exception:
            st.warning(f"⚠️ Could not fetch data for {sym}")
        progress_bar.progress((i + 1) / len(symbols))

    st.success(f"Data fetched for {len(prices)} symbols")

    if prices:
        df = pd.DataFrame(prices).dropna(axis=1, how="any")

        def pct_change(period_days):
            """Return % change over a period (e.g., 21 = ~1 month)."""
            return (df.iloc[-1] / df.iloc[-period_days] - 1) * 100

        perf_1m = pct_change(21)
        perf_3m = pct_change(63)
        perf_6m = pct_change(126)
        perf_12m = pct_change(len(df) - 1)

        # Composite Minervini RS score (equal weighting)
        rs_score = (perf_1m + perf_3m + perf_6m + perf_12m) / 4.0

        # Percentile rank (0 – 100)
        rs_rank = rs_score.rank(pct=True) * 100

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

        st.subheader("🏆 Top 30 % Stocks by Minervini RS Rank")
        st.dataframe(top_stocks.reset_index(drop=True), use_container_width=True)

        st.download_button(
            label="💾 Download Top RS Stocks (CSV)",
            data=top_stocks.to_csv(index=False).encode("utf-8"),
            file_name="top_minervini_rs_stocks.csv",
            mime="text/csv",
        )
    else:
        st.error("No valid stock data retrieved.")
else:
    st.info("Please upload a CSV file to begin.")
