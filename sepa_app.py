import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import date, timedelta
from tqdm import tqdm

# Streamlit App Title
st.title("📈 Minervini RS Ranker – NSE Stocks")

st.markdown("""
Upload a CSV file with a column named **'Symbol'** (e.g., `RELIANCE`, `TCS`, `INFY`).
The app will:
- Fetch 6-month data for each symbol (from Yahoo Finance)
- Compute 6-month % change
- Rank all stocks by relative strength (RS)
- Display the top 30% (RS rank ≥ 70%)
""")

uploaded_file = st.file_uploader("Upload your .csv file", type=['csv'])

if uploaded_file is not None:
    # Read and prepare symbols
    df = pd.read_csv(uploaded_file)
    if "Symbol" not in df.columns:
        st.error("❌ The CSV must have a column named 'Symbol'")
    else:
        symbols = [s.strip().upper() + ".NS" for s in df["Symbol"].dropna().unique()]

        # Date range for 6-month performance
        end_date = date.today()
        start_date = end_date - timedelta(days=180)

        results = []

        st.write("Fetching stock data... please wait ⏳")

        progress = st.progress(0)
        for i, sym in enumerate(symbols):
            try:
                data = yf.download(sym, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    start_price = data["Close"].iloc[0]
                    end_price = data["Close"].iloc[-1]
                    change = ((end_price - start_price) / start_price) * 100
                    results.append((sym.replace(".NS", ""), change))
            except Exception:
                pass  # ignore failed symbols
            progress.progress((i + 1) / len(symbols))

        if len(results) == 0:
            st.error("❌ No valid stock data could be fetched. Check your symbols or internet connection.")
        else:
            perf_df = pd.DataFrame(results, columns=["Symbol", "6M_Percent_Change"])
            perf_df["RS_Rank"] = perf_df["6M_Percent_Change"].rank(pct=True) * 100
            top_rs_df = perf_df[perf_df["RS_Rank"] >= 70].sort_values("RS_Rank", ascending=False)

            st.success(f"✅ Found {len(top_rs_df)} stocks with RS ≥ 70")
            st.dataframe(top_rs_df)

            # Optional download
            csv = top_rs_df.to_csv(index=False)
            st.download_button("📥 Download Top RS Stocks", csv, "top_rs_stocks.csv", "text/csv")
