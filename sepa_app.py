import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

# Configuration
TIMEFRAMES = {
    '1Y': 252,
    '6M': 126,
    '3M': 63,
    '1M': 21
}

WEIGHTS = {
    '1Y': 0.40,
    '6M': 0.20,
    '3M': 0.20,
    '1M': 0.20
}

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period='1y'):
    """Fetch historical data for a single stock"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty or len(df) < 200:  # Need sufficient history
            return None
        return df['Close']
    except:
        return None

def create_composite_index(symbols, period='1y'):
    """Create equal-weighted composite index from stock universe"""
    stock_prices_dict = {}
    failed_symbols = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, symbol in enumerate(symbols):
        nse_symbol = f"{symbol}.NS"
        status_text.text(f"Fetching {symbol}... ({idx+1}/{len(symbols)})")
        
        prices = fetch_stock_data(nse_symbol, period)
        if prices is not None and len(prices) >= TIMEFRAMES['1Y']:
            stock_prices_dict[symbol] = prices
        else:
            failed_symbols.append(symbol)
        
        progress_bar.progress((idx + 1) / len(symbols))
    
    progress_bar.empty()
    status_text.empty()
    
    if not stock_prices_dict:
        return None, stock_prices_dict, failed_symbols
    
    # Use outer join to keep all dates, then forward fill missing values
    all_prices_df = pd.concat(stock_prices_dict.values(), axis=1, join='outer', keys=stock_prices_dict.keys())
    all_prices_df = all_prices_df.sort_index()
    
    # Forward fill missing values (stocks that weren't trading on certain days)
    all_prices_df = all_prices_df.fillna(method='ffill')
    
    # Only keep dates where at least 50% of stocks have data
    threshold = len(stock_prices_dict) * 0.5
    all_prices_df = all_prices_df.dropna(thresh=threshold)
    
    if len(all_prices_df) < TIMEFRAMES['1Y']:
        return None, stock_prices_dict, failed_symbols
    
    # Calculate equal-weighted index (average of normalized prices)
    normalized = all_prices_df.div(all_prices_df.iloc[0]) * 100
    composite_index = normalized.mean(axis=1)
    
    return composite_index, stock_prices_dict, failed_symbols

def calculate_return(prices, days):
    """Calculate percentage return over specified days"""
    if len(prices) < days:
        return np.nan
    return ((prices.iloc[-1] - prices.iloc[-days]) / prices.iloc[-days]) * 100

def calculate_rs_rating(stock_prices_dict, composite_index):
    """Calculate RS rating for all stocks"""
    results = []
    skipped = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (symbol, prices) in enumerate(stock_prices_dict.items()):
        status_text.text(f"Calculating RS for {symbol}... ({idx+1}/{len(stock_prices_dict)})")
        
        # Align stock prices with composite index
        try:
            # Use outer join and forward fill to align dates
            aligned = pd.concat([prices, composite_index], axis=1, join='outer').sort_index()
            aligned.columns = ['stock', 'index']
            aligned = aligned.fillna(method='ffill').dropna()
            
            if len(aligned) < TIMEFRAMES['1Y']:
                skipped.append(f"{symbol} (has {len(aligned)} days, need {TIMEFRAMES['1Y']})")
                continue
            
            stock_prices = aligned['stock']
            index_prices = aligned['index']
            
            stock_returns = {}
            index_returns = {}
            
            # Calculate returns for each timeframe
            for period, days in TIMEFRAMES.items():
                stock_returns[period] = calculate_return(stock_prices, days)
                index_returns[period] = calculate_return(index_prices, days)
            
            # Skip if any return is NaN
            if any(np.isnan(list(stock_returns.values()))) or any(np.isnan(list(index_returns.values()))):
                skipped.append(f"{symbol} (NaN returns)")
                continue
            
            # Calculate relative performance (stock return - index return)
            relative_perf = {
                period: stock_returns[period] - index_returns[period]
                for period in TIMEFRAMES.keys()
            }
            
            results.append({
                'Symbol': symbol,
                **{f'Return_{k}': v for k, v in stock_returns.items()},
                **{f'RelPerf_{k}': v for k, v in relative_perf.items()}
            })
        except Exception as e:
            skipped.append(f"{symbol} (error: {str(e)[:30]})")
            continue
        
        progress_bar.progress((idx + 1) / len(stock_prices_dict))
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        return None, skipped
    
    df = pd.DataFrame(results)
    
    # Calculate percentile ranks for each timeframe
    for period in TIMEFRAMES.keys():
        col = f'RelPerf_{period}'
        df[f'Percentile_{period}'] = df[col].rank(pct=True) * 100
    
    # Calculate composite RS rating
    df['RS_Rating'] = sum(
        df[f'Percentile_{period}'] * WEIGHTS[period]
        for period in TIMEFRAMES.keys()
    )
    
    # Sort by RS rating
    df = df.sort_values('RS_Rating', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    
    return df, skipped

def create_rs_chart(df, top_n=20):
    """Create visualization of top RS rated stocks"""
    plot_df = df.head(top_n).copy()
    
    fig = go.Figure()
    
    colors = ['#2ecc71' if x >= 80 else '#3498db' if x >= 70 else '#95a5a6' 
              for x in plot_df['RS_Rating']]
    
    fig.add_trace(go.Bar(
        x=plot_df['Symbol'],
        y=plot_df['RS_Rating'],
        marker_color=colors,
        text=plot_df['RS_Rating'].round(1),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Stocks by RS Rating',
        xaxis_title='Symbol',
        yaxis_title='RS Rating',
        height=500,
        showlegend=False,
        template='plotly_white',
        yaxis_range=[0, 105]
    )
    
    return fig

def create_performance_heatmap(df, top_n=20):
    """Create heatmap of returns across timeframes"""
    plot_df = df.head(top_n).copy()
    
    heatmap_data = plot_df[[f'Return_{period}' for period in TIMEFRAMES.keys()]].values
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=list(TIMEFRAMES.keys()),
        y=plot_df['Symbol'],
        colorscale='RdYlGn',
        text=np.round(heatmap_data, 2),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title="Return %")
    ))
    
    fig.update_layout(
        title=f'Returns Heatmap - Top {top_n} Stocks',
        xaxis_title='Timeframe',
        yaxis_title='Symbol',
        height=600,
        template='plotly_white'
    )
    
    return fig

# Streamlit App
st.set_page_config(page_title="Minervini RS Rating", layout="wide")

st.title("📊 Mark Minervini RS Rating Calculator")
st.markdown("*Based on SEPA Methodology - Composite Index Benchmark*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload CSV (Symbol column)", type=['csv'])
    
    st.markdown("---")
    st.markdown("**Weights**")
    for period, weight in WEIGHTS.items():
        st.text(f"{period}: {weight*100:.0f}%")

# Main content
if uploaded_file is not None:
    stocks_df = pd.read_csv(uploaded_file)
    
    if 'Symbol' not in stocks_df.columns:
        st.error("❌ CSV must contain a 'Symbol' column")
    else:
        symbols = stocks_df['Symbol'].str.strip().tolist()
        
        st.info(f"📥 Loaded {len(symbols)} symbols")
        
        # Create composite index
        with st.spinner("Creating composite index from stock universe..."):
            composite_index, stock_prices_dict, failed_symbols = create_composite_index(symbols)
        
        if composite_index is None:
            st.error("❌ Failed to create composite index. No valid data retrieved.")
            st.stop()
        
        valid_symbols = list(stock_prices_dict.keys())
        
        # Show data retrieval summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valid Stocks", len(valid_symbols))
        with col2:
            st.metric("Failed/Delisted", len(failed_symbols))
        with col3:
            st.metric("Success Rate", f"{len(valid_symbols)/len(symbols)*100:.1f}%")
        
        if failed_symbols:
            with st.expander("⚠️ View failed symbols"):
                st.text(", ".join(failed_symbols))
        
        # Calculate RS ratings
        with st.spinner("Calculating RS ratings..."):
            results_df, skipped = calculate_rs_rating(stock_prices_dict, composite_index)
        
        if results_df is None or len(results_df) == 0:
            st.error("❌ No valid RS ratings calculated")
            if skipped:
                with st.expander("🔍 Debug: View skipped stocks"):
                    for item in skipped:
                        st.text(item)
        else:
            st.success(f"✅ Processed {len(results_df)} stocks")
            
            if skipped:
                with st.expander(f"⚠️ Skipped {len(skipped)} stocks during RS calculation"):
                    for item in skipped[:50]:  # Show first 50
                        st.text(item)
            
            # Display options
            top_n = st.slider("Top N stocks to display", 10, min(50, len(results_df)), 20)
            
            # Display table
            st.subheader("📋 Rankings")
            
            display_cols = ['Rank', 'Symbol', 'RS_Rating'] + \
                          [f'Return_{p}' for p in TIMEFRAMES.keys()]
            
            st.dataframe(
                results_df[display_cols].head(top_n).style.format({
                    'RS_Rating': '{:.2f}',
                    **{f'Return_{p}': '{:.2f}%' for p in TIMEFRAMES.keys()}
                }),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results",
                data=csv,
                file_name=f"rs_ratings_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Visualizations
            st.subheader("📊 Charts")
            
            tab1, tab2 = st.tabs(["RS Rating", "Returns Heatmap"])
            
            with tab1:
                fig1 = create_rs_chart(results_df, top_n)
                st.plotly_chart(fig1, use_container_width=True)
            
            with tab2:
                fig2 = create_performance_heatmap(results_df, top_n)
                st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👆 Upload CSV to begin")
    st.markdown("""
    **CSV Format:**
    ```
    Symbol
    RELIANCE
    TCS
    INFY
    ```
    """)
