import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from nsepy import get_history
import requests
from datetime import datetime, timedelta
import talib
import warnings
import io
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Minervini SEPA Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


class MinerviniSEPA:
    """Enhanced Mark Minervini SEPA implementation for NSE India"""
    
    def __init__(self, custom_criteria=None):
        self.fundamental_criteria = custom_criteria.get('fundamental', {}) if custom_criteria else {
            'earnings_growth_annual': 25.0,
            'earnings_growth_quarterly': 20.0,
            'sales_growth_quarterly': 20.0,
            'sales_growth_annual': 15.0,
            'roe_minimum': 15.0,
            'debt_equity_max': 0.5,
            'profit_margin_min': 10.0,
            'market_cap_min': 500
        }
        
        self.technical_criteria = custom_criteria.get('technical', {}) if custom_criteria else {}
    
    def get_fundamental_data(self, symbol):
        """Fetch fundamental data for NSE stocks"""
        fundamental_data = {}
        
        try:
            ticker = f"{symbol}.NS"
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info:
                fundamental_data.update({
                    'market_cap': info.get('marketCap', 0) / 10000000,
                    'pe_ratio': info.get('forwardPE', info.get('trailingPE')),
                    'roe': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                    'debt_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                    'profit_margin': info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0,
                    'revenue_growth': info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                    'earnings_growth': info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0,
                    'current_ratio': info.get('currentRatio', 0),
                    'book_value': info.get('bookValue', 0),
                    'dividend_yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                })
                
            try:
                quarterly_financials = stock.quarterly_financials
                annual_financials = stock.financials
                
                if not quarterly_financials.empty and not annual_financials.empty:
                    fundamental_data.update(self._calculate_growth_rates(
                        quarterly_financials, annual_financials, symbol
                    ))
                    
            except Exception as e:
                st.warning(f"Could not fetch detailed financials for {symbol}: {str(e)}")
                
        except Exception as e:
            st.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            
        return fundamental_data
    
    def _calculate_growth_rates(self, quarterly_financials, annual_financials, symbol):
        """Calculate various growth rates from financial data"""
        growth_data = {}
        
        try:
            # Revenue/Sales growth
            if 'Total Revenue' in quarterly_financials.index:
                quarterly_revenue = quarterly_financials.loc['Total Revenue']
                if len(quarterly_revenue) >= 2:
                    latest_q = quarterly_revenue.iloc[0]
                    previous_q = quarterly_revenue.iloc[1]
                    if previous_q != 0:
                        growth_data['sales_growth_quarterly'] = ((latest_q - previous_q) / abs(previous_q)) * 100
                        
                if len(quarterly_revenue) >= 4:
                    current_q = quarterly_revenue.iloc[0]
                    same_q_last_year = quarterly_revenue.iloc[3]
                    if same_q_last_year != 0:
                        growth_data['sales_growth_yoy'] = ((current_q - same_q_last_year) / abs(same_q_last_year)) * 100
            
            if 'Total Revenue' in annual_financials.index:
                annual_revenue = annual_financials.loc['Total Revenue']
                if len(annual_revenue) >= 2:
                    latest_year = annual_revenue.iloc[0]
                    previous_year = annual_revenue.iloc[1]
                    if previous_year != 0:
                        growth_data['sales_growth_annual'] = ((latest_year - previous_year) / abs(previous_year)) * 100
            
            # Net Income growth (earnings)
            if 'Net Income' in quarterly_financials.index:
                quarterly_earnings = quarterly_financials.loc['Net Income']
                if len(quarterly_earnings) >= 2:
                    latest_q = quarterly_earnings.iloc[0]
                    previous_q = quarterly_earnings.iloc[1]
                    if previous_q != 0 and latest_q > 0 and previous_q > 0:
                        growth_data['earnings_growth_quarterly'] = ((latest_q - previous_q) / abs(previous_q)) * 100
                        
                if len(quarterly_earnings) >= 4:
                    current_q = quarterly_earnings.iloc[0]
                    same_q_last_year = quarterly_earnings.iloc[3]
                    if same_q_last_year != 0 and current_q > 0 and same_q_last_year > 0:
                        growth_data['earnings_growth_yoy'] = ((current_q - same_q_last_year) / abs(same_q_last_year)) * 100
            
            if 'Net Income' in annual_financials.index:
                annual_earnings = annual_financials.loc['Net Income']
                if len(annual_earnings) >= 2:
                    latest_year = annual_earnings.iloc[0]
                    previous_year = annual_earnings.iloc[1]
                    if previous_year != 0 and latest_year > 0 and previous_year > 0:
                        growth_data['earnings_growth_annual'] = ((latest_year - previous_year) / abs(previous_year)) * 100
                        
            if 'Net Income' in annual_financials.index:
                annual_earnings = annual_financials.loc['Net Income']
                if len(annual_earnings) >= 4:
                    earnings_list = annual_earnings.head(4).tolist()
                    earnings_list.reverse()
                    
                    if all(e > 0 for e in earnings_list):
                        growth_rates = []
                        for i in range(1, len(earnings_list)):
                            if earnings_list[i-1] != 0:
                                growth_rate = ((earnings_list[i] - earnings_list[i-1]) / abs(earnings_list[i-1])) * 100
                                growth_rates.append(growth_rate)
                        
                        if growth_rates:
                            growth_data['earnings_growth_3yr_avg'] = sum(growth_rates) / len(growth_rates)
                            
        except Exception as e:
            st.warning(f"Error calculating growth rates for {symbol}: {str(e)}")
            
        return growth_data
    
    def check_fundamental_criteria(self, symbol, fundamental_data):
        """Check if stock meets Minervini's fundamental criteria"""
        if not fundamental_data:
            return False, "No fundamental data available", {}
            
        criteria_results = {}
        
        earnings_annual = fundamental_data.get('earnings_growth_annual', 0)
        criteria_results['earnings_growth_annual'] = earnings_annual >= self.fundamental_criteria['earnings_growth_annual']
        
        earnings_quarterly = fundamental_data.get('earnings_growth_quarterly', 0)
        criteria_results['earnings_growth_quarterly'] = earnings_quarterly >= self.fundamental_criteria['earnings_growth_quarterly']
        
        sales_quarterly = fundamental_data.get('sales_growth_quarterly', 0)
        criteria_results['sales_growth_quarterly'] = sales_quarterly >= self.fundamental_criteria['sales_growth_quarterly']
        
        sales_annual = fundamental_data.get('sales_growth_annual', 0)
        criteria_results['sales_growth_annual'] = sales_annual >= self.fundamental_criteria['sales_growth_annual']
        
        roe = fundamental_data.get('roe', 0)
        criteria_results['roe_sufficient'] = roe >= self.fundamental_criteria['roe_minimum']
        
        debt_equity = fundamental_data.get('debt_equity', 999)
        criteria_results['debt_equity_acceptable'] = debt_equity <= self.fundamental_criteria['debt_equity_max']
        
        profit_margin = fundamental_data.get('profit_margin', 0)
        criteria_results['profit_margin_sufficient'] = profit_margin >= self.fundamental_criteria['profit_margin_min']
        
        market_cap = fundamental_data.get('market_cap', 0)
        criteria_results['market_cap_sufficient'] = market_cap >= self.fundamental_criteria['market_cap_min']
        
        current_q_earnings = fundamental_data.get('earnings_growth_quarterly', 0)
        yoy_earnings = fundamental_data.get('earnings_growth_yoy', 0)
        criteria_results['accelerating_earnings'] = current_q_earnings > 0 and yoy_earnings > current_q_earnings
        
        current_ratio = fundamental_data.get('current_ratio', 0)
        criteria_results['current_ratio_healthy'] = current_ratio >= 1.2
        
        fundamental_score = sum(criteria_results.values())
        fundamental_pass = fundamental_score >= 6
        
        return fundamental_pass, criteria_results, fundamental_score
    
    def get_technical_data(self, symbol, start_date, end_date):
        """Fetch technical/price data"""
        try:
            data = get_history(symbol=symbol, start=start_date, end=end_date)
            if data.empty:
                raise Exception("NSEpy failed")
            return data
        except:
            try:
                ticker = f"{symbol}.NS"
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
                return data
            except Exception as e:
                st.error(f"Error fetching technical data for {symbol}: {str(e)}")
                return pd.DataFrame()
    
    def calculate_technical_indicators(self, data):
        """Calculate all technical indicators needed for SEPA"""
        if data.empty:
            return data
            
        data['SMA_10'] = talib.SMA(data['Close'], timeperiod=10)
        data['SMA_30'] = talib.SMA(data['Close'], timeperiod=30)
        data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
        data['SMA_150'] = talib.SMA(data['Close'], timeperiod=150)
        data['SMA_200'] = talib.SMA(data['Close'], timeperiod=200)
        
        data['Volume_SMA_50'] = talib.SMA(data['Volume'], timeperiod=50)
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_50']
        
        data['52_week_high'] = data['High'].rolling(window=252).max()
        data['52_week_low'] = data['Low'].rolling(window=252).min()
        
        data['ATR_14'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
        
        return data
    
    def check_technical_criteria(self, data, symbol):
        """Check technical/trend template criteria"""
        if len(data) < 200:
            return False, "Insufficient technical data", {}, 0
            
        latest = data.iloc[-1]
        results = {}
        
        results['price_above_10_sma'] = latest['Close'] > latest['SMA_10']
        results['price_above_30_sma'] = latest['Close'] > latest['SMA_30']
        results['price_above_50_sma'] = latest['Close'] > latest['SMA_50']
        results['price_above_150_sma'] = latest['Close'] > latest['SMA_150']
        results['price_above_200_sma'] = latest['Close'] > latest['SMA_200']
        
        results['sma_alignment'] = (
            latest['SMA_10'] > latest['SMA_30'] > latest['SMA_50'] > 
            latest['SMA_150'] > latest['SMA_200']
        )
        
        if len(data) >= 220:
            results['sma_200_trending_up'] = latest['SMA_200'] > data.iloc[-20]['SMA_200']
        else:
            results['sma_200_trending_up'] = False
            
        results['price_within_25_percent_of_52wk_high'] = latest['Close'] >= (latest['52_week_high'] * 0.75)
        results['price_above_30_percent_of_52wk_low'] = latest['Close'] >= (latest['52_week_low'] * 1.30)
        
        recent_volume_avg = data['Volume'].tail(10).mean()
        earlier_volume_avg = data['Volume_SMA_50'].iloc[-1]
        results['volume_above_average'] = recent_volume_avg > earlier_volume_avg
        
        vcp_detected, _ = self.detect_vcp_pattern(data)
        results['vcp_pattern'] = vcp_detected
        
        recent_data = data.tail(20)
        avg_close_position = ((recent_data['Close'] - recent_data['Low']) / 
                            (recent_data['High'] - recent_data['Low'])).mean()
        results['strong_price_action'] = avg_close_position > 0.6
        
        technical_score = sum(results.values())
        technical_pass = technical_score >= 6
        
        return technical_pass, results, technical_score
    
    def detect_vcp_pattern(self, data, lookback_period=50):
        """Enhanced VCP pattern detection"""
        if len(data) < lookback_period:
            return False, 0
            
        recent_data = data.tail(lookback_period)
        
        high_low = recent_data['High'] - recent_data['Low']
        high_close = np.abs(recent_data['High'] - recent_data['Close'].shift())
        low_close = np.abs(recent_data['Low'] - recent_data['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(window=14).mean()
        
        latest_atr = atr.iloc[-5:].mean()
        earlier_atr = atr.iloc[-25:-20].mean()
        volatility_contracting = latest_atr < earlier_atr * 0.8
        
        support_level = recent_data['SMA_50'].iloc[-1]
        price_above_support = recent_data['Low'].iloc[-10:].min() > support_level * 0.95
        
        pullback_periods = recent_data[recent_data['Close'] < recent_data['Close'].shift()].tail(5)
        if len(pullback_periods) > 0:
            pullback_volume = pullback_periods['Volume'].mean()
            up_periods = recent_data[recent_data['Close'] > recent_data['Close'].shift()].tail(5)
            up_volume = up_periods['Volume'].mean() if len(up_periods) > 0 else pullback_volume
            volume_dry_up = pullback_volume < up_volume
        else:
            volume_dry_up = True
            
        vcp_score = sum([volatility_contracting, price_above_support, volume_dry_up])
        return vcp_score >= 2, vcp_score
    
    def calculate_relative_strength(self, stock_data, market_data):
        """Calculate relative strength vs market"""
        if len(stock_data) != len(market_data):
            min_len = min(len(stock_data), len(market_data))
            stock_data = stock_data.tail(min_len)
            market_data = market_data.tail(min_len)
        
        rs_line = stock_data['Close'] / market_data['Close']
        rs_momentum = rs_line / rs_line.shift(63)
        rs_rating = ((rs_momentum.rank(pct=True).iloc[-1]) * 100) if len(rs_momentum) > 0 else 50
        
        return rs_rating
    
    def calculate_precise_entry_points(self, data):
        """Calculate precise entry points and risk management levels"""
        latest = data.iloc[-1]
        
        lookback = min(30, len(data))
        recent_data = data.tail(lookback)
        
        resistance = recent_data['High'].max()
        support_50sma = latest['SMA_50']
        support_30sma = latest['SMA_30']
        
        pivot_entry = resistance * 1.015
        pullback_entry = support_30sma * 1.005
        stop_loss = support_30sma * 0.92
        
        current_price = latest['Close']
        risk_per_share = current_price - stop_loss
        risk_reward_ratio = abs((pivot_entry * 1.20) - pivot_entry) / risk_per_share if risk_per_share > 0 else 0
        
        return {
            'pivot_entry': round(pivot_entry, 2),
            'pullback_entry': round(pullback_entry, 2),
            'stop_loss': round(stop_loss, 2),
            'current_price': round(current_price, 2),
            'risk_per_share': round(risk_per_share, 2),
            'risk_reward_ratio': round(risk_reward_ratio, 2),
            'position_size_suggestion': '2-3% of portfolio max'
        }
    
    def complete_sepa_analysis(self, symbol, market_symbol='NIFTY 50', days_back=365):
        """Complete SEPA analysis combining fundamentals and technicals"""
        
        analysis_result = {
            'symbol': symbol,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sepa_score': 0,
            'recommendation': 'NOT QUALIFIED'
        }
        
        # Fundamental Analysis
        fundamental_data = self.get_fundamental_data(symbol)
        if fundamental_data:
            fundamental_pass, fund_criteria, fund_score = self.check_fundamental_criteria(symbol, fundamental_data)
            analysis_result['fundamental_data'] = fundamental_data
            analysis_result['fundamental_pass'] = fundamental_pass
            analysis_result['fundamental_criteria'] = fund_criteria
            analysis_result['fundamental_score'] = fund_score
        else:
            return analysis_result
        
        # Technical Analysis
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back + 50)
        
        stock_data = self.get_technical_data(symbol, start_date, end_date)
        if stock_data.empty:
            return analysis_result
            
        stock_data = self.calculate_technical_indicators(stock_data)
        technical_pass, tech_criteria, tech_score = self.check_technical_criteria(stock_data, symbol)
        
        analysis_result['technical_pass'] = technical_pass
        analysis_result['technical_criteria'] = tech_criteria
        analysis_result['technical_score'] = tech_score
        analysis_result['current_price'] = round(stock_data['Close'].iloc[-1], 2)
        
        # Calculate relative strength
        market_data = self.get_technical_data(market_symbol, start_date, end_date)
        if not market_data.empty:
            rs_rating = self.calculate_relative_strength(stock_data, market_data)
            analysis_result['rs_rating'] = round(rs_rating, 1)
            tech_criteria['rs_rating_above_70'] = rs_rating > 70
        
        # Entry Point Analysis
        if fundamental_pass and technical_pass:
            entry_analysis = self.calculate_precise_entry_points(stock_data)
            analysis_result['entry_analysis'] = entry_analysis
        
        # Overall SEPA Rating
        sepa_score = (fund_score * 0.4) + (tech_score * 0.6)
        analysis_result['sepa_score'] = round(sepa_score, 1)
        
        # Final recommendation
        if fundamental_pass and technical_pass:
            if sepa_score >= 8.0:
                analysis_result['recommendation'] = 'STRONG BUY'
            elif sepa_score >= 6.5:
                analysis_result['recommendation'] = 'BUY'
            else:
                analysis_result['recommendation'] = 'QUALIFIED - MONITOR'
        elif fundamental_pass or technical_pass:
            analysis_result['recommendation'] = 'PARTIAL QUALIFICATION - WATCH'
        
        return analysis_result


# Session state management
def init_session_state():
    """Initialize session state variables"""
    if 'symbol_history' not in st.session_state:
        st.session_state.symbol_history = []
    if 'uploaded_symbols' not in st.session_state:
        st.session_state.uploaded_symbols = []
    if 'screening_results' not in st.session_state:
        st.session_state.screening_results = []
    if 'custom_criteria' not in st.session_state:
        st.session_state.custom_criteria = {
            'fundamental': {
                'earnings_growth_annual': 25.0,
                'earnings_growth_quarterly': 20.0,
                'sales_growth_quarterly': 20.0,
                'sales_growth_annual': 15.0,
                'roe_minimum': 15.0,
                'debt_equity_max': 0.5,
                'profit_margin_min': 10.0,
                'market_cap_min': 500
            }
        }


def save_uploaded_symbols(symbols):
    """Save uploaded symbols to session state and local storage"""
    st.session_state.uploaded_symbols = symbols
    # Save to file for persistence
    with open('uploaded_symbols.json', 'w') as f:
        json.dump(symbols, f)


def load_uploaded_symbols():
    """Load symbols from local storage"""
    try:
        if os.path.exists('uploaded_symbols.json'):
            with open('uploaded_symbols.json', 'r') as f:
                symbols = json.load(f)
                st.session_state.uploaded_symbols = symbols
                return symbols
    except:
        pass
    return []


def main():
    """Main Streamlit application"""
    
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">📈 Minervini SEPA Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Stage Analysis & Precise Entry Points for NSE India</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1f77b4/ffffff?text=SEPA", use_column_width=True)
        st.markdown("---")
        
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Single Stock", "Bulk Screening", "Custom Criteria"],
            help="Choose your analysis approach"
        )
        
        st.markdown("---")
        
        # Market selection
        market_index = st.selectbox(
            "Market Index for RS Calculation",
            ["NIFTY 50", "NIFTY BANK", "NIFTY IT", "NIFTY AUTO"],
            help="Select market index for relative strength calculation"
        )
        
        # Days back for analysis
        days_back = st.slider(
            "Historical Data (Days)",
            min_value=180,
            max_value=730,
            value=365,
            step=30,
            help="Number of days of historical data to analyze"
        )
        
        st.markdown("---")
        
        # File upload for symbols
        st.subheader("📁 Upload Stock List")
        uploaded_file = st.file_uploader(
            "Upload CSV with stock symbols",
            type=['csv'],
            help="CSV should have a column named 'Symbol' or 'SYMBOL'"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                # Try different column name variations
                symbol_col = None
                for col in df.columns:
                    if col.upper() in ['SYMBOL', 'SYMBOLS', 'TICKER', 'STOCK']:
                        symbol_col = col
                        break
                
                if symbol_col:
                    symbols = df[symbol_col].dropna().unique().tolist()
                    save_uploaded_symbols(symbols)
                    st.success(f"✅ Loaded {len(symbols)} symbols")
                    
                    with st.expander("View Uploaded Symbols"):
                        st.write(symbols[:20])
                        if len(symbols) > 20:
                            st.info(f"... and {len(symbols) - 20} more")
                else:
                    st.error("❌ No 'Symbol' column found in CSV")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Load previously uploaded symbols
        if st.button("📂 Load Previous Upload"):
            symbols = load_uploaded_symbols()
            if symbols:
                st.success(f"✅ Loaded {len(symbols)} symbols from previous session")
            else:
                st.info("No previous upload found")
        
        st.markdown("---")
        
        # Download sample CSV
        sample_df = pd.DataFrame({
            'Symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
        })
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_symbols.csv",
            mime="text/csv"
        )
    
    # Main content
    if analysis_mode == "Single Stock":
        render_single_stock_analysis(market_index, days_back)
    
    elif analysis_mode == "Bulk Screening":
        render_bulk_screening(market_index, days_back)
    
    elif analysis_mode == "Custom Criteria":
        render_custom_criteria(market_index, days_back)


def render_single_stock_analysis(market_index, days_back):
    """Render single stock analysis interface"""
    
    st.header("🔍 Single Stock Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol (NSE)",
            value="RELIANCE",
            help="Enter NSE stock symbol without .NS suffix"
        ).upper()
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    if analyze_btn and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            sepa = MinerviniSEPA(st.session_state.custom_criteria)
            result = sepa.complete_sepa_analysis(symbol, market_index, days_back)
            
            if result and result['sepa_score'] > 0:
                # Add to history
                if symbol not in st.session_state.symbol_history:
                    st.session_state.symbol_history.append(symbol)
                
                # Display results
                display_single_stock_results(result)
            else:
                st.error("❌ Unable to analyze stock. Please check symbol and try again.")


def display_single_stock_results(result):
    """Display detailed results for single stock analysis"""
    
    # Overall Score Card
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SEPA Score", f"{result['sepa_score']}/10")
    with col2:
        st.metric("Current Price", f"₹{result['current_price']}")
    with col3:
        st.metric("RS Rating", f"{result.get('rs_rating', 'N/A')}")
    with col4:
        recommendation = result['recommendation']
        if 'STRONG BUY' in recommendation:
            st.markdown('<div class="success-box"><b>✅ ' + recommendation + '</b></div>', unsafe_allow_html=True)
        elif 'BUY' in recommendation:
            st.markdown('<div class="success-box"><b>✓ ' + recommendation + '</b></div>', unsafe_allow_html=True)
        elif 'MONITOR' in recommendation or 'WATCH' in recommendation:
            st.markdown('<div class="warning-box"><b>⚠️ ' + recommendation + '</b></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="danger-box"><b>❌ ' + recommendation + '</b></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Fundamental", "📈 Technical", "🎯 Entry Points", "📋 Summary"])
    
    with tab1:
        st.subheader("Fundamental Analysis")
        
        if 'fundamental_data' in result:
            fund_data = result['fundamental_data']
            fund_criteria = result.get('fundamental_criteria', {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Market Cap", f"₹{fund_data.get('market_cap', 0):.0f} Cr")
                st.metric("ROE", f"{fund_data.get('roe', 0):.1f}%")
                st.metric("P/E Ratio", f"{fund_data.get('pe_ratio', 'N/A')}")
            
            with col2:
                st.metric("Earnings Growth (Annual)", f"{fund_data.get('earnings_growth_annual', 0):.1f}%")
                st.metric("Sales Growth (Quarterly)", f"{fund_data.get('sales_growth_quarterly', 0):.1f}%")
                st.metric("Profit Margin", f"{fund_data.get('profit_margin', 0):.1f}%")
            
            with col3:
                st.metric("Debt/Equity", f"{fund_data.get('debt_equity', 0):.2f}")
                st.metric("Current Ratio", f"{fund_data.get('current_ratio', 0):.2f}")
                st.metric("Dividend Yield", f"{fund_data.get('dividend_yield', 0):.2f}%")
            
            st.markdown("---")
            st.subheader("Fundamental Criteria Check")
            
            criteria_df = pd.DataFrame([
                {"Criteria": "Earnings Growth (Annual)", "Status": "✅" if fund_criteria.get('earnings_growth_annual') else "❌"},
                {"Criteria": "Earnings Growth (Quarterly)", "Status": "✅" if fund_criteria.get('earnings_growth_quarterly') else "❌"},
                {"Criteria": "Sales Growth (Quarterly)", "Status": "✅" if fund_criteria.get('sales_growth_quarterly') else "❌"},
                {"Criteria": "Sales Growth (Annual)", "Status": "✅" if fund_criteria.get('sales_growth_annual') else "❌"},
                {"Criteria": "ROE Sufficient", "Status": "✅" if fund_criteria.get('roe_sufficient') else "❌"},
                {"Criteria": "Debt/Equity Acceptable", "Status": "✅" if fund_criteria.get('debt_equity_acceptable') else "❌"},
                {"Criteria": "Profit Margin Sufficient", "Status": "✅" if fund_criteria.get('profit_margin_sufficient') else "❌"},
                {"Criteria": "Market Cap Sufficient", "Status": "✅" if fund_criteria.get('market_cap_sufficient') else "❌"},
                {"Criteria": "Accelerating Earnings", "Status": "✅" if fund_criteria.get('accelerating_earnings') else "❌"},
                {"Criteria": "Current Ratio Healthy", "Status": "✅" if fund_criteria.get('current_ratio_healthy') else "❌"},
            ])
            
            st.dataframe(criteria_df, use_container_width=True, hide_index=True)
            
            fund_score = result.get('fundamental_score', 0)
            st.progress(fund_score / 10)
            st.caption(f"Fundamental Score: {fund_score}/10 - {'✅ PASS' if result.get('fundamental_pass') else '❌ FAIL'}")
    
    with tab2:
        st.subheader("Technical Analysis")
        
        if 'technical_criteria' in result:
            tech_criteria = result['technical_criteria']
            
            st.markdown("##### Trend Template Criteria")
            
            criteria_df = pd.DataFrame([
                {"Criteria": "Price Above 10 SMA", "Status": "✅" if tech_criteria.get('price_above_10_sma') else "❌"},
                {"Criteria": "Price Above 30 SMA", "Status": "✅" if tech_criteria.get('price_above_30_sma') else "❌"},
                {"Criteria": "Price Above 50 SMA", "Status": "✅" if tech_criteria.get('price_above_50_sma') else "❌"},
                {"Criteria": "Price Above 150 SMA", "Status": "✅" if tech_criteria.get('price_above_150_sma') else "❌"},
                {"Criteria": "Price Above 200 SMA", "Status": "✅" if tech_criteria.get('price_above_200_sma') else "❌"},
                {"Criteria": "SMA Alignment (Bullish)", "Status": "✅" if tech_criteria.get('sma_alignment') else "❌"},
                {"Criteria": "200 SMA Trending Up", "Status": "✅" if tech_criteria.get('sma_200_trending_up') else "❌"},
                {"Criteria": "Within 25% of 52W High", "Status": "✅" if tech_criteria.get('price_within_25_percent_of_52wk_high') else "❌"},
                {"Criteria": "30% Above 52W Low", "Status": "✅" if tech_criteria.get('price_above_30_percent_of_52wk_low') else "❌"},
                {"Criteria": "Volume Above Average", "Status": "✅" if tech_criteria.get('volume_above_average') else "❌"},
                {"Criteria": "VCP Pattern Detected", "Status": "✅" if tech_criteria.get('vcp_pattern') else "❌"},
                {"Criteria": "Strong Price Action", "Status": "✅" if tech_criteria.get('strong_price_action') else "❌"},
                {"Criteria": "RS Rating > 70", "Status": "✅" if tech_criteria.get('rs_rating_above_70') else "❌"},
            ])
            
            st.dataframe(criteria_df, use_container_width=True, hide_index=True)
            
            tech_score = result.get('technical_score', 0)
            st.progress(tech_score / 13)
            st.caption(f"Technical Score: {tech_score}/13 - {'✅ PASS' if result.get('technical_pass') else '❌ FAIL'}")
    
    with tab3:
        st.subheader("Entry Points & Risk Management")
        
        if 'entry_analysis' in result:
            entry = result['entry_analysis']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Entry Points")
                st.metric("Current Price", f"₹{entry['current_price']}")
                st.metric("Pivot Entry (Breakout)", f"₹{entry['pivot_entry']}", 
                         delta=f"+{((entry['pivot_entry'] - entry['current_price']) / entry['current_price'] * 100):.1f}%")
                st.metric("Pullback Entry", f"₹{entry['pullback_entry']}", 
                         delta=f"{((entry['pullback_entry'] - entry['current_price']) / entry['current_price'] * 100):.1f}%")
            
            with col2:
                st.markdown("##### Risk Management")
                st.metric("Stop Loss", f"₹{entry['stop_loss']}", 
                         delta=f"{((entry['stop_loss'] - entry['current_price']) / entry['current_price'] * 100):.1f}%",
                         delta_color="inverse")
                st.metric("Risk per Share", f"₹{entry['risk_per_share']}")
                st.metric("Risk/Reward Ratio", f"{entry['risk_reward_ratio']}:1")
            
            st.markdown("---")
            st.info(f"💡 **Position Size Suggestion:** {entry['position_size_suggestion']}")
            
            # Trading plan
            st.markdown("##### Suggested Trading Plan")
            st.markdown(f"""
            1. **Buy Zone:** ₹{entry['pullback_entry']} - ₹{entry['pivot_entry']}
            2. **Initial Stop Loss:** ₹{entry['stop_loss']}
            3. **First Target:** ₹{entry['pivot_entry'] * 1.15:.2f} (+15%)
            4. **Second Target:** ₹{entry['pivot_entry'] * 1.30:.2f} (+30%)
            5. **Trailing Stop:** Move to breakeven after +10% gain
            """)
        else:
            st.warning("⚠️ Stock doesn't meet both fundamental and technical criteria for entry point calculation.")
    
    with tab4:
        st.subheader("Analysis Summary")
        
        # Create summary table
        summary_data = {
            "Metric": ["Symbol", "Analysis Date", "Current Price", "SEPA Score", "Fundamental Score", 
                       "Technical Score", "RS Rating", "Recommendation"],
            "Value": [
                result['symbol'],
                result['analysis_date'],
                f"₹{result['current_price']}",
                f"{result['sepa_score']}/10",
                f"{result.get('fundamental_score', 0)}/10",
                f"{result.get('technical_score', 0)}/13",
                result.get('rs_rating', 'N/A'),
                result['recommendation']
            ]
        }
        
        st.table(pd.DataFrame(summary_data))
        
        # Export options
        st.markdown("---")
        st.subheader("Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as JSON
            json_str = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"{result['symbol']}_SEPA_analysis.json",
                mime="application/json"
            )
        
        with col2:
            # Export as CSV
            export_df = pd.DataFrame([{
                'Symbol': result['symbol'],
                'Date': result['analysis_date'],
                'Price': result['current_price'],
                'SEPA_Score': result['sepa_score'],
                'Recommendation': result['recommendation']
            }])
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{result['symbol']}_SEPA_analysis.csv",
                mime="text/csv"
            )


def render_bulk_screening(market_index, days_back):
    """Render bulk screening interface"""
    
    st.header("📊 Bulk Stock Screening")
    
    # Source selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        source = st.radio(
            "Stock List Source",
            ["Manual Entry", "Uploaded File", "Predefined Lists"],
            horizontal=True
        )
    
    with col2:
        min_sepa_score = st.slider("Min SEPA Score", 0.0, 10.0, 6.0, 0.5)
    
    stock_list = []
    
    if source == "Manual Entry":
        symbols_input = st.text_area(
            "Enter symbols (comma-separated)",
            value="RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK",
            help="Enter NSE symbols separated by commas"
        )
        stock_list = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    
    elif source == "Uploaded File":
        if st.session_state.uploaded_symbols:
            st.success(f"✅ Using {len(st.session_state.uploaded_symbols)} uploaded symbols")
            stock_list = st.session_state.uploaded_symbols
            
            with st.expander("View Symbols"):
                st.write(stock_list[:50])
                if len(stock_list) > 50:
                    st.info(f"... and {len(stock_list) - 50} more")
        else:
            st.warning("⚠️ No symbols uploaded. Please upload a CSV file from the sidebar.")
    
    elif source == "Predefined Lists":
        predefined = st.selectbox(
            "Select List",
            ["Nifty 50 Sample", "Nifty Bank Sample", "Nifty IT Sample", "Top Growth Stocks"]
        )
        
        if predefined == "Nifty 50 Sample":
            stock_list = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 
                         'ICICIBANK', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT']
        elif predefined == "Nifty Bank Sample":
            stock_list = ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 
                         'BANDHANBNK', 'FEDERALBNK', 'INDUSINDBK']
        elif predefined == "Nifty IT Sample":
            stock_list = ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'LTIM', 'COFORGE']
        else:
            stock_list = ['ADANIPORTS', 'BAJAJFINSV', 'TATAMOTORS', 'TITAN', 'BHARTIARTL']
    
    st.info(f"📋 Ready to screen {len(stock_list)} stocks")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        screen_btn = st.button("🚀 Start Screening", type="primary", use_container_width=True)
    
    if screen_btn and stock_list:
        screen_stocks(stock_list, min_sepa_score, market_index, days_back)


def screen_stocks(stock_list, min_sepa_score, market_index, days_back):
    """Screen multiple stocks"""
    
    qualified_stocks = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    sepa = MinerviniSEPA(st.session_state.custom_criteria)
    
    for i, symbol in enumerate(stock_list):
        progress = (i + 1) / len(stock_list)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing {symbol}... ({i+1}/{len(stock_list)})")
        
        try:
            result = sepa.complete_sepa_analysis(symbol, market_index, days_back)
            
            if result and result['sepa_score'] >= min_sepa_score:
                qualified_stocks.append(result)
        except Exception as e:
            st.warning(f"⚠️ Error analyzing {symbol}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    # Store results in session state
    st.session_state.screening_results = qualified_stocks
    
    # Display results
    display_screening_results(qualified_stocks, len(stock_list))


def display_screening_results(qualified_stocks, total_screened):
    """Display bulk screening results"""
    
    st.markdown("---")
    st.subheader("Screening Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stocks Screened", total_screened)
    with col2:
        st.metric("Qualified Stocks", len(qualified_stocks))
    with col3:
        qualification_rate = (len(qualified_stocks) / total_screened * 100) if total_screened > 0 else 0
        st.metric("Qualification Rate", f"{qualification_rate:.1f}%")
    with col4:
        if qualified_stocks:
            avg_score = sum(s['sepa_score'] for s in qualified_stocks) / len(qualified_stocks)
            st.metric("Avg SEPA Score", f"{avg_score:.1f}")
    
    if qualified_stocks:
        # Sort by SEPA score
        qualified_stocks.sort(key=lambda x: x['sepa_score'], reverse=True)
        
        st.markdown("---")
        st.subheader("📈 Top Qualified Stocks")
        
        # Create results dataframe
        results_data = []
        for stock in qualified_stocks:
            entry = stock.get('entry_analysis', {})
            results_data.append({
                'Symbol': stock['symbol'],
                'SEPA Score': stock['sepa_score'],
                'Price': f"₹{stock['current_price']}",
                'RS Rating': stock.get('rs_rating', 'N/A'),
                'Pivot Entry': f"₹{entry.get('pivot_entry', 'N/A')}" if entry else 'N/A',
                'Stop Loss': f"₹{entry.get('stop_loss', 'N/A')}" if entry else 'N/A',
                'R/R Ratio': f"{entry.get('risk_reward_ratio', 'N/A')}:1" if entry else 'N/A',
                'Recommendation': stock['recommendation']
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Display with color coding
        def highlight_recommendation(row):
            if 'STRONG BUY' in row['Recommendation']:
                return ['background-color: #d4edda'] * len(row)
            elif 'BUY' in row['Recommendation']:
                return ['background-color: #d1ecf1'] * len(row)
            elif 'MONITOR' in row['Recommendation'] or 'WATCH' in row['Recommendation']:
                return ['background-color: #fff3cd'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = results_df.style.apply(highlight_recommendation, axis=1)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Export options
        st.markdown("---")
        st.subheader("Export Watchlist")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"SEPA_Watchlist_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_str = json.dumps(qualified_stocks, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"SEPA_Watchlist_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Create Excel-ready format
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    results_df.to_excel(writer, index=False, sheet_name='Watchlist')
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_buffer,
                    file_name=f"SEPA_Watchlist_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except:
                st.info("Install openpyxl for Excel export")
        
        # Detailed view of top stocks
        with st.expander("🔍 View Detailed Analysis of Top 3 Stocks"):
            for i, stock in enumerate(qualified_stocks[:3], 1):
                st.markdown(f"### #{i} {stock['symbol']}")
                display_mini_analysis(stock)
                if i < 3:
                    st.markdown("---")
    
    else:
        st.warning("😔 No stocks qualified with the current criteria. Try adjusting the minimum SEPA score or screening criteria.")


def display_mini_analysis(stock):
    """Display condensed analysis for a stock"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SEPA Score", f"{stock['sepa_score']}/10")
    with col2:
        st.metric("Price", f"₹{stock['current_price']}")
    with col3:
        st.metric("RS Rating", stock.get('rs_rating', 'N/A'))
    with col4:
        st.metric("Recommendation", stock['recommendation'])
    
    if 'entry_analysis' in stock:
        entry = stock['entry_analysis']
        st.markdown(f"""
        **Entry Points:** Breakout at ₹{entry['pivot_entry']} | Pullback at ₹{entry['pullback_entry']}  
        **Stop Loss:** ₹{entry['stop_loss']} | **Risk/Reward:** {entry['risk_reward_ratio']}:1
        """)


def render_custom_criteria(market_index, days_back):
    """Render custom criteria configuration interface"""
    
    st.header("⚙️ Custom Screening Criteria")
    
    st.info("💡 Adjust the criteria below to create your own screening parameters. Changes apply to all analyses.")
    
    tab1, tab2, tab3 = st.tabs(["Fundamental Criteria", "Presets", "Test Custom Criteria"])
    
    with tab1:
        st.subheader("Fundamental Analysis Criteria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Growth Metrics")
            earnings_annual = st.number_input(
                "Min Earnings Growth (Annual %)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.custom_criteria['fundamental']['earnings_growth_annual'],
                step=5.0
            )
            
            earnings_quarterly = st.number_input(
                "Min Earnings Growth (Quarterly %)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.custom_criteria['fundamental']['earnings_growth_quarterly'],
                step=5.0
            )
            
            sales_quarterly = st.number_input(
                "Min Sales Growth (Quarterly %)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.custom_criteria['fundamental']['sales_growth_quarterly'],
                step=5.0
            )
            
            sales_annual = st.number_input(
                "Min Sales Growth (Annual %)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.custom_criteria['fundamental']['sales_growth_annual'],
                step=5.0
            )
        
        with col2:
            st.markdown("##### Financial Health Metrics")
            roe_minimum = st.number_input(
                "Min ROE (%)",
                min_value=0.0,
                max_value=50.0,
                value=st.session_state.custom_criteria['fundamental']['roe_minimum'],
                step=5.0
            )
            
            debt_equity_max = st.number_input(
                "Max Debt/Equity Ratio",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.custom_criteria['fundamental']['debt_equity_max'],
                step=0.1
            )
            
            profit_margin_min = st.number_input(
                "Min Profit Margin (%)",
                min_value=0.0,
                max_value=50.0,
                value=st.session_state.custom_criteria['fundamental']['profit_margin_min'],
                step=5.0
            )
            
            market_cap_min = st.number_input(
                "Min Market Cap (₹ Crores)",
                min_value=0,
                max_value=10000,
                value=int(st.session_state.custom_criteria['fundamental']['market_cap_min']),
                step=100
            )
        
        if st.button("💾 Save Custom Criteria", type="primary"):
            st.session_state.custom_criteria['fundamental'] = {
                'earnings_growth_annual': earnings_annual,
                'earnings_growth_quarterly': earnings_quarterly,
                'sales_growth_quarterly': sales_quarterly,
                'sales_growth_annual': sales_annual,
                'roe_minimum': roe_minimum,
                'debt_equity_max': debt_equity_max,
                'profit_margin_min': profit_margin_min,
                'market_cap_min': market_cap_min
            }
            st.success("✅ Custom criteria saved successfully!")
    
    with tab2:
        st.subheader("Preset Configurations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔥 Aggressive Growth", use_container_width=True):
                st.session_state.custom_criteria['fundamental'] = {
                    'earnings_growth_annual': 35.0,
                    'earnings_growth_quarterly': 30.0,
                    'sales_growth_quarterly': 30.0,
                    'sales_growth_annual': 25.0,
                    'roe_minimum': 20.0,
                    'debt_equity_max': 0.3,
                    'profit_margin_min': 15.0,
                    'market_cap_min': 1000
                }
                st.success("✅ Aggressive Growth preset applied!")
        
        with col2:
            if st.button("⚖️ Balanced (Default)", use_container_width=True):
                st.session_state.custom_criteria['fundamental'] = {
                    'earnings_growth_annual': 25.0,
                    'earnings_growth_quarterly': 20.0,
                    'sales_growth_quarterly': 20.0,
                    'sales_growth_annual': 15.0,
                    'roe_minimum': 15.0,
                    'debt_equity_max': 0.5,
                    'profit_margin_min': 10.0,
                    'market_cap_min': 500
                }
                st.success("✅ Balanced preset applied!")
        
        with col3:
            if st.button("🛡️ Conservative", use_container_width=True):
                st.session_state.custom_criteria['fundamental'] = {
                    'earnings_growth_annual': 15.0,
                    'earnings_growth_quarterly': 12.0,
                    'sales_growth_quarterly': 12.0,
                    'sales_growth_annual': 10.0,
                    'roe_minimum': 12.0,
                    'debt_equity_max': 0.3,
                    'profit_margin_min': 8.0,
                    'market_cap_min': 2000
                }
                st.success("✅ Conservative preset applied!")
        
        st.markdown("---")
        st.markdown("##### Current Criteria")
        criteria_display = pd.DataFrame([
            {"Parameter": k.replace('_', ' ').title(), "Value": v}
            for k, v in st.session_state.custom_criteria['fundamental'].items()
        ])
        st.table(criteria_display)
    
    with tab3:
        st.subheader("Test Custom Criteria")
        st.write("Test your custom criteria on a stock to see how it performs.")
        
        test_symbol = st.text_input("Enter Stock Symbol", value="RELIANCE").upper()
        
        if st.button("🧪 Test Criteria"):
            with st.spinner(f"Testing criteria on {test_symbol}..."):
                sepa = MinerviniSEPA(st.session_state.custom_criteria)
                result = sepa.complete_sepa_analysis(test_symbol, market_index, days_back)
                
                if result and result['sepa_score'] > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("SEPA Score", f"{result['sepa_score']}/10")
                    with col2:
                        st.metric("Fundamental Score", f"{result.get('fundamental_score', 0)}/10")
                    with col3:
                        st.metric("Technical Score", f"{result.get('technical_score', 0)}/13")
                    
                    st.success(f"✅ {test_symbol} - {result['recommendation']}")
                    
                    if 'fundamental_criteria' in result:
                        with st.expander("View Detailed Criteria Check"):
                            fund_criteria = result['fundamental_criteria']
                            for criterion, passed in fund_criteria.items():
                                status = "✅" if passed else "❌"
                                st.write(f"{status} {criterion.replace('_', ' ').title()}")


# Footer
def render_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p><b>Minervini SEPA Analyzer</b> | Built for NSE India Stock Analysis</p>
        <p style='font-size: 0.8rem;'>⚠️ This tool is for educational purposes only. Not financial advice. Always do your own research.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    render_footer()
