def screen_stocks(stock_list, min_sepa_score, market_index, days_back):
    """Screen multiple stocks with proper RS calculation against universe"""
    
    qualified_stocks = []
    
    # Phase 1: Build the universe data (fetch all stock data first)
    st.info("📊 Phase 1: Building stock universe for RS calculation...")
    universe_progress = st.progress(0)
    universe_status = st.empty()
    
    universe_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 50)
    
    sepa = MinerviniSEPA(st.session_state.custom_criteria)
    
    for i, symbol in enumerate(stock_list):
        progress = (i + 1) / len(stock_list)
        universe_progress.progress(progress)
        universe_status.text(f"Fetching data for {symbol}... ({i+1}/{len(stock_list)})")
        
        try:
            stock_data = sepa.get_technical_data(symbol, start_date, end_date)
            if not stock_data.empty:
                stock_data = sepa.calculate_technical_indicators(stock_data)
                universe_data[symbol] = stock_data
        except Exception as e:
            st.warning(f"⚠️ Could not fetch data for {symbol}: {str(e)}")
            continue
    
    universe_progress.empty()
    universe_status.empty()
    
    st.success(fimport streamlit as st
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
    
    def check_technical_criteria(self, data, symbol, rs_rating=None):
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
        
        # Enhanced VCP detection with quality
        vcp_detected, vcp_score, vcp_details = self.detect_vcp_pattern(data)
        results['vcp_pattern'] = vcp_detected
        results['vcp_quality_score'] = vcp_score
        results['vcp_details'] = vcp_details
        
        recent_data = data.tail(20)
        avg_close_position = ((recent_data['Close'] - recent_data['Low']) / 
                            (recent_data['High'] - recent_data['Low'])).mean()
        results['strong_price_action'] = avg_close_position > 0.6
        
        # Use universe-based RS rating if provided
        if rs_rating is not None:
            results['rs_rating_above_70'] = rs_rating > 70
            results['rs_rating_above_80'] = rs_rating > 80  # Elite performers
        
        technical_score = sum([v for k, v in results.items() 
                              if isinstance(v, bool)])
        technical_pass = technical_score >= 6
        
        return technical_pass, results, technical_score
    
    def detect_vcp_pattern(self, data, lookback_period=50):
        """
        Enhanced VCP pattern detection with quality scoring
        Based on Mark Minervini's criteria from "Trade Like a Stock Market Wizard"
        
        VCP Quality Criteria:
        1. Volatility Contraction (3+ contractions, each smaller than previous)
        2. Volume Dry-Up (volume decreases on pullbacks)
        3. Price Structure (higher lows, tightening)
        4. Base Depth (shallower pullbacks as pattern progresses)
        5. Time Structure (proper base formation time)
        """
        if len(data) < lookback_period:
            return False, 0, {}
            
        recent_data = data.tail(lookback_period)
        
        vcp_quality = {
            'pattern_detected': False,
            'quality_score': 0,  # 0-100 scale
            'stage': 'Unknown',
            'contractions': [],
            'volatility_trend': 'Unknown',
            'volume_behavior': 'Unknown',
            'base_structure': 'Unknown',
            'base_depth': 0,
            'recommendation': 'Not a VCP'
        }
        
        # 1. VOLATILITY ANALYSIS
        # Calculate Average True Range (ATR) for volatility
        high_low = recent_data['High'] - recent_data['Low']
        high_close = np.abs(recent_data['High'] - recent_data['Close'].shift())
        low_close = np.abs(recent_data['Low'] - recent_data['Close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = tr.rolling(window=14).mean()
        
        # Identify contraction phases (divide into segments)
        segment_size = lookback_period // 5
        contractions = []
        
        for i in range(5):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size
            if end_idx <= len(atr):
                segment_atr = atr.iloc[start_idx:end_idx].mean()
                contractions.append(segment_atr)
        
        # Check if volatility is contracting (each phase smaller)
        volatility_contracting = all(contractions[i] > contractions[i+1] 
                                     for i in range(len(contractions)-1) 
                                     if not pd.isna(contractions[i]) and not pd.isna(contractions[i+1]))
        
        if volatility_contracting:
            vcp_quality['quality_score'] += 25
            vcp_quality['volatility_trend'] = 'Contracting ✓'
            vcp_quality['contractions'] = [round(c, 2) for c in contractions if not pd.isna(c)]
        else:
            vcp_quality['volatility_trend'] = 'Not Contracting ✗'
        
        # 2. VOLUME ANALYSIS
        # Volume should dry up on pullbacks and surge on advances
        pullback_periods = recent_data[recent_data['Close'] < recent_data['Close'].shift()]
        advance_periods = recent_data[recent_data['Close'] > recent_data['Close'].shift()]
        
        if len(pullback_periods) > 0 and len(advance_periods) > 0:
            avg_pullback_volume = pullback_periods['Volume'].mean()
            avg_advance_volume = advance_periods['Volume'].mean()
            
            # Volume dry-up: pullback volume should be significantly less
            volume_ratio = avg_pullback_volume / avg_advance_volume if avg_advance_volume > 0 else 1
            
            if volume_ratio < 0.7:  # Pullback volume is <70% of advance volume
                vcp_quality['quality_score'] += 25
                vcp_quality['volume_behavior'] = f'Healthy Dry-Up ✓ (Ratio: {volume_ratio:.2f})'
            elif volume_ratio < 0.85:
                vcp_quality['quality_score'] += 15
                vcp_quality['volume_behavior'] = f'Moderate Dry-Up ~ (Ratio: {volume_ratio:.2f})'
            else:
                vcp_quality['volume_behavior'] = f'No Dry-Up ✗ (Ratio: {volume_ratio:.2f})'
        
        # 3. PRICE STRUCTURE ANALYSIS
        # Look for higher lows and tightening price action
        lows = recent_data['Low'].values
        highs = recent_data['High'].values
        
        # Divide into 3 phases and check for progressive tightening
        third = len(recent_data) // 3
        phase1_range = highs[:third].max() - lows[:third].min()
        phase2_range = highs[third:2*third].max() - lows[third:2*third].min()
        phase3_range = highs[2*third:].max() - lows[2*third:].min()
        
        # Each phase should have tighter range
        price_tightening = phase1_range > phase2_range > phase3_range
        
        if price_tightening:
            vcp_quality['quality_score'] += 20
            vcp_quality['base_structure'] = 'Tightening ✓'
            vcp_quality['base_depth'] = round((phase3_range / phase1_range) * 100, 1)
        else:
            vcp_quality['base_structure'] = 'Not Tightening ✗'
        
        # 4. BASE QUALITY
        # Check if price is holding above key support (50-day SMA)
        support_level = recent_data['SMA_50'].iloc[-1]
        recent_low = recent_data['Low'].iloc[-10:].min()
        
        if recent_low > support_level * 0.95:  # Within 5% of 50 SMA
            vcp_quality['quality_score'] += 15
            vcp_quality['base_structure'] += ' | Above Support ✓'
        
        # 5. PULLBACK DEPTH ANALYSIS (The "T" in VCP - Tight)
        # Latest pullback should be shallowest (ideally <15% from high)
        recent_high = recent_data['High'].max()
        current_price = recent_data['Close'].iloc[-1]
        pullback_depth = ((recent_high - current_price) / recent_high) * 100
        
        if pullback_depth < 10:
            vcp_quality['quality_score'] += 15
            vcp_quality['stage'] = 'Stage 4 - Very Tight (Excellent)'
        elif pullback_depth < 15:
            vcp_quality['quality_score'] += 10
            vcp_quality['stage'] = 'Stage 3 - Tight (Good)'
        elif pullback_depth < 20:
            vcp_quality['quality_score'] += 5
            vcp_quality['stage'] = 'Stage 2 - Moderate'
        else:
            vcp_quality['stage'] = 'Stage 1 - Early/Wide'
        
        # FINAL ASSESSMENT
        final_score = vcp_quality['quality_score']
        
        if final_score >= 75:
            vcp_quality['pattern_detected'] = True
            vcp_quality['recommendation'] = 'High Quality VCP - Strong Buy Setup'
        elif final_score >= 60:
            vcp_quality['pattern_detected'] = True
            vcp_quality['recommendation'] = 'Good Quality VCP - Buy Setup'
        elif final_score >= 45:
            vcp_quality['pattern_detected'] = True
            vcp_quality['recommendation'] = 'Moderate VCP - Monitor Closely'
        elif final_score >= 30:
            vcp_quality['pattern_detected'] = False
            vcp_quality['recommendation'] = 'Weak VCP Pattern - Wait'
        else:
            vcp_quality['pattern_detected'] = False
            vcp_quality['recommendation'] = 'No VCP Pattern'
        
        # Binary detection (for criteria check)
        vcp_detected = final_score >= 45
        
        return vcp_detected, final_score, vcp_quality
    
    def calculate_relative_strength(self, stock_data, market_data):
        """Calculate relative strength vs market (simple version)"""
        if len(stock_data) != len(market_data):
            min_len = min(len(stock_data), len(market_data))
            stock_data = stock_data.tail(min_len)
            market_data = market_data.tail(min_len)
        
        rs_line = stock_data['Close'] / market_data['Close']
        rs_momentum = rs_line / rs_line.shift(63)
        rs_rating = ((rs_momentum.rank(pct=True).iloc[-1]) * 100) if len(rs_momentum) > 0 else 50
        
        return rs_rating
    
    def calculate_rs_rating_vs_universe(self, symbol, stock_data, universe_data_dict):
        """
        Calculate proper RS Rating (0-100) based on Minervini's methodology
        Compares stock performance against entire stock universe
        
        Timeframes used (as per Minervini):
        - Quarter: 63 trading days
        - Half-year: 126 trading days  
        - Year: 252 trading days
        """
        try:
            if symbol not in universe_data_dict or stock_data.empty:
                return 50, {}
            
            # Calculate price performance for different periods
            periods = {
                'quarter': 63,
                'half_year': 126,
                'year': 252
            }
            
            stock_performance = {}
            for period_name, days in periods.items():
                if len(stock_data) >= days:
                    current_price = stock_data['Close'].iloc[-1]
                    past_price = stock_data['Close'].iloc[-days]
                    if past_price > 0:
                        performance = ((current_price - past_price) / past_price) * 100
                        stock_performance[period_name] = performance
                    else:
                        stock_performance[period_name] = 0
                else:
                    stock_performance[period_name] = 0
            
            # Calculate weighted performance (as per Minervini's emphasis on recent performance)
            # 40% weight on quarter, 30% on half-year, 30% on year
            weighted_performance = (
                stock_performance.get('quarter', 0) * 0.40 +
                stock_performance.get('half_year', 0) * 0.30 +
                stock_performance.get('year', 0) * 0.30
            )
            
            # Now compare against universe
            universe_performances = []
            for other_symbol, other_data in universe_data_dict.items():
                if other_data.empty or len(other_data) < 63:
                    continue
                
                other_performance = {}
                for period_name, days in periods.items():
                    if len(other_data) >= days:
                        current = other_data['Close'].iloc[-1]
                        past = other_data['Close'].iloc[-days]
                        if past > 0:
                            perf = ((current - past) / past) * 100
                            other_performance[period_name] = perf
                        else:
                            other_performance[period_name] = 0
                    else:
                        other_performance[period_name] = 0
                
                # Calculate weighted performance for this stock
                weighted_perf = (
                    other_performance.get('quarter', 0) * 0.40 +
                    other_performance.get('half_year', 0) * 0.30 +
                    other_performance.get('year', 0) * 0.30
                )
                universe_performances.append(weighted_perf)
            
            # Calculate percentile rank (RS Rating)
            if universe_performances:
                # Count how many stocks we outperform
                better_than = sum(1 for p in universe_performances if weighted_performance > p)
                rs_rating = (better_than / len(universe_performances)) * 100
            else:
                rs_rating = 50
            
            # Additional RS metrics
            rs_metrics = {
                'rs_rating': round(rs_rating, 1),
                'quarter_performance': round(stock_performance.get('quarter', 0), 2),
                'half_year_performance': round(stock_performance.get('half_year', 0), 2),
                'year_performance': round(stock_performance.get('year', 0), 2),
                'weighted_performance': round(weighted_performance, 2),
                'universe_size': len(universe_performances),
                'percentile': round(rs_rating, 0)
            }
            
            return rs_rating, rs_metrics
            
        except Exception as e:
            st.warning(f"Error calculating RS rating for {symbol}: {str(e)}")
            return 50, {}
    
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
    
    def complete_sepa_analysis(self, symbol, market_symbol='NIFTY 50', days_back=365, universe_data=None):
        """
        Complete SEPA analysis combining fundamentals and technicals
        Now includes proper RS calculation against stock universe
        """
        
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
        
        # Calculate RS Rating against universe if available
        rs_rating = None
        rs_metrics = {}
        
        if universe_data and symbol in universe_data:
            rs_rating, rs_metrics = self.calculate_rs_rating_vs_universe(
                symbol, stock_data, universe_data
            )
            analysis_result['rs_rating'] = round(rs_rating, 1)
            analysis_result['rs_metrics'] = rs_metrics
        else:
            # Fallback to market-based RS if no universe data
            market_data = self.get_technical_data(market_symbol, start_date, end_date)
            if not market_data.empty:
                rs_rating = self.calculate_relative_strength(stock_data, market_data)
                analysis_result['rs_rating'] = round(rs_rating, 1)
        
        # Technical criteria check (now with RS rating)
        technical_pass, tech_criteria, tech_score = self.check_technical_criteria(
            stock_data, symbol, rs_rating
        )
        
        analysis_result['technical_pass'] = technical_pass
        analysis_result['technical_criteria'] = tech_criteria
        analysis_result['technical_score'] = tech_score
        analysis_result['current_price'] = round(stock_data['Close'].iloc[-1], 2)
        
        # Extract VCP quality details
        if 'vcp_details' in tech_criteria:
            analysis_result['vcp_quality'] = tech_criteria['vcp_details']
            analysis_result['vcp_score'] = tech_criteria['vcp_quality_score']
        
        # Entry Point Analysis
        if fundamental_pass and technical_pass:
            entry_analysis = self.calculate_precise_entry_points(stock_data)
            analysis_result['entry_analysis'] = entry_analysis
        
        # Overall SEPA Rating (with RS heavily weighted)
        # Minervini emphasizes RS rating significantly
        rs_component = (rs_rating / 100) * 2 if rs_rating else 0  # RS can contribute up to 2 points
        sepa_score = (fund_score * 0.3) + (tech_score * 0.5) + rs_component
        analysis_result['sepa_score'] = round(sepa_score, 1)
        
        # Final recommendation (stricter with RS requirement)
        if fundamental_pass and technical_pass and rs_rating and rs_rating >= 70:
            if sepa_score >= 8.0 and rs_rating >= 85:
                analysis_result['recommendation'] = 'STRONG BUY - Elite RS'
            elif sepa_score >= 7.0 and rs_rating >= 80:
                analysis_result['recommendation'] = 'STRONG BUY'
            elif sepa_score >= 6.5:
                analysis_result['recommendation'] = 'BUY'
            else:
                analysis_result['recommendation'] = 'QUALIFIED - MONITOR'
        elif fundamental_pass and technical_pass and rs_rating and rs_rating >= 60:
            analysis_result['recommendation'] = 'QUALIFIED - WATCH (Improve RS)'
        elif fundamental_pass or technical_pass:
            analysis_result['recommendation'] = 'PARTIAL QUALIFICATION - WATCH'
        else:
            analysis_result['recommendation'] = 'NOT QUALIFIED'
        
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
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        symbol = st.text_input(
            "Enter Stock Symbol (NSE)",
            value="RELIANCE",
            help="Enter NSE stock symbol without .NS suffix"
        ).upper()
    
    with col2:
        use_universe = st.checkbox(
            "Use Universe RS",
            value=True,
            help="Calculate RS against uploaded/session stocks"
        )
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    if analyze_btn and symbol:
        with st.spinner(f"Analyzing {symbol}..."):
            sepa = MinerviniSEPA(st.session_state.custom_criteria)
            
            # Build universe if requested and available
            universe_data = None
            if use_universe and (st.session_state.uploaded_symbols or st.session_state.symbol_history):
                universe_symbols = list(set(
                    st.session_state.uploaded_symbols + 
                    st.session_state.symbol_history
                ))
                
                # Add current symbol if not in universe
                if symbol not in universe_symbols:
                    universe_symbols.append(symbol)
                
                st.info(f"📊 Building RS universe with {len(universe_symbols)} stocks...")
                
                universe_data = {}
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back + 50)
                
                universe_progress = st.progress(0)
                for i, sym in enumerate(universe_symbols[:100]):  # Limit to 100 for performance
                    universe_progress.progress((i + 1) / min(len(universe_symbols), 100))
                    try:
                        data = sepa.get_technical_data(sym, start_date, end_date)
                        if not data.empty:
                            data = sepa.calculate_technical_indicators(data)
                            universe_data[sym] = data
                    except:
                        continue
                
                universe_progress.empty()
                st.success(f"✅ Universe built with {len(universe_data)} stocks")
            
            result = sepa.complete_sepa_analysis(
                symbol, 
                market_index, 
                days_back,
                universe_data=universe_data
            )
            
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
            
            # RS Rating Section (Prominent Display)
            st.markdown("##### 🎯 Relative Strength Analysis")
            
            if 'rs_metrics' in result:
                rs_metrics = result['rs_metrics']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    rs_val = rs_metrics.get('rs_rating', 0)
                    rs_color = "🟢" if rs_val >= 80 else "🟡" if rs_val >= 70 else "🟠" if rs_val >= 60 else "🔴"
                    st.metric(f"{rs_color} RS Rating", f"{rs_val:.1f}/100")
                with col2:
                    st.metric("Universe Percentile", f"{rs_metrics.get('percentile', 0):.0f}%")
                with col3:
                    st.metric("Universe Size", rs_metrics.get('universe_size', 'N/A'))
                with col4:
                    st.metric("Quarter Performance", f"{rs_metrics.get('quarter_performance', 0):.1f}%")
                
                # Performance breakdown
                with st.expander("📊 Detailed Performance Breakdown"):
                    perf_data = pd.DataFrame([
                        {
                            "Period": "Quarter (63 days)",
                            "Return": f"{rs_metrics.get('quarter_performance', 0):.2f}%",
                            "Weight": "40%"
                        },
                        {
                            "Period": "Half-Year (126 days)",
                            "Return": f"{rs_metrics.get('half_year_performance', 0):.2f}%",
                            "Weight": "30%"
                        },
                        {
                            "Period": "Year (252 days)",
                            "Return": f"{rs_metrics.get('year_performance', 0):.2f}%",
                            "Weight": "30%"
                        },
                        {
                            "Period": "Weighted Average",
                            "Return": f"{rs_metrics.get('weighted_performance', 0):.2f}%",
                            "Weight": "100%"
                        }
                    ])
                    st.table(perf_data)
                    
                    if rs_val >= 80:
                        st.success("✅ **Elite RS Rating** - Stock is outperforming 80%+ of universe")
                    elif rs_val >= 70:
                        st.info("✓ **Strong RS Rating** - Stock is in top 30% of performers")
                    elif rs_val >= 60:
                        st.warning("~ **Moderate RS Rating** - Stock is above average but not exceptional")
                    else:
                        st.error("✗ **Weak RS Rating** - Stock is underperforming the universe")
            else:
                rs_val = result.get('rs_rating', 'N/A')
                st.metric("RS Rating (vs Market)", f"{rs_val}")
                st.info("💡 Upload a stock list to enable universe-based RS calculation")
            
            st.markdown("---")
            
            # VCP Quality Analysis (Detailed)
            if 'vcp_quality' in result:
                st.markdown("##### 🎪 VCP Pattern Quality Analysis")
                vcp_q = result['vcp_quality']
                vcp_score = result.get('vcp_score', 0)
                
                # Overall VCP Score
                col1, col2, col3 = st.columns(3)
                with col1:
                    score_color = "🟢" if vcp_score >= 75 else "🟡" if vcp_score >= 60 else "🟠" if vcp_score >= 45 else "🔴"
                    st.metric(f"{score_color} VCP Quality Score", f"{vcp_score}/100")
                with col2:
                    st.metric("Pattern Stage", vcp_q.get('stage', 'Unknown'))
                with col3:
                    detected = "✅ Yes" if vcp_q.get('pattern_detected') else "❌ No"
                    st.metric("VCP Detected", detected)
                
                # Quality breakdown
                st.markdown("**Quality Components:**")
                
                quality_components = [
                    {
                        "Component": "Volatility Contraction",
                        "Status": vcp_q.get('volatility_trend', 'Unknown'),
                        "Details": f"ATR: {vcp_q.get('contractions', [])}"
                    },
                    {
                        "Component": "Volume Behavior",
                        "Status": vcp_q.get('volume_behavior', 'Unknown'),
                        "Details": "Pullback vs Advance volume"
                    },
                    {
                        "Component": "Base Structure",
                        "Status": vcp_q.get('base_structure', 'Unknown'),
                        "Details": f"Depth: {vcp_q.get('base_depth', 0)}%"
                    }
                ]
                
                quality_df = pd.DataFrame(quality_components)
                st.dataframe(quality_df, use_container_width=True, hide_index=True)
                
                # Recommendation
                recommendation = vcp_q.get('recommendation', 'Unknown')
                if 'High Quality' in recommendation:
                    st.success(f"✅ **{recommendation}**")
                elif 'Good Quality' in recommendation:
                    st.info(f"✓ **{recommendation}**")
                elif 'Moderate' in recommendation:
                    st.warning(f"~ **{recommendation}**")
                else:
                    st.error(f"✗ **{recommendation}**")
                
                # Educational note
                with st.expander("ℹ️ Understanding VCP Quality"):
                    st.markdown("""
                    **VCP (Volatility Contraction Pattern) Quality Criteria:**
                    
                    1. **Volatility Contraction (25 points):** Multiple price contractions, each tighter than the previous
                    2. **Volume Dry-Up (25 points):** Volume decreases on pullbacks vs advances
                    3. **Price Tightening (20 points):** Range gets progressively smaller
                    4. **Support Holding (15 points):** Price stays above 50-day SMA
                    5. **Pullback Depth (15 points):** Latest pullback is shallowest
                    
                    **Scoring:**
                    - 75-100: High Quality VCP - Strong Buy Setup
                    - 60-74: Good Quality VCP - Buy Setup
                    - 45-59: Moderate VCP - Monitor Closely
                    - Below 45: No reliable VCP pattern
                    
                    **Stage Classification:**
                    - Stage 4 (Very Tight): < 10% pullback - Excellent
                    - Stage 3 (Tight): 10-15% pullback - Good
                    - Stage 2 (Moderate): 15-20% pullback - Fair
                    - Stage 1 (Wide): > 20% pullback - Early/Too wide
                    """)
            
            st.markdown("---")
            
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
                {"Criteria": "Strong Price Action", "Status": "✅" if tech_criteria.get('strong_price_action') else "❌"},
                {"Criteria": "RS Rating > 70", "Status": "✅" if tech_criteria.get('rs_rating_above_70') else "❌"},
                {"Criteria": "RS Rating > 80 (Elite)", "Status": "✅" if tech_criteria.get('rs_rating_above_80') else "❌"},
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
    """Screen multiple stocks with proper RS calculation against universe"""
    
    qualified_stocks = []
    
    # Phase 1: Build the universe data (fetch all stock data first)
    st.info("📊 Phase 1: Building stock universe for RS calculation...")
    universe_progress = st.progress(0)
    universe_status = st.empty()
    
    universe_data = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + 50)
    
    sepa = MinerviniSEPA(st.session_state.custom_criteria)
    
    for i, symbol in enumerate(stock_list):
        progress = (i + 1) / len(stock_list)
        universe_progress.progress(progress)
        universe_status.text(f"Fetching data for {symbol}... ({i+1}/{len(stock_list)})")
        
        try:
            stock_data = sepa.get_technical_data(symbol, start_date, end_date)
            if not stock_data.empty:
                stock_data = sepa.calculate_technical_indicators(stock_data)
                universe_data[symbol] = stock_data
        except Exception as e:
            st.warning(f"⚠️ Could not fetch data for {symbol}: {str(e)}")
            continue
    
    universe_progress.empty()
    universe_status.empty()
    
    st.success(f"✅ Built universe with {len(universe_data)} stocks for RS calculation")
    
    # Phase 2: Analyze each stock with universe-based RS
    st.info("🔍 Phase 2: Analyzing stocks with universe-based RS ratings...")
    analysis_progress = st.progress(0)
    analysis_status = st.empty()
    
    for i, symbol in enumerate(stock_list):
        progress = (i + 1) / len(stock_list)
        analysis_progress.progress(progress)
        analysis_status.text(f"Analyzing {symbol}... ({i+1}/{len(stock_list)})")
        
        try:
            result = sepa.complete_sepa_analysis(
                symbol, 
                market_index, 
                days_back,
                universe_data=universe_data
            )
            
            if result and result['sepa_score'] >= min_sepa_score:
                qualified_stocks.append(result)
        except Exception as e:
            st.warning(f"⚠️ Error analyzing {symbol}: {str(e)}")
            continue
    
    analysis_progress.empty()
    analysis_status.empty()
    
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
            rs_metrics = stock.get('rs_metrics', {})
            vcp_score = stock.get('vcp_score', 0)
            
            results_data.append({
                'Symbol': stock['symbol'],
                'SEPA Score': stock['sepa_score'],
                'Price': f"₹{stock['current_price']}",
                'RS Rating': stock.get('rs_rating', 'N/A'),
                'RS Percentile': f"{rs_metrics.get('percentile', 'N/A')}" if rs_metrics else 'N/A',
                'VCP Quality': f"{vcp_score}/100" if vcp_score else 'N/A',
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
        
        # RS and VCP Analysis Summary
        st.markdown("---")
        st.subheader("📊 Universe Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            elite_rs = sum(1 for s in qualified_stocks if s.get('rs_rating', 0) >= 80)
            st.metric("Elite RS (≥80)", f"{elite_rs} stocks")
        
        with col2:
            strong_rs = sum(1 for s in qualified_stocks if 70 <= s.get('rs_rating', 0) < 80)
            st.metric("Strong RS (70-79)", f"{strong_rs} stocks")
        
        with col3:
            high_vcp = sum(1 for s in qualified_stocks if s.get('vcp_score', 0) >= 75)
            st.metric("High Quality VCP", f"{high_vcp} stocks")
        
        with col4:
            good_vcp = sum(1 for s in qualified_stocks if 60 <= s.get('vcp_score', 0) < 75)
            st.metric("Good Quality VCP", f"{good_vcp} stocks")
        
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
