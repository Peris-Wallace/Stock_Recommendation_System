# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yaml
import random 


# Import modules
from src.get_data import (
    get_price_data, 
    get_fundamental_metrics,
    calculate_technical_indicators, 
    calculate_risk_metrics,
    get_market_status,
    validate_symbol,
    get_long_name,
    export_data,
)


# Set page config
st.set_page_config(
    page_title="Market Analytics Dashboard", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()


# Color scheme
COLORS = config['theme']['colors']
primary_color = COLORS['primary']

# Sidebar Configuration
with st.sidebar:

    # Symbol input
    default_symbols = config['symbols']['default_pool'] # Get default symbols

    # Use session state symbol, random default, or fallback to NVDA
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = random.choice(default_symbols)
    
    default_symbol = st.session_state.get('selected_symbol', 'NVDA')
    
    symbol = st.text_input(
        "ENTER SYMBOL 🔍", 
        value=default_symbol,
        placeholder="e.g., NVDA, BTC-USD, ^GSPC",
        help="Enter any ticker symbol (stocks, crypto, indices, etc.)",
        key="symbol_input"
    )
    
    if symbol:
        symbol = symbol.upper().strip()
        # Update session state
        st.session_state.selected_symbol = symbol
    
    # Symbol validation and info
    if symbol:
        with st.container():
            validation_result = validate_symbol(symbol)
            
            if validation_result['valid']:
                st.success(f"✅ {validation_result['name']}")
                if validation_result['sector'] != 'N/A':
                    st.caption(f"Sector: {validation_result['sector']}")
                if validation_result['exchange'] != 'N/A':
                    st.caption(f"Exchange: {validation_result['exchange']}")
            else:
                st.warning(f"⚠️ {validation_result['error']}")
    
    st.divider()
    
    # Time period and interval selection
    period_options = {
        "1mo": "1 Month",
        "3mo": "3 Months", 
        "6mo": "6 Months",
        "1y": "1 Year",
        "2y": "2 Years",
        "5y": "5 Years",
        "10y": "10 Years",
        "ytd": "Year to Date",
        "max": "Maximum"
    }
    
    period = st.selectbox(
        "📅 Time Period", 
        list(period_options.keys()),
        format_func=lambda x: period_options[x],
        index=3,  # Default to 1 year
        help="Select the time period for historical data analysis"
    )
    
    # Interval selection with smart defaults based on period
    interval_options = {
        "1m": "1 Minute",
        "2m": "2 Minutes", 
        "5m": "5 Minutes",
        "15m": "15 Minutes",
        "30m": "30 Minutes",
        "60m": "1 Hour",
        "90m": "90 Minutes",
        "1h": "1 Hour",
        "1d": "1 Day",
        "5d": "5 Days",
        "1wk": "1 Week",
        "1mo": "1 Month",
        "3mo": "3 Months"
    }
    
    # Smart interval suggestions based on period
    if period in ["1mo", "3mo"]:
        suggested_intervals = ["1d", "1wk"]
        default_idx = 0
    elif period in ["6mo", "1y"]:
        suggested_intervals = ["1d", "1wk", "1mo"]
        default_idx = 0
    else:
        suggested_intervals = ["1d", "1wk", "1mo"]
        default_idx = 1
    
    interval = st.selectbox(
        "⏱️ Data Interval", 
        suggested_intervals,
        format_func=lambda x: interval_options.get(x, x),
        index=default_idx,
        help="Choose the frequency of data points"
    )
    
    st.divider()
    
    # Chart and display options
    st.markdown("### 📈 Display Options")
    
    # Chart type with icons
    chart_types = {
        "Candlestick": "🕯️ Candlestick",
        "Line": "📈 Line Chart", 
        "Area": "📊 Area Chart",
        "OHLC": "📋 OHLC Bars"
    }
    
    chart_type = st.selectbox(
        "Chart Style",
        list(chart_types.keys()),
        format_func=lambda x: chart_types[x],
        help="Choose how to display the price data"
    )
    
    # Technical indicators
    st.markdown("#### 🔧 Technical Indicators")
    
    show_ma = st.checkbox("Moving Averages", value=True, help="Show 20 & 50 day moving averages")
    show_volume = st.checkbox("Volume", value=True, help="Display trading volume")
    show_bollinger = st.checkbox("Bollinger Bands", help="Show volatility bands")
    show_rsi = st.checkbox("RSI", help="Relative Strength Index")
    
    # Advanced options in expander
    with st.expander("🔬 Advanced Options"):
        # Export options
        st.markdown("**Export Data**")
        export_format = st.radio(
            "Format",
            ["CSV", "Excel", "JSON"],
            horizontal=True
        )
        
        if st.button("💾 Export Data", use_container_width=True):
            if 'df' in locals() and not df.empty:
                exported_data = export_data(df, export_format)
                if exported_data:
                    st.download_button(
                        label=f"Download {export_format}",
                        data=exported_data,
                        file_name=f"{symbol}_data.{export_format.lower()}",
                        mime=f"application/{export_format.lower()}"
                    )
            else:
                st.warning("No data available to export")
    
    st.divider()

    
    # Watchlist functionality
    st.markdown("### ⭐ Watchlist")
    
    # Initialize watchlist in session state
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    
    # Add current symbol to watchlist
    if symbol and st.button("➕ Add to Watchlist", use_container_width=True):
        if symbol not in st.session_state.watchlist:
            st.session_state.watchlist.append(symbol)
            st.success(f"Added {symbol} to watchlist")
        else:
            st.warning(f"{symbol} already in watchlist")
    
    # Display watchlist
    if st.session_state.watchlist:
        st.markdown("**My Watchlist:**")
        for i, watch_symbol in enumerate(st.session_state.watchlist):
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(watch_symbol, key=f"watch_{i}", use_container_width=True):
                    st.session_state.selected_symbol = watch_symbol
                    st.rerun()
            with col2:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.watchlist.remove(watch_symbol)
                    st.rerun()
        
        if st.button("🗑️ Clear Watchlist"):
            st.session_state.watchlist = []
            st.rerun()
    else:
        st.info("No symbols in watchlist")
    
    st.divider()
    
    # Market status
    st.markdown("### 🌍 Market Status")
    
    # This would typically fetch real market hours data
    now = datetime.now()
    is_trading_hours = 9 <= now.hour < 16 and now.weekday() < 5
    
    if is_trading_hours:
        st.success("🟢 Markets Open")
    else:
        st.error("🔴 Markets Closed")
    
    st.caption(f"Last updated: {now.strftime('%H:%M:%S')}")
    
    st.divider()

    # Refresh data
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh", type="primary", use_container_width=True):
            # Clear cache and refresh data
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear", type="secondary", use_container_width=True):
            # Clear all inputs
            st.session_state.clear()
            st.rerun()


## Visualizations
def create_price_chart(df, long_name, chart_type, show_ma=True, show_volume=True, show_rsi=True):
    """Create a price chart with technical indicators"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{long_name} Price Analysis", "Volume", "RSI"]
    )
    
    # Price chart
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color=COLORS["up"],
            decreasing_line_color=COLORS["down"],
            name="OHLC"
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            line=dict(color=COLORS["primary"], width=2),
            name="Close Price",
            fill='tonexty' if chart_type == "Area" else None,
            fillcolor=f"rgba(99, 102, 241, 0.1)" if chart_type == "Area" else None
        ), row=1, col=1)
    
    # Add moving averages if enabled
    if show_ma and 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            line=dict(color=COLORS["warning"], width=1, dash='dash'),
            name="SMA 20"
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            line=dict(color=COLORS["info"], width=1, dash='dash'),
            name="SMA 50"
        ), row=1, col=1)
    
    # Volume chart
    volume_colors = [COLORS["up"] if c >= o else COLORS["down"] 
                    for c, o in zip(df['Close'], df['Open'])]

    if show_volume:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=volume_colors,
            name="Volume",
            opacity=0.6
        ), row=2, col=1)
    
    # RSI chart
    if show_rsi and 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            line=dict(color=COLORS["secondary"], width=2),
            name="RSI"
        ), row=3, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["danger"], 
                     opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["success"], 
                     opacity=0.5, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        template="plotly_white",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig


def create_fundamentals_display(fundamentals):
    """Create a display for fundamental metrics"""
    st.markdown("### 📊 Fundamental Analysis")
    
    # Organize metrics into categories
    valuation_metrics = {
        'Market Cap': fundamentals.get('Market Cap'),
        'PE Ratio': fundamentals.get('PE Ratio'),
        'Forward PE': fundamentals.get('Forward PE'),
        'PEG Ratio': fundamentals.get('PEG Ratio'),
        'Price to Book': fundamentals.get('Price to Book'),
        'Enterprise Value': fundamentals.get('Enterprise Value')
    }
    
    profitability_metrics = {
        'Revenue': fundamentals.get('Revenue'),
        'EBITDA': fundamentals.get('EBITDA'),
        'EPS': fundamentals.get('EPS'),
        'Profit Margin': fundamentals.get('Profit Margin'),
        'Operating Margin': fundamentals.get('Operating Margin'),
        'Gross Margin': fundamentals.get('Gross Margin'),
        'Return on Equity': fundamentals.get('Return on Equity'),
        'Return on Assets': fundamentals.get('Return on Assets')
    }
    
    financial_health = {
        'Current Ratio': fundamentals.get('Current Ratio'),
        'Quick Ratio': fundamentals.get('Quick Ratio'),
        'Debt to Equity': fundamentals.get('Debt to Equity'),
        'Free Cash Flow': fundamentals.get('Free Cash Flow'),
        'Book Value': fundamentals.get('Book Value')
    }
    
    dividend_info = {
        'Dividend Yield': fundamentals.get('Dividend Yield'),
        'Beta': fundamentals.get('Beta'),
        '52 Week High': fundamentals.get('52 Week High'),
        '52 Week Low': fundamentals.get('52 Week Low')
    }
    
    # Display in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Valuation", "📈 Profitability", "🏦 Financial Health", "💵 Dividend & Risk"])
    
    with tab1:
        display_metric_category(valuation_metrics, "Valuation Metrics")
    
    with tab2:
        display_metric_category(profitability_metrics, "Profitability Metrics")
    
    with tab3:
        display_metric_category(financial_health, "Financial Health Metrics")
    
    with tab4:
        display_metric_category(dividend_info, "Dividend & Risk Metrics")


def display_metric_category(metrics_dict, category_name):
    """Helper function to display a category of metrics"""
    st.markdown(f"#### {category_name}")
    
    cols = st.columns(3)
    col_idx = 0
    
    for metric_name, value in metrics_dict.items():
        with cols[col_idx % 3]:
            if value is not None:
                if isinstance(value, (int, float)):
                    if metric_name in ['Market Cap', 'Revenue', 'EBITDA', 'Enterprise Value', 'Free Cash Flow']:
                        # Format large numbers
                        if abs(value) >= 1e12:
                            formatted_value = f"${value/1e12:.2f}T"
                        elif abs(value) >= 1e9:
                            formatted_value = f"${value/1e9:.2f}B"
                        elif abs(value) >= 1e6:
                            formatted_value = f"${value/1e6:.2f}M"
                        else:
                            formatted_value = f"${value:,.0f}"
                    elif metric_name in ['Profit Margin', 'Operating Margin', 'Gross Margin', 'Return on Equity', 'Return on Assets', 'Dividend Yield']:
                        # Format percentages
                        formatted_value = f"{value*100:.2f}%" if value < 1 else f"{value:.2f}%"
                    elif metric_name in ['52 Week High', '52 Week Low', 'Book Value', 'EPS']:
                        # Format currency
                        formatted_value = f"${value:.2f}"
                    else:
                        # Format regular numbers
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                
                st.metric(metric_name, formatted_value)
            else:
                st.metric(metric_name, "N/A")
        
        col_idx += 1



# Main Dashboard
st.markdown('<div class="main-header">📈 Market Analytics Dashboard</div>', unsafe_allow_html=True)

# Fetch and process data
with st.spinner(f"Loading data for {symbol}..."):
    df, stock_info = get_price_data(symbol, period, interval)

if df.empty:
    st.error(f"❌ No data available for symbol: {symbol}")
    st.info("💡 Try a different symbol or check if the market is open.")
    st.stop()

# Calculate indicators and metrics
df = calculate_technical_indicators(df)
metrics = calculate_risk_metrics(df)

# Stock/Crypto info header
if stock_info:
    company_name = stock_info.get('longName', stock_info.get('shortName', symbol))
    st.markdown(f'<div class="stock-header">{company_name} ({symbol.upper()})</div>', unsafe_allow_html=True)



# Main Price Chart
long_name = get_long_name(symbol)
fig = create_price_chart(df, long_name, chart_type, show_ma, True, show_rsi)
st.plotly_chart(fig, use_container_width=True)

# Technical Analysis Tabs
st.markdown("### 🔍 Technical Analysis")
tab1, tab2, tab3, tab4 = st.tabs(["📊 MACD", "📈 Bollinger Bands", "⚡ Stochastic", "📋 Summary"])

with tab1:
    if all(col in df.columns for col in ['MACD', 'MACD_SIGNAL', 'MACD_HIST']):
        fig_macd = go.Figure()
        
        # MACD Histogram
        fig_macd.add_trace(go.Bar(
            x=df.index, y=df['MACD_HIST'],
            name="MACD Histogram",
            marker_color=[COLORS["up"] if v >= 0 else COLORS["down"] for v in df['MACD_HIST']],
            opacity=0.6
        ))
        
        # MACD and Signal lines
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df['MACD'], 
            name='MACD', 
            line=dict(color=COLORS['primary'], width=2)
        ))
        fig_macd.add_trace(go.Scatter(
            x=df.index, y=df['MACD_SIGNAL'], 
            name='Signal', 
            line=dict(color=COLORS['warning'], width=2)
        ))
        
        fig_macd.update_layout(
            height=400, 
            template='plotly_white', 
            yaxis_title='MACD',
            title="MACD (Moving Average Convergence Divergence)"
        )
        st.plotly_chart(fig_macd, use_container_width=True)
        
        # MACD interpretation
        latest_macd = df['MACD'].iloc[-1]
        latest_signal = df['MACD_SIGNAL'].iloc[-1]
        if latest_macd > latest_signal:
            st.success("🟢 MACD is above signal line - Bullish momentum")
        else:
            st.warning("🔴 MACD is below signal line - Bearish momentum")

with tab2:
    if all(col in df.columns for col in ['BB_UPPER', 'BB_LOWER', 'BB_MIDDLE']):
        fig_bb = go.Figure()
        
        # Price and Bollinger Bands
        fig_bb.add_trace(go.Scatter(
            x=df.index, y=df['Close'], 
            name='Close Price', 
            line=dict(color=COLORS["primary"], width=2)
        ))
        fig_bb.add_trace(go.Scatter(
            x=df.index, y=df['BB_UPPER'], 
            name='Upper Band', 
            line=dict(color=COLORS["danger"], width=1, dash='dash')
        ))
        fig_bb.add_trace(go.Scatter(
            x=df.index, y=df['BB_LOWER'], 
            name='Lower Band', 
            line=dict(color=COLORS["success"], width=1, dash='dash')
        ))
        fig_bb.add_trace(go.Scatter(
            x=df.index, y=df['BB_MIDDLE'], 
            name='Middle Band (SMA)', 
            line=dict(color=COLORS["neutral"], width=1)
        ))
        
        fig_bb.update_layout(
            height=400, 
            template='plotly_white', 
            yaxis_title='Price ($)',
            title="Bollinger Bands"
        )
        st.plotly_chart(fig_bb, use_container_width=True)
        
        # Bollinger Bands interpretation
        current_price = df['Close'].iloc[-1]
        upper_band = df['BB_UPPER'].iloc[-1]
        lower_band = df['BB_LOWER'].iloc[-1]
        
        if current_price > upper_band:
            st.warning("⚠️ Price is above upper Bollinger Band - Potentially overbought")
        elif current_price < lower_band:
            st.info("💡 Price is below lower Bollinger Band - Potentially oversold")
        else:
            st.success("✅ Price is within normal Bollinger Band range")

with tab3:
    if 'STOCH' in df.columns:
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(
            x=df.index, y=df['STOCH'],
            name='Stochastic %K',
            line=dict(color=COLORS["secondary"], width=2)
        ))
        
        # Stochastic levels
        fig_stoch.add_hline(y=80, line_dash="dash", line_color=COLORS["danger"], opacity=0.5)
        fig_stoch.add_hline(y=20, line_dash="dash", line_color=COLORS["success"], opacity=0.5)
        
        fig_stoch.update_layout(
            height=400, 
            template='plotly_white', 
            yaxis_title='Stochastic %K',
            title="Stochastic Oscillator"
        )
        st.plotly_chart(fig_stoch, use_container_width=True)
        
        # Stochastic interpretation
        current_stoch = df['STOCH'].iloc[-1]
        if current_stoch > 80:
            st.warning("🔴 Stochastic indicates overbought conditions")
        elif current_stoch < 20:
            st.success("🟢 Stochastic indicates oversold conditions")
        else:
            st.info("🔵 Stochastic in neutral range")

with tab4:
    st.markdown("#### 📋 Technical Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Trend Analysis**")
        current_price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else 0
        sma_50 = df['SMA_50'].iloc[-1] if 'SMA_50' in df.columns else 0
        
        if current_price > sma_20 > sma_50:
            st.success("🟢 Strong Uptrend")
        elif current_price < sma_20 < sma_50:
            st.error("🔴 Strong Downtrend")
        else:
            st.warning("🟡 Sideways/Mixed Trend")
    
    with col2:
        st.markdown("**Risk Assessment**")
        volatility = metrics.get('volatility', 0)
        if volatility > 0.3:
            st.error("🔴 High Volatility")
        elif volatility > 0.15:
            st.warning("🟡 Moderate Volatility")
        else:
            st.success("🟢 Low Volatility")

# Additional Metrics Section
st.markdown("### 📈 Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Risk Metrics")
    st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
    st.metric("Volatility (Annual)", f"{metrics.get('volatility', 0)*100:.1f}%")
    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")

with col2:
    st.markdown("#### Return Analysis")
    st.metric("Best Day", f"{metrics.get('best_day', 0)*100:.2f}%")
    st.metric("Worst Day", f"{metrics.get('worst_day', 0)*100:.2f}%")
    st.metric("Avg Daily Return", f"{df['Returns'].mean()*100:.3f}%")

with col3:
    st.markdown("#### Trading Activity")
    st.metric("Positive Days", f"{metrics.get('positive_days', 0):.1f}%")
    st.metric("Negative Days", f"{metrics.get('negative_days', 0):.1f}%")
    st.metric("Avg Volume", f"{metrics.get('avg_volume', 0):,.0f}")

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer"> Data from Yahoo Finance • Last updated: {}</div>'.format(
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ), 
    unsafe_allow_html=True
)