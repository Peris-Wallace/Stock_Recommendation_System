# Import libraries
import streamlit as st
import numpy as np
import pandas as pd


# Convert large numbers to readable format (K, M, B)
def human_format(num, precision=2):
    if num is None or num != num:  # Check for NaN
        return "N/A"
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000:
            return f"{num:.{precision}f}{unit}"
        num /= 1000.0
    return f"{num:.{precision}f}P"


# Stock summary metrics
def render_summary_cards(df):
    current_price = df['Close'].iloc[-1]
    previous_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - previous_price
    pct_change = (price_change / previous_price) * 100 if previous_price != 0 else 0
    
    # Determine change class
    if price_change > 0:
        change_class = "positive"
        change_icon = ""
    elif price_change < 0:
        change_class = "negative"
        change_icon = ""
    else:
        change_class = "neutral"
        change_icon = "⏸"
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card no-shadow">
            <div class="metric-label">Current</div>
            <div class="metric-value">${human_format(current_price)}</div>
            <div class="metric-change {change_class}">
                {change_icon} {price_change:+.2f} ({pct_change:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)


    with col2:
        day_high = df['High'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Day High</div>
            <div class="metric-value">${human_format(day_high)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        day_low = df['Low'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Day Low</div>
            <div class="metric-value">${human_format(day_low)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].tail(20).mean()
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volume</div>
            <div class="metric-value">{human_format(volume)}</div>
            <div class="metric-change {'positive' if volume_ratio > 1.2 else 'negative' if volume_ratio < 0.8 else 'neutral'}">
                {volume_ratio:.1f}x avg
            </div>
        </div>
        """, unsafe_allow_html=True)


@st.cache_data
def get_fundamental_metrics(info):
    """Extract fundamental financial metrics from stock info"""
    if not info:
        return {}
    
    fundamental_metrics = {
        'Market Cap': info.get('marketCap'),
        'PE Ratio': info.get('trailingPE'),
        'EPS': info.get('trailingEps'),
        'Revenue': info.get('totalRevenue'),
        'Profit Margin': info.get('profitMargins'),
        'Dividend Yield': info.get('dividendYield'),
        'Book Value': info.get('bookValue'),
        'Price to Book': info.get('priceToBook'),
        'EBITDA': info.get('ebitda'),
        'Debt to Equity': info.get('debtToEquity'),
        'Return on Equity': info.get('returnOnEquity'),
        'Operating Margin': info.get('operatingMargins'),
        'Gross Margin': info.get('grossMargins'),
        '52 Week High': info.get('fiftyTwoWeekHigh'),
        '52 Week Low': info.get('fiftyTwoWeekLow'),
        'Shares Outstanding': info.get('sharesOutstanding'),
        'Float': info.get('floatShares'),
        'Insider Holdings': info.get('heldPercentInsiders'),
        'Institution Holdings': info.get('heldPercentInstitutions')
    }
    
    return fundamental_metrics


@st.cache_data
def calculate_risk_metrics(df):
    """Calculate comprehensive risk and performance metrics"""
    if df.empty or 'Returns' not in df.columns:
        return {}
    
    returns = df['Returns'].dropna()
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2] if len(df) > 1 else current_price
    
    # Performance metrics
    daily_return = (current_price - prev_close) / prev_close
    total_return = (current_price - df['Close'].iloc[0]) / df['Close'].iloc[0]
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(252)  # Annualized
    downside_deviation = returns[returns < 0].std() * np.sqrt(252)
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
    sortino_ratio = (returns.mean() / downside_deviation) * np.sqrt(252) if downside_deviation > 0 else 0
    
    # Drawdown calculation
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    current_drawdown = drawdown.iloc[-1]
    
    # Value at Risk (VaR)
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)
    
    # Expected Shortfall (Conditional VaR)
    es_95 = returns[returns <= var_95].mean()
    es_99 = returns[returns <= var_99].mean()
    
    # Additional performance metrics
    positive_days = (returns > 0).sum() / len(returns) * 100
    negative_days = (returns < 0).sum() / len(returns) * 100
    win_loss_ratio = (returns > 0).sum() / (returns < 0).sum() if (returns < 0).sum() > 0 else np.inf
    
    # Skewness and Kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Calmar Ratio
    calmar_ratio = (returns.mean() * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        "current_price": current_price,
        "daily_return": daily_return,
        "total_return": total_return,
        "volatility": volatility,
        "downside_deviation": downside_deviation,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "max_drawdown": max_drawdown,
        "current_drawdown": current_drawdown,
        "var_95": var_95,
        "var_99": var_99,
        "es_95": es_95,
        "es_99": es_99,
        "rsi": df['RSI'].iloc[-1] if 'RSI' in df.columns else 0,
        "macd": df['MACD'].iloc[-1] if 'MACD' in df.columns else 0,
        "best_day": returns.max(),
        "worst_day": returns.min(),
        "positive_days": positive_days,
        "negative_days": negative_days,
        "win_loss_ratio": win_loss_ratio,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "avg_volume": df['Volume'].mean(),
        "current_volume": df['Volume'].iloc[-1],
        "volume_ratio": df['Volume'].iloc[-1] / df['Volume'].mean() if df['Volume'].mean() > 0 else 0
    }


def show_analysis_summary(df, metrics, stock_info=None):
    # Trend & Volatility
    col1, col2 = st.columns(2)
    with col1:
        current = df['Close'].iloc[-1]
        sma20 = df.get('SMA_20', pd.Series([0])).iloc[-1]
        sma50 = df.get('SMA_50', pd.Series([0])).iloc[-1]

        if current > sma20 > sma50:
            trend_text = "🟢 Strong Uptrend"
            trend_color = "#16c784"
        elif current < sma20 < sma50:
            trend_text = "🔴 Strong Downtrend"
            trend_color = "#dc3545"
        else:
            trend_text = "🟡 Sideways Trend"
            trend_color = "#ffc107"

        st.markdown(f"""
        <div class="metric-card-small">
            <div class="metric-value" style="color: {trend_color};">{trend_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        vol = metrics.get("volatility", 0)
        if vol > 0.3:
            vol_text = "🔴 High Volatility"
            vol_color = "#dc3545"
        elif vol > 0.15:
            vol_text = "🟡 Moderate Volatility"
            vol_color = "#ffc107"
        else:
            vol_text = "🟢 Low Volatility"
            vol_color = "#16c784"

        st.markdown(f"""
        <div class="metric-card-small">
            <div class="metric-value" style="color: {vol_color};">{vol_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Performance Metrics Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div style="color: #092f53; font-family: 'Libertinus Math', system-ui, sans-serif; font-size: 14px; font-weight: 600; margin-bottom: 6px;">RISK METRICS
            </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-value">{metrics.get('max_drawdown', 0)*100:.2f}%</div></div>
        <div class="metric-card"><div class="metric-label">Volatility</div><div class="metric-value">{metrics.get('volatility', 0)*100:.1f}%</div></div>
        <div class="metric-card"><div class="metric-label">Sharpe Ratio</div><div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div></div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="color: #092f53; font-family: 'Libertinus Math', system-ui, sans-serif; font-size: 14px; font-weight: 600; margin-bottom: 6px;">RETURNS ANALYSIS
            </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card"><div class="metric-label">Best Day</div><div class="metric-value">{metrics.get('best_day', 0)*100:.2f}%</div></div>
        <div class="metric-card"><div class="metric-label">Worst Day</div><div class="metric-value">{metrics.get('worst_day', 0)*100:.2f}%</div></div>
        <div class="metric-card"><div class="metric-label">Avg Daily Return</div><div class="metric-value">{df['Returns'].mean()*100:.3f}%</div></div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style="color: #092f53; font-family: 'Libertinus Math', system-ui, sans-serif; font-size: 14px; font-weight: 600; margin-bottom: 6px;">TRADING ACTIVITY
            </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card"><div class="metric-label">Positive Days</div><div class="metric-value">{metrics.get('positive_days', 0):.1f}%</div></div>
        <div class="metric-card"><div class="metric-label">Negative Days</div><div class="metric-value">{metrics.get('negative_days', 0):.1f}%</div></div>
        <div class="metric-card"><div class="metric-label">Avg Volume</div><div class="metric-value">{metrics.get('avg_volume', 0):,.0f}</div></div>
        """, unsafe_allow_html=True)

    # Key Fundamentals
    if stock_info:
        st.divider()
        st.markdown("""
            <div style="color: #092f53; font-family: 'Libertinus Math', system-ui, sans-serif; font-size: 14px; font-weight: 600; margin-bottom: 6px;">
                KEY FUNDAMENTALS
            </div>
        """, unsafe_allow_html=True)


        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">52W High</div>
                <div class="metric-value">${human_format(stock_info.get('fiftyTwoWeekHigh'))}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">52W Low</div>
                <div class="metric-value">${human_format(stock_info.get('fiftyTwoWeekLow'))}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Market Cap</div>
                <div class="metric-value">{human_format(stock_info.get('marketCap'))}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">P/E Ratio</div>
                <div class="metric-value">{human_format(stock_info.get('trailingPE'))}</div>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            dy = stock_info.get('dividendYield')
            dy_display = f"{dy * 100:.2f}%" if dy else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dividend Yield</div>
                <div class="metric-value">{dy_display}</div>
            </div>
            """, unsafe_allow_html=True)
