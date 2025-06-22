# Import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import datetime
import pytz


@st.cache_data(ttl=300)
def get_price_data(ticker, period, interval):
    """Fetch stock price data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame(), None
        
        # Get financial info
        info = stock.info
        return df, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame(), None


@st.cache_data
def get_fundamental_metrics(info):
    """Extract fundamental financial metrics from stock info"""
    if not info:
        return {}
    
    metrics = {
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
    
    return metrics


@st.cache_data
def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    if df.empty:
        return df
    
    # Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # Momentum Indicators
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['STOCH'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['STOCH_SIGNAL'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
    df['WILLIAMS_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
    
    # Volatility Indicators
    df['BB_UPPER'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_LOWER'] = ta.volatility.bollinger_lband(df['Close'])
    df['BB_MIDDLE'] = ta.volatility.bollinger_mavg(df['Close'])
    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['KELTNER_UPPER'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'])
    df['KELTNER_LOWER'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'])
    
    # Trend Indicators
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_SIGNAL'] = ta.trend.macd_signal(df['Close'])
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    df['AROON_UP'] = ta.trend.aroon_up(df['High'], df['Low'])
    df['AROON_DOWN'] = ta.trend.aroon_down(df['High'], df['Low'])
    
    # Volume Indicators
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])
    
    # Returns and Performance
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Cumulative_Returns'] = (1 + df['Returns']).cumprod() - 1
    
    # Support and Resistance levels
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    
    return df


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


def get_market_status():

    try:
        # US Eastern Time
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        
        # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        is_weekday = now.weekday() < 5
        market_open_time = 9.5  
        market_close_time = 16  
        current_time_decimal = now.hour + now.minute/60
        
        is_trading_hours = (market_open_time <= current_time_decimal <= market_close_time) and is_weekday
        
        return {
            'is_open': is_trading_hours,
            'current_time': now.strftime('%H:%M:%S ET'),
            'is_weekday': is_weekday,
            'market_opens_at': '09:30 ET',
            'market_closes_at': '16:00 ET'
        }
    except:
        now = datetime.datetime.now()
        is_trading_hours = 9 <= now.hour < 16 and now.weekday() < 5
        
        return {
            'is_open': is_trading_hours,
            'current_time': now.strftime('%H:%M:%S'),
            'is_weekday': now.weekday() < 5,
            'market_opens_at': '09:30 ET',
            'market_closes_at': '16:00 ET'
        }


def validate_symbol(symbol):
    """Validate if a symbol exists and get basic info"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if info and len(info) > 1:  # Valid symbol should have info
            return {
                'valid': True,
                'name': info.get('longName', info.get('shortName', symbol)),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market': info.get('market', 'N/A'),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'N/A')
            }
        else:
            return {'valid': False, 'error': 'Symbol not found'}
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def long_name(symbol):
    try:
        info = yf.Ticker(symbol).info
        return info.get("longName", symbol.upper())
    except Exception:
        return symbol.upper()

def export_data(data, format_type='CSV'):
    """Export data in specified format"""
    import io
    
    try:
        if format_type == 'CSV':
            return data.to_csv().encode('utf-8')
        elif format_type == 'Excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                data.to_excel(writer, sheet_name='Stock_Data')
            return output.getvalue()
        elif format_type == 'JSON':
            return data.to_json(orient='records', date_format='iso').encode('utf-8')
        
        return None
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        return None
