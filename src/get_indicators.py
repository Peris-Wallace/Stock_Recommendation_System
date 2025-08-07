# Import libraries
import streamlit as st
import numpy as np
import ta


@st.cache_data
def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    if df.empty:
        return df
    
    # Moving Averages
    # Daily Moving Averages
    df['SMA_20_day'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50_day'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA_100_day'] = ta.trend.sma_indicator(df['Close'], window=100)
    df['SMA_200_day'] = ta.trend.sma_indicator(df['Close'], window=200)

    # Weekly Moving Averages (approximated by multiplying days)
    df['SMA_4_week'] = ta.trend.sma_indicator(df['Close'], window= 4 * 5)
    df['SMA_20_week'] = ta.trend.sma_indicator(df['Close'], window=20 * 5)
    df['SMA_50_week'] = ta.trend.sma_indicator(df['Close'], window=50 * 5)
    df['SMA_100_week'] = ta.trend.sma_indicator(df['Close'], window=100 * 5)
    df['SMA_200_week'] = ta.trend.sma_indicator(df['Close'], window=200 * 5)

    # EMAs
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


