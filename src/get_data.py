# Import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
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
