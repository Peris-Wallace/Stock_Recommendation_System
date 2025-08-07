# Import libraries
import yaml
import streamlit as st
import os
from datetime import datetime

# Import modules
from src.sidebar import shared_sidebar
from src.get_data import get_price_data
from src.get_metrics import render_summary_cards, calculate_risk_metrics, show_analysis_summary
from src.get_indicators import calculate_technical_indicators
from src.get_charts import create_price_chart


# Import tab views
from tabs.technical_analysis import show_technical_analysis


# Set page config
st.set_page_config(
    page_title="Stock Market Analytics", 
    layout="wide", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Load CSS file
@st.cache_resource
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")


# Load configuration
if not os.path.exists('config.yaml'):
    st.error("Configuration file 'config.yaml' not found. Please ensure it exists in the app directory.")
    st.stop()

@st.cache_data
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

config = load_config()
COLORS = config['theme']['colors']


# Get values for the sidebar
symbol, period, interval, chart_type, selected_mas = shared_sidebar()


# Tabs
st.markdown("""
<style>
    /* Target the tabs container */
    div[role="tablist"] {
        justify-content: flex-end !important;
        font-family: 'Libertinus Math', system-ui, sans-serif !important;
        font-size: 16px; 
        font-weight: 600; 
    }
</style>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Dashboard", "Technical Analysis"])


with tab1:
    # Fetch and process data
    with st.spinner(f"Loading data for {symbol}..."):
        df, stock_info = get_price_data(symbol, period, interval)

    if df.empty:
        st.error(f"❌ No data available for symbol: {symbol}")
        st.info("💡 Try a different symbol or check if the market is open.")
        st.stop()

    # Stock info header
    if stock_info:
        company_name = stock_info.get('longName', stock_info.get('shortName', symbol))
        st.markdown(f'<div class="stock-header">{company_name} ({symbol.upper()})</div>', unsafe_allow_html=True)


    # Show summary cards
    render_summary_cards(df)

    st.divider()

    # Main Price Chart
    long_name = stock_info.get('longName', symbol)
    df = calculate_technical_indicators(df)
    fig = create_price_chart(df, chart_type, selected_mas=selected_mas)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()


    # Additional Metrics Display
    metrics = calculate_risk_metrics(df)
    show_analysis_summary(df, metrics, stock_info)
    

    # Footer
    st.markdown(f"""
        <div style="
            font-size: 12px;
            font-family: 'Libertinus Math', system-ui, sans-serif;
            margin-top: 100px;
            text-align: left;
        ">Data from Yahoo Finance • Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    """, unsafe_allow_html=True)


with tab2:
    show_technical_analysis(symbol, period, interval, chart_type, selected_mas=selected_mas)
