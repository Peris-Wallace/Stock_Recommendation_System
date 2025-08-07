# Import libraries
import streamlit as st
import random
import yaml
from datetime import datetime


@st.cache_data
def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)


def shared_sidebar():
    config = load_config()

    # Load config
    default_symbols = config.get('symbols', {}).get('default_pool', ['NVDA'])
    chart_config = config.get('chart', {})
    default_period = chart_config.get('default_period', 'ytd')
    period_labels = chart_config.get('period_labels', {})
    interval_labels = chart_config.get('interval_labels', {})
    interval_suggestions = chart_config.get('interval_suggestions', {})
    chart_types = chart_config.get('chart_types', {
        'candlestick': 'Candlestick',
        'line': 'Line',
        'heikin_ashi': 'Heikin Ashi'
    })

    # Session State
    if 'selected_symbol' not in st.session_state:
        st.session_state.selected_symbol = random.choice(default_symbols)
    default_symbol = st.session_state.get('selected_symbol', 'NVDA')

    with st.sidebar:
        # Ticker Input
        symbol = st.text_input(
            "Enter Symbol",
            value=default_symbol,
            placeholder="e.g., NVDA, BTC-USD, ^GSPC",
            help="Enter a valid ticker symbol",
            key="symbol_input"
        ).upper().strip()

        if symbol:
            st.session_state.selected_symbol = symbol

        # Time Period Selection
        period_keys = list(period_labels.keys())
        period = st.selectbox(
            "Select Time Period",
            options=period_keys,
            format_func=lambda x: period_labels.get(x, x),
            index=period_keys.index(default_period) if default_period in period_keys else 0
        )

        # Interval Selection 
        suggested_intervals = interval_suggestions.get(period, ["1d"])
        interval = st.selectbox(
            "Select Interval",
            options=suggested_intervals,
            format_func=lambda x: interval_labels.get(x, x)
        )

        st.divider()

        # Chart Type Selection
        chart_type_keys = list(chart_types.keys())
        chart_type = st.selectbox(
            "Chart Type",
            options=chart_type_keys,
            format_func=lambda x: chart_types.get(x, x)
        )

        # Indicators
        st.markdown("Indicators")
        st.subheader("Moving Averages")

        ma_choices = {
            "20-day": 20,
            "50-day": 50,
            "100-day": 100,
            "200-day": 200,
            "4-week": 4 * 5,
            "20-week": 20 * 5,
            "50-week": 50 * 5,
            "100-week": 100 * 5,
            "200-week": 200 * 5,
        }

        selected_mas = [
            (label, period) for label, period in ma_choices.items()
            if st.sidebar.selectbox(label, value=False)
        ]

        st.divider()

        # Market Status
        now = datetime.now()
        is_trading_hours = 9 <= now.hour < 16 and now.weekday() < 5
        st.success("🟢 Markets Open") if is_trading_hours else st.error("🔴 Markets Closed")
        st.caption(f"Last updated: {now.strftime('%H:%M:%S')}")

        st.divider()

        # Refresh + Clear
        if st.button("Refresh"):
            st.session_state.selected_symbol = symbol
            st.rerun()


    return symbol, period, interval, chart_type, selected_mas




#     with st.expander("🔬 Advanced Options"):
#         # Export options
#         st.markdown("**Export Data**")
#         export_format = st.radio(
#             "Format",
#             ["CSV", "Excel", "JSON"],
#             horizontal=True
#         )
        
#         if st.button("💾 Export Data", use_container_width=True):
#             if 'df' in locals() and not df.empty:
#                 exported_data = export_data(df, export_format)
#                 if exported_data:
#                     st.download_button(
#                         label=f"Download {export_format}",
#                         data=exported_data,
#                         file_name=f"{symbol}_data.{export_format.lower()}",
#                         mime=f"application/{export_format.lower()}"
#                     )
#             else:
#                 st.warning("No data available to export")

# # Watchlist functionality
#     st.markdown("### ⭐ Watchlist")
    
#     # Initialize watchlist in session state
#     if 'watchlist' not in st.session_state:
#         st.session_state.watchlist = []
    
#     # Add current symbol to watchlist
#     if symbol and st.button("➕ Add to Watchlist", use_container_width=True):
#         if symbol not in st.session_state.watchlist:
#             st.session_state.watchlist.append(symbol)
#             st.success(f"Added {symbol} to watchlist")
#         else:
#             st.warning(f"{symbol} already in watchlist")
    
#     # Display watchlist
#     if st.session_state.watchlist:
#         st.markdown("**My Watchlist:**")
#         for i, watch_symbol in enumerate(st.session_state.watchlist):
#             col1, col2 = st.columns([3, 1])
#             with col1:
#                 if st.button(watch_symbol, key=f"watch_{i}", use_container_width=True):
#                     st.session_state.selected_symbol = watch_symbol
#                     st.rerun()
#             with col2:
#                 if st.button("❌", key=f"remove_{i}"):
#                     st.session_state.watchlist.remove(watch_symbol)
#                     st.rerun()
        
#         if st.button("🗑️ Clear Watchlist"):
#             st.session_state.watchlist = []
#             st.rerun()
#     else:
#         st.info("No symbols in watchlist")
    
#     st.divider()
