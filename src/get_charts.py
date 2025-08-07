# Import libraries
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import os
import numpy as np

@st.cache_data
def load_config():
    """Load configuration from YAML file"""
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        st.error(f"Configuration file not found: {config_path}")
        st.stop()

    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


config = load_config()
COLORS = config['theme']['colors']


def add_price_trace(fig, df, chart_type):
    if chart_type == "candlestick":
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

    elif chart_type == "heikin_ashi":
        ha_df = df.copy()
        ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

        ha_open = []
        for i in range(len(df)):
            if i == 0:
                ha_open.append((df['Open'].iloc[0] + df['Close'].iloc[0]) / 2)
            else:
                ha_open.append((ha_open[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2)
        ha_df['HA_Open'] = ha_open

        ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
        ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)

        fig.add_trace(go.Candlestick(
            x=ha_df.index,
            open=ha_df['HA_Open'],
            high=ha_df['HA_High'],
            low=ha_df['HA_Low'],
            close=ha_df['HA_Close'],
            increasing_line_color=COLORS["up"],
            decreasing_line_color=COLORS["down"],
            name="Heikin Ashi"
        ), row=1, col=1)

    elif chart_type == "line":
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            line=dict(color=COLORS["primary"], width=2),
            name="Close Price"
        ), row=1, col=1)


def add_moving_averages(fig, df, selected_mas):
    if df.empty or 'Close' not in df.columns or not selected_mas:
        return

    color_map = {
        "20_day": COLORS.get("warning", "orange"),
        "50_day": COLORS.get("info", "blue"),
        "100_day": COLORS.get("success", "green"),
        "200_day": COLORS.get("primary", "black"),
        "4_week": COLORS.get("warning", "orange"),
        "20_week": COLORS.get("warning", "orange"),
        "50_week": COLORS.get("info", "blue"),
        "100_week": COLORS.get("success", "green"),
        "200_week": COLORS.get("primary", "black")
    }

    # Plot selected MAs
    for label, window in selected_mas:
        clean_label = label.replace("-", "_")
        col_name = f"SMA_{clean_label}"
        if col_name not in df.columns:
            df[col_name] = df['Close'].rolling(window=window).mean()
        ma_col = f"SMA_{clean_label}"

        if ma_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[ma_col],
                line=dict(
                    color=color_map.get(clean_label, "gray"),
                    width=1.5,
                    dash="dash"
                ),
                name=label,
                hovertemplate=f"<b>{label}:</b> %{{y:.2f}}<extra></extra>"
            ), row=1, col=1)

    # Add crossover signals if at least two MAs are selected
    if len(selected_mas) >= 2:
        sorted_mas = sorted(selected_mas, key=lambda x: x[1])

        for i in range(len(sorted_mas) - 1):
            fast_label = sorted_mas[i][0].replace("-", "_")
            slow_label = sorted_mas[i + 1][0].replace("-", "_")

            fast_col = f"SMA_{fast_label}"
            slow_col = f"SMA_{slow_label}"

            if fast_col in df.columns and slow_col in df.columns:
                fast_ma = df[fast_col]
                slow_ma = df[slow_col]

                crossover = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
                crossdown = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

                # Cross above
                for idx in df.index[crossover]:
                    fig.add_trace(go.Scatter(
                        x=[idx],
                        y=[df.loc[idx, 'Close']],
                        mode='markers',
                        marker=dict(symbol='triangle-up', color='green', size=10),
                        name="Bullish crossover",
                        showlegend=False
                    ))

                # Cross below
                for idx in df.index[crossdown]:
                    fig.add_trace(go.Scatter(
                        x=[idx],
                        y=[df.loc[idx, 'Close']],
                        mode='markers',
                        marker=dict(symbol='triangle-down', color='red', size=10),
                        name="Bearish crossover",
                        showlegend=False
                    ))


def add_volume_bars(fig, df):
    """Add volume bars to the chart"""
    volume_colors = [
        COLORS["up"] if close >= open_ else COLORS["down"]
        for close, open_ in zip(df['Close'], df['Open'])
    ]

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        marker_color=volume_colors,
        name="Volume",
        opacity=0.9
    ), row=2, col=1)


def style_axes(fig):
    # Apply axis titles and font styles
    axis_font = dict(size=10, family='Poppins')
    title_font = dict(size=12, family='Poppins', color=COLORS["text_secondary"])

    fig.update_yaxes(
        title_text="Price ($)", row=1, col=1,
        title_font=title_font, tickfont=axis_font
    )
    fig.update_yaxes(
        title_text="Volume", row=2, col=1,
        title_font=title_font, tickfont=axis_font
    )
    fig.update_xaxes(
        title_text="Date", row=2, col=1,
        title_font=title_font, tickfont=axis_font
    )


def analyze_trend_and_volume(df):
    """Analyze recent price trend and volume strength."""
    recent = df[-20:]  # last ~20 intervals
    price_change = recent['Close'].iloc[-1] - recent['Close'].iloc[0]
    pct_change = (price_change / recent['Close'].iloc[0]) * 100
    slope = np.polyfit(range(len(recent)), recent['Close'], 1)[0]
    recent_vol = recent['Volume'].mean()
    avg_vol = df['Volume'].mean()

    # Determine trend direction
    if pct_change > 2 and slope > 0:
        trend = "📈 Uptrend"
        color = COLORS["success"]
    elif pct_change < -2 and slope < 0:
        trend = "📉 Downtrend"
        color = COLORS["danger"]
    else:
        trend = "🔁 Sideways market"
        color = COLORS["neutral"]

    # Determine volume strength
    if recent_vol > avg_vol * 1.2:
        volume_note = "with strong volume participation"
    elif recent_vol < avg_vol * 0.8:
        volume_note = "on weak volume"
    else:
        volume_note = "with average volume"

    return f"{trend} {volume_note}", color


def create_price_chart(df, chart_type: str,  selected_mas: list):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )

    add_price_trace(fig, df, chart_type)
    add_moving_averages(fig, df, selected_mas)
    add_volume_bars(fig, df)
    style_axes(fig)
    title_text, title_color = analyze_trend_and_volume(df)
    fig.update_layout(
        title=dict(
            text=title_text,
            x=0.5,
            xanchor='center',
            font=dict(size=18, family='Poppins', color=title_color)
        ),
        height=400,
        template="simple_white",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(t=70, b=50, l=40, r=40),
        legend=dict(
            orientation="h",
            y=1.05,
            x=1,
            xanchor="right",
            font=dict(family='Poppins', size=12)
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#000000",
            font_size=15,
            font_family="Roboto"
        )
    )

    return fig