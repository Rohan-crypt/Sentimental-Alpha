import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import os
from engine import get_market_data 

# System Configuration
st.set_page_config(page_title="Sentimental-Alpha v2.0", layout="wide", initial_sidebar_state="collapsed")

# Professional UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .control-panel {
        background-color: #161b22;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

API_URL = "http://localhost:8000"

st.title("Sentimental-Alpha: Research Terminal")

# Dashboard Control Panel
with st.expander("MARKET CONTROLS & WATCHLIST", expanded=True):
    c1, c2, c3 = st.columns([2, 2, 1])
    
    with c1:
        watchlist = ["RELIANCE.NS", "NIFTY_50.NS", "AAPL", "GOOGL", "TSLA", "BTC-USD", "CUSTOM"]
        selected_ticker = st.selectbox("Market Watchlist", watchlist)
    
    with c2:
        if selected_ticker == "CUSTOM":
            ticker = st.text_input("Custom Ticker", "RELIANCE.NS").upper()
        else:
            ticker = selected_ticker
            st.write(f"**Target:** {ticker}")
            
    with c3:
        st.write("") 
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()

st.write(f"Analyzing Market Intelligence for **{ticker}**") 

with st.spinner(f"Querying AI Inference Engine for {ticker}..."):
    try:
        # Load market data for visualization
        df = get_market_data(ticker)
        
        # Load AI signals from microservice
        api_response = requests.get(f"{API_URL}/predict/{ticker}", timeout=15)
        api_status = api_response.status_code == 200
        if api_status:
            ai_data = api_response.json()
        else:
            st.warning("AI Service Offline: Ensure api.py is running on port 8000")
            
    except Exception as e:
        st.error(f"Inference Connection Error: {e}")
        df = None

if df is not None and not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    curr_price = df['Close'].iloc[-1]
    currency_sym = "Rs" if ".NS" in ticker else "$"
    
    m1.metric(f"{ticker} PRICE", f"{currency_sym}{curr_price:,.2f}")
    
    if api_status:
        signal = ai_data['signal']
        m2.metric("AI SIGNAL", signal, delta=f"{ai_data['confidence']}% Confidence")
        
        rsi = ai_data['rsi']
        rsi_state = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"
        m3.metric("RSI (14)", f"{rsi:.2f}", delta=rsi_state, delta_color="inverse" if rsi > 70 else "normal")
        
        ema = ai_data['ema']
        trend = "BULLISH" if curr_price > ema else "BEARISH"
        m4.metric("EMA (20)", f"{ema:,.1f}", delta=trend)
    
    # Technical Charting
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1.5), name="EMA 20"))
    
    fig.update_layout(
        template="plotly_dark", 
        xaxis_rangeslider_visible=False, 
        height=550,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Neural Intelligence")
        if api_status:
            sentiment = ai_data['sentiment']
            if sentiment > 0.1:
                st.success(f"MARKET MOOD: BULLISH ({sentiment:+.2f})")
            elif sentiment < -0.1:
                st.error(f"MARKET MOOD: BEARISH ({sentiment:+.2f})")
            else:
                st.warning(f"MARKET MOOD: NEUTRAL ({sentiment:+.2f})")
            
            st.write(f"The PPO Policy Agent has executed a state analysis for {ticker} and issued a **{ai_data['signal']}** command.")
            st.caption(f"Inference Timestamp: {ai_data['timestamp']}")

    with col_r:
        st.subheader("Market Intelligence Feed")
        from run_sentiment import get_news
        headlines = get_news(ticker)
        for h in headlines[:5]:
            st.write(f"- {h}")
        st.caption("Active headlines processed by FinBERT Sentiment Engine.")
else:
    st.error("Market data unavailable: Verify ticker symbol or connection.")
