import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from engine import get_market_data 

st.set_page_config(page_title="Sentimental-Alpha v2.0", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

st.title("Sentimental-Alpha: Research Terminal")

with st.expander("MARKET CONTROLS & WATCHLIST", expanded=True):
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        watchlist = ["RELIANCE.NS", "^NSEI", "AAPL", "GOOGL", "TSLA", "BTC-USD", "CUSTOM"]
        selected_ticker = st.selectbox("Market Watchlist", watchlist)
    with c2:
        ticker = st.text_input("Custom Ticker", "RELIANCE.NS").upper() if selected_ticker == "CUSTOM" else selected_ticker
        if selected_ticker != "CUSTOM": st.write(f"**Target:** {ticker}")
    with c3:
        st.write("") 
        if st.button("Refresh Data", use_container_width=True): st.rerun()

st.write(f"Analyzing Market Intelligence for **{ticker}**") 

with st.spinner(f"Querying AI Inference Engine for {ticker}..."):
    try:
        df = get_market_data(ticker)
        api_response = requests.get(f"{API_URL}/predict/{ticker}", timeout=20)
        api_status = api_response.status_code == 200
        if api_status: 
            ai_data = api_response.json()
        else: 
            st.error(f"AI Service Error ({api_response.status_code}): {api_response.text}")
            api_status = False
    except Exception as e:
        st.warning(f"Connection Failed: {e}")
        api_status = False
        df = None

if df is not None and not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    curr_price = df['Raw_Close'].iloc[-1]
    currency_sym = "Rs" if ".NS" in ticker or ticker == "^NSEI" else "$"
    m1.metric(f"{ticker} PRICE", f"{currency_sym}{curr_price:,.2f}")
    
    if api_status:
        m2.metric("AI SIGNAL", ai_data['signal'], delta=f"{ai_data['confidence']}% Confidence")
        rsi = ai_data['rsi']
        m3.metric("RSI (14)", f"{rsi:.2f}", delta="OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL")
        ema = ai_data['ema']
        m4.metric("EMA (20)", f"{ema:,.1f}", delta="BULLISH" if curr_price > ema else "BEARISH")
    
    # Restored Candlestick Visualization
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Raw_Open'], high=df['Raw_High'], 
        low=df['Raw_Low'], close=df['Raw_Close'], name="Price Action"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20_Raw'], line=dict(color='orange', width=1.5), name="EMA 20"))
    
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.subheader("Neural Intelligence")
        if api_status:
            sentiment = ai_data['sentiment']
            if sentiment > 0.1: st.success(f"MARKET MOOD: BULLISH ({sentiment:+.2f})")
            elif sentiment < -0.1: st.error(f"MARKET MOOD: BEARISH ({sentiment:+.2f})")
            else: st.warning(f"MARKET MOOD: NEUTRAL ({sentiment:+.2f})")
            st.write(f"The PPO Policy Agent has issued a **{ai_data['signal']}** command.")
    with col_r:
        st.subheader("Market Intelligence Feed")
        from run_sentiment import get_news
        headlines = get_news(ticker)
        for h in headlines[:5]: st.write(f"- {h}")
else:
    st.error("Market data unavailable.")
