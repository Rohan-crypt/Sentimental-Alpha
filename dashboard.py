import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from engine import get_market_data
import os

# Dashboard ki wide screen setting taaki charts sahi dikhen
st.set_page_config(page_title="Sentimental-Alpha v1.0", layout="wide")

# DATA PIPELINE: AI ke results load kar rahe hain ---
if os.path.exists("last_results.csv"):
    ai_data = pd.read_csv("last_results.csv")
    ai_pl = ai_data['final_pl'].iloc[0]
    ai_sent = ai_data['sentiment'].iloc[0]
    ai_conf = ai_data['confidence'].iloc[0]
    ticker = ai_data['ticker'].iloc[0]
else:
    # Agar file nahi mili toh default values (Fallback)
    ai_pl, ai_sent, ai_conf, ticker = 0.0, 0.5, 0.0, "AAPL"

st.title("🚀 Sentimental-Alpha: Research Terminal")
st.write(f"_Internal Build: v0.6-Alpha ({ticker} Focus)_") 

# Data load hote waqt spinner dikhana achha rehta hai
with st.spinner(f'Live market state fetch ho raha hai...'):
    df = get_market_data(ticker)

if df is not None and not df.empty:
    # TOP ROW: Saare main metrics (KPIs)
    m1, m2, m3, m4 = st.columns(4)
    curr_price = df['Close'].iloc[-1]
    currency_sym = "₹" if ".NS" in ticker else "$"
    
    # Metric 1: Current Market Price
    m1.metric(f"{ticker} PRICE", f"{currency_sym}{curr_price:,.2f}")
    
    # Metric 2: AI ka Backtest Result (Ab ye real-time dynamic hai)
    m2.metric("BACKTEST P/L", f"{ai_pl} pts", delta="WINNING" if ai_pl > 0 else "LOSING")
    
    # Technical Indicators nikal rahe hain DataFrame se
    rsi_val = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 0
    m3.metric("RSI (14)", f"{rsi_val:.2f}")

    ema_val = df['EMA_20'].iloc[-1] if 'EMA_20' in df.columns else 0
    m4.metric("EMA (20)", f"{ema_val:,.1f}")
    # CHART SECTION: Candlestick visualization ---
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], 
        low=df['Low'], close=df['Close'], name="Market"
    )])
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    # AI BRAIN LOGIC: Yahan hum Neural Network output dikha rahe hain ---
    st.divider()
    st.markdown("### 🤖 Agent Prediction Logic (PPO + FinBERT)")
    l, r = st.columns([1, 2])
    
    with l:
        # Sentiment score ke basis pe BUY/SELL signal
        signal = "BUY" if ai_sent > 0.5 else "SELL"
        if signal == "BUY":
            st.success(f"CURRENT SIGNAL: {signal}")
        else:
            st.error(f"CURRENT SIGNAL: {signal}")
        st.info(f"Agent Confidence: {ai_conf}%")
    
    with r:
        st.write("**Neural Network Rationale (Logic):**")
        sentiment_desc = "Bullish" if ai_sent > 0.5 else "Bearish"
        # The caption section has been removed to clean up the UI.
else:
    st.error("Bhai Connection Error hai! Terminal check kar ya internet.")