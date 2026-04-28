import yfinance as yf
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuration for FinBERT model
FINBERT_MODEL = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

def analyze_sentiment(text):
    if not text: return 0.0
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][0].item() - probs[0][2].item() 

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    gain_values, loss_values = gain.values, loss.values
    avg_gain_values, avg_loss_values = avg_gain.values, avg_loss.values
    for i in range(window, len(data)):
        avg_gain_values[i] = (avg_gain_values[i-1] * (window - 1) + gain_values[i]) / window
        avg_loss_values[i] = (avg_loss_values[i-1] * (window - 1) + loss_values[i]) / window
    rs = pd.Series(avg_gain_values, index=data.index) / pd.Series(avg_loss_values, index=data.index)
    return 100 - (100 / (1 + rs))

def get_market_data(ticker="RELIANCE.NS", period="2y"):
    """
    Standardizes market data with technical indicators and normalized returns.
    Preserves raw OHLC for visual dashboard.
    """
    raw_data = yf.download(ticker, period=period, interval="1d", progress=False)
    if raw_data.empty: return pd.DataFrame()
    if isinstance(raw_data.columns, pd.MultiIndex): raw_data.columns = raw_data.columns.get_level_values(0)

    # 1. Preserve Raw OHLC for Charting
    raw_data['Raw_Open'] = raw_data['Open']
    raw_data['Raw_High'] = raw_data['High']
    raw_data['Raw_Low'] = raw_data['Low']
    raw_data['Raw_Close'] = raw_data['Close']
    
    # 2. Technical Indicators
    raw_data['EMA_20_Raw'] = raw_data['Close'].ewm(span=20, adjust=False).mean()
    raw_data['RSI_14'] = calculate_rsi(raw_data['Close'], window=14)
    
    ema_12 = raw_data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = raw_data['Close'].ewm(span=26, adjust=False).mean()
    raw_data['MACD'] = ema_12 - ema_26
    raw_data['MACD_Signal'] = raw_data['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. Trend Momentum
    raw_data['Return_1d'] = raw_data['Close'].pct_change()
    raw_data['Return_3d'] = raw_data['Close'].pct_change(periods=3)
    raw_data['Return_5d'] = raw_data['Close'].pct_change(periods=5)
    
    # 4. Dummy target for compatibility
    raw_data['Target_Next_5d'] = 0 

    # 5. Normalization for AI Observation Space
    raw_data['RSI_14_Norm'] = raw_data['RSI_14'] / 100.0
    raw_data['EMA_Dist'] = (raw_data['Close'] / raw_data['EMA_20_Raw']) - 1.0
    
    # Normalize MACD by dividing by rolling standard deviation (volatility)
    volatility = raw_data['Close'].rolling(20).std()
    raw_data['MACD_Norm'] = np.clip(raw_data['MACD'] / volatility, -1, 1).fillna(0)
    raw_data['MACD_Signal_Norm'] = np.clip(raw_data['MACD_Signal'] / volatility, -1, 1).fillna(0)
    
    # Normalize Volume (percent change)
    if 'Volume' in raw_data.columns:
        raw_data['Volume_Norm'] = np.clip(raw_data['Volume'].pct_change() * 10, -1, 1).fillna(0)
    else:
        raw_data['Volume_Norm'] = 0.0

    # Add a continuous Sentiment score (since running FinBERT on 500+ historical rows is too slow, we simulate it based on momentum)
    raw_data['Sentiment'] = np.clip((raw_data['RSI_14'] - 50) / 50.0 + (raw_data['Close'].pct_change(periods=3) * 10), -1, 1).fillna(0)
    
    for col in ['Return_1d', 'Return_3d', 'Return_5d']:
        raw_data[col] = np.clip(raw_data[col] * 20, -1, 1)
    
    for col in ['Open', 'High', 'Low', 'Close']:
        raw_data[col] = raw_data[col].pct_change() * 20
        
    raw_data.dropna(inplace=True)
    return raw_data

if __name__ == "__main__":
    sample = get_market_data("AAPL")
    print(sample.tail())
