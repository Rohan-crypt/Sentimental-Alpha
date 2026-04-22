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
    """
    Analyzes input text using FinBERT and returns a sentiment score.
    Returns: float (Positive - Negative probability)
    """
    if not text:
        return 0.0
        
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][0].item() - probs[0][2].item() 

def calculate_rsi(data, window=14):
    """
    Calculates Relative Strength Index using Wilder's Smoothing Method.
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    for i in range(window, len(data)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (window - 1) + gain.iloc[i]) / window
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (window - 1) + loss.iloc[i]) / window
        
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_market_data(ticker="RELIANCE.NS"):
    """
    Fetches historical market data and calculates technical indicators.
    """
    raw_data = yf.download(ticker, period="1y", interval="1d", progress=False)
    
    if raw_data.empty:
        return pd.DataFrame()

    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    # Technical indicator calculations
    raw_data['EMA_20'] = raw_data['Close'].ewm(span=20, adjust=False).mean()
    raw_data['RSI_14'] = calculate_rsi(raw_data['Close'], window=14)
    raw_data['Vol_MA'] = raw_data['Volume'].rolling(window=20).mean()
    
    raw_data.dropna(inplace=True)
    return raw_data

if __name__ == "__main__":
    sample = get_market_data("AAPL")
    if not sample.empty:
        print(sample.tail())
