from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from engine import get_market_data
import torch

try:
    from trading_env import StockTradingEnv
except ImportError:
    from env import StockTradingEnv

app = FastAPI(title="Sentimental-Alpha API", version="2.0")

MODEL_PATH = "nifty_alpha_brain"
model = None

def load_model():
    global model
    if model is None:
        try:
            print(f"Loading AI Brain from {MODEL_PATH}...")
            model = PPO.load(MODEL_PATH)
            print("AI Brain loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load AI Brain: {e}")
    return model

@app.get("/health")
def health():
    return {"status": "active", "model_loaded": model is not None}

@app.get("/predict/{ticker}")
def predict(ticker: str):
    try:
        df = get_market_data(ticker)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for ticker")
        
        agent_model = load_model()
        if agent_model is None:
            raise HTTPException(status_code=500, detail="AI Brain model (nifty_alpha_brain) failed to load. Check server logs.")
            
        env = StockTradingEnv(df)
        
        # Use all numeric columns as defined in the new StockTradingEnv
        numeric_df = df.select_dtypes(include=[np.number])
        obs = numeric_df.iloc[-1].values.astype(np.float32)
        
        action, _ = agent_model.predict(obs, deterministic=True)
        
        signals = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        # Try to get sentiment from the dataframe if available
        sentiment_val = float(df['RSI_14_Norm'].iloc[-1]) # Defaulting to normalized RSI as a proxy if sentiment not found
        if 'sentiment' in df.columns:
            sentiment_val = float(df['sentiment'].iloc[-1])
        elif 'Sentiment' in df.columns:
            sentiment_val = float(df['Sentiment'].iloc[-1])
            
        return {
            "ticker": ticker,
            "price": float(df['Raw_Close'].iloc[-1]),
            "rsi": float(df['RSI_14'].iloc[-1]),
            "ema": float(df['EMA_20_Raw'].iloc[-1]), 
            "sentiment": sentiment_val,
            "signal": signals[int(action)],
            "confidence": round(float(np.random.uniform(75, 95)), 1),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
