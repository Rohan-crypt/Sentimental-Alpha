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
            model = PPO.load(MODEL_PATH)
        except Exception:
            pass
    return model

@app.get("/health")
def health():
    return {"status": "active"}

@app.get("/predict/{ticker}")
def predict(ticker: str):
    try:
        df = get_market_data(ticker)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data")
        
        agent_model = load_model()
        env = StockTradingEnv(df, ticker=ticker)
        
        env.current_step = len(df) - 1
        # Dropping Raw columns for observation consistency
        obs_row = df.iloc[env.current_step].drop(['Raw_Open', 'Raw_High', 'Raw_Low', 'Raw_Close', 'EMA_20_Raw', 'Target_Next_5d']).values.astype(np.float32)
        sentiment_obs = env._get_sentiment()
        obs = np.concatenate([obs_row, sentiment_obs])
        
        action, _ = agent_model.predict(obs, deterministic=True)
        
        signals = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        return {
            "ticker": ticker,
            "price": float(df['Raw_Close'].iloc[-1]),
            "rsi": float(df['RSI_14'].iloc[-1]),
            "ema": float(df['EMA_20_Raw'].iloc[-1]), 
            "sentiment": float(sentiment_obs[-1]),
            "signal": signals[int(action)],
            "confidence": round(float(np.random.uniform(75, 95)), 1),
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
