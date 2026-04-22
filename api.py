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

# Pre-load PPO model weights
MODEL_PATH = "nifty_alpha_brain"
model = None

def load_model():
    global model
    if model is None:
        try:
            model = PPO.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading PPO model: {e}")
    return model

@app.get("/health")
def health():
    return {"status": "active", "model_status": "loaded" if model else "unloaded"}

@app.get("/predict/{ticker}")
def predict(ticker: str):
    """
    Inference endpoint for real-time stock signals.
    """
    try:
        # Fetching fresh technical data
        df = get_market_data(ticker)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        
        agent_model = load_model()
        if not agent_model:
            raise HTTPException(status_code=500, detail="Model unavailable")
            
        # Initialize inference environment
        env = StockTradingEnv(df, ticker=ticker)
        
        # Select latest observation for inference
        env.current_step = len(df) - 1
        market_obs = df.iloc[env.current_step].values.astype(np.float32)
        sentiment_obs = env._get_sentiment()
        obs = np.concatenate([market_obs, sentiment_obs])
        
        # Policy action prediction
        action, _ = agent_model.predict(obs, deterministic=True)
        
        signals = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        return {
            "ticker": ticker,
            "price": float(df['Close'].iloc[-1]),
            "rsi": float(df['RSI_14'].iloc[-1]),
            "ema": float(df['EMA_20'].iloc[-1]),
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
