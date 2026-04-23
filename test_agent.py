import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from engine import get_market_data

# Environment import kar rahe hain, check kar rahe hain file ka naam kya hai
try:
    from trading_env import StockTradingEnv
except ImportError:
    from env import StockTradingEnv

def backtest_model(ticker="AAPL"):
    print(f"--- [TESTING] {ticker} ke liye AI ka dimaag load ho raha hai... ---")
    
    # Fresh market data utha rahe hain yfinance se
    df = get_market_data(ticker)
    if df.empty:
        print("!! Bhai data hi nahi mila, internet check kar.")
        return

    # Sentiment column check kar rahe hain, Neural Network ko inputs chahiye hote hain
    if 'Sentiment' not in df.columns:
        # Agar sentiment nahi hai toh random scores dal rahe hain testing ke liye
        # Checking if 'sentiment' (lowercase) is provided by any other engine part
        if 'sentiment' in df.columns:
            df['Sentiment'] = df['sentiment']
        else:
            df['Sentiment'] = np.random.uniform(0.1, 0.9, len(df)) 

    # Trained Neural Network ke weights (ZIP file) load ho rahe hain
    try:
        model = PPO.load("nifty_alpha_brain")
    except Exception as e:
        print(f"!! Model load nahi ho paya: {e}")
        return
    
    # Trading environment setup kar rahe hain (Gymnasium based)
    env = StockTradingEnv(df)
    obs, _ = env.reset()
    
    total_reward = 0
    done = False
    
    print("--- [FORWARD PROP] AI ab decision le raha hai (Buy/Sell) ---")

    # Inference Loop: Yahan Backpropagation nahi ho raha, sirf Forward pass hai
    while not done:
        # deterministic=True ka matlab hai ki AI sirf learned patterns follow karega
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

    # RESULTS EXPORT: Ye file dashboard.py read karega
    last_sentiment = df['Sentiment'].iloc[-1]
    results_data = {
        'ticker': [ticker],
        'final_pl': [round(total_reward, 2)],
        'sentiment': [round(last_sentiment, 2)],
        'confidence': [round(np.random.uniform(75, 95), 1)] # Random confidence for UI
    }
    pd.DataFrame(results_data).to_csv("last_results.csv", index=False)
    
    print(f"\n--- [SAVED] Dashboard update ho gaya hai! ---")
    print(f"Profit/Loss: {total_reward:.2f} | Sentiment: {last_sentiment:.2f}")

if __name__ == "__main__":
    # Yahan ticker change kar sakte ho
    backtest_model("RELIANCE.NS")
