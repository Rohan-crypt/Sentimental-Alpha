import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from engine import get_market_data

try:
    from trading_env import StockTradingEnv
except ImportError:
    from env import StockTradingEnv

def backtest_model(ticker="AAPL"):
    df = get_market_data(ticker)
    if df.empty:
        return

    model = PPO.load("nifty_alpha_brain")
    env = StockTradingEnv(df, ticker=ticker)
    obs, _ = env.reset()

    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    print(f"Backtest Result for {ticker}: {total_reward:.2f}")

if __name__ == "__main__":
    backtest_model("RELIANCE.NS")
