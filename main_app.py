import os
from stable_baselines3 import PPO
from engine import get_market_data

try:
    from trading_env import StockTradingEnv
except ImportError:
    from env import StockTradingEnv

def start_training():
    ticker = "AAPL" 
    df = get_market_data(ticker)
    
    if df.empty:
        return

    env = StockTradingEnv(df, ticker=ticker)
    agent = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./logs/")

    # 50,000 steps for production-grade training
    agent.learn(total_timesteps=50000)

    save_path = "nifty_alpha_brain"
    agent.save(save_path)

if __name__ == "__main__":
    start_training()
