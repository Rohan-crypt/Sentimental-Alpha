import os
from stable_baselines3 import PPO
from engine import get_market_data

try:
    from trading_env import StockTradingEnv
except ImportError:
    from env import StockTradingEnv

def start_training():
    # To stop the model from just 'Buying Apple', we train on a mix of diverse stocks
    training_tickers = ["AAPL", "GOOGL", "RELIANCE.NS"]
    
    # 1. Initialize with first ticker
    df = get_market_data(training_tickers[0], period="2y")
    env = StockTradingEnv(df, ticker=training_tickers[0])
    
    agent = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        device="cpu", 
        tensorboard_log="./logs/",
        learning_rate=0.0003,
        ent_coef=0.1 # Very high entropy to force exploration of Sell/Hold
    )

    # 2. Sequential Multi-Asset Training
    for ticker in training_tickers:
        print(f"\n--- [TRAINING PHASE] Learning from: {ticker} ---")
        df = get_market_data(ticker, period="2y")
        env = StockTradingEnv(df, ticker=ticker)
        agent.set_env(env)
        
        # 50k steps per ticker (Total 150k)
        agent.learn(total_timesteps=50000, reset_num_timesteps=False)

    save_path = "nifty_alpha_brain"
    agent.save(save_path)
    print(f"\n--- [SUCCESS] Diverse brain saved as {save_path}.zip ---")

if __name__ == "__main__":
    start_training()
