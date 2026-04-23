import os
from stable_baselines3 import PPO
from engine import get_market_data

try:
    from trading_env import StockTradingEnv
except ImportError:
    from env import StockTradingEnv

def start_training():
    # Training on a mix of Growth (AAPL), Crash (PYPL), Volatile (ARKK), and Sideways (T)
    training_tickers = ["AAPL", "PYPL", "ARKK", "T"]
    
    # 1. Initialize with first ticker
    df = get_market_data(training_tickers[0], period="2y")
    env = StockTradingEnv(df)
    
    # 🎯 UPGRADED BRAIN: Deeper Neural Network for complex pattern recognition
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256])
    )

    agent = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1, 
        device="cpu", 
        tensorboard_log="./logs/",
        learning_rate=0.0002, # Slightly lower LR for deeper networks
        ent_coef=0.1 
    )

    # 2. Sequential Multi-Asset Training
    for ticker in training_tickers:
        print(f"\n--- [TRAINING PHASE] Learning from: {ticker} ---")
        df = get_market_data(ticker, period="2y")
        env = StockTradingEnv(df)
        agent.set_env(env)
        
        # 50k steps per ticker (Higher complexity, more steps might be needed but let's start with 50k for speed)
        agent.learn(total_timesteps=50000, reset_num_timesteps=False)

    save_path = "nifty_alpha_brain"
    agent.save(save_path)
    print(f"\n--- [SUCCESS] Diverse brain saved as {save_path}.zip ---")

if __name__ == "__main__":
    start_training()
