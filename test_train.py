import os
from stable_baselines3 import PPO
from engine import get_market_data
from trading_env import StockTradingEnv

def start_training():
    training_tickers = ["AAPL"]
    df = get_market_data(training_tickers[0], period="1mo")
    env = StockTradingEnv(df)
    
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    agent = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        verbose=1, 
        device="cpu"
    )

    print("--- [TEST] Starting very short training to verify environment ---")
    agent.learn(total_timesteps=100)
    print("--- [TEST] Training step successful ---")

if __name__ == "__main__":
    start_training()
