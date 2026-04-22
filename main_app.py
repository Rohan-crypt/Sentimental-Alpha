import os
from stable_baselines3 import PPO
from engine import get_market_data

# Filename mismatch handle karne ke liye try-except lagaya hai
try:
    from trading_env import StockTradingEnv
except ImportError:
    from env import StockTradingEnv

def start_training():
    # Pehle Nifty 50 tha, par validation ke liye Apple (AAPL) use kar rahe hain
    ticker = "AAPL" 
    
    print(f"--- [Step 1] Data Engine start ho raha hai: {ticker} ke liye ---")
    df = get_market_data(ticker)
    
    if df.empty:
        print("!! Data nahi mila. Internet ya ticker symbol check karo.")
        return

    print("--- [Step 2] RL Environment initialize ho raha hai ---")
    # Ticker pass kar rahe hain news engine ke liye
    env = StockTradingEnv(df, ticker=ticker)

    # Policy: MlpPolicy (Multi-Layer Perceptron) best hai numerical data ke liye
    agent = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./logs/")

    print("--- [Step 3] PPO Agent ki training start ho rahi hai ---")
    # Training the model
    agent.learn(total_timesteps=50000)

    # Final 'brain' (model) ko save kar rahe hain
    save_path = "nifty_alpha_brain"
    agent.save(save_path)
    
    print(f"--- [DONE] Model save ho gaya: {save_path}.zip ---")

if __name__ == "__main__":
    start_training()
