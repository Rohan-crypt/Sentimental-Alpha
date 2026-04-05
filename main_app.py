import os
from stable_baselines3 import PPO
from engine import get_market_data

# Filename mismatch handle karne ke liye try-except lagaya hai
# Kabhi 'trading_env' hota hai toh kabhi 'env', isse code crash nahi hoga
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

    # Sentiment logic: Abhi ke liye neutral (0.5) placeholder rakha hai
    # Presentation ke baad real news headlines map karenge dates ke saath
    if 'Sentiment' not in df.columns:
        df['Sentiment'] = 0.5 

    print("--- [Step 2] RL Environment initialize ho raha hai ---")
    env = StockTradingEnv(df)

    # Policy: MlpPolicy (Multi-Layer Perceptron) best hai numerical data ke liye
    # verbose=1 rakha hai taaki training ke logs terminal pe dikhte rahein
    agent = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log="./logs/")

    print("--- [Step 3] PPO Agent ki training start ho rahi hai ---")
    # 50,000 steps rakhe hain taaki presentation tak model 'theek-thaak' seekh jaye
    agent.learn(total_timesteps=50000)

    # Final 'brain' (model) ko save kar rahe hain
    save_path = "nifty_alpha_brain"
    agent.save(save_path)
    
    print(f"--- [DONE] Model save ho gaya: {save_path}.zip ---")

if __name__ == "__main__":
    # Script execution yahan se start hogi
    start_training()