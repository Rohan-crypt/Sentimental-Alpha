import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from engine import get_market_data
from trading_env import StockTradingEnv

def run_validation(ticker="AAPL"):
    print(f"--- Performance Validation: {ticker} ---")
    
    # 1. Load Data & Model
    df = get_market_data(ticker)
    if df.empty:
        print("Error: Could not fetch data.")
        return
        
    model = PPO.load("nifty_alpha_brain")
    env = StockTradingEnv(df, ticker=ticker)
    obs, _ = env.reset()
    
    # 2. Execution Loop
    done = False
    rewards = []
    actions = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
    
    # 3. Calculate Metrics
    total_steps = len(rewards)
    buy_actions = actions.count(1)
    sell_actions = actions.count(2)
    hold_actions = actions.count(0)
    
    strategy_return = sum(rewards)
    
    # Buy & Hold Benchmark
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    benchmark_return = final_price - initial_price
    
    # Win Rate (Simplified: % of non-hold steps where reward > 0)
    trades = [r for a, r in zip(actions, rewards) if a != 0]
    win_rate = (len([t for r in trades if r > 0]) / len(trades)) * 100 if trades else 0

    # 4. Professional Output
    print("\n" + "="*30)
    print(f" FINAL VALIDATION REPORT: {ticker}")
    print("="*30)
    print(f"Testing Period: {total_steps} trading days")
    print(f"Total AI Signal Return: {strategy_return:.2f} pts")
    print(f"Buy & Hold Return:      {benchmark_return:.2f} pts")
    print(f"Model Win Rate:         {win_rate:.1f}%")
    print("-" * 30)
    print(f"AI Decision Split: Hold: {hold_actions} | Buy: {buy_actions} | Sell: {sell_actions}")
    
    if strategy_return > benchmark_return:
        print("\nRESULT: Model OUTPERFORMED the market.")
    else:
        print("\nRESULT: Model UNDERPERFORMED the market (Needs more training).")
    print("="*30)

if __name__ == "__main__":
    # Test on a fresh ticker the model hasn't focused on
    run_validation("GOOGL")
