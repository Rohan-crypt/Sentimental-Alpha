import pandas as pd
import numpy as np
import sys
from stable_baselines3 import PPO
from engine import get_market_data
from trading_env import StockTradingEnv

def run_validation(ticker="AAPL"):
    print(f"\n--- Performance Validation: {ticker} ---")
    
    df = get_market_data(ticker)
    if df.empty:
        print(f"Error: Could not fetch data for {ticker}.")
        return
        
    try:
        model = PPO.load("nifty_alpha_brain")
    except Exception:
        print("Error: Brain model not found. Please train the model first.")
        return

    env = StockTradingEnv(df, ticker=ticker)
    obs, _ = env.reset()
    
    done = False
    rewards = []
    actions = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
    
    total_steps = len(rewards)
    buy_actions = actions.count(1)
    sell_actions = actions.count(2)
    hold_actions = actions.count(0)
    
    strategy_return = sum(rewards)
    initial_price = df['Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    benchmark_return = final_price - initial_price
    
    trades = [r for a, r in zip(actions, rewards) if a != 0]
    win_rate = (len([r for r in trades if r > 0]) / len(trades)) * 100 if trades else 0

    print("\n" + "="*40)
    print(f" FINAL VALIDATION REPORT: {ticker}")
    print("="*40)
    print(f"Testing Period:     {total_steps} trading days")
    print(f"Total AI Return:    {strategy_return:.2f} pts")
    print(f"Buy & Hold Return:  {benchmark_return:.2f} pts")
    print(f"Model Win Rate:     {win_rate:.1f}%")
    print("-" * 40)
    print(f"AI Actions -> Buy: {buy_actions} | Sell: {sell_actions} | Hold: {hold_actions}")
    
    if strategy_return > benchmark_return:
        print("\nRESULT: Model OUTPERFORMED the market.")
    else:
        print("\nRESULT: Model UNDERPERFORMED the market.")
    print("="*40)

if __name__ == "__main__":
    # Check if ticker passed as argument
    target = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_validation(target)
