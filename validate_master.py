import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from engine import get_market_data
from trading_env import StockTradingEnv
import sys
import os

def create_validation_diagram():
    """Generates a professional-looking workflow diagram of the validation process."""
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    plt.style.use('dark_background')
    
    # Define boxes and text
    boxes = [
        (0.05, 0.4, "Market Data\n(OHLCV + Indicators)", "#2c3e50"),
        (0.25, 0.4, "Sentiment Analysis\n(FinBERT Engine)", "#2980b9"),
        (0.45, 0.4, "RL Environment\n(Observation Space)", "#27ae60"),
        (0.65, 0.4, "PPO Agent\n(Neural Policy)", "#8e44ad"),
        (0.85, 0.4, "Validation Engine\n(Backtest & Metrics)", "#c0392b")
    ]
    
    for x, y, text, color in boxes:
        plt.text(x + 0.07, y + 0.1, text, ha='center', va='center', 
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8),
                 color='white', fontsize=10, fontweight='bold')
        
        if x < 0.8:
            plt.arrow(x + 0.15, y + 0.1, 0.05, 0, head_width=0.03, head_length=0.02, fc='gray', ec='gray')

    plt.title("Sentimental-Alpha: ML Model Validation Workflow", fontsize=16, pad=20)
    plt.savefig("validation_workflow.png", dpi=300, bbox_inches='tight')
    print("Workflow diagram saved to validation_workflow.png")

def run_comprehensive_validation(ticker="AAPL"):
    print(f"Running Comprehensive Validation for {ticker}...")
    
    df = get_market_data(ticker)
    if df.empty: return

    try:
        model = PPO.load("nifty_alpha_brain.zip")
    except:
        print("Model not found.")
        return

    env = StockTradingEnv(df)
    obs, _ = env.reset()
    
    done = False
    strategy_returns = []
    benchmark_returns = []
    step = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        curr_price = df.iloc[step]['Raw_Close']
        
        if step > 0:
            bench_ret = (curr_price - df.iloc[step-1]['Raw_Close']) / df.iloc[step-1]['Raw_Close']
        else:
            bench_ret = 0
        benchmark_returns.append(bench_ret)
        
        obs, reward, done, _, _ = env.step(action)
        
        if env.in_position:
            strategy_returns.append(bench_ret)
        else:
            strategy_returns.append(0.0)
        step += 1

    strat_ret = pd.Series(strategy_returns)
    bench_ret = pd.Series(benchmark_returns)
    
    # --- CALCULATE ADVANCED METRICS ---
    cum_strat = (1 + strat_ret).cumprod()
    cum_bench = (1 + bench_ret).cumprod()
    
    # 1. Rolling Sharpe (60-day)
    rolling_sharpe = strat_ret.rolling(60).apply(lambda x: np.sqrt(252) * x.mean() / (x.std() + 1e-9))
    
    # 2. Monte Carlo Simulation (100 paths)
    mc_paths = []
    for _ in range(100):
        # Sample with replacement from actual returns
        samples = np.random.choice(strategy_returns, size=len(strategy_returns), replace=True)
        mc_paths.append((1 + pd.Series(samples)).cumprod())
    
    # --- PLOTTING MASTER DASHBOARD ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2)
    
    # A. Equity Curve with Confidence Intervals (MC)
    ax1 = fig.add_subplot(gs[0, 0])
    for path in mc_paths:
        ax1.plot(path, color='gray', alpha=0.05)
    ax1.plot(cum_strat, color='#00ff00', linewidth=3, label='AI Strategy (Actual)')
    ax1.plot(cum_bench, color='#888888', linestyle='--', label='Benchmark')
    ax1.set_title("Strategy Robustness: Monte Carlo Paths & Cumulative Returns", fontsize=14)
    ax1.legend()
    
    # B. Monthly Returns Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    # Mock some dates for the heatmap since we might not have them in the simple df
    dates = pd.date_range(end='2024-01-01', periods=len(strat_ret), freq='D')
    heat_df = pd.DataFrame({'Returns': strat_ret.values}, index=dates)
    monthly_ret = heat_df['Returns'].resample('M').apply(lambda x: (1+x).prod() - 1)
    
    # Pivot for heatmap (Year x Month)
    # Since we might have < 1 year, we'll just show the distribution if we can't pivot nicely
    if len(monthly_ret) > 1:
        monthly_ret_df = pd.DataFrame({
            'Month': monthly_ret.index.month,
            'Year': monthly_ret.index.year,
            'Ret': monthly_ret.values * 100
        })
        pivot = monthly_ret_df.pivot(index='Year', columns='Month', values='Ret')
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap='RdYlGn', center=0, ax=ax2)
        ax2.set_title("Validation Heatmap: Monthly Performance (%)", fontsize=14)
    else:
        ax2.text(0.5, 0.5, "Insufficient Data for Heatmap", ha='center')

    # C. Rolling Risk Metrics (Sharpe)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(rolling_sharpe, color='#33bbff', label='Rolling Sharpe (60d)')
    ax3.axhline(0, color='white', linestyle='--', alpha=0.3)
    ax3.set_title("Performance Stability: Rolling Sharpe Ratio", fontsize=14)
    ax3.legend()

    # D. Trade Return Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    trade_returns = [r for r in strategy_returns if r != 0]
    if trade_returns:
        sns.histplot(trade_returns, kde=True, ax=ax4, color='#f1c40f')
        ax4.axvline(0, color='white', linestyle='--')
        ax4.set_title("Risk Profile: Trade Return Distribution", fontsize=14)
    else:
        ax4.text(0.5, 0.5, "No Active Trades", ha='center')

    # E. Drawdown Underwater Plot
    ax5 = fig.add_subplot(gs[2, :])
    rolling_max = cum_strat.cummax()
    drawdown = (cum_strat - rolling_max) / rolling_max
    ax5.fill_between(range(len(drawdown)), drawdown * 100, color='#ff3333', alpha=0.4)
    ax5.plot(drawdown * 100, color='#ff3333', linewidth=1)
    ax5.set_title("Stress Test: Drawdown Depth (%)", fontsize=14)
    ax5.set_ylabel("Percentage")

    plt.tight_layout()
    output_path = f"validation_comprehensive_{ticker}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Comprehensive validation saved to {output_path}")

if __name__ == "__main__":
    create_validation_diagram()
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_comprehensive_validation(ticker)
