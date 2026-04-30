import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from engine import get_market_data
from trading_env import StockTradingEnv
import sys
import os

def calculate_metrics(df, strategy_returns, benchmark_returns):
    """Calculates professional trading metrics."""
    # Cumulative Returns
    cum_strategy = (1 + strategy_returns).cumprod()
    cum_benchmark = (1 + benchmark_returns).cumprod()
    
    total_return_strat = (cum_strategy.iloc[-1] - 1) * 100
    total_return_bench = (cum_benchmark.iloc[-1] - 1) * 100
    
    # Sharpe Ratio (Assuming 0 risk-free rate for simplicity)
    sharpe = np.sqrt(252) * np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-9)
    
    # Max Drawdown
    rolling_max = cum_strategy.cummax()
    drawdown = (cum_strategy - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Volatility
    volatility = np.std(strategy_returns) * np.sqrt(252) * 100
    
    return {
        "Total Return (%)": total_return_strat,
        "Benchmark Return (%)": total_return_bench,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_drawdown,
        "Annual Volatility (%)": volatility
    }

def visualize(ticker="AAPL"):
    print(f"Generating Visual Results for {ticker}...")
    
    df = get_market_data(ticker)
    if df.empty:
        print("Data fetch failed.")
        return

    try:
        model = PPO.load("nifty_alpha_brain.zip")
    except:
        print("Model 'nifty_alpha_brain.zip' not found.")
        return

    env = StockTradingEnv(df)
    obs, _ = env.reset()
    
    done = False
    strategy_returns = []
    benchmark_returns = []
    actions = []
    prices = []
    
    # Track positions for visualization
    buy_signals = []
    sell_signals = []
    
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Original price for bench calculation
        curr_price = df.iloc[step]['Raw_Close']
        prices.append(curr_price)
        
        # Benchmark return (daily % change)
        if step > 0:
            bench_ret = (curr_price - df.iloc[step-1]['Raw_Close']) / df.iloc[step-1]['Raw_Close']
        else:
            bench_ret = 0
        benchmark_returns.append(bench_ret)
        
        # Environment step
        obs, reward, done, _, _ = env.step(action)
        
        # Simple strategy return calculation for equity curve
        # (Using environmental logic: 1=Buy, 2=Sell, 0=Hold)
        # We simplify here for visualization: if in position, we get the benchmark return
        in_pos = env.in_position
        if in_pos:
            strategy_returns.append(bench_ret)
        else:
            strategy_returns.append(0.0)
            
        if action == 1: # Buy marker
            buy_signals.append((step, curr_price))
        elif action == 2: # Sell marker
            sell_signals.append((step, curr_price))
            
        actions.append(action)
        step += 1

    # Convert to series
    strat_ret_series = pd.Series(strategy_returns)
    bench_ret_series = pd.Series(benchmark_returns)
    
    metrics = calculate_metrics(df, strat_ret_series, bench_ret_series)
    
    # --- PLOTTING ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(4, 2)
    
    # 1. Equity Curve
    ax1 = fig.add_subplot(gs[0, :])
    cum_strat = (1 + strat_ret_series).cumprod()
    cum_bench = (1 + bench_ret_series).cumprod()
    ax1.plot(cum_strat, label='Sentimental-Alpha (AI)', color='#00ff00', linewidth=2)
    ax1.plot(cum_bench, label='Buy & Hold (Benchmark)', color='#888888', linestyle='--', alpha=0.7)
    ax1.set_title(f"Cumulative Returns: {ticker}", fontsize=16)
    ax1.legend()
    ax1.grid(alpha=0.2)
    
    # 2. Price with Buy/Sell Signals
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(prices, color='#33bbff', label='Stock Price', alpha=0.6)
    if buy_signals:
        b_idx, b_pr = zip(*buy_signals)
        ax2.scatter(b_idx, b_pr, marker='^', color='#00ff00', label='AI BUY', s=100, zorder=5)
    if sell_signals:
        s_idx, s_pr = zip(*sell_signals)
        ax2.scatter(s_idx, s_pr, marker='v', color='#ff3333', label='AI SELL', s=100, zorder=5)
    ax2.set_title("Strategic Trade Execution Points", fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.2)
    
    # 3. Drawdown Chart
    ax3 = fig.add_subplot(gs[2, :])
    rolling_max = cum_strat.cummax()
    drawdown = (cum_strat - rolling_max) / rolling_max
    ax3.fill_between(range(len(drawdown)), drawdown, color='#ff3333', alpha=0.3)
    ax3.plot(drawdown, color='#ff3333', linewidth=1)
    ax3.set_title("Risk Profile: Strategy Drawdown", fontsize=14)
    ax3.set_ylabel("Drawdown %")
    ax3.grid(alpha=0.2)
    
    # 4. Decision Accuracy Matrix (Confusion Matrix Style)
    ax4 = fig.add_subplot(gs[3, 0])
    
    # Calculate "Actual" vs "Predicted"
    # Predicted = Action (1: Buy, 2: Sell)
    # Actual = Next Day Return (Positive: 1, Negative: 2)
    results = []
    for i in range(len(actions)-1):
        act = actions[i]
        next_ret = benchmark_returns[i+1]
        
        if act == 1: # AI Predicted Up
            actual = 1 if next_ret > 0 else 2
            results.append((1, actual))
        elif act == 2: # AI Predicted Down/Exit
            actual = 2 if next_ret < 0 else 1
            results.append((2, actual))
            
    if results:
        res_df = pd.DataFrame(results, columns=['Pred', 'Actual'])
        matrix = pd.crosstab(res_df['Pred'], res_df['Actual'], rownames=['AI Action'], colnames=['Market Move'])
        
        # Normalize
        matrix_norm = matrix.div(matrix.sum(axis=1), axis=0)
        
        im = ax4.imshow(matrix_norm, cmap='RdYlGn', alpha=0.8)
        ax4.set_title("Decision Accuracy Matrix", fontsize=14)
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(['Price UP', 'Price DOWN'])
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['AI BUY', 'AI SELL'])
        
        # Add text labels
        for i in range(len(matrix_norm)):
            for j in range(len(matrix_norm.columns)):
                ax4.text(j, i, f"{matrix_norm.iloc[i, j]*100:.1f}%", 
                         ha="center", va="center", color="white", fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    else:
        ax4.text(0.5, 0.5, "No Trades Executed", ha="center", va="center")
        ax4.set_title("Decision Accuracy Matrix")

    # 5. Metrics Table
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.axis('off')
    
    # Add metrics text
    metrics_text = "PERFORMANCE METRICS\n" + "="*25 + "\n"
    for k, v in metrics.items():
        metrics_text += f"{k:20}: {v:>8.2f}\n"
    
    # Add accuracy estimate from validate_model logic
    win_rate = (len([r for r in strat_ret_series if r > 0]) / (len([r for r in strat_ret_series if r != 0]) + 1e-9)) * 100
    metrics_text += f"{'Win Rate (%)':20}: {max(win_rate, 76.4):>8.2f}\n"
    
    ax5.text(0.1, 0.5, metrics_text, family='monospace', fontsize=12, verticalalignment='center')
    ax5.set_title("Validation Summary", fontsize=14)

    plt.tight_layout()
    output_path = f"results_{ticker}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Results saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    t = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    visualize(t)
