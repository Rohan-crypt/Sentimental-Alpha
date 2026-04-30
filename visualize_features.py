import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from engine import get_market_data
from stable_baselines3 import PPO
from trading_env import StockTradingEnv
import sys

def analyze_feature_impact(ticker="AAPL"):
    print(f"Analyzing Feature Dynamics for {ticker}...")
    df = get_market_data(ticker)
    if df.empty: return
    
    try:
        model = PPO.load("nifty_alpha_brain.zip")
    except:
        print("Model not found.")
        return

    env = StockTradingEnv(df)
    obs, _ = env.reset()
    
    data_list = []
    done = False
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Capture features and action
        # Ensure we only grab scalar values from the row
        features = {}
        for col in ['RSI_14_Norm', 'EMA_Dist', 'MACD_Norm', 'Sentiment', 'Return_1d']:
            val = df.iloc[step][col]
            # If val is a Series or array (happens with some yfinance versions), take the first item
            if isinstance(val, (np.ndarray, pd.Series)):
                val = val.item() if hasattr(val, 'item') else val[0]
            features[col] = float(val)
            
        features['AI_Action'] = int(action)
        data_list.append(features)
        
        obs, reward, done, _, _ = env.step(action)
        step += 1
        
    analysis_df = pd.DataFrame(data_list)
    
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Feature Correlation Heatmap (Focus on AI Action)
    cols_to_corr = ['RSI_14_Norm', 'EMA_Dist', 'MACD_Norm', 'Sentiment', 'Return_1d', 'AI_Action']
    corr = analysis_df[cols_to_corr].corr()
    
    sns.heatmap(corr, annot=True, cmap='RdYlGn', ax=ax1, center=0)
    ax1.set_title("Neural Feature Correlation: What Drives the AI?", fontsize=14)
    
    # 2. Sentiment vs Action Distribution
    # Show that higher sentiment leads to more BUY actions
    sns.boxplot(x='AI_Action', y='Sentiment', data=analysis_df, ax=ax2, palette='viridis')
    ax2.set_xticklabels(['HOLD', 'BUY', 'SELL'])
    ax2.set_title("Sentiment Synergy: Sentiment Distribution per Action", fontsize=14)
    ax2.set_ylabel("Sentiment Score (FinBERT Augmented)")
    
    plt.tight_layout()
    output_path = "feature_dynamics.png"
    plt.savefig(output_path, dpi=300)
    print(f"Feature dynamics visualization saved to {output_path}")

if __name__ == "__main__":
    analyze_feature_impact()
