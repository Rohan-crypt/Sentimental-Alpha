import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    """
    Custom environment integrating FinBERT sentiment and Financial Stress 
    for Optimized Portfolio Performance (Based on 2025 Research Papers).
    """
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        
        # Define strictly normalized features for the Neural Network Observation Space
        # This prevents Raw_Close (e.g., 150.0) from blowing up the PPO gradients
        self.feature_cols = [
            'Close', 'Open', 'High', 'Low',
            'Return_1d', 'Return_3d', 'Return_5d',
            'RSI_14_Norm', 'EMA_Dist', 'MACD_Norm', 'MACD_Signal_Norm',
            'Volume_Norm', 'Sentiment'
        ]
        self.feature_cols = [c for c in self.feature_cols if c in self.df.columns]
        
        self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        
        # Obs space: features + pos_flag + unrealized_profit + stress_index
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols) + 3,), dtype=np.float32
        )
        self.current_step = 0
        self.in_position = False
        self.entry_price = 0.0
        self.step_history = []

    def _get_obs(self):
        obs = self.df.iloc[self.current_step][self.feature_cols].values.astype(np.float32)
        pos_flag = 1.0 if self.in_position else 0.0
        unrealized = 0.0
        if self.in_position and self.entry_price > 0:
            current_raw_close = self.df.iloc[self.current_step]['Raw_Close']
            unrealized = ((current_raw_close - self.entry_price) / self.entry_price) * 10.0 # Scaled for network
            
        # Paper 3: Financial Stress Index (Volatility + Sentiment Gating)
        # Increases when volatility is high and sentiment is negative
        sentiment = self.df.iloc[self.current_step]['Sentiment'] if 'Sentiment' in self.df.columns else 0.0
        volatility = np.std(self.step_history[-5:]) if len(self.step_history) > 5 else 0.0
        stress_index = volatility * (1.0 - sentiment)
        
        return np.append(obs, [pos_flag, np.clip(unrealized, -1, 1), np.clip(stress_index, -1, 1)]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.in_position = False
        self.entry_price = 0.0
        self.step_history = []
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = 0.0
        
        if not done:
            current_raw_close = self.df.iloc[self.current_step]['Raw_Close']
            prev_raw_close = self.df.iloc[self.current_step - 1]['Raw_Close']
            
            # Safe step return calculation to avoid divide-by-zero
            step_return = (current_raw_close - prev_raw_close) / prev_raw_close if prev_raw_close > 0 else 0.0
            self.step_history.append(step_return)
            
            sentiment = self.df.iloc[self.current_step]['Sentiment'] if 'Sentiment' in self.df.columns else 0.0
            volatility = np.std(self.step_history[-5:]) if len(self.step_history) > 5 else 0.0
            
            # Paper 3: Sentiment-Stress Synergy Model - Dynamic Risk Penalty
            stress_penalty = volatility * (1.0 - sentiment) * 50.0
            
            if action == 1: # BUY
                if not self.in_position:
                    self.in_position = True
                    self.entry_price = current_raw_close
                    reward = -0.1 # Small transaction cost
                else:
                    reward = step_return * 100 # Holding a long position
            elif action == 2: # SELL
                if self.in_position:
                    realized_profit = ((current_raw_close - self.entry_price) / self.entry_price) * 100
                    reward = realized_profit
                    self.in_position = False
                else:
                    reward = -0.5 # Paper 1: PPO Optimization - penalize invalid sell (shorting disabled)
            else: # HOLD
                if self.in_position:
                    reward = step_return * 100
                else:
                    # Reward patience during high stress, penalize missed opportunity during strong positive sentiment
                    if sentiment > 0.5 and volatility < 0.01:
                        reward = -0.2 # Missed a good setup
                    else:
                        reward = 0.1 # Good patience
            
            # Apply gating mechanism if in position
            if self.in_position:
                reward -= stress_penalty
                
        return self._get_obs(), reward, done, False, {}
