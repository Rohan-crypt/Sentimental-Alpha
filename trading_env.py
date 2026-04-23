import gymnasium as gym
from gymnasium import spaces
import numpy as np
from run_sentiment import get_news, get_sentiment

class StockTradingEnv(gym.Env):
    """
    Sniper v6: Balanced Active Trader.
    Strictly designed to punish 'Always Buy' or 'Always Hold' strategies.
    """
    def __init__(self, df, ticker="AAPL"):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.ticker = ticker
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1] - 1 + 4,), dtype=np.float32
        )
        self.current_step = 0
        self.cached_sentiment = None

    def _get_sentiment(self):
        if self.cached_sentiment is None:
            try:
                headlines = get_news(self.ticker)
                data = get_sentiment(headlines)
                self.cached_sentiment = np.array(
                    [data['pos'], data['neg'], data['neu'], data['sentiment']], 
                    dtype=np.float32
                )
            except Exception:
                self.cached_sentiment = np.array([0.33, 0.33, 0.34, 0.0], dtype=np.float32)
        return self.cached_sentiment

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cached_sentiment = None
        row = self.df.iloc[self.current_step].drop('Raw_Close').values.astype(np.float32)
        sentiment_obs = self._get_sentiment()
        return np.concatenate([row, sentiment_obs]), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        reward = 0
        if not done:
            price_now = self.df.iloc[self.current_step]['Raw_Close']
            price_next = self.df.iloc[self.current_step + 1]['Raw_Close']
            move = (price_next - price_now) / price_now
            
            # --- AGGRESSIVE SYMMETRIC REWARDS ---
            threshold = 0.003 # 0.3% move
            
            if action == 1: # BUY
                if move > threshold:
                    reward = 10.0 # Correct Buy
                elif move < -threshold:
                    reward = -20.0 # Punish wrong buy heavily
                else:
                    reward = -5.0 # Punish buying in flat market
                    
            elif action == 2: # SELL
                if move < -threshold:
                    reward = 10.0 # Correct Sell
                elif move > threshold:
                    reward = -20.0 # Punish wrong sell heavily
                else:
                    reward = -5.0 # Punish selling in flat market
                    
            else: # HOLD
                if abs(move) > 0.01: # Market moved > 1%
                    reward = -10.0 # Heavy punishment for being lazy during big moves
                else:
                    reward = 2.0 # Small reward for being patient in noise

        row = self.df.iloc[self.current_step].drop('Raw_Close').values.astype(np.float32)
        sentiment_obs = self._get_sentiment()
        obs = np.concatenate([row, sentiment_obs])
        return obs, reward, done, False, {}
