import gymnasium as gym
from gymnasium import spaces
import numpy as np
from run_sentiment import get_news, get_sentiment

class StockTradingEnv(gym.Env):
    def __init__(self, df, ticker="AAPL"):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.ticker = ticker
        self.action_space = spaces.Discrete(3)
        
        # Observation space must drop raw columns used for dashboard
        raw_cols = ['Raw_Open', 'Raw_High', 'Raw_Low', 'Raw_Close', 'EMA_20_Raw', 'Target_Next_5d']
        obs_shape = df.shape[1] - len(raw_cols) + 4
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
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
        
        raw_cols = ['Raw_Open', 'Raw_High', 'Raw_Low', 'Raw_Close', 'EMA_20_Raw', 'Target_Next_5d']
        row = self.df.iloc[self.current_step].drop(raw_cols).values.astype(np.float32)
        sentiment_obs = self._get_sentiment()
        return np.concatenate([row, sentiment_obs]), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        reward = 0
        if not done:
            current_price = self.df.iloc[self.current_step]['Raw_Close']
            next_price = self.df.iloc[self.current_step + 1]['Raw_Close']
            move = (next_price - current_price) / current_price
            
            threshold = 0.003
            if action == 1: # BUY
                reward = 10.0 if move > threshold else -20.0
            elif action == 2: # SELL
                reward = 10.0 if move < -threshold else -20.0
            else: # HOLD
                reward = 2.0 if abs(move) < 0.01 else -10.0

        raw_cols = ['Raw_Open', 'Raw_High', 'Raw_Low', 'Raw_Close', 'EMA_20_Raw', 'Target_Next_5d']
        row = self.df.iloc[self.current_step].drop(raw_cols).values.astype(np.float32)
        sentiment_obs = self._get_sentiment()
        obs = np.concatenate([row, sentiment_obs])
        return obs, reward, done, False, {}
