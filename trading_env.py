import gymnasium as gym
from gymnasium import spaces
import numpy as np
from run_sentiment import get_news, get_sentiment

class StockTradingEnv(gym.Env):
    """
    Standard Gymnasium environment for stock trading simulation.
    Combines market indicators and real-time news sentiment.
    """
    def __init__(self, df, ticker="AAPL"):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.ticker = ticker
        
        # 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Market data features + 4 Sentiment features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1] + 4,), dtype=np.float32
        )

        self.current_step = 0
        self.cached_sentiment = None

    def _get_sentiment(self):
        """
        Calculates sentiment scores once per environment instance.
        """
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
        market_obs = self.df.iloc[self.current_step].values.astype(np.float32)
        sentiment_obs = self._get_sentiment()
        return np.concatenate([market_obs, sentiment_obs]), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        reward = 0
        if not done:
            current_close = self.df.iloc[self.current_step]['Close']
            next_close = self.df.iloc[self.current_step + 1]['Close']
            diff = next_close - current_close
            
            if action == 1:  # Buy
                reward = diff
            elif action == 2:  # Sell
                reward = -diff
            else:  # Hold
                reward = -0.01 

        market_obs = self.df.iloc[self.current_step].values.astype(np.float32)
        sentiment_obs = self._get_sentiment()
        obs = np.concatenate([market_obs, sentiment_obs])
        return obs, reward, done, False, {}
