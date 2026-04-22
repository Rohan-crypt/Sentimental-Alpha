import gymnasium as gym
from gymnasium import spaces
import numpy as np
from run_sentiment import get_news, get_sentiment

class StockTradingEnv(gym.Env):
    """
    Custom environment for trading.
    Observation space includes market data and sentiment.
    Now uses ticker-specific news via yfinance.
    """
    def __init__(self, df, ticker="AAPL"):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.ticker = ticker
        
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # 🔥 +4 sentiment features added (pos, neg, neu, sentiment)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(df.shape[1] + 4,), dtype=np.float32
        )

        self.current_step = 0
        self.cached_sentiment = None

    def _get_sentiment(self):
        """
        Fetches live sentiment for the specific ticker.
        """
        if self.cached_sentiment is None:
            try:
                # Fetches ticker-specific news (NO API KEY NEEDED)
                headlines = get_news(self.ticker)
                sentiment_data = get_sentiment(headlines)
                
                self.cached_sentiment = np.array(
                    [sentiment_data['pos'], sentiment_data['neg'], sentiment_data['neu'], sentiment_data['sentiment']], 
                    dtype=np.float32
                )
            except Exception as e:
                print(f"[WARN] Sentiment fetch failed: {e}. Using defaults.")
                self.cached_sentiment = np.array([0.33, 0.33, 0.34, 0.0], dtype=np.float32)

        return self.cached_sentiment

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cached_sentiment = None

        market_obs = self.df.iloc[self.current_step].values.astype(np.float32)
        sentiment_obs = self._get_sentiment()

        obs = np.concatenate([market_obs, sentiment_obs])
        return obs, {}

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
