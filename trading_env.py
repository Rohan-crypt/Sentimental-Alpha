import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StockTradingEnv(gym.Env):
    """
    Custom environment taaki humara agent Nifty 50 ya AAPL trade kar sake.
    Observation space mein price, technical indicators aur sentiment sab mixed hai.
    """
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        # Select only numeric columns for the observation space
        self.df = df.select_dtypes(include=[np.number])
        
        # Actions: 0 = Hold (Baitho), 1 = Buy (Kharido), 2 = Sell (Becho)
        self.action_space = spaces.Discrete(3)
        
        # Observation space humara engine wala dataframe hai
        # np.float32 use kar rahe hain kyunki PyTorch aur Stable Baselines ismein fast hain
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.df.shape[1],), dtype=np.float32
        )
        self.current_step = 0

    def reset(self, seed=None, options=None):
        # Har episode ke start mein environment ko reset karne ke liye
        super().reset(seed=seed)
        self.current_step = 0
        
        # Pehli row se start karenge
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs, {}

    def step(self, action):
        # Ek step aage badhte hain data mein
        self.current_step += 1
        
        # Check kar rahe hain ki data khatam toh nahi ho gaya
        done = self.current_step >= len(self.df) - 1
        
        # REWARD LOGIC (The most tricky part)
        # Basic idea: Agar Buy kiya aur price badha -> Reward milega.
        # Agar Sell kiya aur price gira -> Toh bhi profit (Shorting logic).
        reward = 0
        if not done:
            # We use 'Close' for reward calculation if available, else the first column (assuming it's price-related)
            price_col = 'Close' if 'Close' in self.df.columns else self.df.columns[0]
            current_close = self.df.iloc[self.current_step][price_col]
            next_close = self.df.iloc[self.current_step + 1][price_col]
            diff = next_close - current_close
            
            if action == 1: # Buy
                reward = diff
            elif action == 2: # Sell
                reward = -diff
            else: # Action 0: Hold
                # Chhota sa penalty taaki agent lazy na ban jaye aur trade karna seekhe
                reward = -0.01 
        
        # Naya state (observation) nikal rahe hain
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        
        # Gym requirements: obs, reward, terminated, truncated, info
        return obs, reward, done, False, {}
