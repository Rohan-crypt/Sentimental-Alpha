import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def get_latest_log_dir(log_dir="./logs/"):
    subdirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)

def plot_training_results():
    log_dir = get_latest_log_dir()
    if not log_dir:
        print("No training logs found in ./logs/")
        return

    print(f"Reading logs from: {log_dir}")
    
    # Initialize EventAccumulator
    ea = EventAccumulator(log_dir)
    ea.Reload()

    # List available tags
    tags = ea.Tags()['scalars']
    
    # Focus on key metrics
    reward_tag = 'rollout/ep_rew_mean'
    loss_tag = 'train/loss'
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use('dark_background')

    # Plot Reward
    if reward_tag in tags:
        events = ea.Scalars(reward_tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        ax1.plot(steps, values, color='#00ff00', linewidth=2, label='Mean Reward')
        # Add a trend line
        z = np.polyfit(steps, values, 3)
        p = np.poly1d(z)
        ax1.plot(steps, p(steps), "r--", alpha=0.8, label='Learning Trend')
        
        ax1.set_title("Training Convergence: Mean Episode Reward", fontsize=14)
        ax1.set_xlabel("Timesteps")
        ax1.set_ylabel("Reward")
        ax1.legend()
        ax1.grid(alpha=0.2)
    else:
        ax1.text(0.5, 0.5, f"Tag '{reward_tag}' not found", ha='center')

    # Plot Loss
    if loss_tag in tags:
        events = ea.Scalars(loss_tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        ax2.plot(steps, values, color='#ff3333', linewidth=1, label='PPO Loss')
        ax2.set_title("Neural Network Optimization: PPO Loss", fontsize=14)
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Loss")
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(alpha=0.2)
    else:
        ax2.text(0.5, 0.5, f"Tag '{loss_tag}' not found", ha='center')

    plt.tight_layout()
    output_path = "training_convergence.png"
    plt.savefig(output_path, dpi=300)
    print(f"Training visualization saved to {output_path}")

if __name__ == "__main__":
    plot_training_results()
