import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_system_model():
    """Generates a high-quality system architecture diagram for Sentimental-Alpha."""
    plt.figure(figsize=(15, 10))
    plt.axis('off')
    plt.style.use('default') # Use clean white background for the main architecture
    
    # Define layers
    layers = [
        {"name": "Data Ingestion Layer", "color": "#e1f5fe", "items": ["Yahoo Finance API", "Real-time News Feed", "FinBERT Sentiment Engine"]},
        {"name": "Intelligence Layer (RL Agent)", "color": "#e8f5e9", "items": ["PPO Policy Network", "Custom Trading Environment", "State Normalizer"]},
        {"name": "Execution & Service Layer", "color": "#fff3e0", "items": ["FastAPI Signal Server", "Streamlit Dashboard", "Automated Backtester"]}
    ]
    
    y_pos = 0.8
    for layer in layers:
        # Draw layer box
        rect = patches.Rectangle((0.1, y_pos - 0.2), 0.8, 0.25, linewidth=2, edgecolor='black', facecolor=layer['color'], alpha=0.5)
        plt.gca().add_patch(rect)
        
        # Layer title
        plt.text(0.5, y_pos + 0.08, layer['name'], ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Items in layer
        x_items = np.linspace(0.2, 0.8, len(layer['items']))
        for x, item in zip(x_items, layer['items']):
            plt.text(x, y_pos - 0.1, item, ha='center', va='center', fontsize=10, 
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=1))
        
        y_pos -= 0.35

    # Arrows for flow
    plt.arrow(0.5, 0.55, 0, -0.05, head_width=0.02, head_length=0.03, fc='black', ec='black')
    plt.arrow(0.5, 0.2, 0, -0.05, head_width=0.02, head_length=0.03, fc='black', ec='black')

    plt.title("Sentimental-Alpha: System Architecture (v2.0)", fontsize=18, fontweight='bold', pad=30)
    plt.savefig("system_model_v2.png", dpi=300, bbox_inches='tight')
    print("System model diagram saved to system_model_v2.png")

if __name__ == "__main__":
    import numpy as np
    generate_system_model()
