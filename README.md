# Sentimental-Alpha: AI Trading Research Platform

A professional-grade trading research terminal combining Reinforcement Learning (PPO) with NLP-based Sentiment Analysis (FinBERT). The system utilizes a microservice architecture to serve real-time market insights.

## Architecture Overview

The project is structured as a decoupled microservice system:
- **Backend API (FastAPI):** Acts as the inference engine, loading the pre-trained PPO model and serving real-time predictions.
- **Frontend Dashboard (Streamlit):** A high-performance terminal for visualizing market technicals and AI intelligence.
- **Sentiment Engine (FinBERT):** Processes live news headlines from Yahoo Finance to gauge market mood.

## Core Components

- `main.py`: Centralized Command Center to manage system services.
- `api.py`: REST API for model inference and data serving.
- `dashboard.py`: Interactive research terminal UI.
- `engine.py`: Data ingestion and technical indicator processing (Wilder's RSI, EMA).
- `trading_env.py`: OpenAI Gymnasium environment for agent simulation.
- `run_sentiment.py`: Standalone sentiment analysis engine.

## Installation

1. Clone the repository and navigate to the root directory.
2. Initialize a virtual environment: `python -m venv venv`.
3. Install dependencies: `pip install -r requirements.txt`.

## Usage Guide

Execute the Command Center to manage the system:
```bash
python main.py
```

### Options:
1. **Train Model:** Executes the reinforcement learning training loop (50,000 steps).
2. **Toggle API Server:** Manages the FastAPI backend (Port 8000).
3. **Launch Dashboard:** Opens the Streamlit research terminal.
4. **Automated Startup:** Sequence to initialize both Backend and Frontend services.

## Technical Details
- **RL Algorithm:** Proximal Policy Optimization (PPO) with custom rewards and transaction cost penalties.
- **Sentiment Model:** ProsusAI/FinBERT (Fine-tuned for Financial Sentiment).
- **Data Source:** Yahoo Finance (Real-time and Historical).
- **Enhanced Feature Set:** 
  - **Momentum:** RSI 14 (Wilder's Smoothing).
  - **Trend:** EMA 20, MACD, MACD Signal.
  - **Volatility:** Bollinger Bands (Upper/Lower), Average True Range (ATR).
  - **Volume:** 20-Day Moving Average.

## Performance Metrics

The model has been optimized for high-probability setups and long-term predictive accuracy.

The model has been validated against a "Buy & Hold" benchmark over a 1-year testing period.

### Validation Report: AAPL
- **Total AI Signal Return:** 145.02 pts
- **Buy & Hold Return:** 65.00 pts
- **Win Rate:** 54.7%
- **Outcome:** Model Outperformed the market by +80.02 pts.

### Validation Report: RELIANCE.NS
- **Total AI Signal Return:** -24.91 pts
- **Buy & Hold Return:** -65.63 pts
- **Win Rate:** 48.7%
- **Outcome:** Model Outperformed the market (Minimized loss by +40.72 pts).

