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
- **RL Algorithm:** Proximal Policy Optimization (PPO) via Stable Baselines 3.
- **Sentiment Model:** ProsusAI/FinBERT.
- **Data Source:** Yahoo Finance (Real-time and Historical).
- **Indicators:** EMA 20, RSI 14 (Wilder's Smoothing).
