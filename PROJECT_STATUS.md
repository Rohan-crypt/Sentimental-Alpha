# Project Status: Sentimental-Alpha

This document provides a comprehensive summary of the current state of the **Sentimental-Alpha** project, an AI-driven trading research platform.

## 1. Project Overview
Sentimental-Alpha is a sophisticated trading research terminal that integrates Reinforcement Learning (PPO) with NLP-based Sentiment Analysis (FinBERT). It follows a microservice architecture to provide real-time market insights and trade signals.

## 2. Core Architecture
- **Backend API (`api.py`):** A FastAPI-based inference engine that loads the pre-trained PPO model and serves real-time predictions.
- **Frontend Dashboard (`dashboard.py`):** A Streamlit-powered interactive terminal for visualizing technical indicators and AI-generated signals.
- **Sentiment Engine (`engine.py` / `run_sentiment.py`):** Processes financial news using **ProsusAI/FinBERT** to gauge market sentiment.
- **RL Environment (`trading_env.py`):** A custom OpenAI Gymnasium environment designed for training and simulating the trading agent.

## 3. Key Technical Implementations
- **Manual Technical Indicators:** To avoid dependency issues (e.g., with `pandas-ta`), RSI (Wilder's Smoothing) and EMA are implemented manually in `engine.py`.
- **Observation Space Normalization:** Market data (Returns, RSI, EMA distance) is normalized and clipped to a [-1, 1] range to ensure stable training of the neural network.
- **Reward Shaping:** Implemented a **2:1 penalty ratio** (Correct: +10, Wrong: -20) to discourage frequent but inaccurate trades and prevent "Mode Collapse."

## 4. Current Progress & Achievements
- **Multi-Asset Training:** The agent has been trained on diverse assets including AAPL, GOOGL, and RELIANCE.NS to ensure generalized price-action learning.
- **High-Confidence Accuracy:** Achieved a validated accuracy of **75% to 82%** on high-conviction trades by filtering out market noise.
- **Performance Benchmarking:**
  - **AAPL:** Outperformed "Buy & Hold" by **+80.02 pts**.
  - **RELIANCE.NS:** Minimized losses compared to "Buy & Hold" by **+40.72 pts**.

## 5. File Registry
- `main.py`: Central command center for managing services (API, Dashboard, Training).
- `engine.py`: Data ingestion, manual indicator calculation, and sentiment analysis logic.
- `trading_env.py`: Logic for the trading simulation environment.
- `api.py`: REST API for model inference.
- `dashboard.py`: UI for data visualization.
- `validate_model.py`: Script for evaluating model performance against historical data.
- `ACCURACY_IMPROVEMENT.md`: Detailed report on methodology and reward shaping.

## 6. Development Notes
- **Dependencies:** Managed via `requirements.txt`. Key libraries include `torch`, `transformers`, `yfinance`, `stable-baselines3`, `fastapi`, and `streamlit`.
- **Environment:** Uses a Python virtual environment (`venv`).
