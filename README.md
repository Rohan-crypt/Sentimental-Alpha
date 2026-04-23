# Sentimental-Alpha: AI Trading Research Platform

Sentimental-Alpha is a professional-grade trading research terminal that integrates **Reinforcement Learning (PPO)** with **NLP-based Sentiment Analysis (FinBERT)**. The system is designed with a decoupled microservice architecture to deliver real-time market insights and high-confidence trade signals.

---

## 🏗️ Project Architecture & Tech Stack

The project is structured into four core layers, each utilizing specialized technologies:

### 1. **Intelligence Layer (AI & ML)**
This layer is the "brain" of the system, responsible for processing data and making decisions.
- **Stable-Baselines3 (PPO):** Implements the Proximal Policy Optimization algorithm, a state-of-the-art Reinforcement Learning method used to train the trading agent.
- **HuggingFace Transformers (FinBERT):** A specialized NLP model (ProsusAI/finbert) fine-tuned for financial sentiment analysis. It processes news headlines to gauge market "mood."
- **Gymnasium (OpenAI Gym):** Provides the standardized environment (`trading_env.py`) where the RL agent "plays" the market to learn optimal trading strategies.
- **PyTorch:** The underlying deep learning framework that powers both the RL policy network and the Sentiment model.

### 2. **Data & Engineering Layer**
Handles data ingestion, technical analysis, and feature engineering.
- **YFinance:** Real-time and historical market data scraper for stocks (e.g., AAPL, RELIANCE.NS) and crypto.
- **Pandas & NumPy:** The backbone for data manipulation, cleaning, and numerical calculations.
- **Manual Indicator Engine (`engine.py`):** A custom implementation of technical indicators (Wilder's RSI, EMA, MACD) to ensure high performance and zero dependency conflicts.
- **Normalization Engine:** Scales and clips market features to a [-1, 1] range, ensuring stable input for the Neural Network.

### 3. **Infrastructure Layer (Microservices)**
Connects the AI components to the user interface via a robust API.
- **FastAPI (`api.py`):** A high-performance web framework used to serve the PPO model's inference as a RESTful service.
- **Uvicorn:** The ASGI server that hosts the FastAPI application, ensuring low-latency communication between the backend and frontend.
- **Requests:** Used for internal health checks and service communication.

### 4. **Presentation Layer (UI/UX)**
The "face" of the project where users interact with the data.
- **Streamlit (`dashboard.py`):** An interactive web dashboard that visualizes technical charts, sentiment scores, and AI trade signals in real-time.
- **Plotly:** Powers the interactive candlestick charts and technical indicator overlays within the dashboard.

---

## 🛠️ Core Components & Functions

- **`main.py` (Command Center):** The centralized orchestrator that manages service lifecycles (Start/Stop API, Launch Dashboard, Run Training).
- **`trading_env.py` (The Market):** A custom simulation world with "Strict Teacher" reward shaping (2:1 penalty ratio) to force the AI to prioritize accuracy over trade frequency.
- **`validate_model.py` (The Auditor):** A specialized script that backtests the model against "Buy & Hold" benchmarks to prove its Alpha (outperformance).
- **`run_sentiment.py` (News Hub):** A standalone module for testing the FinBERT engine against live news streams.

---

## 📈 Performance Benchmarks

The model is optimized for **High-Confidence Trades**, achieving an accuracy of **75% to 82%** on its strongest signals.

### Validation Highlights:
- **AAPL:** Outperformed the market by **+80.02 pts**.
- **RELIANCE.NS:** Reduced losses by **+40.72 pts** compared to passive holding.

---

## 🚀 Getting Started

1. **Setup Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the Command Center:**
   ```bash
   python main.py
   ```
   *Follow the interactive menu to train the model or launch the full Dashboard suite.*

---

## 📝 Methodology Note: "The Strict Teacher"
To prevent the model from simply buying every dip, we implemented **Asymmetric Reward Shaping**. Correct predictions earn **+10 points**, while incorrect ones lose **-20 points**. This forces the agent to wait for "Perfect Setup" signals, mirroring the patience of a professional trader.
