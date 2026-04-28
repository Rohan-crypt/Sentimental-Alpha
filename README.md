# Sentimental-Alpha: AI Trading Research Platform (v2.0)

Sentimental-Alpha is a professional-grade trading research terminal that integrates **Deep Reinforcement Learning (PPO)** with **NLP-based Sentiment Analysis (FinBERT)**. The system is built on a research-backed architecture (2024-2026) to deliver high-conviction trade signals and robust risk management.

---

## 🏗️ Project Architecture & Tech Stack

![Sentimental-Alpha System Model](./system_model.jpg)

### 1. **Intelligence Layer (AI & ML)**
- **Stable-Baselines3 (PPO):** Implements Proximal Policy Optimization with an optimized entropy coefficient to prevent mode collapse.
- **HuggingFace Transformers (FinBERT):** A specialized NLP model (ProsusAI/finbert) used to convert unstructured financial news into dense sentiment vectors.
- **Sentiment-Stress Synergy Model:** A dynamic risk-gating mechanism that adjusts the RL agent's risk appetite based on market volatility and news sentiment.

### 2. **Data & Engineering Layer**
- **Augmented State Space:** A 16-dimensional observation vector including RSI, MACD, Volume Momentum, Normalized Returns, and the Financial Stress Index.
- **Manual Indicator Engine (`engine.py`):** Custom high-performance implementations of technical indicators ensuring zero dependency conflicts.
- **Strict Normalization:** All neural network inputs are scaled and clipped to a [-1, 1] range for stable gradient descent.

### 3. **Infrastructure Layer (Microservices)**
- **FastAPI (`api.py`):** Serves real-time inference using the trained PPO "Brain."
- **Streamlit (`dashboard.py`):** Interactive terminal for visualizing market intelligence, neural signals, and sentiment feeds.

---

## 🛠️ Core Components

- **`main.py`:** Central Command Center for managing service lifecycles.
- **`trading_env.py`:** Custom Gymnasium environment featuring the "Strict Teacher" reward loop (+10/-20) and automated volatility gating.
- **`RESEARCH_METHODOLOGY.md`:** Detailed technical report on the 2024-2026 research papers used to build this system.
- **`validate_model.py`:** Performance auditor that evaluates the model against historical benchmarks.

---

## 📝 Research & Methodology
The v2.0 update incorporates findings from several key research papers:
1. **IEEE (2024):** PPO Performance Benchmarking with FinBERT.
2. **arXiv (2025):** News-Aware Direct Reinforcement Trading.
3. **Journal of Financial Data Science (2025):** Sentiment-Stress Synergy & MDD Reduction.

For a deep dive into the math and logic behind these improvements, see [RESEARCH_METHODOLOGY.md](./RESEARCH_METHODOLOGY.md).

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
   *Follow the interactive menu to train the model, start the API, or launch the Dashboard.*

---

## 📈 Performance Targets
The model is optimized for **High-Confidence Trades**, targeting a validated accuracy of **75% to 82%** on its strongest signals by filtering out market noise.
