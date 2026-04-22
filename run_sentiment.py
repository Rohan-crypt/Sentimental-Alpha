import yfinance as yf
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# 📰 FETCH NEWS (via yfinance - FREE & NO KEY)
# =========================
def get_news(query="AAPL"):
    """
    Fetches latest news for a specific ticker using yfinance.
    Defaults to AAPL if no ticker is provided.
    """
    print(f"[LOG] Fetching news for {query} via yfinance...")
    try:
        ticker = yf.Ticker(query)
        news = ticker.news
        
        # Extracting titles from yfinance news format (nested under 'content')
        headlines = []
        for item in news[:10]:
            if 'content' in item and 'title' in item['content']:
                headlines.append(item['content']['title'])
            elif 'title' in item:
                headlines.append(item['title'])
        
        if not headlines:
            print(f"[WARN] No news found for {query}. Using fallback.")
            headlines = ["Market is stable", "Trading volume remains consistent"]
            
        return headlines
    except Exception as e:
        print(f"[ERROR] yfinance news fetch failed: {e}")
        return ["Market showing neutral trend"]

# =========================
# 🤖 LOAD FINBERT
# =========================
print("Loading FinBERT...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# =========================
# 📊 SENTIMENT ENGINE
# =========================
def get_sentiment(headlines):
    """
    Analyzes a list of headlines and returns aggregated sentiment scores.
    """
    if not headlines:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "sentiment": 0.0}

    inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).numpy()

    pos = probs[:, 0].mean()
    neg = probs[:, 1].mean()
    neu = probs[:, 2].mean()

    # Sentiment Score: Positive minus Negative
    sentiment = pos - neg

    return {
        "pos": float(pos),
        "neg": float(neg),
        "neu": float(neu),
        "sentiment": float(sentiment)
    }

# =========================
# 🚀 TEST RUN
# =========================
if __name__ == "__main__":
    ticker_to_test = "AAPL"
    headlines = get_news(ticker_to_test)

    print(f"\n📰 Latest News for {ticker_to_test}:\n")
    for h in headlines:
        print("-", h)

    result = get_sentiment(headlines)
    print("\n📊 Aggregated Sentiment Result:\n")
    print(result)
