import yfinance as yf
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# News fetching logic using yfinance
def get_news(ticker_name="AAPL"):
    """
    Retrieves latest news titles for a specific ticker.
    """
    try:
        ticker = yf.Ticker(ticker_name)
        news_data = ticker.news
        headlines = []
        for item in news_data[:10]:
            if 'content' in item and 'title' in item['content']:
                headlines.append(item['content']['title'])
            elif 'title' in item:
                headlines.append(item['title'])
        return headlines if headlines else ["Market stable", "Volume neutral"]
    except Exception:
        return ["Sentiment fetch unavailable"]

# FinBERT Sentiment Engine Initialization
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment(headlines):
    """
    Analyzes headlines and returns mean sentiment scores.
    """
    if not headlines:
        return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "sentiment": 0.0}

    inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1).numpy()
    pos = float(probs[:, 0].mean())
    neg = float(probs[:, 1].mean())
    neu = float(probs[:, 2].mean())
    sentiment = pos - neg

    return {"pos": pos, "neg": neg, "neu": neu, "sentiment": sentiment}

if __name__ == "__main__":
    ticker = "AAPL"
    titles = get_news(ticker)
    print(f"News: {titles}")
    print(f"Sentiment: {get_sentiment(titles)}")
