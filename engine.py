import yfinance as yf
import pandas as pd
import pandas_ta_classic as ta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# NLP SETUP 
# FinBERT use kar rahe hain kyunki normal BERT ko market sentiment samajh nahi aata
# ProsusAI wala model specifically financial text pe trained hai
FINBERT_MODEL = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)

def analyze_sentiment(text):
    """
    Headline input lega aur score return karega.
    Positive result = Bullish signal, Negative = Bearish signal.
    """
    if not text:
        return 0.0 # Empty news headlines pe code crash nahi hona chahiye
        
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Logic: Positive probability minus Negative probability.
    # Softmax se values normalize ho jati hain [0, 1] range mein
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0][0].item() - probs[0][2].item() 

def get_market_data(ticker="RELIANCE.NS"):
    """
    Yahoo Finance se data utha raha hai aur clean kar raha hai.
    Presentation ke liye Nifty 50 defaults rakha hai, but AAPL/RELIANCE bhi chalega.
    """
    print(f"[LOG] {ticker} ke latest bars fetch ho rahe hain...")
    # 1 saal ka data kaafi hai indicators ko 'settle' hone ke liye (RSI/EMA calculation)
    raw_data = yf.download(ticker, period="1y", interval="1d")
    
    if raw_data.empty:
        print(f"[ERROR] {ticker} ka data nahi mila. Check karo ticker ya internet.")
        return pd.DataFrame()
    #FIXING COLUMN NAMES
    # yfinance kabhi-kabhi MultiIndex columns bhejta hai jo code phod deta hai
    # Isko flatten karna zaroori hai varna pandas-ta index error dega
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    # --- TECHNICALS (The Agent's Eyes) ---
    # EMA (20): Trend direction ke liye
    # RSI (14): Overbought ya Oversold levels detect karne ke liye
    raw_data.ta.ema(length=20, append=True)
    raw_data.ta.rsi(length=14, append=True)
    
    # ADX: Ye dekhne ke liye ki market 'trending' hai ya sirf 'sideways' move kar raha hai
    raw_data.ta.adx(length=14, append=True) 
    
    # NaN values (shuruat ke rows) drop kar rahe hain taaki agent confuse na ho
    raw_data.dropna(inplace=True)
    
    print(f"[SUCCESS] {ticker} ready hai. Total {len(raw_data)} steps mil gaye training ke liye.")
    return raw_data

if __name__ == "__main__":
    # Demo run: Sirf check karne ke liye ki indicators sahi se compute ho rahe hain
    sample = get_market_data("AAPL")
    if not sample.empty:
        print(sample.tail())