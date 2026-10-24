# News Sentiment Setup Guide

## What It Does

The news sentiment analyzer:
1. Fetches news articles about stocks from free APIs
2. Analyzes sentiment (bullish/bearish/neutral) per article
3. Calculates aggregate sentiment score per stock
4. Generates buy/sell signals based on news sentiment

---

## Free API Options

### 1. Alpha Vantage (Recommended)
- **URL:** https://www.alphavantage.co/support/#api-key
- **Free tier:** 25 calls/day
- **Includes:** Article-level sentiment, ticker-level sentiment, relevance scores
- **Setup:** Get key, paste in `news_sentiment.py` line 15

### 2. Marketaux
- **URL:** https://www.marketaux.com/register
- **Free tier:** 100 calls/day
- **Includes:** Sentiment score, entity detection
- **Setup:** Get key, paste in line 16, change `NEWS_SOURCE = "marketaux"`

### 3. Finnhub
- **URL:** https://finnhub.io/register
- **Free tier:** 60 calls/minute
- **Includes:** Company news, news sentiment (bullish/bearish %)
- **Setup:** Get key, paste in line 17, change `NEWS_SOURCE = "finnhub"`

---

## How to Run

```powershell
cd C:\Users\david_wncs0ps\jarvis\trading
python news_sentiment.py
```

**Output:**
- Sentiment scores for 25 stocks
- Bullish/bearish signals
- Top articles per stock
- Saved to `news_sentiment_results.json`

---

## Integrating with Screener

The screener and news sentiment work together:

| Screener Signal | News Signal | Action |
|---|---|---|
| BUY (score ≥ 4) | BULLISH | Strong buy — both technicals and news align |
| BUY (score ≥ 4) | BEARISH | Caution — technicals say buy but news is negative |
| SELL (score ≤ -4) | BEARISH | Strong sell — both align |
| SELL (score ≤ -4) | BULLISH | Caution — technicals say sell but news is positive |

**Run both daily and compare signals.**

---

## Next Steps

1. **Get a free API key** from one of the three providers
2. **Paste it** in `news_sentiment.py`
3. **Run it** alongside the screener
4. **Compare signals** — do technical and news agree?

---

**Files saved:**
- `C:\Users\david_wncs0ps\jarvis\trading\news_sentiment.py`
