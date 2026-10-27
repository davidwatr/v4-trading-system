# Trading Tools Setup Guide

## What I Built

| File | What it does |
|---|---|
| `screener.py` | Scans stocks + crypto for swing trading signals |
| `telegram_bot.py` | Sends trade alerts to your phone via Telegram |

---

## How to Use

### Step 1: Run the Screener

```powershell
cd C:\Users\david_wncs0ps\jarvis\trading
python screener.py
```

This will:
- Screen 44 stocks + 24 cryptos
- Calculate 6 indicators (RSI, MACD, MA, BB, Volume, Support/Resistance)
- Generate buy/sell signals with scores
- Save results to `screener_results.json`

### Step 2: Set Up Telegram Bot (Optional)

1. Open Telegram, search **@BotFather**
2. Send `/newbot`
3. Follow the instructions
4. Copy the bot token
5. Open `telegram_bot.py`, paste your token
6. Send a message to your bot
7. Get your chat ID from: `https://api.telegram.org/bot<TOKEN>/getUpdates`
8. Paste your chat ID in the script

### Step 3: Run the Bot

```powershell
python telegram_bot.py
```

Choose option 2 to send screener results to your phone.

---

## Trade Journal (Google Sheets)

Create a Google Sheet with these columns:

| Column | What to enter |
|---|---|
| **Date** | When you entered the trade |
| **Symbol** | Stock/crypto ticker |
| **Direction** | Buy or Sell |
| **Entry Price** | Price you bought at |
| **Exit Price** | Price you sold at |
| **Quantity** | How many shares/coins |
| **P&L** | Profit or loss in $ |
| **P&L %** | Profit or loss percentage |
| **Signals** | What signals triggered the trade |
| **Notes** | What you learned |

**Formula for P&L:**
- Stocks: `(Exit Price - Entry Price) * Quantity`
- Crypto: Same formula

**Formula for P&L %:**
- `((Exit Price - Entry Price) / Entry Price) * 100`

---

## Indicator Reference

| Indicator | What it tells you | Bullish signal | Bearish signal |
|---|---|---|---|
| **RSI** | Overbought/oversold | < 30 (oversold) | > 70 (overbought) |
| **MACD** | Momentum | MACD > Signal | MACD < Signal |
| **MA** | Trend direction | MA20 > MA50 | MA20 < MA50 |
| **Bollinger Bands** | Volatility | Price touches lower band | Price touches upper band |
| **Volume** | Strength of move | Volume > 1.5x average | Low volume |
| **Support/Resistance** | Key levels | Price at support | Price at resistance |

---

## Scoring System

The screener assigns points:

| Signal | Points |
|---|---|
| RSI oversold | +2 |
| RSI overbought | -2 |
| MACD bullish | +1 |
| MACD bearish | -1 |
| MA bullish (MA20 > MA50) | +1 |
| MA bearish | -1 |
| BB oversold (price < lower band) | +2 |
| BB overbought | -2 |
| Volume spike (>1.5x avg) | +1 |
| At support | +2 |
| At resistance | -2 |

**Score >= 3:** Strong buy signal
**Score <= -3:** Strong sell signal

---

## Next Steps

1. **Run the screener daily** — check for new signals
2. **Paper trade first** — don't use real money until you prove you can pick winners
3. **Track every trade** — use the Google Sheet
4. **Review weekly** — what worked? What didn't?
5. **Adjust** — tweak the screener parameters based on results

---

## Important Disclaimers

- **This is not financial advice**
- **Most traders lose money** — especially beginners
- **Paper trade first** — prove your strategy works before risking real money
- **Never invest more than you can afford to lose**
- **Past performance ≠ future results**

---

**Files saved:**
- `C:\Users\david_wncs0ps\jarvis\trading\screener.py`
- `C:\Users\david_wncs0ps\jarvis\trading\telegram_bot.py`
- `C:\Users\david_wncs0ps\jarvis\trading\SETUP_GUIDE.md`
