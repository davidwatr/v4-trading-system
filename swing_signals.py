#!/usr/bin/env python3
"""
Swing 2d PB Signal Generator — for Alpaca Executor
Replaces the heuristic_signals.py mean reversion approach.

Rules (backtest-validated on fixed 501-day window, Sharpe 1.54):
- Macro: SPY must be above its 50-day SMA (no entries when below)
- Trend: Target asset close must be above its 50-day SMA
- Entry: 2 consecutive down days (pullback) + today's volume > 20-day avg
- TP: +4% | SL: -2% | Time stop: 5 trading days
- Universe: Financials + Tech (no consumer, no dead symbols)
"""

import numpy as np
import logging
import time
from production_v18 import fetch_data

logger = logging.getLogger(__name__)

# Swing universe — same as backtest (drop AAPL, MSFT, ADBE, MS, INTC, AMZN)
SWING_UNIVERSE = [
    'GOOGL', 'META', 'NVDA', 'AMD', 'CRM', 'PYPL',
    'JPM', 'BAC', 'GS', 'V', 'MA',
]

def sma(data, period, end_idx):
    """SMA ending at end_idx (inclusive)."""
    if end_idx < period - 1:
        return None
    return sum(data[end_idx - period + 1:end_idx + 1]) / period

def generate_swing_signals():
    """
    Generate swing 2d PB signals.
    Returns: list of signal dicts compatible with AlpacaExecutor.
    """
    results = []

    # ---- 1. Fetch SPY for macro filter ----
    spy_closes, _, _, _ = fetch_data("SPY", days=260)
    if not spy_closes or len(spy_closes) < 50:
        logger.warning("Failed to fetch SPY data for macro filter")
        return []

    spy_sma50 = sma(spy_closes, 50, len(spy_closes) - 1)
    if spy_sma50 is None:
        return []
    spy_above_sma = spy_closes[-1] > spy_sma50
    logger.info(f"SPY macro filter: {'ACTIVE' if spy_above_sma else 'INACTIVE'} "
                f"(SPY=${spy_closes[-1]:.2f} vs SMA50=${spy_sma50:.2f})")

    if not spy_above_sma:
        logger.info("SPY below SMA50 — no new entries today")
        return []

    # ---- 2. Scan each symbol ----
    for symbol in SWING_UNIVERSE:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=260)
            if closes is None or len(closes) < 60:
                logger.debug(f"{symbol}: insufficient data ({len(closes) if closes else 0} bars)")
                continue

            # Trend filter: close > SMA50
            sma50 = sma(closes, 50, len(closes) - 1)
            if sma50 is None or closes[-1] <= sma50:
                continue

            # Pullback: 2 consecutive down days
            if len(closes) < 3:
                continue
            day0 = closes[-1]   # today
            day1 = closes[-2]   # yesterday
            day2 = closes[-3]   # day before
            if not (day0 < day1 and day1 < day2):
                continue  # not a pullback

            # Volume spike: today > 20-day avg volume
            if len(volumes) < 20:
                continue
            vol_today = volumes[-1]
            vol_avg20 = sum(volumes[-20:]) / 20
            if vol_today <= vol_avg20:
                continue

            # ---- SIGNAL FOUND ----
            entry_price = round(closes[-1], 2)
            stop_loss = round(entry_price * 0.98, 2)       # -2%
            take_profit = round(entry_price * 1.04, 2)     # +4%

            # Position sizing: fixed fractional (12.5% of equity = 1/8 max positions)
            # Kelly size of 0.125 = 12.5% of portfolio per position
            kelly_size = 0.125

            results.append({
                'symbol': symbol,
                'features': {
                    'current_price': entry_price,
                    'atr': entry_price * 0.02,  # fallback, not used for bracket
                    'rsi': 50,  # placeholder
                    'sma50': sma50,
                    'vol_ratio': vol_today / vol_avg20,
                    'pullback_days': 2,
                },
                'kelly_size': kelly_size,
                'win_prob': 0.528,  # backtest win rate
                'upper_mult': 0.04,   # +4% TP
                'lower_mult': 0.02,   # -2% SL
                'signal_source': 'swing_2d_pb',
                'sector': get_sector(symbol),
                'strength': 1.0,
            })

        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            continue

        time.sleep(0.15)

    # Sort by sector diversity, then by volume ratio (strongest reversal first)
    results.sort(key=lambda x: (-x['features']['vol_ratio']))

    logger.info(f"Generated {len(results)} swing signals")
    return results


def get_sector(symbol):
    """Sector mapping for the swing universe."""
    sectors = {
        'GOOGL': 'tech', 'META': 'tech', 'NVDA': 'tech',
        'AMD': 'tech', 'CRM': 'tech', 'PYPL': 'tech',
        'JPM': 'financials', 'BAC': 'financials',
        'GS': 'financials', 'V': 'financials', 'MA': 'financials',
    }
    return sectors.get(symbol, 'unknown')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    signals = generate_swing_signals()
    print(f"\n{signals and len(signals) or 0} signals generated:")
    for s in signals:
        f = s['features']
        print(f"  {s['symbol']:6s} | ${f['current_price']:.2f} | "
              f"Vol ratio: {f['vol_ratio']:.2f}x | "
              f"TP: ${s['features']['current_price']*1.04:.2f} | "
              f"SL: ${s['features']['current_price']*0.98:.2f} | "
              f"Sector: {s['sector']}")
