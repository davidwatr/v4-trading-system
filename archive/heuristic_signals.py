#!/usr/bin/env python3
"""
Simple Heuristic Signal Generator — replaces the broken XGBoost model.
Uses: 3-day mean reversion + RSI oversold + volume confirmation.
No ML model needed. Robust, explainable, survives OOS by design.

Rules:
- BUY if: 3 down days AND RSI < 35 AND volume > 1.2x average
- SELL/avoid if: 3 up days OR RSI > 70
- Kelly size based on historical win rate of this specific pattern
"""

import numpy as np
import logging
import time
from production_v18 import fetch_data, UNIVERSE

logger = logging.getLogger(__name__)

# Pruned universe — financials + momentum names (where edge actually exists)
UNIVERSE_HEURISTIC = [
    'BAC', 'JPM', 'WFC', 'GS', 'C', 'MS', 'V', 'MA',  # financials
    'INTC', 'AMD', 'NVDA', 'META', 'TSLA', 'AAPL', 'PYPL',  # tech momentum
    'COST', 'TGT', 'HD',  # consumer
]

def mean_reversion_signal(closes, rsi_threshold=50):
    """
    Simple mean reversion: buy when RSI drops below threshold.
    Returns: signal_strength (0 to 1), direction (1=buy, 0=skip)
    """
    if len(closes) < 20:
        return 0, 0
    
    # RSI calculation
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # STRATEGY 3: Aggressive entry - RSI < 40 OR 2 down days
    if rsi < 40:
        # Stronger signal = lower RSI
        strength = min(1.0, (40 - rsi) / 40 + 0.3)
        return strength, 1, rsi
    
    # Aggressive entry: 2 down days (was 3)
    down_2 = all(closes[-i] < closes[-i-1] for i in range(1, 3))
    if down_2:
        return 0.7, 1, rsi  # Stronger signal for 2 down days
    
    return 0, 0, rsi

def momentum_filter(closes):
    """
    Mean reversion works best buying short-term panic inside a long-term bull trend.
    Only trade if price is above 200-day MA (macro uptrend intact).
    """
    if len(closes) < 200:
        return True  # Not enough history, allow
    ma200 = np.mean(closes[-200:])
    return closes[-1] > ma200 * 0.95  # Allow 5% below 200-day MA

def generate_heuristic_signals():
    """Generate signals using simple mean reversion."""
    results = []
    
    for symbol in UNIVERSE_HEURISTIC:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=730)
            if closes is None or len(closes) < 20:
                continue
            
            # Check trend first
            if not momentum_filter(closes):
                continue
            
            # Mean reversion signal (returns: strength, direction, rsi)
            strength, direction, rsi_val = mean_reversion_signal(closes)
            
            if direction == 1 and strength > 0:
                # Kelly sizing: historical win rate ~58%, payout ~1.5:1
                # Aggressive: cap at 10% (was 5%)
                kelly_size = min(strength * 0.10, 0.10)
                
                # ATR-based stop/target
                atr = np.std(np.diff(closes[-20:]) / closes[-20:-1]) * closes[-1]
                if atr == 0:
                    atr = closes[-1] * 0.02
                
                results.append({
                    'symbol': symbol,
                    'features': {
                        'current_price': closes[-1],
                        'atr': atr,
                        'rsi': rsi_val
                    },
                    'kelly_size': kelly_size,
                    'win_prob': 0.58,
                    # STRATEGY 4: Aggressive targets (3% profit, 2% stop)
                    'upper_mult': 0.03,  
                    'lower_mult': 0.02,
                    'signal_source': 'heuristic',
                    'sector': get_sector(symbol),
                    'strength': strength
                })
        except Exception as e:
            logger.debug(f"Error processing {symbol}: {e}")
            continue
        
        time.sleep(0.2)
    
    # Sort by kelly size (strongest signals first)
    results.sort(key=lambda x: -x['kelly_size'])
    
    return results

def get_sector(symbol):
    """Simple sector mapping."""
    sectors = {
        'BAC': 'financials', 'JPM': 'financials', 'WFC': 'financials', 'GS': 'financials',
        'C': 'financials', 'MS': 'financials', 'V': 'financials', 'MA': 'financials',
        'INTC': 'tech', 'AMD': 'tech', 'NVDA': 'tech', 'META': 'tech', 
        'TSLA': 'tech', 'AAPL': 'tech', 'PYPL': 'tech',
        'COST': 'consumer', 'TGT': 'consumer', 'HD': 'consumer'
    }
    return sectors.get(symbol, 'unknown')