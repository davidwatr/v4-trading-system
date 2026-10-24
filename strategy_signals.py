#!/usr/bin/env python3
"""
Strategy Signal Generators
Two independent modules:
- Strategy A: Trend Following (breakouts, MA crossovers, trailing stops)
- Strategy B: Mean Reversion (oversold bounces, tight targets)
"""

import numpy as np
from production_v18 import calc_rsi, calc_macd, calc_bollinger

# Local SMA — production_v18 has no calc_sma
def calc_sma(closes, period):
    """Simple moving average of last `period` closes."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


# ============================================================
# STRATEGY A: TREND FOLLOWING
# ============================================================
def tf_signal(closes, day_idx, highs=None, lows=None):
    """
    Trend Following entry signal.
    
    Two entry triggers (both require a clear signal, not just proximity):
    1. Donchian Breakout: Price CROSSES ABOVE the prior 20-day high (new high, not just near it)
       → momentum continuation entry
    2. MA Pullback: Price > SMA50 (uptrend) AND price pulls back to/touches SMA20
       → buy the dip in confirmed uptrend
    
    Per-symbol cooldown: no re-entry within 5 days of previous exit (enforced by backtest).
    
    Returns:
    (strength, direction, rsi, entry_type)
    - strength: 0.0 to 1.0 (confidence score)
    - direction: 1 (long) or 0 (no signal)
    - rsi: current RSI(14)
    - entry_type: 'breakout' | 'pullback' | None
    
    Asymmetric exits (applied by backtest engine, not here):
    - Hard stop: 2% (always)
    - Take profit: Trailing stop at 3% trail (no fixed target — let winners run)
    """
    if day_idx < 50:
        return 0.0, 0, 50, None
    
    closes_slice = closes[:day_idx+1]
    current_price = closes_slice[-1]
    
    if len(closes_slice) < 50:
        return 0.0, 0, 50, None
    
    sma20 = calc_sma(closes_slice, 20)
    sma50 = calc_sma(closes_slice, 50)
    
    if sma20 is None or sma50 is None:
        return 0.0, 0, 50, None
    
    rsi_val = calc_rsi(closes_slice, 14)
    rsi_val = rsi_val if rsi_val is not None else 50
    
    strength = 0.0
    entry_type = None
    
    # --- Trigger 1: Donchian Breakout (RESUMPTION, not proximity) ---
    # Only trigger if price CROSSED above the 20-day high TODAY (was below yesterday)
    # AND the breakout is substantial (>3.5% above prior high) to filter noise
    # AND there's a gap up (momentum initiating, not just grinding)
    # AND price is above SMA50 (uptrend confirmed)
    if len(closes_slice) >= 22:
        prev_high = max(closes_slice[-22:-1])  # 20-day high ending yesterday
        lookback_high = max(closes_slice[-21:-1])  # 20-day high ending today (excl. today)
        
        yesterday_price = closes_slice[-2]
        
        was_below_prior_high = yesterday_price <= prev_high
        breakout_margin = (current_price - lookback_high) / lookback_high if lookback_high > 0 else 0
        gap_from_close = (current_price - yesterday_price) / yesterday_price if yesterday_price > 0 else 0
        
        # High conviction only: big breakout (>3.5%) + gap (>0.5%) + above SMA50 + was below prior high
        if was_below_prior_high and breakout_margin > 0.035 and gap_from_close > 0.005 and current_price > sma50:
            strength = min(0.12, 0.02 + breakout_margin * 2.5)
            entry_type = 'breakout'
    
    # --- Trigger 2: MA Pullback (buy the dip in an uptrend) ---
    # Price above SMA50 (uptrend) AND price within 2% of SMA20 (pullback to short MA)
    # Only trigger if we haven't already breakout-triggered
    if strength < 0.03 and current_price > sma50:
        dist_to_sma20 = (current_price - sma20) / sma20
        if -0.02 <= dist_to_sma20 <= 0.01:  # At or slightly below SMA20
            # Strength based on how deep the pullback + RSI confirmation
            pullback_depth = abs(dist_to_sma20)
            rsi_filter = max(0, 1 - rsi_val / 60)  # Lower RSI = better entry
            strength = min(0.10, 0.03 + pullback_depth * 2 * rsi_filter)
            entry_type = 'pullback'
    
    if strength > 0:
        return round(strength, 4), 1, round(rsi_val, 1), entry_type
    
    return 0.0, 0, round(rsi_val, 1), None


def tf_signal_short(closes, day_idx, highs=None, lows=None):
    """
    Trend Following SHORT signal (for bear regimes).
    
    Inverse of TF long:
    - Break below 20-day low → momentum continuation short
    - Price < SMA50 AND pullback to/near SMA20 from above → short the rally
    
    Returns same format as tf_signal.
    """
    if day_idx < 50:
        return 0.0, -1, 50, None  # -1 = short direction
    
    closes_slice = closes[:day_idx+1]
    current_price = closes_slice[-1]
    
    if len(closes_slice) < 50:
        return 0.0, -1, 50, None
    
    sma20 = calc_sma(closes_slice, 20)
    sma50 = calc_sma(closes_slice, 50)
    
    if sma20 is None or sma50 is None:
        return 0.0, -1, 50, None
    
    rsi_val = calc_rsi(closes_slice, 14)
    rsi_val = rsi_val if rsi_val is not None else 50
    
    strength = 0.0
    entry_type = None
    
    # --- Trigger 1: Breakdown (RESUMPTION, not proximity) ---
    # Below 20-day low by >3%, with gap down, and below SMA50 (downtrend confirmed)
    if len(closes_slice) >= 22:
        prev_low = min(closes_slice[-22:-1])
        lookback_low = min(closes_slice[-21:-1])
        
        yesterday_price = closes_slice[-2]
        
        was_above_prior_low = yesterday_price >= prev_low
        breakdown_margin = (lookback_low - current_price) / lookback_low if lookback_low > 0 else 0
        gap_from_close = (current_price - yesterday_price) / yesterday_price if yesterday_price > 0 else 0
        
        # High conviction short: big breakdown (>3%) + gap down (>0.5%) + below SMA50
        if was_above_prior_low and breakdown_margin > 0.03 and gap_from_close < -0.005 and current_price < sma50:
            strength = min(0.12, 0.02 + breakdown_margin * 2.5)
            entry_type = 'breakdown'
    
    # --- Trigger 2: Short the rally in a downtrend ---
    if strength < 0.03 and current_price < sma50:
        dist_to_sma20 = (current_price - sma20) / sma20
        if 0.01 <= dist_to_sma20 <= 0.03:  # At or slightly above SMA20 in a downtrend
            rally_depth = dist_to_sma20
            rsi_filter = max(0, rsi_val / 60 - 0.5)  # Higher RSI = better short
            strength = min(0.10, 0.03 + rally_depth * 2 * rsi_filter)
            entry_type = 'rally_bounce'
    
    if strength > 0:
        return round(strength, 4), -1, round(rsi_val, 1), entry_type
    
    return 0.0, -1, round(rsi_val, 1), None


# ============================================================
# STRATEGY B: MEAN REVERSION
# ============================================================
def mr_signal(closes, day_idx, volumes=None):
    """
    Mean Reversion entry signal.
    
    Entry trigger:
    - RSI(14) < 30 (oversold)
    - Price near or below lower Bollinger Band (2σ)
    - Momentum filter: no strong downtrend (price > 200-day SMA × 0.95)
    
    Returns:
    (strength, direction, rsi, bb_position)
    - strength: 0.0 to 1.0
    - direction: 1 (long bounce expected) or 0
    - rsi: current RSI
    - bb_position: where price is relative to BB (0=center, -1=lower band)
    
    Asymmetric exits:
    - Hard stop: 2% (always)
    - Take profit: Fixed 3% target (mean reversion doesn't trend, so no trailing)
    - Time exit: 5 days max hold
    """
    if day_idx < 20:
        return 0.0, 0, 50, 0
    
    closes_slice = closes[:day_idx+1]
    current_price = closes_slice[-1]
    
    if len(closes_slice) < 20:
        return 0.0, 0, 50, 0
    
    rsi_val = calc_rsi(closes_slice, 14)
    if rsi_val is None:
        rsi_val = 50
    
    # Bollinger Band position
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes_slice, 20, 2)
    
    bb_position = 0
    if bb_upper and bb_lower and bb_mid:
        bb_range = bb_upper - bb_lower
        if bb_range > 0:
            bb_position = (current_price - bb_lower) / bb_range - 0.5  # 0 at mid, -0.5 at lower
    
    # Momentum filter: is price still above 200-day MA? (no strong trend = good for MR)
    sma200 = calc_sma(closes_slice, 200) if len(closes_slice) >= 200 else None
    trend_filter_pass = True
    if sma200:
        trend_filter_pass = current_price > sma200 * 0.95  # Within 5% of SMA200
    
    if not trend_filter_pass:
        return 0.0, 0, round(rsi_val, 1), bb_position
    
    # Entry condition: RSI oversold OR price near lower BB (relaxed for more signals)
    # RSI < 35 (slightly above "oversold") OR near/through lower Bollinger Band
    rsi_oversold = rsi_val < 35
    near_lower_bb = bb_position < -0.2  # Within 20% of lower band range
    
    if rsi_oversold and near_lower_bb:
        # Strong MR setup: oversold + near lower band
        oversold_depth = (35 - rsi_val) / 35
        bb_depth = abs(bb_position) if bb_position < 0 else 0
        strength = min(0.12, oversold_depth * 0.07 + bb_depth * 0.05)
        return round(strength, 4), 1, round(rsi_val, 1), round(bb_position, 3)
    
    # Single-condition signal: just RSI < 30 (classic oversold bounce)
    if rsi_val < 30:
        strength = min(0.07, (30 - rsi_val) / 30 * 0.06)
        return round(strength, 4), 1, round(rsi_val, 1), round(bb_position, 3)
    
    # Weaker signal: RSI < 38 + near lower BB
    if rsi_val < 38 and bb_position < -0.1:
        strength = min(0.04, (38 - rsi_val) / 38 * 0.03)
        if strength > 0.015:
            return round(strength, 4), 1, round(rsi_val, 1), round(bb_position, 3)
    
    return 0.0, 0, round(rsi_val, 1), round(bb_position, 3)


# ============================================================
# STRATEGY B: MEAN REVERSION — SECTOR FILTER
# ============================================================
# Financials tickers (best performing MR sector per our analysis)
FINANCIALS = {'JPM', 'BAC', 'GS', 'MS', 'V', 'MA'}
# Tech tickers (second best)
TECH = {'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE'}

# Consumer performed worst (40% WR) — excluded from MR universe
# All other tickers are fair game

MR_UNIVERSE = FINANCIALS | TECH


def mr_signal_with_sector(symbol, closes, day_idx):
    """
    Mean Reversion signal with sector filter.
    Only generates signals for Financials + Tech (excludes Consumer).
    
    Returns same format as mr_signal.
    """
    if symbol not in MR_UNIVERSE:
        return 0.0, 0, 50, 0  # Not in our MR universe
    
    return mr_signal(closes, day_idx)


# ============================================================
# TRAILING STOP CALCULATION (for TF module)
# ============================================================
def calculate_trailing_stop(entry_price, current_price, trail_pct=0.03, activation_pct=0.02):
    """
    Calculate trailing stop price for a long position.
    
    Logic:
    - Stop is NOT activated until price has moved up by activation_pct (2%) from entry
    - Once activated, stop trails at trail_pct (3%) below the highest price reached
    - If price hasn't reached activation, stop stays at entry_price * (1 - 0.02) hard stop
    
    Returns:
    (stop_price, is_activated, highest_price)
    - stop_price: current stop level
    - is_activated: has the trailing stop been activated?
    - highest_price: highest closing price since entry (for trail calculation)
    """
    hard_stop = entry_price * (1 - 0.02)  # Always active: 2% hard stop
    
    activation_price = entry_price * (1 + activation_pct)  # 2% above entry
    
    # For simplicity in backtest, we track highest price externally
    # This function just calculates current stop based on highest price seen
    
    if current_price <= activation_price:
        # Not yet activated → use hard stop
        return hard_stop, False, current_price
    
    # Activated → trail at trail_pct below highest price
    if current_price > activation_price:
        highest = max(current_price, entry_price * (1 + activation_pct))
    else:
        highest = current_price
    
    trail_stop = highest * (1 - trail_pct)
    
    # Never move stop below hard stop
    effective_stop = max(trail_stop, hard_stop)
    
    return round(effective_stop, 2), True, round(highest, 2)


def check_trailing_stop_exit(entry_price, highest_price, current_price, trail_pct=0.03):
    """
    Check if price has hit the trailing stop.
    
    Returns True if current_price <= trailing stop level.
    """
    if current_price <= entry_price * (1 - 0.02):
        return True  # Hard stop hit
    
    if highest_price <= entry_price * (1 + 0.02):
        return False  # Not yet activated
    
    trailing_stop = highest_price * (1 - trail_pct)
    return current_price <= trailing_stop
