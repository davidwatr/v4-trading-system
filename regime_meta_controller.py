#!/usr/bin/env python3
"""
Regime Meta-Controller
Classifies market regime using ADX, ATR, and SMA alignment.
Returns strategy allocation: which strategies to run, how much capital, risk parameters.
"""

import numpy as np
from production_v18 import calc_adx, calc_atr

# Local SMA (production_v18 has no calc_sma)
def calc_sma(closes, period):
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


# ============================================================
# REGIME FEATURE EXTRACTION
# ============================================================
def extract_regime_features(closes, highs, lows, day_idx):
    """
    Compute all features needed for regime classification at a given day.
    
    Returns dict with:
    - adx: ADX(14) value
    - atr: ATR(14) value
    - atr_ratio: ATR / current price (volatility relative to price level)
    - sma20, sma50, sma200: Simple moving averages
    - price: current price
    - sma20_slope: rate of change of SMA20 (trend momentum)
    - ma_alignment: 0-2 score (SMA20>SMA50 + SMA50>SMA200)
    - price_vs_sma50: (price - SMA50) / SMA50
    - price_vs_sma200: (price - SMA200) / SMA200
    """
    if day_idx < 50:
        return None
    
    closes_slice = closes[:day_idx+1]
    highs_slice = highs[:day_idx+1] if highs is not None else closes_slice
    lows_slice = lows[:day_idx+1] if lows is not None else closes_slice
    
    if len(closes_slice) < 50:
        return None
    
    current_price = closes_slice[-1]
    
    sma20 = calc_sma(closes_slice, 20)
    sma50 = calc_sma(closes_slice, 50)
    sma200 = calc_sma(closes_slice, 200)
    
    if sma20 is None or sma50 is None or sma200 is None:
        return None
    
    adx_val = calc_adx(highs_slice, lows_slice, closes_slice, 14)
    atr_val = calc_atr(highs_slice, lows_slice, closes_slice, 14)
    
    if adx_val is None:
        adx_val = 20  # Default to neutral
    
    if atr_val is None:
        atr_val = current_price * 0.02
    
    # SMA20 slope: compare current SMA20 to SMA20 from 10 days ago
    if len(closes_slice) >= 30:
        older_slice = closes_slice[:-10]
        if len(older_slice) >= 20:
            sma20_older = calc_sma(older_slice, 20)
            sma20_slope = (sma20 - sma20_older) / (sma20_older + 1e-10) if sma20_older else 0
        else:
            sma20_slope = 0
    else:
        sma20_slope = 0
    
    # MA alignment: bullish alignment = SMA20 > SMA50 > SMA200
    ma_alignment = 0
    if sma20 > sma50:
        ma_alignment += 1
    if sma50 > sma200:
        ma_alignment += 1
    
    return {
        'price': current_price,
        'sma20': sma20,
        'sma50': sma50,
        'sma200': sma200,
        'adx': adx_val,
        'atr': atr_val,
        'atr_ratio': atr_val / current_price if current_price > 0 else 0.02,
        'sma20_slope': sma20_slope,
        'ma_alignment': ma_alignment,
        'price_vs_sma50': (current_price - sma50) / sma50 if sma50 > 0 else 0,
        'price_vs_sma200': (current_price - sma200) / sma200 if sma200 > 0 else 0,
    }


# ============================================================
# REGIME CLASSIFICATION
# ============================================================
def classify_regime(features):
    """
    Classify market regime into one of 5 states.
    
    Thresholds:
    - ADX > 25: Trending market
    - ADX < 20: Ranging / choppy market
    - ADX 20-25: Transition zone
    - ATR/Price > 2.5%: High volatility override
    
    Returns:
    - 'trend_bull': Strong uptrend → TF long only
    - 'trend_bear': Strong downtrend → Short TF (with vol override)
    - 'range': Choppy/ranging → Mean Reversion only
    - 'high_vol': Volatility spike → Defense: reduce to cash
    - 'transition': Gray zone → Split allocation
    """
    if features is None:
        return 'transition', {'reason': 'insufficient_data'}
    
    adx = features['adx']
    atr_ratio = features['atr_ratio']
    price_vs_sma200 = features['price_vs_sma200']
    ma_align = features['ma_alignment']
    sma20_slope = features['sma20_slope']
    
    # --- High Vol Override (checks FIRST — safety first) ---
    # If volatility is extreme, reduce to defense regardless of trend
    if atr_ratio > 0.035:  # ATR > 3.5% of price = extreme volatility
        return 'high_vol', {
            'reason': 'extreme_volatility',
            'atr_ratio': atr_ratio,
            'note': 'ATR > 3.5% of price — defensive posture, minimize exposure'
        }
    
    # --- Trending Regimes ---
    if adx > 25:
        if price_vs_sma200 > 0:
            # Uptrend confirmed
            return 'trend_bull', {
                'reason': 'strong_uptrend',
                'adx': adx,
                'price_vs_sma200': price_vs_sma200,
                'ma_alignment': ma_align,
            }
        elif price_vs_sma200 < -0.03:
            # Downtrend confirmed (>3% below SMA200)
            return 'trend_bear', {
                'reason': 'strong_downtrend',
                'adx': adx,
                'price_vs_sma200': price_vs_sma200,
                'ma_alignment': ma_align,
            }
        else:
            # ADX high but price near SMA200 — choppy trend, treat as range
            return 'range', {
                'reason': 'high_adx_but_flat_price',
                'adx': adx,
                'price_vs_sma200': price_vs_sma200,
            }
    
    # --- Ranging Regime ---
    if adx < 20:
        return 'range', {
            'reason': 'low_adx_chop',
            'adx': adx,
        }
    
    # --- Transition Zone (ADX 20-25) ---
    # Determine dominant direction
    if price_vs_sma200 > 0.02 and sma20_slope > 0:
        return 'transition', {
            'reason': 'mild_uptrend_gray_zone',
            'adx': adx,
            'price_vs_sma200': price_vs_sma200,
            'allocation': '70_tf_30_mr',
        }
    elif price_vs_sma200 < -0.02:
        return 'transition', {
            'reason': 'mild_downtrend_gray_zone',
            'adx': adx,
            'price_vs_sma200': price_vs_sma200,
            'allocation': '70_mr_30_tf',
        }
    else:
        return 'transition', {
            'reason': 'neutral_gray_zone',
            'adx': adx,
            'allocation': '50_50',
        }


# ============================================================
# STRATEGY ALLOCATION
# ============================================================
def get_strategy_allocation(regime, regime_reason):
    """
    Given a classified regime, return the strategy allocation parameters.
    
    Returns dict:
    {
        'tf_enabled': bool,       — Trend Following active?
        'mr_enabled': bool,       — Mean Reversion active?
        'short_enabled': bool,    — Short-selling active?
        'tf_weight': float,       — Capital allocation weight for TF (0-1)
        'mr_weight': float,       — Capital allocation weight for MR (0-1)
        'short_weight': float,    — Capital allocation weight for shorts (0-1)
        'kelly_multiplier': float,— Global Kelly scaling factor
        'max_positions': int,     — Max concurrent positions
        'max_position_kelly': float, — Max Kelly fraction per position
        'slippage_bps': float,    — Slippage in basis points
        'regime_label': str,      — Human-readable label
        'note': str,              — Why this allocation
    }
    """
    if regime == 'trend_bull':
        return {
            'tf_enabled': True,
            'mr_enabled': False,
            'short_enabled': False,
            'tf_weight': 1.0,
            'mr_weight': 0.0,
            'short_weight': 0.0,
            'kelly_multiplier': 1.0,
            'max_positions': 8,
            'max_position_kelly': 0.15,
            'slippage_bps': 10,
            'regime_label': 'TREND BULL',
            'note': 'Strong uptrend — full TF allocation, no MR (would get chopped), no shorts',
        }
    
    elif regime == 'trend_bear':
        return {
            'tf_enabled': False,  # No long TF in bear
            'mr_enabled': False,  # No MR in strong trend
            'short_enabled': True,  # Short TF active
            'tf_weight': 0.0,
            'mr_weight': 0.0,
            'short_weight': 1.0,
            'kelly_multiplier': 0.75,  # Slightly reduced for short risk
            'max_positions': 4,  # Fewer shorts than longs
            'max_position_kelly': 0.10,
            'slippage_bps': 15,  # Wider slippage for shorts
            'regime_label': 'TREND BEAR (SHORT)',
            'note': 'Strong downtrend — short TF only, half-Kelly for safety',
        }
    
    elif regime == 'range':
        return {
            'tf_enabled': False,
            'mr_enabled': True,
            'short_enabled': False,
            'tf_weight': 0.0,
            'mr_weight': 1.0,
            'short_weight': 0.0,
            'kelly_multiplier': 0.75,
            'max_positions': 6,
            'max_position_kelly': 0.10,
            'slippage_bps': 12,
            'regime_label': 'RANGE / CHOP',
            'note': 'Choppy market — MR only, no breakout trades (false breakouts), no shorts',
        }
    
    elif regime == 'high_vol':
        return {
            'tf_enabled': False,
            'mr_enabled': False,
            'short_enabled': False,
            'tf_weight': 0.0,
            'mr_weight': 0.0,
            'short_weight': 0.0,
            'kelly_multiplier': 0.25,  # Near-zero — almost all cash
            'max_positions': 1,  # At most 1 tiny position
            'max_position_kelly': 0.03,  # Tiny position size
            'slippage_bps': 20,
            'regime_label': 'DEFENSE (HIGH VOL)',
            'note': 'Extreme volatility — near 100% cash, tiny positions only if absolutely necessary',
        }
    
    else:  # transition
        info = regime_reason.get('allocation', '70_tf_30_mr')
        
        if '70_tf' in info:
            return {
                'tf_enabled': True,
                'mr_enabled': True,
                'short_enabled': False,
                'tf_weight': 0.7,
                'mr_weight': 0.3,
                'short_weight': 0.0,
                'kelly_multiplier': 0.6,
                'max_positions': 5,
                'max_position_kelly': 0.10,
                'slippage_bps': 12,
                'regime_label': 'TRANSITION (TF bias)',
                'note': 'Gray zone with uptrend bias — mostly TF, some MR, reduced size',
            }
        elif '70_mr' in info:
            return {
                'tf_enabled': True,
                'mr_enabled': True,
                'short_enabled': False,
                'tf_weight': 0.3,
                'mr_weight': 0.7,
                'short_weight': 0.0,
                'kelly_multiplier': 0.6,
                'max_positions': 5,
                'max_position_kelly': 0.10,
                'slippage_bps': 12,
                'regime_label': 'TRANSITION (MR bias)',
                'note': 'Gray zone with downtrend bias — mostly MR, some TF, reduced size',
            }
        else:
            return {
                'tf_enabled': True,
                'mr_enabled': True,
                'short_enabled': False,
                'tf_weight': 0.5,
                'mr_weight': 0.5,
                'short_weight': 0.0,
                'kelly_multiplier': 0.5,
                'max_positions': 4,
                'max_position_kelly': 0.08,
                'slippage_bps': 15,
                'regime_label': 'TRANSITION (neutral)',
                'note': 'Neutral gray zone — equal TF/MR, half-Kelly, minimal exposure',
            }


# ============================================================
# VOLATILITY OVERRIDE CHECK (for short module)
# ============================================================
def check_volatility_override(features, threshold=0.03):
    """
    Check if volatility has spiked beyond safe threshold for short positions.
    
    If ATR/Price > threshold (default 3%), return True → force exit to cash.
    This protects against crash volatility where short squeezes can wipe accounts.
    
    Called every bar for active short positions.
    """
    if features is None:
        return False
    
    atr_ratio = features['atr_ratio']
    return atr_ratio > threshold


def get_volatility_override_action(features, threshold=0.03):
    """
    Returns action dict for volatility override:
    {
        'override': bool,       — True if volatility exceeds threshold
        'action': str,          — 'exit_all_shorts' or 'hold'
        'atr_ratio': float,     — Current ATR/Price ratio
        'threshold': float,     — Threshold that was breached
    }
    """
    override = check_volatility_override(features, threshold)
    return {
        'override': override,
        'action': 'exit_all_shorts' if override else 'hold',
        'atr_ratio': features['atr_ratio'] if features else 0,
        'threshold': threshold,
    }
