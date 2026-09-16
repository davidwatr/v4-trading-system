#!/usr/bin/env python3
"""
Production Model v18 — Bull Market Cruise Control
Market regime-aware position sizing and trade filtering.
"""

import requests
import json
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

UNIVERSE = ["PYPL", "ORCL", "BRK-B", "BAC", "JPM", "GS", "MS", "WFC", "C", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "INTC", "AMD", "CSCO", "CRM", "ADBE", "NFLX", "DIS", "V", "MA"]

SECTOR_ETFS = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "NVDA": "XLK", "META": "XLK",
    "INTC": "XLK", "AMD": "XLK", "CSCO": "XLK", "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK",
    "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF", "WFC": "XLF", "C": "XLF",
    "BRK-B": "XLF", "V": "XLF", "MA": "XLF", "PYPL": "XLF",
    "AMZN": "XLY", "TSLA": "XLY", "DIS": "XLY", "NFLX": "XLY",
}

MODEL_FILE = Path("ml_model.pkl")

# ============================================
# INDICATORS
# ============================================
def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1: return None
    trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(1, len(closes))]
    return sum(trs[-period:]) / period

def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(0, change)); losses.append(max(0, -change))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calc_bollinger(closes, period=20, std=2):
    if len(closes) < period: return None, None, None
    sma = sum(closes[-period:]) / period
    variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
    std_dev = variance ** 0.5
    return sma + std * std_dev, sma, sma - std * std_dev

def calc_vwap(closes, volumes):
    if not closes or not volumes: return None
    total_pv = sum(c * v for c, v in zip(closes, volumes))
    total_v = sum(volumes)
    return total_pv / total_v if total_v > 0 else None

def calc_iv_rank(atr_history, current_atr, period=252):
    if not atr_history or len(atr_history) < min(period, len(atr_history)): return None
    recent = atr_history[-min(period, len(atr_history)):]
    low, high = min(recent), max(recent)
    if high == low: return 50.0
    return (current_atr - low) / (high - low) * 100

def calc_beta(stock_returns, market_returns, period=60):
    if len(stock_returns) < period or len(market_returns) < period: return 1.0
    s = stock_returns[-period:]
    m = market_returns[-period:]
    cov = np.cov(s, m)[0][1]
    var = np.var(m)
    return cov / var if var > 0 else 1.0

def calc_residual_returns(stock_closes, market_closes, period=60):
    if len(stock_closes) < period + 1 or len(market_closes) < period + 1: return [0.0]
    stock_ret = [(stock_closes[i] - stock_closes[i-1]) / stock_closes[i-1] for i in range(1, len(stock_closes))]
    mkt_ret = [(market_closes[i] - market_closes[i-1]) / market_closes[i-1] for i in range(1, len(market_closes))]
    beta = calc_beta(stock_ret, mkt_ret, period)
    return [stock_ret[i] - beta * mkt_ret[i] for i in range(len(stock_ret))]

def calc_relative_strength(stock_closes, sector_closes, period=20):
    if len(stock_closes) < period or len(sector_closes) < period: return 0.0
    stock_perf = (stock_closes[-1] - stock_closes[-period]) / stock_closes[-period]
    sector_perf = (sector_closes[-1] - sector_closes[-period]) / sector_closes[-period]
    return stock_perf - sector_perf

def calc_rvol(volumes, period=20):
    if len(volumes) < period: return 1.0
    avg = sum(volumes[-period:]) / period
    return volumes[-1] / avg if avg > 0 else 1.0

# ============================================
# 1. FEATURE SELECTION
# ============================================
def select_features(model, feature_names, threshold=0.02):
    """Strip features with importance < threshold (2%)"""
    importances = model.feature_importances_
    total = sum(importances)
    selected = []
    selected_importances = []
    for i, (name, imp) in enumerate(zip(feature_names, importances)):
        normalized_imp = imp / total
        if normalized_imp >= threshold:
            selected.append(name)
            selected_importances.append(normalized_imp)
    return selected, selected_importances

# ============================================
# 2. MARKET REGIME FILTERING (CRUISE CONTROL)
# ============================================
def get_market_regime(spy_closes, spy_highs, spy_lows):
    """
    Determine market regime:
    - bull: SPY > MA50 > MA200 (confirmed uptrend) → Cruise Control ON
    - bear: SPY < MA50 < MA200 (confirmed downtrend) → Defensive Mode
    - sideways: mixed → Defensive Mode
    - high_vol: ATR expanding rapidly → Defensive Mode
    
    Returns: (regime, penalty, big_cap_max, small_cap_max, kelly_multiplier)
    """
    if not spy_closes or len(spy_closes) < 200:
        return 'unknown', 1.0, 2, 1, 0.5
    
    current_price = spy_closes[-1]
    ma50 = sum(spy_closes[-50:]) / 50
    ma200 = sum(spy_closes[-200:]) / 200
    
    # ATR trend (volatility regime)
    atr_14 = calc_atr(spy_highs, spy_lows, spy_closes, 14)
    atr_50 = calc_atr(spy_highs, spy_lows, spy_closes, 50)
    atr_expanding = atr_14 > atr_50 * 1.2 if atr_14 and atr_50 else False
    
    # Trend regime
    if current_price > ma50 > ma200:
        regime = 'bull'
        penalty = 1.0  # No penalty
        big_cap_max = 8  # Allow more positions (was 5)
        small_cap_max = 4  # (was 3)
        kelly_multiplier = 1.5  # Scale up sizing
    elif current_price < ma50 < ma200:
        regime = 'bear'
        penalty = 0.7
        big_cap_max = 4  # (was 2)
        small_cap_max = 2  # (was 1)
        kelly_multiplier = 0.5  # Scale down
    else:
        regime = 'sideways'
        penalty = 0.85
        big_cap_max = 6  # (was 3)
        small_cap_max = 3  # (was 2)
        kelly_multiplier = 0.75
    
    # Volatility override
    if atr_expanding:
        regime += '_high_vol'
        penalty *= 0.8
        big_cap_max = max(1, big_cap_max - 1)
        small_cap_max = max(0, small_cap_max - 1)
        kelly_multiplier *= 0.7
    
    return regime, penalty, big_cap_max, small_cap_max, kelly_multiplier

# ============================================
# 3. DYNAMIC TRIPLE BARRIERS
# ============================================
def dynamic_triple_barriers(atr, closes, iv_rank):
    """
    Dynamic barriers based on:
    - ATR (volatility)
    - IV Rank (options market expectations)
    - Recent price compression/expansion
    """
    base_upper = 2.0
    base_lower = 1.5
    base_hold = 14
    
    if iv_rank is not None:
        if iv_rank > 80:
            upper_mult = base_upper * 1.3
            lower_mult = base_lower * 1.3
            max_hold = int(base_hold * 1.2)
        elif iv_rank < 20:
            upper_mult = base_upper * 0.8
            lower_mult = base_lower * 0.8
            max_hold = int(base_hold * 0.8)
        else:
            upper_mult = base_upper
            lower_mult = base_lower
            max_hold = base_hold
    else:
        upper_mult = base_upper
        lower_mult = base_lower
        max_hold = base_hold
    
    if len(closes) >= 20:
        recent_range = max(closes[-20:]) - min(closes[-20:])
        avg_price = sum(closes[-20:]) / 20
        compression = recent_range / avg_price if avg_price > 0 else 0
        
        if compression < 0.03:
            upper_mult *= 0.7
            lower_mult *= 0.7
            max_hold = int(max_hold * 0.7)
        elif compression > 0.10:
            upper_mult *= 1.3
            lower_mult *= 1.3
            max_hold = int(max_hold * 1.3)
    
    return upper_mult, lower_mult, max_hold

# ============================================
# TRIPLE BARRIER LABELS
# ============================================
def triple_barrier_labels(closes, highs, lows, atr_series, upper_mult=2.0, lower_mult=1.5, max_hold=14):
    labels = []
    for i in range(len(closes)):
        if i + max_hold >= len(closes) or atr_series[i] is None:
            labels.append(0); continue
        entry = closes[i]; atr = atr_series[i]
        upper = entry + atr * upper_mult; lower = entry - atr * lower_mult
        label = 0
        for j in range(1, max_hold + 1):
            if i + j >= len(closes): break
            if highs[i + j] >= upper: label = 1; break
            elif lows[i + j] <= lower: label = -1; break
        labels.append(label)
    return labels

# ============================================
# KELLY POSITION SIZING
# ============================================
def kelly_criterion(win_prob, payout_ratio, max_risk=0.1):
    """Fractional Kelly: f* = min(max_risk, (p*q - (1-p)) / q)"""
    if payout_ratio <= 0: return 0.0
    kelly = (win_prob * payout_ratio - (1 - win_prob)) / payout_ratio
    kelly = kelly / 2  # Half-Kelly
    return max(0.0, min(kelly, max_risk))

# ============================================
# DATA FETCHING
# ============================================
def fetch_data(symbol, days=730):
    try:
        end = int(time.time()); start = end - (days * 86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start}&period2={end}&interval=1d"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json'
        }
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 429: time.sleep(15); resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200: return None, None, None, None
        result = resp.json()["chart"]["result"][0]
        closes = [c for c in result["indicators"]["quote"][0]["close"] if c is not None]
        volumes = [v for v in result["indicators"]["quote"][0]["volume"] if v is not None]
        highs = [h for h in result["indicators"]["quote"][0]["high"] if h is not None]
        lows = [l for l in result["indicators"]["quote"][0]["low"] if l is not None]
        if len(closes) < 100: return None, None, None, None
        return closes, volumes, highs, lows
    except: return None, None, None, None

# ============================================
# FEATURE EXTRACTION
# ============================================
def calc_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal: return None, None, None
    ema_fast = closes[0]; ema_slow = closes[0]
    for i in range(1, len(closes)):
        alpha_fast = 2 / (fast + 1); alpha_slow = 2 / (slow + 1)
        ema_fast = closes[i] * alpha_fast + ema_fast * (1 - alpha_fast)
        ema_slow = closes[i] * alpha_slow + ema_slow * (1 - alpha_slow)
    macd_line = ema_fast - ema_slow
    return macd_line, macd_line * 0.9, macd_line * 0.8

def calc_adx(highs, lows, closes, period=14):
    """
    Proper ADX calculation.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    if len(closes) < period * 2 + 1:
        return None
    
    if highs is None:
        highs = closes
    if lows is None:
        lows = closes
    
    n = len(closes)
    
    # Step 1: Calculate +DM, -DM, and TR for each period
    plus_dm = []
    minus_dm = []
    tr_list = []
    
    for i in range(1, n):
        high = highs[i]
        low = lows[i]
        prev_high = highs[i-1]
        prev_low = lows[i-1]
        prev_close = closes[i-1]
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = max(tr1, tr2, tr3)
        tr_list.append(tr)
        
        # +DM and -DM
        up_move = high - prev_high
        down_move = prev_low - low
        
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
            minus_dm.append(0)
        elif down_move > up_move and down_move > 0:
            plus_dm.append(0)
            minus_dm.append(down_move)
        else:
            plus_dm.append(0)
            minus_dm.append(0)
    
    if len(tr_list) < period:
        return None
    
    # Step 2: Smooth using Wilder's smoothing (RMA)
    # First `period` values are simple sums, then Wilder's smoothing
    def wilder_smooth(values, p):
        if len(values) < p:
            return None
        # First value: simple sum of first p
        smoothed = [sum(values[:p])]
        for i in range(p, len(values)):
            smoothed.append(smoothed[-1] - smoothed[-1]/p + values[i])
        return smoothed
    
    tr_smooth = wilder_smooth(tr_list, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)
    
    if tr_smooth is None or plus_dm_smooth is None or minus_dm_smooth is None:
        return None
    
    # Step 3: Calculate +DI and -DI
    plus_di = []
    minus_di = []
    for i in range(len(tr_smooth)):
        if tr_smooth[i] > 0:
            plus_di.append(100 * plus_dm_smooth[i] / tr_smooth[i])
            minus_di.append(100 * minus_dm_smooth[i] / tr_smooth[i])
        else:
            plus_di.append(0)
            minus_di.append(0)
    
    # Step 4: Calculate DX and ADX
    dx_values = []
    for i in range(len(plus_di)):
        denom = plus_di[i] + minus_di[i]
        if denom > 0:
            dx = 100 * abs(plus_di[i] - minus_di[i]) / denom
        else:
            dx = 0
        dx_values.append(dx)
    
    # ADX is Wilder-smoothed average of DX
    if len(dx_values) < period:
        return None
    
    adx_values = wilder_smooth(dx_values, period)
    if adx_values is None or len(adx_values) == 0:
        return None
    
    return adx_values[-1]

def calc_stochastic(closes, highs, lows, period=14):
    if len(closes) < period: return None
    recent_high = max(highs[-period:]); recent_low = min(lows[-period:])
    if recent_high == recent_low: return 50.0
    return (closes[-1] - recent_low) / (recent_high - recent_low) * 100

def calc_mfi(highs, lows, closes, volumes, period=14):
    if len(closes) < period + 1: return None
    return 50.0

def calc_cci(highs, lows, closes, period=20):
    if len(closes) < period: return None
    return 0.0

def calc_williams_r(highs, lows, closes, period=14):
    if len(closes) < period: return None
    recent_high = max(highs[-period:]); recent_low = min(lows[-period:])
    if recent_high == recent_low: return -50.0
    return (recent_high - closes[-1]) / (recent_high - recent_low) * -100

def extract_features(closes, volumes, highs, lows, symbol, spy_closes, sector_closes):
    if len(closes) < 200: return None
    current_price = closes[-1]
    rsi = calc_rsi(closes)
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    vwap = calc_vwap(closes, volumes)
    atr = calc_atr(highs, lows, closes)
    ma_short = sum(closes[-20:]) / 20
    ma_long = sum(closes[-50:]) / 50
    ma_200 = sum(closes[-200:]) / 200
    avg_vol = sum(volumes[-20:]) / 20
    vol_ratio = volumes[-1] / avg_vol if avg_vol > 0 else 1
    support = min(lows[-20:])
    resistance = max(highs[-20:])
    atr_history = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(14, len(closes))]
    atr_history = [a for a in atr_history if a is not None]
    iv_rank = calc_iv_rank(atr_history, atr, period=min(252, len(atr_history)))
    atr_series = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
    tb_labels = triple_barrier_labels(closes, highs, lows, atr_series, upper_mult=2.0, lower_mult=1.5, max_hold=14)
    tb_label = tb_labels[-1] if tb_labels[-1] is not None else 0
    residual_returns = calc_residual_returns(closes, spy_closes, period=60)
    residual_return = residual_returns[-1] if residual_returns else 0.0
    rel_strength = calc_relative_strength(closes, sector_closes, period=20)
    rvol = calc_rvol(volumes, period=20)
    atr_normalized = atr / current_price if current_price > 0 and atr else 0.0
    bb_pos = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper and bb_lower and bb_upper != bb_lower else 0.5
    vwap_pos = (current_price - vwap) / vwap if vwap and vwap > 0 else 0
    
    macd, macd_signal, macd_hist = calc_macd(closes)
    adx = calc_adx(highs, lows, closes)
    stoch = calc_stochastic(closes, highs, lows)
    mfi = calc_mfi(highs, lows, closes, volumes)
    cci = calc_cci(highs, lows, closes)
    williams_r = calc_williams_r(highs, lows, closes)
    
    ma20 = ma_short
    price_vs_ma20 = (current_price - ma20) / ma20 if ma20 > 0 else 0
    price_vs_ma50 = (current_price - ma_long) / ma_long if ma_long > 0 else 0
    price_vs_ma200 = (current_price - ma_200) / ma_200 if ma_200 > 0 else 0
    
    # Rate of change
    roc_5 = (current_price - closes[-6]) / closes[-6] if len(closes) >= 6 and closes[-6] > 0 else 0
    roc_10 = (current_price - closes[-11]) / closes[-11] if len(closes) >= 11 and closes[-11] > 0 else 0
    roc_20 = (current_price - closes[-21]) / closes[-21] if len(closes) >= 21 and closes[-21] > 0 else 0
    
    # Volume vs average (longer period)
    avg_vol_50 = sum(volumes[-50:]) / 50 if len(volumes) >= 50 else avg_vol
    vol_vs_avg = volumes[-1] / avg_vol_50 if avg_vol_50 > 0 else 1
    
    # Trend: price slope vs MA200
    trend = price_vs_ma200
    
    return {
        'rsi': rsi, 'macd': macd or 0, 'macd_signal': macd_signal or 0,
        'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_pos': bb_pos,
        'vwap': vwap, 'vwap_pos': vwap_pos,
        'ma_short': ma_short, 'ma_long': ma_long, 'ma_200': ma_200,
        'price_vs_ma20': price_vs_ma20, 'price_vs_ma50': price_vs_ma50, 'price_vs_ma200': price_vs_ma200,
        'vol_ratio': vol_ratio, 'support': support, 'resistance': resistance, 'atr': atr,
        'adx': adx or 25, 'stoch': stoch or 50, 'mfi': mfi or 50, 'cci': cci or 0, 'williams_r': williams_r or -50,
        'iv_rank': iv_rank, 'tb_label': tb_label, 'residual_return': residual_return,
        'rel_strength': rel_strength, 'rvol': rvol, 'atr_normalized': atr_normalized,
        'current_price': current_price,
        'roc_5': roc_5, 'roc_10': roc_10, 'roc_20': roc_20, 'vol_vs_avg': vol_vs_avg, 'trend': trend
    }

# ============================================
# TRADE EXECUTION CARD
# ============================================
def print_trade_card(stock, entry_price, atr, win_prob, kelly_size, upper_mult, lower_mult, max_hold, regime, regime_penalty, max_positions, kelly_multiplier):
    """Print a clean, actionable trade execution card"""
    stop_loss = entry_price - (lower_mult * atr)
    take_profit = entry_price + (upper_mult * atr)
    risk_per_share = entry_price - stop_loss
    reward_per_share = take_profit - entry_price
    risk_reward = reward_per_share / risk_per_share if risk_per_share > 0 else 0
    
    # Position sizing for $10,000 account
    portfolio_risk = 10000 * kelly_size
    shares = int(portfolio_risk / entry_price) if entry_price > 0 else 0
    dollar_risk = shares * risk_per_share
    
    print("\n" + "=" * 70)
    print("  TRADE EXECUTION CARD — " + datetime.now().strftime('%Y-%m-%d %H:%M'))
    print("=" * 70)
    print(f"  STOCK:           {stock}")
    print(f"  DIRECTION:       LONG")
    print(f"-" * 70)
    print(f"  ENTRY PRICE:     ${entry_price:,.2f}")
    print(f"  STOP-LOSS:       ${stop_loss:,.2f}  (-{lower_mult:.2f} × ATR = ${lower_mult * atr:,.2f})")
    print(f"  TAKE-PROFIT:     ${take_profit:,.2f}  (+{upper_mult:.2f} × ATR = ${upper_mult * atr:,.2f})")
    print(f"  MAX HOLD:        {max_hold} days")
    print(f"-" * 70)
    print(f"  ATR (14-day):    ${atr:,.2f}")
    print(f"  RISK/SHARE:      ${risk_per_share:,.2f}")
    print(f"  REWARD/SHARE:    ${reward_per_share:,.2f}")
    print(f"  RISK/REWARD:     1 : {risk_reward:.2f}")
    print(f"-" * 70)
    print(f"  MARKET REGIME:   {regime.upper()}")
    print(f"  REGIME PENALTY:  {regime_penalty:.0%}")
    print(f"  MAX POSITIONS:   {max_positions}")
    print(f"  KELLY MULTIPLIER:{kelly_multiplier:.1f}x")
    print(f"-" * 70)
    print(f"  WIN PROBABILITY: {win_prob:.2%}")
    print(f"  KELLY SIZE:      {kelly_size:.2%}")
    print(f"-" * 70)
    print(f"  PORTFOLIO RISK:  ${portfolio_risk:,.2f}  (on $10,000 account)")
    print(f"  SHARES:          {shares} shares")
    print(f"  DOLLAR RISK:     ${dollar_risk:,.2f}")
    print("=" * 70)

# ============================================
# MAIN
# ============================================
def main():
    print("=" * 70)
    print("  PRODUCTION MODEL v18 — Bull Market Cruise Control")
    print("=" * 70)
    
    # Load XGBoost model
    if not MODEL_FILE.exists():
        print("ERROR: XGBoost model not found."); return
    with open(MODEL_FILE, 'rb') as f: xgb_saved = pickle.load(f)
    xgb_model = xgb_saved['model']; xgb_features = xgb_saved['feature_names']
    
    # 1. Feature Selection — use all 24 features (model expects 24)
    selected_features = xgb_features
    print(f"\nFeatures: {len(selected_features)} (using all)")
    
    # Fetch SPY
    print("\nFetching SPY...")
    spy_closes, _, spy_highs, spy_lows = fetch_data("SPY", days=730)
    if not spy_closes: print("ERROR: Failed to fetch SPY"); return
    
    # 2. Market Regime
    regime, regime_penalty, max_positions, kelly_multiplier = get_market_regime(spy_closes, spy_highs, spy_lows)
    print(f"Market Regime: {regime} (penalty: {regime_penalty:.0%}, max positions: {max_positions}, Kelly mult: {kelly_multiplier}x)")
    
    results = []
    for symbol in UNIVERSE:
        closes, volumes, highs, lows = fetch_data(symbol, days=730)
        if not closes: continue
        sector_etf = SECTOR_ETFS.get(symbol, "SPY")
        sector_closes, _, _, _ = fetch_data(sector_etf, days=730)
        if not sector_closes: sector_closes = spy_closes
        features = extract_features(closes, volumes, highs, lows, symbol, spy_closes, sector_closes)
        if features: results.append({'symbol': symbol, 'features': features, 'closes': closes}); time.sleep(0.3)
    
    # Sort by residual return
    results.sort(key=lambda x: x['features']['residual_return'], reverse=True)
    top = results[0]
    f = top['features']
    
    # 3. Dynamic Triple Barriers
    upper_mult, lower_mult, max_hold = dynamic_triple_barriers(f['atr'], closes, f['iv_rank'])
    
    # Calculate Kelly using XGBoost features
    X = [[f[name] for name in selected_features]]
    raw_prob = xgb_model.predict_proba(X)[0][1]
    
    # Apply regime penalty
    penalized_prob = raw_prob * regime_penalty
    
    # Kelly with multiplier
    payout_ratio = upper_mult / lower_mult
    base_kelly = kelly_criterion(penalized_prob, payout_ratio, max_risk=0.1)
    kelly_size = min(base_kelly * kelly_multiplier, 0.25)  # Cap at 25%
    
    # Print trade card
    print_trade_card(
        top['symbol'], f['current_price'], f['atr'],
        penalized_prob, kelly_size,
        upper_mult, lower_mult, max_hold,
        regime, regime_penalty, big_cap_max, small_cap_max, kelly_mult
    )

if __name__ == "__main__":
    main()
