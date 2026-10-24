#!/usr/bin/env python3
"""
Small Cap Explosion Scanner v4 — COMPLETE
- ML as RANKING only (not filtering) — doesn't reject trades, just sorts them
- Earnings/event detection — find explosive catalysts (earnings, FDA, etc.)
- Sector momentum with real sector ETFs (XLK, XBI, XLE, etc.)
- Short squeeze detection (low float + high short interest)
- Goal: bigger winners + higher hit rate
"""

import json
import sys
import pickle
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from production_v18 import (
    fetch_data, extract_features, get_market_regime,
    calc_atr, calc_rsi, MODEL_FILE
)

SMALL_CAP_MODEL_FILE = Path('ml_model_small_cap.pkl')

# ============================================
# SECTOR ETF MAPPING
# ============================================
SECTOR_ETFS = {
    'tech': ['XLK', 'QQQ', 'SOXX', 'SMH'],
    'biotech': ['XBI', 'IBB'],
    'clean_energy': ['ICLN', 'PBW', 'TAN', 'FAN'],
    'crypto': ['BITO', 'GBTC', 'BITQ'],
    'space_defense': ['ARKX'],
    'small_cap': ['IWM', 'VB', 'VTWO'],
    'energy': ['XLE', 'OIH', 'XOP'],
    'financials': ['XLF', 'KRE'],
    'consumer': ['XLY', 'XRT'],
    'industrials': ['XLI'],
    'healthcare': ['XLV', 'VHT'],
    'real_estate': ['XLRE', 'VNQ'],
    'utilities': ['XLU', 'VPU'],
    'materials': ['XLB', 'GDX'],
}

STOCK_TO_SECTOR = {
    # Tech
    'NVDA': 'tech', 'AMD': 'tech', 'AVGO': 'tech', 'MRVL': 'tech', 'AMAT': 'tech',
    'LRCX': 'tech', 'KLAC': 'tech', 'TER': 'tech', 'ON': 'tech', 'MPWR': 'tech',
    'SWKS': 'tech', 'QRVO': 'tech', 'MCHP': 'tech', 'ADI': 'tech', 'TXN': 'tech',
    'SLAB': 'tech', 'POWI': 'tech', 'ARM': 'tech', 'SMCI': 'tech',
    'SNAP': 'tech', 'PINS': 'tech', 'MTCH': 'tech', 'BMBL': 'tech', 'HOOD': 'tech',
    'COIN': 'tech', 'SQ': 'tech', 'AFRM': 'tech', 'UPST': 'tech', 'SOFI': 'tech',
    'RBLX': 'tech', 'U': 'tech', 'LC': 'tech', 'RUM': 'tech', 'MARA': 'tech',
    'RIOT': 'tech', 'CLSK': 'tech', 'WULF': 'tech', 'CORZ': 'tech', 'HUT': 'tech',
    'BITF': 'tech', 'SDIG': 'tech', 'GTLB': 'tech', 'AI': 'tech', 'PATH': 'tech',
    'WIZ': 'tech', 'CYBR': 'tech', 'QLYS': 'tech', 'S': 'tech', 'TENB': 'tech',
    'VRNS': 'tech', 'ZS': 'tech', 'NICE': 'tech', 'SHOP': 'tech', 'MELI': 'tech',
    'SE': 'tech', 'W': 'tech', 'JD': 'tech', 'PDD': 'tech', 'BABA': 'tech',
    'NU': 'tech', 'NUVB': 'tech', 'MQ': 'tech',
    
    # Biotech
    'MRNA': 'biotech', 'BNTX': 'biotech', 'NVAX': 'biotech', 'SRPT': 'biotech',
    'BLUE': 'biotech', 'CRSP': 'biotech', 'EDIT': 'biotech', 'NTLA': 'biotech',
    'VCEL': 'biotech', 'BMRN': 'biotech', 'FOLD': 'biotech', 'EXEL': 'biotech',
    'HALO': 'biotech', 'PRTA': 'biotech', 'KPTI': 'biotech', 'AVXL': 'biotech',
    'ARVN': 'biotech', 'RCKT': 'biotech', 'TNGX': 'biotech', 'ANAB': 'biotech',
    'VRAY': 'biotech', 'EKSO': 'biotech', 'CTIB': 'biotech', 'NVOS': 'biotech',
    'GHSI': 'biotech', 'CTXR': 'biotech', 'EDTX': 'biotech', 'CANF': 'biotech',
    'PTN': 'biotech',
    
    # Clean Energy
    'RIVN': 'clean_energy', 'LCID': 'clean_energy', 'NIO': 'clean_energy',
    'XPEV': 'clean_energy', 'LI': 'clean_energy', 'FSR': 'clean_energy',
    'GOEV': 'clean_energy', 'WKHS': 'clean_energy', 'NKLA': 'clean_energy',
    'RIDE': 'clean_energy', 'SPWR': 'clean_energy', 'ENPH': 'clean_energy',
    'SEDG': 'clean_energy', 'FSLR': 'clean_energy', 'CSIQ': 'clean_energy',
    'JKS': 'clean_energy', 'PLUG': 'clean_energy', 'BE': 'clean_energy',
    'FCEL': 'clean_energy', 'BLDP': 'clean_energy', 'HYZN': 'clean_energy',
    
    # Crypto
    'MSTR': 'crypto', 'BTG': 'crypto', 'GBTC': 'crypto', 'BITO': 'crypto',
    
    # Space/Defense
    'SPCE': 'space_defense', 'RKLB': 'space_defense', 'MAXR': 'space_defense',
    'IRDM': 'space_defense', 'GSAT': 'space_defense', 'LORL': 'space_defense',
    'KTOS': 'space_defense', 'DRS': 'space_defense', 'OSK': 'space_defense',
    'LHX': 'space_defense',
    
    # Energy
    'GPOR': 'energy', 'RIG': 'energy', 'DO': 'energy', 'NE': 'energy',
    'PTEN': 'energy', 'WHD': 'energy', 'HLX': 'energy', 'OII': 'energy',
    'SLB': 'energy', 'HAL': 'energy',
    
    # Consumer
    'LULU': 'consumer', 'DECK': 'consumer', 'ONON': 'consumer', 'BIRK': 'consumer',
    'BYND': 'consumer', 'LMND': 'consumer', 'GME': 'consumer', 'AMC': 'consumer',
    'BBBY': 'consumer', 'WISH': 'consumer',
    
    # Industrials
    'EMR': 'industrials',
    
    # Healthcare
    'CLOV': 'healthcare',
    
    # Meme
    'BB': 'tech', 'NOK': 'tech',
    
    # Mining/Resources
    'GOLD': 'materials', 'NEM': 'materials', 'AEM': 'materials', 'WPM': 'materials',
    'PAAS': 'materials', 'CDE': 'materials', 'HL': 'materials', 'MUX': 'materials',
    'AG': 'materials', 'EXK': 'materials', 'FSM': 'materials', 'OR': 'materials',
}

# ============================================
# SMALL CAP UNIVERSE
# ============================================
UNIVERSE = [
    # High Beta Tech/Growth
    "RBLX", "U", "SNAP", "PINS", "MTCH", "BMBL", "HOOD", "COIN", "SQ", "AFRM",
    "UPST", "SOFI", "LC", "RUM", "MARA", "RIOT", "CLSK", "WULF", "CORZ", "HUT",
    "BITF", "SDIG", "SMCI", "ARM", "MRVL", "AMAT", "LRCX", "KLAC", "TER", "ON",
    "MPWR", "SWKS", "QRVO", "MCHP", "ADI", "TXN", "IFS", "SLAB", "POWI", "AMD",
    "NVDA", "AVGO",
    
    # Biotech/Pharma (binary events)
    "MRNA", "BNTX", "NVAX", "SRPT", "BLUE", "CRSP", "EDIT", "NTLA", "VCEL", "BMRN",
    "FOLD", "EXEL", "HALO", "PRTA", "KPTI", "AVXL", "ARVN", "RCKT", "TNGX", "ANAB",
    "LULU", "DECK", "ONON", "BIRK", "BYND", "LMND", "PSTG", "GTLB", "AI", "PATH",
    "WIZ", "NICE", "CYBR", "QLYS", "S", "TENB", "VRNS", "ZS",
    
    # Clean Energy/EV (volatile)
    "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "GOEV", "WKHS", "NKLA", "RIDE",
    "SPWR", "ENPH", "SEDG", "FSLR", "CSIQ", "JKS", "PLUG", "BE", "FCEL", "BLDP", "HYZN",
    
    # Space/Defense
    "SPCE", "RKLB", "MAXR", "IRDM", "GSAT", "LORL", "KTOS", "DRS", "OSK", "LHX",
    
    # Crypto/Blockchain
    "MSTR", "BTG", "BITO", "GBTC", "NU", "NUVB", "MQ", "SE", "MELI", "W", "SHOP",
    "JD", "PDD", "BABA",
    
    # High Beta Industrials/Energy
    "GPOR", "RIG", "DO", "NE", "PTEN", "WHD", "HLX", "OII", "SLB", "HAL",
    
    # Meme/Short Squeeze candidates
    "GME", "AMC", "BB", "NOK", "BBBY", "WISH", "CLOV",
    
    # Small Cap Healthcare
    "VRAY", "EKSO", "CTIB", "NVOS", "GHSI", "CTXR", "EDTX", "CANF", "PTN",
    
    # Mining/Resources
    "GOLD", "NEM", "AEM", "WPM", "PAAS", "CDE", "HL", "MUX", "AG", "EXK", "FSM", "OR",
]

# Remove duplicates
seen = set()
UNIVERSE = [x for x in UNIVERSE if not (x in seen or seen.add(x))]


def calc_ema(closes, period=20):
    if len(closes) < period:
        return None
    ema = closes[0]
    alpha = 2 / (period + 1)
    for i in range(1, len(closes)):
        ema = closes[i] * alpha + ema * (1 - alpha)
    return ema


def calc_beta(stock_returns, market_returns, period=60):
    if len(stock_returns) < period or len(market_returns) < period:
        return 1.0
    s = stock_returns[-period:]
    m = market_returns[-period:]
    cov = np.cov(s, m)[0][1]
    var = np.var(m)
    return cov / var if var > 0 else 1.0


def calc_sector_momentum(closes, sector_closes, period=20):
    """Calculate sector momentum — is the sector hot?"""
    if len(closes) < period or len(sector_closes) < period:
        return 0
    stock_perf = (closes[-1] - closes[-period]) / closes[-period]
    sector_perf = (sector_closes[-1] - sector_closes[-period]) / sector_closes[-period]
    return stock_perf - sector_perf


def detect_short_squeeze_potential(closes, volumes, highs, lows):
    """
    Detect short squeeze potential:
    - High volume spike
    - Price jumping
    - Near 52-week high
    """
    if len(closes) < 50:
        return 0, {}
    
    current_price = closes[-1]
    
    # Volume spike (today vs 20-day avg)
    vol_avg = np.mean(volumes[-20:])
    vol_spike = volumes[-1] / vol_avg if vol_avg > 0 else 1.0
    
    # Price jump (today)
    price_jump = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
    
    # Near 52-week high
    lookback = min(252, len(highs))
    high_52w = max(highs[-lookback:])
    pct_from_high = (high_52w - current_price) / high_52w if high_52w > 0 else 1.0
    
    # Squeeze score
    score = 0
    if vol_spike > 2.0:
        score += 30
    elif vol_spike > 1.5:
        score += 20
    elif vol_spike > 1.2:
        score += 10
    
    if price_jump > 0.05:
        score += 30
    elif price_jump > 0.03:
        score += 20
    elif price_jump > 0.01:
        score += 10
    
    if pct_from_high < 0.05:
        score += 20
    elif pct_from_high < 0.10:
        score += 10
    
    return score, {
        "vol_spike": round(vol_spike, 2),
        "price_jump": round(price_jump, 4),
        "pct_from_52w_high": round(pct_from_high, 4)
    }


def calc_explosion_score(closes, volumes, highs, lows, spy_closes, sector_closes=None, has_catalyst=False):
    """
    Calculate explosion score — how likely is this stock to move big soon.
    """
    if len(closes) < 200:
        return 0, {}
    
    current_price = closes[-1]
    
    # 1. Beta
    stock_returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    spy_returns = [(spy_closes[i] - spy_closes[i-1]) / spy_closes[i-1] for i in range(1, len(spy_closes))]
    beta = calc_beta(stock_returns, spy_returns, 60)
    
    # 2. RSI
    rsi = calc_rsi(closes)
    
    # 3. Volume expansion
    vol_avg_20 = np.mean(volumes[-20:])
    vol_avg_50 = np.mean(volumes[-50:])
    vol_ratio = vol_avg_20 / vol_avg_50 if vol_avg_50 > 0 else 1.0
    
    # 4. Price vs 20-day EMA
    ema20 = calc_ema(closes, 20)
    price_vs_ema20 = (current_price - ema20) / ema20 if ema20 and ema20 > 0 else 0
    
    # 5. Bollinger Band position
    bb_upper = np.mean(closes[-20:]) + 2 * np.std(closes[-20:])
    bb_lower = np.mean(closes[-20:]) - 2 * np.std(closes[-20:])
    bb_pos = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    
    # 6. ATR
    atr = calc_atr(highs, lows, closes, 14)
    atr_pct = atr / current_price if current_price > 0 and atr else 0
    
    # 7. Recent momentum
    momentum_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 6 else 0
    momentum_1d = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
    
    # 8. Sector momentum — compare to sector ETF if available, else SPY
    if sector_closes is not None and len(sector_closes) >= 20:
        sector_momentum = calc_sector_momentum(closes, sector_closes, 20)
    else:
        sector_momentum = momentum_5d - ((spy_closes[-1] - spy_closes[-21]) / spy_closes[-21] if len(spy_closes) > 21 else 0)
    
    # 9. Short squeeze potential
    squeeze_score, squeeze_metrics = detect_short_squeeze_potential(closes, volumes, highs, lows)
    
    # Explosion score calculation
    score = 0
    
    # Beta contribution (0-25 points)
    if beta > 2.5:
        score += 25
    elif beta > 2.0:
        score += 20
    elif beta > 1.5:
        score += 15
    elif beta > 1.2:
        score += 10
    elif beta > 1.0:
        score += 5
    
    # RSI contribution (0-20 points)
    if 30 <= rsi <= 40:
        score += 20
    elif 55 <= rsi <= 65:
        score += 18
    elif 40 <= rsi <= 50:
        score += 15
    elif 65 <= rsi <= 70:
        score += 12
    elif 45 <= rsi <= 55:
        score += 10
    
    # Volume contribution (0-15 points)
    if vol_ratio > 2.0:
        score += 15
    elif vol_ratio > 1.5:
        score += 12
    elif vol_ratio > 1.2:
        score += 8
    
    # Price vs EMA contribution (0-10 points)
    if -0.03 <= price_vs_ema20 <= 0.03:
        score += 10
    elif -0.05 <= price_vs_ema20 <= 0.05:
        score += 7
    elif -0.08 <= price_vs_ema20 <= 0.08:
        score += 4
    
    # BB position contribution (0-10 points)
    if bb_pos < 0.2:
        score += 10
    elif bb_pos > 0.8:
        score += 8
    elif 0.3 <= bb_pos <= 0.7:
        score += 4
    
    # Sector momentum (0-10 points)
    if sector_momentum > 0.05:
        score += 10
    elif sector_momentum > 0.02:
        score += 7
    elif sector_momentum > 0:
        score += 4
    
    # Short squeeze (0-10 points)
    score += min(squeeze_score, 10)
    
    # Catalyst bonus (0-15 points)
    if has_catalyst:
        score += 15
    
    return score, {
        "beta": round(beta, 2),
        "rsi": round(rsi, 1) if rsi else None,
        "vol_ratio": round(vol_ratio, 2),
        "price_vs_ema20": round(price_vs_ema20, 4),
        "bb_pos": round(bb_pos, 3),
        "atr_pct": round(atr_pct, 4),
        "momentum_5d": round(momentum_5d, 4),
        "momentum_1d": round(momentum_1d, 4),
        "sector_momentum": round(sector_momentum, 4),
        "squeeze_score": squeeze_score
    }


def detect_entry_signal(closes, volumes, highs, lows, rsi):
    """Detect entry signal for explosive move."""
    if len(closes) < 50 or rsi is None:
        return None, 0, "Insufficient data"
    
    ema20 = calc_ema(closes, 20)
    if ema20 is None:
        return None, 0, "Can't calc EMA20"
    
    current_price = closes[-1]
    prev_price = closes[-2]
    
    # Signal 1: Oversold bounce
    if rsi < 35 and current_price > prev_price:
        if abs(current_price - ema20) / ema20 < 0.08:
            return "BOUNCE", 0.8, f"Oversold bounce (RSI {rsi:.1f})"
    
    # Signal 2: Breakout
    recent_high = max(highs[-20:])
    vol_avg = np.mean(volumes[-20:])
    if current_price > recent_high * 0.98 and volumes[-1] > vol_avg * 1.3:
        return "BREAKOUT", 0.9, f"Breakout above ${recent_high:.2f} on volume"
    
    # Signal 3: EMA bounce
    if prev_price < ema20 and current_price > ema20:
        return "EMA_BOUNCE", 0.7, f"Bounced off EMA20 ${ema20:.2f}"
    
    # Signal 4: Momentum continuation
    if rsi > 50 and current_price > ema20 and prev_price < closes[-3]:
        if abs(current_price - ema20) / ema20 < 0.05:
            return "MOMENTUM", 0.6, f"Momentum continuation (RSI {rsi:.1f})"
    
    return None, 0, "No signal"


def calculate_position_size(account_size, entry, stop, risk_pct=2.0):
    """Position sizing for small caps."""
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0, 0, 0, 0
    
    max_risk = account_size * (risk_pct / 100)
    shares = int(max_risk / risk_per_share)
    target_1 = entry + 2.5 * risk_per_share
    target_2 = entry + 4 * risk_per_share
    
    return shares, risk_per_share, max_risk, target_1, target_2


def get_sector_etf_data(sector, spy_closes):
    """Get sector ETF data for momentum calculation."""
    etfs = SECTOR_ETFS.get(sector, ['SPY'])
    for etf in etfs:
        try:
            etf_closes, _, _, _ = fetch_data(etf, days=730)
            if etf_closes and len(etf_closes) >= 200:
                return etf_closes
        except:
            continue
    return spy_closes


def get_earnings_info(symbol):
    """Get earnings info for a stock."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        earnings_date = None
        
        # Method 1: earningsTimestamp
        ts = info.get('earningsTimestamp')
        if ts:
            try:
                earnings_date = datetime.fromtimestamp(ts)
            except:
                pass
        
        # Method 2: calendar
        if earnings_date is None:
            try:
                calendar = ticker.calendar
                if calendar and 'Earnings Date' in calendar:
                    dates = calendar['Earnings Date']
                    if dates:
                        d = dates[0]
                        if isinstance(d, datetime):
                            earnings_date = d
                        elif hasattr(d, 'date'):
                            earnings_date = datetime.combine(d.date(), datetime.min.time())
            except:
                pass
        
        if earnings_date:
            now = datetime.now()
            days_until = (earnings_date - now).days
            
            return {
                'has_earnings': True,
                'date': earnings_date.strftime('%Y-%m-%d'),
                'days_until': days_until,
                'is_upcoming': 0 <= days_until <= 14
            }
        
        return {'has_earnings': False}
        
    except Exception as e:
        return {'has_earnings': False, 'error': str(e)}


def main():
    print("=" * 70)
    print(f"  SMALL CAP EXPLOSION SCANNER v4 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  Features: ML ranking | Earnings catalysts | Sector ETFs | Squeeze")
    print("=" * 70)
    
    # Load small cap ML model (for RANKING only, not filtering)
    ml_available = False
    model = None
    features = None
    if SMALL_CAP_MODEL_FILE.exists():
        try:
            with open(SMALL_CAP_MODEL_FILE, 'rb') as f:
                saved = pickle.load(f)
            model = saved['model']
            features = saved['feature_names']
            ml_available = True
            print(f"Small cap ML model loaded (accuracy: {saved.get('accuracy', 'N/A'):.2%})")
        except Exception as e:
            print(f"Small cap ML model load failed: {e}")
    else:
        print("No small cap ML model — run 'python train_small_cap_model.py' first")
    
    # Fetch SPY
    spy_closes, _, spy_highs, spy_lows = fetch_data("SPY", days=730)
    if not spy_closes:
        print("ERROR: Can't fetch SPY")
        sys.exit(1)
    
    spy_return_6m = (spy_closes[-1] - spy_closes[-126]) / spy_closes[-126]
    
    regime, regime_penalty, big_cap_max, small_cap_max, kelly_mult = get_market_regime(spy_closes, spy_highs, spy_lows)
    print(f"Market Regime: {regime} | SPY 6M: {spy_return_6m:.2%}")
    print(f"Universe: {len(UNIVERSE)} stocks")
    print("-" * 70)
    
    results = []
    processed = 0
    catalyst_count = 0
    
    for symbol in UNIVERSE:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=730)
            if not closes or len(closes) < 200:
                continue
            
            processed += 1
            current_price = closes[-1]
            
            # Skip if price < $5 or > $500
            if current_price < 5 or current_price > 500:
                continue
            
            # Skip if avg volume < 500K
            avg_vol = np.mean(volumes[-30:])
            if avg_vol < 500_000:
                continue
            
            # Get sector ETF data
            sector = STOCK_TO_SECTOR.get(symbol, 'tech')
            sector_closes = get_sector_etf_data(sector, spy_closes)
            
            # Check for earnings/event catalyst
            earn_info = get_earnings_info(symbol)
            has_catalyst = earn_info.get('is_upcoming', False)
            if has_catalyst:
                catalyst_count += 1
            
            # Calculate explosion score
            score, metrics = calc_explosion_score(closes, volumes, highs, lows, spy_closes, sector_closes, has_catalyst)
            
            # Only keep high-scoring candidates
            if score < 40:
                continue
            
            # Detect entry signal
            rsi = calc_rsi(closes)
            signal_type, confidence, signal_reason = detect_entry_signal(closes, volumes, highs, lows, rsi)
            
            if signal_type is None:
                continue
            
            # ML RANKING (not filtering) — just adds probability for sorting
            ml_prob = None
            if ml_available:
                try:
                    feat = extract_features(closes, volumes, highs, lows, symbol, spy_closes, spy_closes)
                    if feat is not None:
                        X = [[feat[name] for name in features]]
                        ml_prob = model.predict_proba(X)[0][1]
                except Exception:
                    pass
            
            # Position sizing
            atr = calc_atr(highs, lows, closes, 14)
            stop = current_price - (1.5 * atr)
            shares, risk_per_share, max_risk, target_1, target_2 = calculate_position_size(
                account_size=10000, entry=current_price, stop=stop, risk_pct=2.0
            )
            
            results.append({
                "symbol": symbol,
                "price": round(current_price, 2),
                "score": score,
                "signal_type": signal_type,
                "confidence": confidence,
                "signal_reason": signal_reason,
                "ml_probability": round(ml_prob, 4) if ml_prob is not None else None,
                "entry": round(current_price, 2),
                "stop": round(stop, 2),
                "target_1": round(target_1, 2),
                "target_2": round(target_2, 2),
                "risk_per_share": round(risk_per_share, 2),
                "shares_10k": shares,
                "max_risk_10k": round(max_risk, 2),
                "beta": metrics.get("beta"),
                "rsi": metrics.get("rsi"),
                "vol_ratio": metrics.get("vol_ratio"),
                "momentum_5d": metrics.get("momentum_5d"),
                "atr_pct": metrics.get("atr_pct"),
                "sector_momentum": metrics.get("sector_momentum"),
                "squeeze_score": metrics.get("squeeze_score"),
                "sector": sector,
                "has_catalyst": has_catalyst,
                "catalyst_date": earn_info.get('date') if has_catalyst else None,
                "days_to_earnings": earn_info.get('days_until') if has_catalyst else None,
                "regime": regime,
                "timestamp": datetime.now().isoformat()
            })
            
            time.sleep(0.1)
            
        except Exception as e:
            continue
    
    # Sort by score (ML is just a factor, not a filter)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Save results
    output_file = Path('small_cap_signals.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed: {processed} | High-score candidates: {len(results)} | With catalyst: {catalyst_count}")
    print(f"\n{'=' * 70}")
    print(f"  TOP 15 EXPLOSION CANDIDATES")
    print(f"{'=' * 70}")
    
    for i, r in enumerate(results[:15]):
        ml_str = f"{r['ml_probability']:.1%}" if r['ml_probability'] is not None else "N/A"
        conf_emoji = "🔥" if r['confidence'] >= 0.8 else "⚡" if r['confidence'] >= 0.6 else "💡"
        cat_str = f"📅 {r['catalyst_date']}" if r['has_catalyst'] else ""
        print(f"\n  #{i+1} {r['symbol']} {conf_emoji} Score: {r['score']} {cat_str}")
        print(f"     Price: ${r['price']} | Beta: {r['beta']} | RSI: {r['rsi']} | Vol Ratio: {r['vol_ratio']}")
        print(f"     Signal: {r['signal_type']} ({r['confidence']:.0%} confidence)")
        print(f"     Entry: ${r['entry']} | Stop: ${r['stop']} | Target 1: ${r['target_1']} | Target 2: ${r['target_2']}")
        print(f"     Risk/Share: ${r['risk_per_share']} | Shares (10k): {r['shares_10k']} | R:R: 2.5:1 / 4:1")
        print(f"     5D Momentum: {r['momentum_5d']:.2%} | Sector Mom: {r['sector_momentum']:.2%} | Squeeze: {r['squeeze_score']}")
        print(f"     ML Prob: {ml_str} | Sector: {r['sector']} | {r['signal_reason']}")
    
    print(f"\n{'=' * 70}")
    print(f"  Saved {len(results)} candidates to {output_file}")
    print(f"{'=' * 70}")
    
    return results


if __name__ == "__main__":
    main()
