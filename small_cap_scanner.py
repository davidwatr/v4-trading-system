#!/usr/bin/env python3
"""
Small Cap Explosion Scanner — Find high-beta, high-momentum small caps
that can explode 5-20% in days. Run this ONCE to find candidates,
then just manage the trades (hold until SL or TP).
"""

import json
import sys
import pickle
import time
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from production_v18 import (
    fetch_data, extract_features, get_market_regime,
    calc_atr, calc_rsi, calc_bollinger, calc_vwap, calc_iv_rank,
    MODEL_FILE
)

# ============================================
# SMALL CAP EXPLOSION UNIVERSE
# High beta, high momentum, liquid enough to trade
# These stocks can move 5-20% in days
# ============================================
UNIVERSE = [
    # Small/Mid Cap Tech (high beta)
    "RBLX", "U", "SNAP", "PINS", "MTCH", "BMBL", "HOOD", "COIN", "SQ", "AFRM",
    "UPST", "SOFI", "LC", "OPAD", "RUM", "BIRD", "MVIS", "VUZI", "IDEX", "MARA",
    "RIOT", "CLSK", "HBT", "BTBT", "WULF", "CORZ", "SDIG", "HUT", "BITF", "SDIG",
    
    # Small/Mid Cap Growth (high momentum)
    "NVDA", "AMD", "SMCI", "ARM", "AVGO", "MRVL", "AMAT", "LRCX", "KLAC", "TER",
    "ON", "MPWR", "SWKS", "QRVO", "MCHP", "ADI", "TXN", "IFS", "SLAB", "POWI",
    
    # Biotech/Pharma (binary event potential)
    "MRNA", "BNTX", "NVAX", "SRPT", "BLUE", "CRSP", "EDIT", "NTLA", "VCEL", "BMRN",
    "FOLD", "EXEL", "HALO", "PRTA", "KPTI", "AVXL", "ARVN", "RCKT", "TNGX", "ANAB",
    
    # Consumer/Retail (high momentum)
    "LULU", "DECK", "ONON", "BIRK", "SBET", "ALL", "BYND", "LMND", "PSTG", "GTLB",
    "AI", "PATH", "WIZ", "NICE", "CYBR", "QLYS", "S", "TENB", "VRNS", "ZS",
    
    # Clean Energy/EV (volatile)
    "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "GOEV", "WKHS", "NKLA", "RIDE",
    "SPWR", "ENPH", "SEDG", "FSLR", "CSIQ", "JKS", "PLUG", "BE", "FCEL", "BLDP",
    
    # Space/Defense (high momentum)
    "SPCE", "RKLB", "MAXR", "IRDM", "GSAT", "LORL", "KTOS", "DRS", "OSK", "LHX",
    
    # Crypto/Blockchain (explosive)
    "MSTR", "RIOT", "MARA", "CLSK", "COIN", "HOOD", "SI", "BLOK", "BITO", "GBTC",
    
    # Small Cap FinTech
    "NU", "NUVB", "MQ", "SE", "MELI", "W", "SHOP", "JD", "PDD", "BABA",
    
    # High Beta Industrials
    "GPOR", "RIG", "DO", "NE", "PTEN", "WHD", "HLX", "OII", "SLB", "HAL",
    
    # REITs (select high beta)
    "INN", "AHT", "CLDT", "PEB", "SHO", "DRH", "HT", "RLJ", "APLE", "CHSP",
    
    # High Momentum/Short Squeeze candidates
    "GME", "AMC", "BB", "NOK", "BBBY", "WISH", "CLOV", "WKHS", "SKS", "SDC",
    
    # Small Cap Healthcare
    "VRAY", "EKSO", "CTIB", "NVOS", "GHSI", "CTXR", "EDTX", "CANF", "PTN", "OBS",
    
    # Mining/Resources (volatile)
    "GOLD", "NEM", "AEM", "WPM", "PAAS", "CDE", "HL", "MUX", "AG", "EXK",
    "FSM", "OR", "SA", "BTG", "EGO", "AUY", "IAG", "HL", "MUX", "AG",
]

# Remove duplicates while preserving order
seen = set()
UNIVERSE = [x for x in UNIVERSE if not (x in seen or seen.add(x))]


def calc_ema(closes, period=20):
    """Calculate Exponential Moving Average."""
    if len(closes) < period:
        return None
    ema = closes[0]
    alpha = 2 / (period + 1)
    for i in range(1, len(closes)):
        ema = closes[i] * alpha + ema * (1 - alpha)
    return ema


def calc_beta(stock_returns, market_returns, period=60):
    """Calculate beta vs market."""
    if len(stock_returns) < period or len(market_returns) < period:
        return 1.0
    s = stock_returns[-period:]
    m = market_returns[-period:]
    cov = np.cov(s, m)[0][1]
    var = np.var(m)
    return cov / var if var > 0 else 1.0


def calc_explosion_score(closes, volumes, highs, lows, spy_closes, sector_closes=None):
    """
    Calculate an "explosion score" — how likely is this stock to move big soon.
    Higher score = more potential for 5-20% move in days.
    """
    if len(closes) < 200:
        return 0, {}
    
    current_price = closes[-1]
    
    # 1. Beta (higher = more volatile = more explosive)
    stock_returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
    spy_returns = [(spy_closes[i] - spy_closes[i-1]) / spy_closes[i-1] for i in range(1, len(spy_closes))]
    beta = calc_beta(stock_returns, spy_returns, 60)
    
    # 2. RSI (oversold = potential bounce, overbought = potential continuation)
    rsi = calc_rsi(closes)
    
    # 3. Volume expansion (recent volume > average = interest building)
    vol_avg_20 = np.mean(volumes[-20:])
    vol_avg_50 = np.mean(volumes[-50:])
    vol_ratio = vol_avg_20 / vol_avg_50 if vol_avg_50 > 0 else 1.0
    
    # 4. Price vs 20-day EMA (near EMA = potential entry)
    ema20 = calc_ema(closes, 20)
    price_vs_ema20 = (current_price - ema20) / ema20 if ema20 and ema20 > 0 else 0
    
    # 5. Bollinger Band position (near lower band = oversold, near upper band = strong)
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)
    bb_pos = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper and bb_lower and bb_upper != bb_lower else 0.5
    
    # 6. ATR (volatility — higher = more explosive)
    atr = calc_atr(highs, lows, closes, 14)
    atr_pct = atr / current_price if current_price > 0 and atr else 0
    
    # 7. Recent momentum (5-day return)
    momentum_5d = (closes[-1] - closes[-6]) / closes[-6] if len(closes) > 6 else 0
    
    # 8. Short-term reversal (1-day return — oversold bounce potential)
    momentum_1d = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0
    
    # Explosion score calculation
    score = 0
    
    # Beta contribution (0-30 points)
    if beta > 2.0:
        score += 30
    elif beta > 1.5:
        score += 25
    elif beta > 1.2:
        score += 20
    elif beta > 1.0:
        score += 15
    elif beta > 0.8:
        score += 10
    
    # RSI contribution (0-20 points)
    # Oversold (RSI 30-40) = bounce potential
    # Strong momentum (RSI 55-65) = continuation potential
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
    
    # Volume contribution (0-20 points)
    if vol_ratio > 2.0:
        score += 20
    elif vol_ratio > 1.5:
        score += 15
    elif vol_ratio > 1.2:
        score += 10
    
    # Price vs EMA contribution (0-15 points)
    # Near EMA = good entry
    if -0.03 <= price_vs_ema20 <= 0.03:
        score += 15
    elif -0.05 <= price_vs_ema20 <= 0.05:
        score += 12
    elif -0.08 <= price_vs_ema20 <= 0.08:
        score += 8
    
    # BB position contribution (0-10 points)
    if bb_pos < 0.2:
        score += 10  # Near lower band = oversold
    elif bb_pos > 0.8:
        score += 8   # Near upper band = strong
    elif 0.3 <= bb_pos <= 0.7:
        score += 5
    
    # ATR contribution (0-5 points)
    if atr_pct > 0.05:
        score += 5
    elif atr_pct > 0.03:
        score += 3
    elif atr_pct > 0.02:
        score += 2
    
    return score, {
        "beta": round(beta, 2),
        "rsi": round(rsi, 1) if rsi else None,
        "vol_ratio": round(vol_ratio, 2),
        "price_vs_ema20": round(price_vs_ema20, 4),
        "bb_pos": round(bb_pos, 3),
        "atr_pct": round(atr_pct, 4),
        "momentum_5d": round(momentum_5d, 4),
        "momentum_1d": round(momentum_1d, 4)
    }


def detect_entry_signal(closes, volumes, highs, lows, rsi):
    """
    Detect entry signal for explosive move.
    Returns: (signal_type, confidence, reason)
    """
    if len(closes) < 50 or rsi is None:
        return None, 0, "Insufficient data"
    
    ema20 = calc_ema(closes, 20)
    if ema20 is None:
        return None, 0, "Can't calc EMA20"
    
    current_price = closes[-1]
    prev_price = closes[-2]
    
    # Signal 1: Oversold bounce (RSI < 35, price near EMA, green candle)
    if rsi < 35 and current_price > prev_price:
        if abs(current_price - ema20) / ema20 < 0.08:
            return "BOUNCE", 0.8, f"Oversold bounce (RSI {rsi:.1f})"
    
    # Signal 2: Breakout (price above recent high on volume)
    recent_high = max(highs[-20:])
    vol_avg = np.mean(volumes[-20:])
    if current_price > recent_high * 0.98 and volumes[-1] > vol_avg * 1.3:
        return "BREAKOUT", 0.9, f"Breakout above ${recent_high:.2f} on volume"
    
    # Signal 3: EMA pullback bounce (price touched EMA and bounced)
    if prev_price < ema20 and current_price > ema20:
        return "EMA_BOUNCE", 0.7, f"Bounced off EMA20 ${ema20:.2f}"
    
    # Signal 4: Momentum continuation (strong trend, slight pullback)
    if rsi > 50 and current_price > ema20 and prev_price < closes[-3]:
        if abs(current_price - ema20) / ema20 < 0.05:
            return "MOMENTUM", 0.6, f"Momentum continuation (RSI {rsi:.1f})"
    
    return None, 0, "No signal"


def calculate_position_size(account_size, entry, stop, risk_pct=2.0):
    """
    Position sizing for small caps — can be more aggressive.
    """
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0, 0, 0, 0
    
    max_risk = account_size * (risk_pct / 100)
    shares = int(max_risk / risk_per_share)
    target_1 = entry + 2.5 * risk_per_share  # 2.5:1 R:R for small caps
    target_2 = entry + 4 * risk_per_share    # 4:1 R:R for runners
    
    return shares, risk_per_share, max_risk, target_1, target_2


def main():
    print("=" * 70)
    print(f"  SMALL CAP EXPLOSION SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Load ML model (optional — for confirmation)
    ml_available = False
    model = None
    features = None
    if MODEL_FILE.exists():
        try:
            with open(MODEL_FILE, 'rb') as f:
                saved = pickle.load(f)
            model = saved['model']
            features = saved['feature_names']
            ml_available = True
            print("ML model loaded")
        except Exception as e:
            print(f"ML model load failed: {e}")
    else:
        print("No ML model — using explosion score only")
    
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
    
    for symbol in UNIVERSE:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=730)
            if not closes or len(closes) < 200:
                continue
            
            processed += 1
            current_price = closes[-1]
            
            # Skip if price < $5 (too risky) or > $500 (not explosive enough)
            if current_price < 5 or current_price > 500:
                continue
            
            # Skip if avg volume < 500K (can't trade)
            avg_vol = np.mean(volumes[-30:])
            if avg_vol < 500_000:
                continue
            
            # Calculate explosion score
            score, metrics = calc_explosion_score(closes, volumes, highs, lows, spy_closes)
            
            # Only keep high-scoring candidates
            if score < 40:
                continue
            
            # Detect entry signal
            rsi = calc_rsi(closes)
            signal_type, confidence, signal_reason = detect_entry_signal(closes, volumes, highs, lows, rsi)
            
            if signal_type is None:
                continue
            
            # ML confirmation (optional)
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
            stop = current_price - (1.5 * atr)  # 1.5 ATR stop
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
                "regime": regime,
                "timestamp": datetime.now().isoformat()
            })
            
            time.sleep(0.1)
            
        except Exception as e:
            continue
    
    # Sort by score (highest = most explosive)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Save results
    output_file = Path('small_cap_signals.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed: {processed} | High-score candidates: {len(results)}")
    print(f"\n{'=' * 70}")
    print(f"  TOP 15 EXPLOSION CANDIDATES")
    print(f"{'=' * 70}")
    
    for i, r in enumerate(results[:15]):
        ml_str = f"{r['ml_probability']:.1%}" if r['ml_probability'] is not None else "N/A"
        conf_emoji = "🔥" if r['confidence'] >= 0.8 else "⚡" if r['confidence'] >= 0.6 else "💡"
        print(f"\n  #{i+1} {r['symbol']} {conf_emoji} Score: {r['score']}")
        print(f"     Price: ${r['price']} | Beta: {r['beta']} | RSI: {r['rsi']} | Vol Ratio: {r['vol_ratio']}")
        print(f"     Signal: {r['signal_type']} ({r['confidence']:.0%} confidence)")
        print(f"     Entry: ${r['entry']} | Stop: ${r['stop']} | Target 1: ${r['target_1']} | Target 2: ${r['target_2']}")
        print(f"     Risk/Share: ${r['risk_per_share']} | Shares (10k): {r['shares_10k']} | R:R: 2.5:1 / 4:1")
        print(f"     5D Momentum: {r['momentum_5d']:.2%} | ML: {ml_str}")
        print(f"     {r['signal_reason']}")
    
    print(f"\n{'=' * 70}")
    print(f"  Saved {len(results)} candidates to {output_file}")
    print(f"{'=' * 70}")
    
    return results


if __name__ == "__main__":
    main()
