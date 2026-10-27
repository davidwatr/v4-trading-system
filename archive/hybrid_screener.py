#!/usr/bin/env python3
"""
Hybrid Swing Screener — Playbook-First Strategy
Playbook technical setup is primary signal. ML is secondary ranking bonus.
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
    dynamic_triple_barriers, calc_atr, calc_rsi,
    MODEL_FILE
)

# ============================================
# EXPANDED UNIVERSE — ~262 liquid stocks
# ============================================
UNIVERSE = [
    # Tech/Growth
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "ADBE", "AVGO", "QCOM", "TXN", "MU", "LRCX", "KLAC", "MRVL", "NXPI", "SWKS",
    "MCHP", "MPWR", "ON", "ASML", "SNPS", "CDNS", "ANSS", "PAYX", "VRSK", "WDAY",
    "ZS", "CRWD", "OKTA", "SPLK", "NET", "DDOG", "MDB", "SNOW", "PLTR", "RBLX",
    "U", "TTWO", "EA", "ATVI", "MTCH", "PINS", "SNAP", "SPOT", "DASH", "ABNB",
    "RIVN", "LCID", "NIO", "XPEV", "LI", "FSR", "GOEV", "WKHS", "NKLA", "RIDE",
    
    # FinTech / Payments
    "V", "MA", "PYPL", "SQ", "COIN", "HOOD", "AFRM", "UPST", "SOFI", "LC",
    
    # Healthcare / Biotech
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "DHR", "ABT", "BMY",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "ILMN", "DXCM", "ISRG", "ZTS", "HUM",
    "CVS", "CI", "HCA", "EW", "BSX", "SYK", "MDT", "HIMS", "TLRY",
    
    # Consumer / Retail
    "COST", "WMT", "HD", "LOW", "TGT", "NKE", "LULU", "DECK", "ONON",
    "SBUX", "MCD", "CMG", "YUM", "DPZ", "CHTR", "T", "VZ", "TMUS", "CMCSA",
    
    # Industrials / Defense
    "CAT", "DE", "BA", "LMT", "RTX", "NOC", "GD", "HII", "TDG", "AXON",
    "GE", "HON", "UPS", "FDX", "CSX", "UNP", "ITW", "ETN", "PH", "EMR",
    
    # Energy / Materials
    "XOM", "CVX", "COP", "SLB", "OXY", "EOG", "MPC", "VLO", "PSX", "WMB",
    "LIN", "APD", "SHW", "FCX", "NEM", "ECL", "DOW", "DD", "LYB", "CE",
    
    # Crypto-adjacent
    "MSTR", "RIOT", "MARA", "CLSK",
    
    # Meme / High momentum
    "GME", "AMC", "AAL", "UAL", "DAL", "CCL", "NCLH", "RCL", "EXPE", "BKNG",
    "UBER", "LYFT",
    
    # Semiconductors (expanded)
    "AMAT", "TER", "ADI", "IFS", "TSM",
    
    # Software / Cloud
    "INTU", "NOW", "PANW", "FTNT", "CYBR", "QLYS",
    
    # E-commerce / Digital
    "EBAY", "ETSY", "CHWY", "W", "SHOP", "MELI", "SE", "BABA", "JD", "PDD",
    
    # Biotech / Pharma (expanded)
    "MRNA", "BNTX", "NVAX", "ZBH", "UHS", "LH", "DGX",
    
    # Clean Energy / EV
    "SPWR", "ENPH", "SEDG", "FSLR", "CSIQ", "JKS", "PLUG", "BE", "FCEL",
    "BLDP", "HYZN",
    
    # REITs (select)
    "AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "DLR", "WELL", "AVB",
    
    # Transportation
    "NSC", "KNX", "LSTR", "JBHT", "ODFL",
    
    # Aerospace / Defense (expanded)
    "LHX", "KTOS",
    
    # Media / Entertainment
    "DIS", "NFLX", "FOXA", "PARA", "WBD", "LYV",
    
    # Space / Satellite
    "SPCE", "RKLB", "MAXR", "IRDM", "GSAT", "LORL", "TRMB", "GRMN",
    
    # Quantum Computing
    "IONQ", "RGTI", "QBTS", "ARQQ", "QUBT",
    
    # Cannabis
    "TLRY", "CGC", "ACB", "CRON", "SNDL", "OGI", "VFF", "HEXO",
    
    # Misc high-beta
    "BB", "NOK", "BBBY", "WISH", "CLOV", "WEN",
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


def check_trend(closes, highs, lows):
    """
    Relaxed trend check — just needs 50-day SMA above 200-day SMA.
    The setup detection handles the rest.
    """
    if len(closes) < 200:
        return False, "Insufficient data"
    
    ma50 = sum(closes[-50:]) / 50
    ma200 = sum(closes[-200:]) / 200
    
    if ma50 < ma200:
        return False, "50-day SMA below 200-day SMA"
    
    return True, "Uptrend confirmed"


def detect_setup(closes, volumes, highs, lows, rsi):
    """
    Simplified setup detection — returns setup type and reason.
    Relaxed for automated screening.
    """
    if len(closes) < 50 or rsi is None:
        return None, "Insufficient data"
    
    ema20 = calc_ema(closes, 20)
    if ema20 is None:
        return None, "Can't calc EMA20"
    
    current_price = closes[-1]
    
    # Setup A: Price near 20-day EMA (within 8%)
    ema_zone_low = ema20 * 0.92
    ema_zone_high = ema20 * 1.08
    in_ema_zone = ema_zone_low <= current_price <= ema_zone_high
    
    # Or price recently touched EMA
    recent_touch = any(ema_zone_low <= c <= ema_zone_high for c in closes[-10:])
    
    if in_ema_zone or recent_touch:
        # Check RSI in pullback zone (35-65)
        if 35 <= rsi <= 65:
            return "A", f"EMA pullback (RSI {rsi:.1f})"
    
    # Setup B: Price breaking above recent high
    recent_high = max(highs[-20:])
    if current_price >= recent_high * 0.98:
        # Check RSI in breakout zone (50-75)
        if 50 <= rsi <= 75:
            return "B", f"Breakout (RSI {rsi:.1f})"
    
    # Setup C: Strong uptrend with ML confirmation (fallback)
    if rsi > 50 and current_price > ema20:
        return "C", f"Uptrend continuation (RSI {rsi:.1f})"
    
    return None, "No setup"


def calculate_position_size(account_size, entry, stop, risk_pct=1.5):
    """
    Playbook position sizing: 1-2% risk per trade.
    """
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0, 0, 0, 0
    
    max_risk = account_size * (risk_pct / 100)
    shares = int(max_risk / risk_per_share)
    target_1 = entry + 2 * risk_per_share  # 2:1 R:R
    
    return shares, risk_per_share, max_risk, target_1


def main():
    print("=" * 70)
    print(f"  HYBRID SWING SCREENER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Load ML model (optional — for ranking only)
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
            print("ML model loaded (for ranking)")
        except Exception as e:
            print(f"ML model load failed: {e}")
    else:
        print("No ML model — using playbook signals only")
    
    # Fetch SPY for relative strength
    spy_closes, _, spy_highs, spy_lows = fetch_data("SPY", days=730)
    if not spy_closes:
        print("ERROR: Can't fetch SPY")
        sys.exit(1)
    
    spy_return_6m = (spy_closes[-1] - spy_closes[-126]) / spy_closes[-126]
    spy_return_3m = (spy_closes[-1] - spy_closes[-63]) / spy_closes[-63]
    
    regime, regime_penalty, big_cap_max, small_cap_max, kelly_mult = get_market_regime(spy_closes, spy_highs, spy_lows)
    print(f"Market Regime: {regime} | SPY 6M: {spy_return_6m:.2%}")
    print(f"Universe: {len(UNIVERSE)} stocks")
    print("-" * 70)
    
    results = []
    processed = 0
    filtered_out = 0
    
    for symbol in UNIVERSE:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=730)
            if not closes or len(closes) < 200:
                filtered_out += 1
                continue
            
            processed += 1
            current_price = closes[-1]
            
            # === BASIC FILTERS ===
            
            # Price > $5 (avoid penny stocks)
            if current_price < 5:
                filtered_out += 1
                continue
            
            # Average volume > 500K shares/day
            avg_vol = sum(volumes[-30:]) / 30
            if avg_vol < 500_000:
                filtered_out += 1
                continue
            
            # Trend check (relaxed)
            is_uptrend, trend_reason = check_trend(closes, highs, lows)
            if not is_uptrend:
                filtered_out += 1
                continue
            
            # === SETUP DETECTION ===
            rsi = calc_rsi(closes)
            setup_type, setup_reason = detect_setup(closes, volumes, highs, lows, rsi)
            
            if setup_type is None:
                filtered_out += 1
                continue
            
            # === ML RANKING (optional — doesn't filter, just ranks) ===
            ml_prob = None
            if ml_available:
                try:
                    feat = extract_features(closes, volumes, highs, lows, symbol, spy_closes, spy_closes)
                    if feat is not None:
                        X = [[feat[name] for name in features]]
                        ml_prob = model.predict_proba(X)[0][1]
                except Exception:
                    pass
            
            # === POSITION SIZING (Playbook style) ===
            if setup_type == "A":
                # Stop 2% below swing low
                stop = min(lows[-20:]) * 0.98
            elif setup_type == "B":
                # Stop just below breakout pivot
                stop = max(highs[-20:]) * 0.98
            else:
                # Stop 3% below current price
                stop = current_price * 0.97
            
            entry = current_price
            shares, risk_per_share, max_risk, target_1 = calculate_position_size(
                account_size=10000, entry=entry, stop=stop, risk_pct=1.5
            )
            
            # Calculate stock returns for relative strength
            stock_return_6m = (closes[-1] - closes[-126]) / closes[-126] if len(closes) > 126 else 0
            stock_return_3m = (closes[-1] - closes[-63]) / closes[-63] if len(closes) > 63 else 0
            
            results.append({
                "symbol": symbol,
                "price": round(current_price, 2),
                "setup": setup_type,
                "setup_reason": setup_reason,
                "rsi": round(rsi, 1),
                "ml_probability": round(ml_prob, 4) if ml_prob is not None else None,
                "regime": regime,
                "entry": round(entry, 2),
                "stop": round(stop, 2),
                "target_1": round(target_1, 2),
                "risk_per_share": round(risk_per_share, 2),
                "shares_10k": shares,
                "max_risk_10k": round(max_risk, 2),
                "reward_risk": round((target_1 - entry) / risk_per_share, 2) if risk_per_share > 0 else 0,
                "stock_return_6m": round(stock_return_6m, 4),
                "spy_return_6m": round(spy_return_6m, 4),
                "avg_volume": int(avg_vol),
                "timestamp": datetime.now().isoformat()
            })
            
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            filtered_out += 1
            continue
    
    # Sort by ML probability if available, otherwise by setup type
    if ml_available:
        results.sort(key=lambda x: x['ml_probability'] if x['ml_probability'] is not None else 0, reverse=True)
    else:
        # Sort by setup type (A > B > C) then by RSI
        setup_order = {"A": 0, "B": 1, "C": 2}
        results.sort(key=lambda x: (setup_order.get(x['setup'], 3), -x['rsi']))
    
    # Save results
    output_file = Path('hybrid_screener_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed: {processed} | Filtered out: {filtered_out} | Passing: {len(results)}")
    print(f"\n{'=' * 70}")
    print(f"  TOP SIGNALS (Ranked by ML Probability)")
    print(f"{'=' * 70}")
    
    for i, r in enumerate(results[:15]):
        ml_str = f"{r['ml_probability']:.1%}" if r['ml_probability'] is not None else "N/A"
        print(f"\n  #{i+1} {r['symbol']} — Setup {r['setup']}")
        print(f"     Price: ${r['price']} | RSI: {r['rsi']} | ML Prob: {ml_str}")
        print(f"     Entry: ${r['entry']} | Stop: ${r['stop']} | Target: ${r['target_1']}")
        print(f"     Risk/Share: ${r['risk_per_share']} | Shares (10k): {r['shares_10k']} | R:R: {r['reward_risk']}:1")
        print(f"     6M Return: {r['stock_return_6m']:.1%} (vs SPY {r['spy_return_6m']:.1%})")
        print(f"     {r['setup_reason']}")
    
    print(f"\n{'=' * 70}")
    print(f"  Saved {len(results)} signals to {output_file}")
    print(f"{'=' * 70}")
    
    return results


if __name__ == "__main__":
    main()
