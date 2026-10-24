#!/usr/bin/env python3
"""
Daily Trade Screener — reads market data, generates signals,
and outputs a JSON file of buy/sell recommendations.
"""

import json
import sys
import pickle
import time
from pathlib import Path
from datetime import datetime

# Add trading dir to path
sys.path.insert(0, str(Path(__file__).parent))

from production_v18 import (
    fetch_data, extract_features, get_market_regime,
    dynamic_triple_barriers, calc_atr,
    UNIVERSE, MODEL_FILE
)

def main():
    print(f"=== Daily Trade Screener {datetime.now().strftime('%Y-%m-%d %H:%M')} ===")

    if not MODEL_FILE.exists():
        print("ERROR: Model not found")
        sys.exit(1)

    with open(MODEL_FILE, 'rb') as f:
        saved = pickle.load(f)
    model = saved['model']
    features = saved['feature_names']

    # Fetch SPY
    spy_closes, _, spy_highs, spy_lows = fetch_data("SPY", days=730)
    if not spy_closes:
        print("ERROR: Can't fetch SPY")
        sys.exit(1)

    regime, regime_penalty, big_cap_max, small_cap_max, kelly_mult = get_market_regime(spy_closes, spy_highs, spy_lows)
    print(f"Regime: {regime}")

    results = []
    for symbol in UNIVERSE:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=730)
            if not closes or len(closes) < 200:
                continue

            sector_etf = {"AAPL":"XLK","MSFT":"XLK","GOOGL":"XLK","NVDA":"XLK","META":"XLK",
                          "INTC":"XLK","AMD":"XLK","CSCO":"XLK","ORCL":"XLK","CRM":"XLK","ADBE":"XLK",
                          "JPM":"XLF","BAC":"XLF","GS":"XLF","MS":"XLF","WFC":"XLF","C":"XLF",
                          "BRK-B":"XLF","V":"XLF","MA":"XLF","PYPL":"XLF",
                          "AMZN":"XLY","TSLA":"XLY","DIS":"XLY","NFLX":"XLY"}.get(symbol, "SPY")
            sector_closes, _, _, _ = fetch_data(sector_etf, days=730)
            if not sector_closes:
                sector_closes = spy_closes

            feat = extract_features(closes, volumes, highs, lows, symbol, spy_closes, sector_closes)
            if feat is None:
                continue

            X = [[feat[name] for name in features]]
            prob = float(model.predict_proba(X)[0][1])

            # Score: positive = buy, negative = sell
            score = prob - 0.5

            # Classify
            if prob > 0.6:
                signal = "BUY"
            elif prob < 0.4:
                signal = "SELL"
            else:
                signal = "HOLD"

            results.append({
                "symbol": symbol,
                "price": round(float(feat['current_price']), 2),
                "probability": round(float(prob), 4),
                "signal": signal,
                "score": round(float(score), 4),
                "rsi": round(float(feat.get('rsi', 0)), 2),
                "regime": regime,
                "timestamp": datetime.now().isoformat()
            })

            time.sleep(0.25)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue

    # Sort by score descending (best buys first)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Save results
    output_file = Path('screener_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} signals to {output_file}")
    print(f"\nTop 5 Buy Signals:")
    for r in results[:5]:
        if r['signal'] == 'BUY':
            print(f"  {r['symbol']:6s} ${r['price']:>8.2f} | P={r['probability']:.2%} | RSI={r['rsi']:.1f}")

    print(f"\nTop 5 Sell Signals:")
    for r in results[-5:]:
        if r['signal'] == 'SELL':
            print(f"  {r['symbol']:6s} ${r['price']:>8.2f} | P={r['probability']:.2%} | RSI={r['rsi']:.1f}")

    return results

if __name__ == "__main__":
    main()
