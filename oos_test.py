#!/usr/bin/env python3
"""Quick OOS test: does the 78% model still work on recent unseen data?"""
import pickle
import numpy as np
import time
from production_v18 import fetch_data, extract_features, dynamic_triple_barriers, calc_atr

# Load model
with open('ml_model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
features = data['feature_names']

print(f"Model: {type(model).__name__}, Acc: {data.get('accuracy', 0):.2%}")
print(f"Features: {len(features)}")

# Test on recent 3 months of unseen data per symbol
# We'll use forward returns to check directional accuracy
from production_v18 import UNIVERSE

correct = 0
total = 0
results = []

for symbol in UNIVERSE[:10]:
    closes, volumes, highs, lows = fetch_data(symbol, days=730)
    if not closes or len(closes) < 250:
        continue
    
    # Use only the LAST 63 days (~3 months) as OOS test
    oos_start = len(closes) - 63
    
    oos_correct = 0
    oos_total = 0
    
    for i in range(oos_start, len(closes) - 5):
        # Features at day i
        c = closes[:i+1]
        v = volumes[:i+1]
        h = highs[:i+1]
        l = lows[:i+1]
        
        feat = extract_features(c, v, h, l, symbol, c, c)
        if feat is None:
            continue
        
        # Add engineered features
        n = len(c)
        if n >= 5:
            feat['roc_5'] = (c[-1] - c[-5]) / c[-5] if c[-5] != 0 else 0
        if n >= 10:
            feat['roc_10'] = (c[-1] - c[-10]) / c[-10] if c[-10] != 0 else 0
        if n >= 20:
            feat['roc_20'] = (c[-1] - c[-20]) / c[-20] if c[-20] != 0 else 0
        if n >= 20:
            avg_vol = np.mean(v[-20:])
            feat['vol_vs_avg'] = v[-1] / avg_vol if avg_vol > 0 else 1
        if n >= 50:
            ma_10 = np.mean(c[-10:])
            ma_50 = np.mean(c[-50:])
            feat['trend'] = (ma_10 - ma_50) / ma_50 if ma_50 > 0 else 0
        
        X = [[feat[name] for name in features]]
        
        # Predict
        pred = model.predict(X)[0]
        
        # Actual: 5-day forward return
        future_idx = min(i + 5, len(closes) - 1)
        actual_return = (closes[future_idx] - closes[i]) / closes[i]
        actual = 1 if actual_return > 0 else 0
        
        if pred == actual:
            oos_correct += 1
        oos_total += 1
    
    if oos_total > 0:
        acc = oos_correct / oos_total
        results.append((symbol, acc, oos_total))
        correct += oos_correct
        total += oos_total
    
    time.sleep(0.3)

print(f"\n=== OOS TEST (Last 3 Months) ===")
for sym, acc, n in results:
    status = "✓" if acc > 0.5 else "✗"
    print(f"  {status} {sym}: {acc:.2%} ({n} samples)")

if total > 0:
    print(f"\nOverall OOS accuracy: {correct/total:.2%}")
    if correct/total > 0.52:
        print("Model has real (small) edge")
    elif correct/total > 0.48:
        print("Model is coin-flip random")
    else:
        print("Model is actively bad — trades backwards")
