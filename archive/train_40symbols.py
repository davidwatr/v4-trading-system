#!/usr/bin/env python3
"""
40 symbols × 500 days training run.
Adds 14 more symbols to the existing 26.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trading'))

import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings, time
warnings.filterwarnings('ignore')

from production_v18 import (
    fetch_data, extract_features,
    calc_atr, triple_barrier_labels
)

MODEL_FILE = Path("ml_model.pkl")

# 40 symbols: original 26 + 14 more
UNIVERSE_40 = [
    # Original 26
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO', 'QCOM',
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'AXP', 'V', 'MA', 'PYPL',
    # 14 more
    'JNJ', 'PFE', 'UNH', 'ABBV', 'LLY',
    'WMT', 'HD', 'COST', 'NKE', 'MCD',
    'XOM', 'CVX', 'SLB', 'COP',
]

SECTOR_ETFS_40 = {s: 'SPY' for s in UNIVERSE_40}

def build_dataset():
    all_features = []
    all_labels = []
    feature_names = None
    
    spy_c, spy_v, spy_h, spy_l = fetch_data('SPY', days=500)
    if not spy_c:
        return None, None, None
    
    for symbol in UNIVERSE_40:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=500)
            if not closes or len(closes) < 200:
                continue
            
            sector_closes = spy_c
            
            atr_series = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
            labels = triple_barrier_labels(closes, highs, lows, atr_series)
            
            # Sample every 3 days
            for i in range(200, len(closes) - 14, 3):
                sub_closes = closes[:i+1]
                sub_volumes = volumes[:i+1]
                sub_highs = highs[:i+1]
                sub_lows = lows[:i+1]
                sub_sector = sector_closes[:i+1]
                
                features = extract_features(sub_closes, sub_volumes, sub_highs, sub_lows, symbol, sub_sector, sub_sector)
                if features is None:
                    continue
                
                if feature_names is None:
                    feature_names = sorted([k for k in features.keys() 
                                           if isinstance(features[k], (int, float)) 
                                           and k != 'tb_label'])
                
                feature_vec = [features.get(name, 0.0) for name in feature_names]
                label = labels[i] if i < len(labels) else 0
                all_features.append(feature_vec)
                all_labels.append(label)
            
            print(f"  {symbol}: OK")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")
            continue
    
    return np.array(all_features), np.array(all_labels), feature_names

def train():
    start = time.time()
    print("Building dataset (40 symbols × 500 days)...")
    result = build_dataset()
    if result[0] is None:
        return
    X, y, feature_names = result
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    mask = y >= 0
    X = X[mask]
    y = y[mask]
    
    print(f"After filtering: {X.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    
    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([{c: w for c, w in zip(classes, weights)}[y] for y in y_train])
    
    # === Experiments ===
    experiments = [
        # RF
        ("RF-100-5", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-300-10", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-500-8", RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-500-10", RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),
        
        # XGB
        ("XGB-100-7", XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.05, random_state=42, eval_metric='logloss')),
        ("XGB-300-5", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("XGB-300-7", XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("XGB-500-5", XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        
        # GB
        ("GB-100-3", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
        ("GB-300-5", GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42)),
    ]
    
    print(f"\nRunning {len(experiments)} experiments...")
    print("=" * 60)
    
    results = []
    for i, (name, model) in enumerate(experiments):
        t0 = time.time()
        try:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        except:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        elapsed = time.time() - t0
        
        results.append({
            'name': name,
            'accuracy': acc,
            'time': elapsed,
            'model': model
        })
        print(f"  [{i+1:2d}/{len(experiments)}] {name:15s} | Test: {acc:.2%} | {elapsed:.1f}s")
    
    # Rank
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "=" * 60)
    print("RANKINGS")
    print("=" * 60)
    for i, r in enumerate(results):
        marker = " 🏆" if i == 0 else ""
        print(f"{i+1:2d}. {r['name']:15s} | {r['accuracy']:.2%} | {r['time']:.1f}s{marker}")
    
    # Save winner
    winner = results[0]
    print(f"\n🏆 WINNER: {winner['name']} ({winner['accuracy']:.2%})")
    
    importances = winner['model'].feature_importances_
    total = sum(importances)
    print(f"\nTop 10 Features:")
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fname:25s}: {imp/total:.2%}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': winner['model'],
            'feature_names': feature_names,
            'accuracy': winner['accuracy']
        }, f)
    
    elapsed = time.time() - start
    print(f"\nSaved to {MODEL_FILE} ({elapsed:.1f}s)")

if __name__ == "__main__":
    train()
