#!/usr/bin/env python3
"""
Train 40 symbols with feature engineering.
Save the best single model (not ensemble) for the executor.
"""

import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings, time
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from production_v18 import (
    fetch_data, extract_features,
    calc_atr, triple_barrier_labels
)

MODEL_FILE = Path("ml_model.pkl")

UNIVERSE_40 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO', 'QCOM',
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'AXP', 'V', 'MA', 'PYPL',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'LLY',
    'WMT', 'HD', 'COST', 'NKE', 'MCD',
    'XOM', 'CVX', 'SLB', 'COP',
]

def engineer_features(closes, volumes, highs, lows, symbol, spy_c, sector_c):
    base = extract_features(closes, volumes, highs, lows, symbol, spy_c, sector_c)
    if base is None:
        return None
    base.pop('tb_label', None)
    
    n = len(closes)
    if n >= 5:
        base['roc_5'] = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] != 0 else 0
    if n >= 10:
        base['roc_10'] = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] != 0 else 0
    if n >= 20:
        base['roc_20'] = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] != 0 else 0
    if n >= 20:
        avg_vol = np.mean(volumes[-20:])
        base['vol_vs_avg'] = volumes[-1] / avg_vol if avg_vol > 0 else 1
    if n >= 50:
        ma_10 = np.mean(closes[-10:])
        ma_50 = np.mean(closes[-50:])
        base['trend'] = (ma_10 - ma_50) / ma_50 if ma_50 > 0 else 0
    
    return base

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
            
            atr_series = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
            labels = triple_barrier_labels(closes, highs, lows, atr_series)
            
            for i in range(200, len(closes) - 14, 3):
                features = engineer_features(
                    closes[:i+1], volumes[:i+1], highs[:i+1], lows[:i+1],
                    symbol, spy_c[:i+1], spy_c[:i+1]
                )
                if features is None:
                    continue
                
                if feature_names is None:
                    feature_names = sorted(features.keys())
                
                feature_vec = [features.get(name, 0.0) for name in feature_names]
                label = labels[i] if i < len(labels) else 0
                all_features.append(feature_vec)
                all_labels.append(label)
        except:
            continue
    
    return np.array(all_features), np.array(all_labels), feature_names

def train():
    start = time.time()
    print("Building dataset (40 symbols × 500 days + features)...")
    X, y, feature_names = build_dataset()
    if X is None:
        return
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    mask = y >= 0
    X = X[mask]
    y = y[mask]
    print(f"Filtered: {X.shape[0]} samples")
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([{c: w for c, w in zip(classes, weights)}[yy] for yy in y_train])
    
    models = [
        ("RF-100-5", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-300-8", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-300-10", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-500-8", RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("XGB-100-5", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')),
        ("XGB-300-5", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("XGB-500-5", XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("GB-100-3", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
        ("GB-300-5", GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42)),
    ]
    
    results = []
    for name, model in models:
        t0 = time.time()
        try:
            model.fit(X_train, y_train, sample_weight=sample_weights)
        except:
            model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        elapsed = time.time() - t0
        results.append({'name': name, 'accuracy': acc, 'time': elapsed, 'model': model})
        print(f"  {name:15s} | {acc:.2%} | {elapsed:.1f}s")
    
    # Pick best single model (not ensemble)
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    winner = results[0]
    
    print("\n" + "=" * 60)
    print("RANKINGS")
    print("=" * 60)
    for i, r in enumerate(results):
        marker = " 🏆" if i == 0 else ""
        print(f"{i+1:2d}. {r['name']:15s} | {r['accuracy']:.2%}{marker}")
    
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
