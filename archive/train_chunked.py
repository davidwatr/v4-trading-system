#!/usr/bin/env python3
"""
Chunked training — two-pass approach:
Pass 1: Collect unified feature names across all symbols
Pass 2: Build feature vectors using unified names
"""

import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
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
CACHE_DIR = Path("data_cache")
CHUNK_DIR = Path("training_chunks")
CHUNK_DIR.mkdir(exist_ok=True)

UNIVERSE_80 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO', 'QCOM',
    'IBM', 'SAP', 'SHOP', 'SQ',
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'AXP', 'V', 'MA',
    'PYPL', 'COIN', 'SOFI', 'BRK-B', 'USB', 'PNC',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR',
    'BMY', 'AMGN', 'GILD',
    'WMT', 'HD', 'COST', 'TGT', 'NKE', 'MCD', 'SBUX', 'TJX', 'LOW',
    'DIS', 'CMCSA', 'PEP',
    'XOM', 'CVX', 'SLB', 'COP', 'EOG', 'VLO', 'MPC', 'PSX',
    'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'DE',
    'LIN', 'APD', 'SHW', 'FCX',
]

def get_cached_data(symbol, days=1000):
    cache_file = CACHE_DIR / f"{symbol}_{days}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    data = fetch_data(symbol, days=days)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except:
        pass
    return data

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

def get_unified_feature_names():
    """Pass 1: Collect all possible feature names across all symbols."""
    all_feature_names = set()
    
    spy_data = get_cached_data('SPY', days=1000)
    if not spy_data[0]:
        return None
    spy_c = spy_data[0]
    
    for symbol in UNIVERSE_80:
        data = get_cached_data(symbol, days=1000)
        if not data[0] or len(data[0]) < 250:
            continue
        
        closes, volumes, highs, lows = data
        # Sample a few time points to collect feature names
        for i in [300, 400, 500, 600, 700]:
            if i >= len(closes) - 14:
                continue
            features = engineer_features(
                closes[:i+1], volumes[:i+1], highs[:i+1], lows[:i+1],
                symbol, spy_c[:i+1], spy_c[:i+1]
            )
            if features:
                all_feature_names.update(features.keys())
    
    return sorted(all_feature_names)

def build_dataset(unified_names):
    """Pass 2: Build dataset using unified feature names."""
    all_features = []
    all_labels = []
    
    spy_data = get_cached_data('SPY', days=1000)
    if not spy_data[0]:
        return None, None
    spy_c = spy_data[0]
    
    for idx, symbol in enumerate(UNIVERSE_80):
        try:
            data = get_cached_data(symbol, days=1000)
            if not data[0] or len(data[0]) < 250:
                continue
            
            closes, volumes, highs, lows = data
            atr_series = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
            labels = triple_barrier_labels(closes, highs, lows, atr_series)
            
            for i in range(250, len(closes) - 14, 2):
                features = engineer_features(
                    closes[:i+1], volumes[:i+1], highs[:i+1], lows[:i+1],
                    symbol, spy_c[:i+1], spy_c[:i+1]
                )
                if features is None:
                    continue
                
                feature_vec = [features.get(name, 0.0) for name in unified_names]
                label = labels[i] if i < len(labels) else 0
                all_features.append(feature_vec)
                all_labels.append(label)
            
            print(f"  [{idx+1:2d}/80] {symbol}: OK")
        except Exception as e:
            print(f"  [{idx+1:2d}/80] {symbol}: {e}")
            continue
    
    if all_features:
        return np.array(all_features), np.array(all_labels)
    return None, None

def train():
    print("=" * 60)
    print("TWO-PASS CHUNKED TRAINING")
    print("=" * 60)
    
    # Pass 1: Get unified feature names
    print("\nPass 1: Collecting unified feature names...")
    unified_names = get_unified_feature_names()
    if unified_names is None:
        print("ERROR: Could not collect feature names")
        return
    print(f"  {len(unified_names)} features collected")
    
    # Pass 2: Build dataset
    print("\nPass 2: Building dataset...")
    X, y = build_dataset(unified_names)
    if X is None:
        print("ERROR: Could not build dataset")
        return
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    mask = y >= 0
    X = X[mask]
    y = y[mask]
    print(f"Filtered: {X.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
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
    
    # Ensemble
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    top3 = results[:3]
    
    probs = np.zeros((len(y_test), 2))
    for r in top3:
        probs += r['model'].predict_proba(X_test)
    probs /= 3
    ensemble_acc = accuracy_score(y_test, np.argmax(probs, axis=1))
    print(f"\n  Ensemble-Top3  | {ensemble_acc:.2%}")
    
    results.append({'name': 'Ensemble', 'accuracy': ensemble_acc, 'time': 0, 'model': None})
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "=" * 60)
    print("RANKINGS")
    print("=" * 60)
    for i, r in enumerate(results):
        marker = " 🏆" if i == 0 else ""
        print(f"{i+1:2d}. {r['name']:15s} | {r['accuracy']:.2%}{marker}")
    
    winner = results[0]
    print(f"\n🏆 WINNER: {winner['name']} ({winner['accuracy']:.2%})")
    
    if winner['model'] is not None:
        importances = winner['model'].feature_importances_
        total = sum(importances)
        print(f"\nTop 10 Features:")
        for fname, imp in sorted(zip(unified_names, importances), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {fname:25s}: {imp/total:.2%}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': winner['model'],
            'feature_names': unified_names,
            'accuracy': winner['accuracy']
        }, f)
    
    print(f"\nSaved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
