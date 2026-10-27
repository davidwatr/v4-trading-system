#!/usr/bin/env python3
"""
Smart training: 50 symbols, 730 days, every 3 days.
~12,000 samples instead of 234. Should finish in 2-3 min.
"""

import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings, time
warnings.filterwarnings('ignore')

from production_v18 import (
    fetch_data, extract_features,
    calc_atr, triple_barrier_labels
)

MODEL_FILE = Path("ml_model.pkl")

UNIVERSE_50 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO', 'QCOM',
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'AXP', 'V', 'MA', 'PYPL',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO',
    'WMT', 'HD', 'COST', 'NKE', 'MCD', 'SBUX', 'DIS',
    'XOM', 'CVX', 'SLB', 'COP',
    'CAT', 'BA', 'HON', 'UPS', 'GE',
    'LIN', 'APD', 'FCX',
    'NEE', 'DUK', 'SO',
    'PLD', 'AMT', 'CCI',
    'T', 'VZ',
]

SECTOR_ETFS_50 = {s: 'SPY' for s in UNIVERSE_50}  # Use SPY for all (simpler)

def build_dataset():
    all_features = []
    all_labels = []
    feature_names = None
    
    spy_c, spy_v, spy_h, spy_l = fetch_data('SPY', days=730)
    if not spy_c:
        return None, None, None
    
    for symbol in UNIVERSE_50:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=730)
            if not closes or len(closes) < 250:
                continue
            
            sector_closes = spy_c  # Use SPY for all sectors
            
            atr_series = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
            labels = triple_barrier_labels(closes, highs, lows, atr_series)
            
            # Sample every 3 days (good balance of data vs speed)
            for i in range(250, len(closes) - 14, 3):
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
            
            print(f"  {symbol}: OK ({len(all_features)} total samples so far)")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")
            continue
    
    return np.array(all_features), np.array(all_labels), feature_names

def walk_forward_validation(X, y, model, n_splits=5):
    n = len(X)
    fold_size = n // (n_splits + 1)
    scores = []
    
    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        
        if test_end > n or len(np.unique(y[test_start:test_end])) < 2:
            continue
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.array([{c: w for c, w in zip(classes, weights)}[y] for y in y_train])
        
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_train, y_train, sample_weight=sample_weights)
        score = m.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0

def train():
    start = time.time()
    print("Building dataset (50 symbols, 730 days, every 3 days)...")
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
    
    # === Random Forest ===
    print("\n=== Random Forest (Walk-Forward) ===")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=8, min_samples_leaf=4,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_score = walk_forward_validation(X, y, rf)
    print(f"RF Mean Accuracy: {rf_score:.2%}")
    
    # === XGBoost ===
    print("\n=== XGBoost (Walk-Forward) ===")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=7, learning_rate=0.03, random_state=42,
        eval_metric='logloss', reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8
    )
    xgb_score = walk_forward_validation(X, y, xgb)
    print(f"XGB Mean Accuracy: {xgb_score:.2%}")
    
    # === Pick Winner ===
    print("\n" + "=" * 50)
    if rf_score >= xgb_score:
        winner, win_name, win_score = rf, "RandomForest", rf_score
    else:
        winner, win_name, win_score = xgb, "XGBoost", xgb_score
    
    print(f"🏆 WINNER: {win_name} ({win_score:.2%})")
    
    # Final train on full dataset
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([{c: w for c, w in zip(classes, weights)}[y] for y in y_train])
    
    winner.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = winner.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred)
    print(f"Final test accuracy: {final_acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    # Feature importances
    importances = winner.feature_importances_
    total = sum(importances)
    print(f"\nTop 10 Features ({win_name}):")
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fname:25s}: {imp/total:.2%}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': winner,
            'feature_names': feature_names,
            'accuracy': win_score
        }, f)
    
    elapsed = time.time() - start
    print(f"\nSaved to {MODEL_FILE} ({elapsed:.1f}s)")

if __name__ == "__main__":
    train()
