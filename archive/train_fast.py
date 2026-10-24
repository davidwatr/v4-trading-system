#!/usr/bin/env python3
"""
Fast training on existing 26-symbol dataset.
Uses class balancing + hyperparameter tuning. No new data fetching.
"""

import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

from production_v18 import (
    fetch_data, extract_features, UNIVERSE, SECTOR_ETFS,
    calc_atr, triple_barrier_labels
)

MODEL_FILE = Path("ml_model.pkl")

def build_dataset():
    all_features = []
    all_labels = []
    feature_names = None
    
    spy_c, spy_v, spy_h, spy_l = fetch_data('SPY', days=500)
    
    for symbol in UNIVERSE:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=500)
            if not closes or len(closes) < 200:
                continue
            
            sector_etf = SECTOR_ETFS.get(symbol, 'SPY')
            sector_closes, _, _, _ = fetch_data(sector_etf, days=500)
            if not sector_closes:
                sector_closes = spy_c
            
            atr_series = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
            labels = triple_barrier_labels(closes, highs, lows, atr_series)
            
            for i in range(200, len(closes) - 14, 7):
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
        except:
            continue
    
    return np.array(all_features), np.array(all_labels), feature_names

def train():
    print("Building dataset...")
    X, y, feature_names = build_dataset()
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    mask = y >= 0
    X = X[mask]
    y = y[mask]
    
    print(f"After filtering: {X.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Split FIRST
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"Test class dist: {np.bincount(y_test.astype(int))}")
    
    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([{c: w for c, w in zip(classes, weights)}[y] for y in y_train])
    
    # === Random Forest ===
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=8, min_samples_leaf=4,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"RF Accuracy: {rf_acc:.2%}")
    print(classification_report(y_test, rf_pred, target_names=['Down', 'Up']))
    
    # === XGBoost ===
    print("=== XGBoost ===")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.03, random_state=42,
        eval_metric='logloss', reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    xgb_pred = xgb.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"XGB Accuracy: {xgb_acc:.2%}")
    print(classification_report(y_test, xgb_pred, target_names=['Down', 'Up']))
    
    # === Winner ===
    print("\n" + "=" * 50)
    if rf_acc >= xgb_acc:
        winner, win_name, win_score = rf, "RandomForest", rf_acc
    else:
        winner, win_name, win_score = xgb, "XGBoost", xgb_acc
    
    print(f"🏆 WINNER: {win_name} ({win_score:.2%})")
    
    # Feature importances
    importances = winner.feature_importances_
    total = sum(importances)
    print(f"\nTop 10 Features:")
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fname:25s}: {imp/total:.2%}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': winner,
            'feature_names': feature_names,
            'accuracy': win_score
        }, f)
    
    print(f"\nSaved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
