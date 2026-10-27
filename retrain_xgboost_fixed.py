#!/usr/bin/env python3
"""
Retrain XGBoost — fix data leakage (tb_label removed from features).
"""
import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from production_v18 import (
    fetch_data, extract_features, UNIVERSE, SECTOR_ETFS,
    calc_atr, triple_barrier_labels
)

MODEL_FILE = Path("ml_model.pkl")

def build_dataset():
    all_features = []
    all_labels = []
    feature_names = None
    
    for symbol in UNIVERSE:
        print(f"  Processing {symbol}...")
        closes, volumes, highs, lows = fetch_data(symbol, days=500)
        if not closes or len(closes) < 200:
            continue
        
        sector_etf = SECTOR_ETFS.get(symbol, "SPY")
        sector_closes, _, _, _ = fetch_data(sector_etf, days=500)
        if not sector_closes:
            sector_closes = closes
        
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
            
            # FIX: Exclude tb_label from features
            if feature_names is None:
                feature_names = sorted([k for k in features.keys() 
                                       if isinstance(features[k], (int, float)) 
                                       and k != 'tb_label'])  # Remove data leak
            
            feature_vec = [features.get(name, 0.0) for name in feature_names]
            
            label = labels[i] if i < len(labels) else 0
            all_features.append(feature_vec)
            all_labels.append(label)
    
    return np.array(all_features), np.array(all_labels), feature_names

def train():
    print("Building dataset (no leakage)...")
    X, y, feature_names = build_dataset()
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features ({len(feature_names)}): {feature_names}")
    
    mask = y >= 0
    X = X[mask]
    y = y[mask]
    
    print(f"After filtering: {X.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    importances = model.feature_importances_
    total = sum(importances)
    print("Top 10 Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name:25s}: {imp/total:.2%}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy
        }, f)
    
    print(f"\nSaved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
