#!/usr/bin/env python3
"""
Retrain XGBoost with expanded features (24+ instead of 5).
Fixes the feature-starved model issue.
"""
import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from production_v18 import (
    fetch_data, extract_features, UNIVERSE, SECTOR_ETFS,
    calc_atr, triple_barrier_labels
)

MODEL_FILE = Path("ml_model.pkl")

def build_dataset():
    """Build training dataset with all features."""
    all_features = []
    all_labels = []
    
    for symbol in UNIVERSE:
        print(f"Processing {symbol}...")
        closes, volumes, highs, lows = fetch_data(symbol, days=730)
        if not closes or len(closes) < 200:
            continue
        
        sector_etf = SECTOR_ETFS.get(symbol, "SPY")
        sector_closes, _, _, _ = fetch_data(sector_etf, days=730)
        if not sector_closes:
            sector_closes = closes
        
        # Get triple barrier labels
        atr_series = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
        labels = triple_barrier_labels(closes, highs, lows, atr_series)
        
        # Extract features for each day
        for i in range(200, len(closes) - 14):
            sub_closes = closes[:i+1]
            sub_volumes = volumes[:i+1]
            sub_highs = highs[:i+1]
            sub_lows = lows[:i+1]
            sub_sector = sector_closes[:i+1]
            
            features = extract_features(sub_closes, sub_volumes, sub_highs, sub_lows, symbol, sub_sector, sub_sector)
            if features is None:
                continue
            
            # Only take the features that are numeric
            feature_vec = []
            feature_names = []
            for k, v in sorted(features.items()):
                if isinstance(v, (int, float)) and v is not None:
                    feature_vec.append(v)
                    feature_names.append(k)
            
            all_features.append(feature_vec)
            all_labels.append(labels[i] if i < len(labels) else 0)
    
    return np.array(all_features), np.array(all_labels), feature_names

def train():
    """Retrain XGBoost with expanded features."""
    print("Building dataset...")
    X, y, feature_names = build_dataset()
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Filter to only label 0 and 1 (drop -1 for simplicity)
    mask = y >= 0
    X = X[mask]
    y = y[mask]
    
    print(f"After filtering: {X.shape[0]} samples")
    print(f"Features: {feature_names}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric='logloss'
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.2%}")
    
    # Feature importances
    importances = model.feature_importances_
    total = sum(importances)
    print("\nFeature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:25s}: {imp/total:.2%}")
    
    # Save
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy
        }, f)
    
    print(f"\nSaved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
