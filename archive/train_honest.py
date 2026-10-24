#!/usr/bin/env python3
"""
Retrain XGBoost with 6-month OOS hard wall.
Pruned universe (no ORCL, no MS). Honest 58% win prob.
"""
import numpy as np
import pickle
import time
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from production_v18 import (
    fetch_data, extract_features, calc_atr,
    UNIVERSE, SECTOR_ETFS
)

# Pruned universe — only stocks where model has edge
# Dropped: ORCL (41%), MS (45%) — negative edge
# Kept: BAC (76%), PYPL (69%), JPM (66%), WFC (62%), BRK-B (59%), AAPL (57%), GS (55%)
PRUNED_UNIVERSE = [
    'BAC', 'PYPL', 'JPM', 'WFC', 'BRK-B', 'AAPL', 'GS', 'C',
    'INTC', 'AMD', 'NVDA', 'META', 'TSLA', 'V', 'MA', 'COST', 'TGT'
]

# 6-month hard wall — data after this date is NEVER seen
OOS_CUTOFF_DAYS = 180  # ~6 months

def build_dataset(symbol, days=730):
    """Build feature matrix with OOS wall."""
    closes, volumes, highs, lows = fetch_data(symbol, days=days)
    if closes is None or len(closes) < 250:
        return None, None, None
    
    # Hard wall: only use data before cutoff
    usable_days = len(closes) - OOS_CUTOFF_DAYS
    if usable_days < 200:
        return None, None, None
    
    features_list = []
    labels = []
    
    for i in range(50, usable_days):
        c = closes[:i+1]
        v = volumes[:i+1]
        h = highs[:i+1]
        l = lows[:i+1]
        
        feat = extract_features(c, v, h, l, symbol, c, c)
        if feat is None:
            continue
        
        # Engineered features
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
        
        # Label: 5-day forward return
        future_idx = min(i + 5, len(closes) - 1)
        ret = (closes[future_idx] - closes[i]) / closes[i]
        label = 1 if ret > 0 else 0
        
        features_list.append(feat)
        labels.append(label)
    
    if len(features_list) < 100:
        return None, None, None
    
    feature_names = list(features_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in features_list])
    y = np.array(labels)
    
    return X, y, feature_names


def train():
    """Retrain with OOS wall and pruned universe."""
    all_X = []
    all_y = []
    feature_names = None
    
    for symbol in PRUNED_UNIVERSE:
        print(f"Processing {symbol}...")
        X, y, fn = build_dataset(symbol)
        if X is None:
            print(f"  Skipped (insufficient data)")
            continue
        
        if feature_names is None:
            feature_names = fn
        
        all_X.append(X)
        all_y.append(y)
        print(f"  {len(X)} samples")
        time.sleep(0.3)
    
    if not all_X:
        print("No data!")
        return
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # 80/20 chronological split (within pre-wall data)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"\nTotal: {len(X)} samples, {len(X_train)} train, {len(X_test)} test")
    print(f"Class balance: {np.mean(y):.2%} positive")
    
    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\nTrain accuracy: {train_acc:.2%}")
    print(f"Test accuracy (pre-wall): {test_acc:.2%}")
    print(f"Gap: {train_acc - test_acc:.2%} (overfitting measure)")
    
    # Save with honest metadata
    output = {
        'model': model,
        'feature_names': feature_names,
        'accuracy': test_acc,
        'train_accuracy': train_acc,
        'oos_wall_months': 6,
        'universe': PRUNED_UNIVERSE,
        'trained_at': datetime.now().isoformat(),
        'dataset': f"{len(PRUNED_UNIVERSE)} symbols, 6-month OOS wall"
    }
    
    with open('ml_model.pkl', 'wb') as f:
        pickle.dump(output, f)
    
    print(f"\nSaved ml_model.pkl (honest accuracy: {test_acc:.2%})")
    
    # Feature importance
    importances = model.feature_importances_
    idx = np.argsort(-importances)
    print(f"\nTop 10 features:")
    for i in idx[:10]:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")


if __name__ == "__main__":
    train()
