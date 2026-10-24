#!/usr/bin/env python3
"""
ULTIMATE TRAINING PIPELINE
- 80 symbols × 1000 days
- Data caching (fetch once, reuse forever)
- Feature engineering (momentum, volume profile, volatility regime)
- Walk-forward validation
- Ensemble of best models
"""

import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings, time, json, hashlib
warnings.filterwarnings('ignore')

# Add trading dir to path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from production_v18 import (
    fetch_data, extract_features,
    calc_atr, triple_barrier_labels
)

MODEL_FILE = Path("ml_model.pkl")
CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ============================================================
# 80 SYMBOLS
# ============================================================
UNIVERSE_80 = [
    # Tech (20)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO', 'QCOM',
    'IBM', 'SAP', 'SHOP', 'SQ',
    # Finance (16)
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'AXP', 'V', 'MA',
    'PYPL', 'COIN', 'SOFI', 'BRK-B', 'USB', 'PNC',
    # Healthcare (12)
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR',
    'BMY', 'AMGN', 'GILD',
    # Consumer (12)
    'WMT', 'HD', 'COST', 'TGT', 'NKE', 'MCD', 'SBUX', 'TJX', 'LOW',
    'DIS', 'CMCSA', 'PEP',
    # Energy (8)
    'XOM', 'CVX', 'SLB', 'COP', 'EOG', 'VLO', 'MPC', 'PSX',
    # Industrial (8)
    'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'DE',
    # Materials (4)
    'LIN', 'APD', 'SHW', 'FCX',
]

# ============================================================
# DATA CACHING
# ============================================================
def get_cached_data(symbol, days=1000):
    """Fetch data with caching. Returns (closes, volumes, highs, lows)."""
    cache_key = f"{symbol}_{days}"
    cache_file = CACHE_DIR / f"{symbol}_{days}.pkl"
    
    # Check cache (valid for 24 hours)
    if cache_file.exists():
        cache_age = time.time() - cache_file.stat().st_mtime
        if cache_age < 86400:  # 24 hours
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
    
    # Fetch from Yahoo
    data = fetch_data(symbol, days=days)
    
    # Cache it
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except:
        pass
    
    return data

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def engineer_features(closes, volumes, highs, lows, symbol, spy_c, sector_c):
    """Add engineered features on top of base features."""
    base_features = extract_features(closes, volumes, highs, lows, symbol, spy_c, sector_c)
    if base_features is None:
        return None
    
    # Remove data leak
    base_features.pop('tb_label', None)
    
    n = len(closes)
    
    # === Momentum Features ===
    # Rate of change (multiple timeframes)
    if n >= 5:
        base_features['roc_5'] = (closes[-1] - closes[-5]) / closes[-5] if closes[-5] != 0 else 0
    if n >= 10:
        base_features['roc_10'] = (closes[-1] - closes[-10]) / closes[-10] if closes[-10] != 0 else 0
    if n >= 20:
        base_features['roc_20'] = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] != 0 else 0
    
    # Acceleration (change in momentum)
    if n >= 10:
        roc_5_prev = (closes[-6] - closes[-10]) / closes[-10] if closes[-10] != 0 else 0
        base_features['accel_5'] = base_features.get('roc_5', 0) - roc_5_prev
    
    # === Volume Features ===
    if n >= 20:
        avg_vol_20 = np.mean(volumes[-20:])
        base_features['vol_vs_20avg'] = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1
        base_features['vol_trend'] = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if np.mean(volumes[-20:]) > 0 else 1
    
    # === Volatility Regime ===
    if n >= 20:
        returns = np.diff(closes[-20:]) / closes[-21:-1]
        base_features['volatility_20'] = np.std(returns) if len(returns) > 0 else 0
        base_features['vol_regime'] = 1 if base_features['volatility_20'] > 0.02 else 0
    
    # === Price Action ===
    if n >= 10:
        # Distance from 10-day high/low
        high_10 = max(highs[-10:])
        low_10 = min(lows[-10:])
        range_10 = high_10 - low_10
        base_features['dist_from_10high'] = (high_10 - closes[-1]) / range_10 if range_10 > 0 else 0.5
        base_features['dist_from_10low'] = (closes[-1] - low_10) / range_10 if range_10 > 0 else 0.5
    
    # === Trend Strength ===
    if n >= 50:
        ma_10 = np.mean(closes[-10:])
        ma_50 = np.mean(closes[-50:])
        base_features['trend_strength'] = (ma_10 - ma_50) / ma_50 if ma_50 > 0 else 0
    
    # === Mean Reversion ===
    if n >= 20:
        ma_20 = np.mean(closes[-20:])
        base_features['dist_from_ma20'] = (closes[-1] - ma_20) / ma_20 if ma_20 > 0 else 0
    
    return base_features

# ============================================================
# DATASET BUILDER
# ============================================================
def build_dataset():
    all_features = []
    all_labels = []
    feature_names = None
    
    spy_data = get_cached_data('SPY', days=1000)
    if not spy_data[0]:
        return None, None, None
    spy_c, spy_v, spy_h, spy_l = spy_data
    
    for idx, symbol in enumerate(UNIVERSE_80):
        try:
            data = get_cached_data(symbol, days=1000)
            if not data[0] or len(data[0]) < 250:
                continue
            
            closes, volumes, highs, lows = data
            
            atr_series = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
            labels = triple_barrier_labels(closes, highs, lows, atr_series)
            
            # Sample every 2 days for maximum data
            for i in range(250, len(closes) - 14, 2):
                sub_closes = closes[:i+1]
                sub_volumes = volumes[:i+1]
                sub_highs = highs[:i+1]
                sub_lows = lows[:i+1]
                sub_sector = spy_c[:i+1]
                
                features = engineer_features(sub_closes, sub_volumes, sub_highs, sub_lows, symbol, spy_c[:i+1], sub_sector)
                if features is None:
                    continue
                
                if feature_names is None:
                    feature_names = sorted(features.keys())
                
                feature_vec = [features.get(name, 0.0) for name in feature_names]
                label = labels[i] if i < len(labels) else 0
                all_features.append(feature_vec)
                all_labels.append(label)
            
            print(f"  [{idx+1:2d}/80] {symbol}: OK")
        except Exception as e:
            print(f"  [{idx+1:2d}/80] {symbol}: ERROR - {e}")
            continue
    
    return np.array(all_features), np.array(all_labels), feature_names

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================
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
        try:
            m.fit(X_train, y_train, sample_weight=sample_weights)
        except:
            m.fit(X_train, y_train)
        score = m.score(X_test, y_test)
        scores.append(score)
    
    return np.mean(scores) if scores else 0.0

# ============================================================
# MAIN TRAINING
# ============================================================
def train():
    start = time.time()
    print("=" * 60)
    print("ULTIMATE TRAINING PIPELINE")
    print("80 symbols × 1000 days × engineered features")
    print("=" * 60)
    
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
    
    # ============================================================
    # EXPERIMENTS
    # ============================================================
    experiments = [
        # RF (simple → complex)
        ("RF-100-5", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-300-8", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-300-10", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-500-8", RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-500-10", RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),
        
        # XGBoost
        ("XGB-100-5", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')),
        ("XGB-300-5", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("XGB-300-7", XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("XGB-500-5", XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("XGB-500-7", XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        
        # XGBoost with regularization
        ("XGB-reg-1", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')),
        ("XGB-reg-2", XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.02, reg_alpha=0.2, reg_lambda=2.0, subsample=0.7, colsample_bytree=0.7, random_state=42, eval_metric='logloss')),
        
        # Gradient Boosting
        ("GB-100-3", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
        ("GB-300-5", GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42)),
        ("GB-500-5", GradientBoostingClassifier(n_estimators=500, max_depth=5, learning_rate=0.02, random_state=42)),
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
        
        # Walk-forward CV
        cv_score = walk_forward_validation(X, y, model)
        
        elapsed = time.time() - t0
        
        results.append({
            'name': name,
            'accuracy': acc,
            'cv_score': cv_score,
            'time': elapsed,
            'model': model
        })
        print(f"  [{i+1:2d}/{len(experiments)}] {name:15s} | Test: {acc:.2%} | CV: {cv_score:.2%} | {elapsed:.1f}s")
    
    # ============================================================
    # ENSEMBLE
    # ============================================================
    print(f"\nBuilding ensemble...")
    
    # Get top 3 models
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    top3 = results[:3]
    
    # Average probabilities from top 3
    probs = np.zeros((len(y_test), 2))
    for r in top3:
        prob = r['model'].predict_proba(X_test)
        probs += prob
    probs /= 3
    ensemble_pred = np.argmax(probs, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    print(f"  Ensemble of top 3: {ensemble_acc:.2%}")
    
    # Add ensemble to results
    results.append({
        'name': 'Ensemble-Top3',
        'accuracy': ensemble_acc,
        'cv_score': 0,
        'time': 0,
        'model': None
    })
    
    # ============================================================
    # RANKINGS
    # ============================================================
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "=" * 60)
    print("FINAL RANKINGS")
    print("=" * 60)
    print(f"{'Rank':4s} | {'Name':15s} | {'Test':8s} | {'CV':8s} | {'Time':5s}")
    print("-" * 60)
    for i, r in enumerate(results):
        marker = " 🏆" if i == 0 else ""
        print(f"{i+1:4d} | {r['name']:15s} | {r['accuracy']:.2%} | {r['cv_score']:.2%} | {r['time']:.1f}s{marker}")
    
    # ============================================================
    # SAVE WINNER
    # ============================================================
    winner = results[0]
    print(f"\n🏆 WINNER: {winner['name']} ({winner['accuracy']:.2%})")
    
    if winner['model'] is not None:
        importances = winner['model'].feature_importances_
        total = sum(importances)
        print(f"\nTop 10 Features:")
        for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {fname:25s}: {imp/total:.2%}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': winner['model'],
            'feature_names': feature_names,
            'accuracy': winner['accuracy'],
            'top3_models': [r['model'] for r in top3],
            'type': 'single' if winner['model'] is not None else 'ensemble'
        }, f)
    
    elapsed = time.time() - start
    print(f"\nSaved to {MODEL_FILE} ({elapsed:.1f}s)")

if __name__ == "__main__":
    train()
