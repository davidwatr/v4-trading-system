#!/usr/bin/env python3
"""
Hyperparameter search: multiple runs, different configs, find the best.
Starts simple, increases complexity.
"""

import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score
import warnings, time
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

def run_experiment(name, model, X_train, y_train, X_test, y_test, sample_weights):
    """Run a single experiment and return results."""
    start = time.time()
    
    # Handle models that don't support sample_weight
    try:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    except TypeError:
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # Cross-validation on training set
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
    except:
        cv_mean = 0.0
    
    elapsed = time.time() - start
    
    return {
        'name': name,
        'accuracy': acc,
        'cv_mean': cv_mean,
        'time': elapsed,
        'model': model
    }

def train():
    print("Building dataset...")
    X, y, feature_names = build_dataset()
    
    mask = y >= 0
    X = X[mask]
    y = y[mask]
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
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
    # EXPERIMENT GRID
    # ============================================================
    experiments = [
        # === Random Forest (simple → complex) ===
        ("RF-100-5", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-100-8", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-100-10", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-300-5", RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-300-8", RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-300-10", RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-500-8", RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-500-10", RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)),
        ("RF-500-12", RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42, class_weight='balanced', n_jobs=-1)),
        
        # === XGBoost (simple → complex) ===
        ("XGB-100-3", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric='logloss')),
        ("XGB-100-5", XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')),
        ("XGB-100-7", XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.05, random_state=42, eval_metric='logloss')),
        ("XGB-300-5", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("XGB-300-7", XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("XGB-500-5", XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        ("XGB-500-7", XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.03, random_state=42, eval_metric='logloss')),
        
        # === XGBoost with regularization ===
        ("XGB-reg-1", XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')),
        ("XGB-reg-2", XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.03, reg_alpha=0.1, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')),
        ("XGB-reg-3", XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.02, reg_alpha=0.2, reg_lambda=2.0, subsample=0.7, colsample_bytree=0.7, random_state=42, eval_metric='logloss')),
        
        # === Gradient Boosting ===
        ("GB-100-3", GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
        ("GB-300-5", GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42)),
        ("GB-500-5", GradientBoostingClassifier(n_estimators=500, max_depth=5, learning_rate=0.02, random_state=42)),
    ]
    
    # ============================================================
    # RUN ALL EXPERIMENTS
    # ============================================================
    print(f"\nRunning {len(experiments)} experiments...")
    print("=" * 70)
    
    results = []
    for i, (name, model) in enumerate(experiments):
        result = run_experiment(name, model, X_train, y_train, X_test, y_test, sample_weights)
        results.append(result)
        print(f"  [{i+1:2d}/{len(experiments)}] {name:15s} | Test: {result['accuracy']:.2%} | CV: {result['cv_mean']:.2%} | {result['time']:.1f}s")
    
    # ============================================================
    # RANK RESULTS
    # ============================================================
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n" + "=" * 70)
    print("FINAL RANKINGS (by test accuracy)")
    print("=" * 70)
    print(f"{'Rank':4s} | {'Name':15s} | {'Test Acc':8s} | {'CV Mean':8s} | {'Time':5s}")
    print("-" * 70)
    for i, r in enumerate(results):
        marker = " 🏆" if i == 0 else ""
        print(f"{i+1:4d} | {r['name']:15s} | {r['accuracy']:.2%} | {r['cv_mean']:.2%} | {r['time']:.1f}s{marker}")
    
    # ============================================================
    # SAVE WINNER
    # ============================================================
    winner = results[0]
    print(f"\n🏆 WINNER: {winner['name']} ({winner['accuracy']:.2%})")
    
    # Feature importances
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
    
    print(f"\nSaved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
