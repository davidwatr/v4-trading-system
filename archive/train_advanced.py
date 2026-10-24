#!/usr/bin/env python3
"""
Advanced model training pipeline:
- More symbols (80+ instead of 26)
- Longer history (1000 days instead of 500)
- Daily sampling (instead of every 7 days)
- Walk-forward validation (rolling window)
- Hyperparameter tuning
- Ensemble of best models
"""

import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

from production_v18 import (
    fetch_data, extract_features,
    calc_atr, triple_barrier_labels
)

MODEL_FILE = Path("ml_model.pkl")

# Expanded universe - stocks with good liquidity
UNIVERSE_EXTENDED = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO', 'QCOM',
    'IBM', 'SAP', 'SHOP', 'SQ',
    # Finance
    'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'AXP', 'V', 'MA',
    'PYPL', 'COIN', 'SOFI',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR',
    'BMY', 'AMGN', 'GILD', 'REGN', 'VRTX',
    # Consumer
    'WMT', 'HD', 'COST', 'TGT', 'NKE', 'MCD', 'SBUX', 'TJX', 'LOW',
    'DIS', 'CMCSA', 'PEP', 'KO', 'MO', 'PM',
    # Energy
    'XOM', 'CVX', 'SLB', 'COP', 'EOG', 'PXD', 'VLO', 'MPC',
    # Industrial
    'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM', 'DE',
    # Materials
    'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'ECL',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE',
    # Real Estate
    'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG',
    # Telecom
    'T', 'VZ', 'TMUS',
]

SECTOR_ETFS_EXTENDED = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLK', 'AMZN': 'XLC', 'META': 'XLC',
    'NVDA': 'XLK', 'TSLA': 'XLY', 'NFLQ': 'XLC', 'ADBE': 'XLK', 'CRM': 'XLK',
    'ORCL': 'XLK', 'INTC': 'XLK', 'AMD': 'XLK', 'CSCO': 'XLK', 'AVGO': 'XLK',
    'QCOM': 'XLK', 'IBM': 'XLK', 'SAP': 'XLK', 'SHOP': 'XLK', 'SQ': 'XLF',
    'JPM': 'XLF', 'BAC': 'XLF', 'GS': 'XLF', 'MS': 'XLF', 'WFC': 'XLF',
    'C': 'XLF', 'BLK': 'XLF', 'AXP': 'XLF', 'V': 'XLF', 'MA': 'XLF',
    'PYPL': 'XLF', 'COIN': 'XLF', 'SOFI': 'XLF',
    'JNJ': 'XLV', 'PFE': 'XLV', 'UNH': 'XLV', 'ABBV': 'XLV', 'MRK': 'XLV',
    'LLY': 'XLV', 'TMO': 'XLV', 'ABT': 'XLV', 'DHR': 'XLV', 'BMY': 'XLV',
    'AMGN': 'XLV', 'GILD': 'XLV', 'REGN': 'XLV', 'VRTX': 'XLV',
    'WMT': 'XLP', 'HD': 'XLY', 'COST': 'XLP', 'TGT': 'XLY', 'NKE': 'XLY',
    'MCD': 'XLY', 'SBUX': 'XLY', 'TJX': 'XLY', 'LOW': 'XLY',
    'DIS': 'XLC', 'CMCSA': 'XLC', 'PEP': 'XLP', 'KO': 'XLP', 'MO': 'XLP', 'PM': 'XLP',
    'XOM': 'XLE', 'CVX': 'XLE', 'SLB': 'XLE', 'COP': 'XLE', 'EOG': 'XLE',
    'PXD': 'XLE', 'VLO': 'XLE', 'MPC': 'XLE',
    'CAT': 'XLI', 'BA': 'XLI', 'HON': 'XLI', 'UPS': 'XLI', 'RTX': 'XLI',
    'LMT': 'XLI', 'GE': 'XLI', 'MMM': 'XLI', 'DE': 'XLI',
    'LIN': 'XLB', 'APD': 'XLB', 'SHW': 'XLB', 'FCX': 'XLB', 'NEM': 'XLB', 'ECL': 'XLB',
    'NEE': 'XLU', 'DUK': 'XLU', 'SO': 'XLU', 'D': 'XLU', 'AEP': 'XLU', 'SRE': 'XLU',
    'PLD': 'XLRE', 'AMT': 'XLRE', 'CCI': 'XLRE', 'EQIX': 'XLRE', 'PSA': 'XLRE', 'SPG': 'XLRE',
    'T': 'XLC', 'VZ': 'XLC', 'TMUS': 'XLC',
}

def build_dataset_extended():
    """Build dataset with more symbols, longer history, daily sampling."""
    all_features = []
    all_labels = []
    feature_names = None
    
    spy_c, spy_v, spy_h, spy_l = fetch_data('SPY', days=1000)
    if not spy_c:
        print("ERROR: Failed to fetch SPY")
        return None, None, None
    
    for symbol in UNIVERSE_EXTENDED:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=1000)
            if not closes or len(closes) < 250:
                continue
            
            sector_etf = SECTOR_ETFS_EXTENDED.get(symbol, 'SPY')
            sector_closes, _, _, _ = fetch_data(sector_etf, days=1000)
            if not sector_closes:
                sector_closes = spy_c
            
            atr_series = [calc_atr(highs[:i+1], lows[:i+1], closes[:i+1]) for i in range(len(closes))]
            labels = triple_barrier_labels(closes, highs, lows, atr_series)
            
            # Daily sampling for maximum data
            for i in range(250, len(closes) - 14):
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
            
            print(f"  {symbol}: {len([l for l in all_labels if l is not None])} samples")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")
            continue
    
    return np.array(all_features), np.array(all_labels), feature_names

def walk_forward_validation(X, y, model, n_splits=5):
    """Walk-forward validation (rolling window)."""
    n = len(X)
    fold_size = n // (n_splits + 1)
    scores = []
    
    for i in range(n_splits):
        train_end = fold_size * (i + 2)
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        
        if test_end > n:
            break
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]
        
        # Skip if only one class in test
        if len(np.unique(y_test)) < 2:
            continue
        
        # Compute class weights
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.array([{c: w for c, w in zip(classes, weights)}[y] for y in y_train])
        
        # Clone model
        from sklearn.base import clone
        m = clone(model)
        m.fit(X_train, y_train, sample_weight=sample_weights)
        score = m.score(X_test, y_test)
        scores.append(score)
        print(f"    Fold {i+1}: {score:.2%} (train: {len(y_train)}, test: {len(y_test)})")
    
    return np.mean(scores), scores

def train():
    print("Building extended dataset...")
    result = build_dataset_extended()
    if result[0] is None:
        return
    X, y, feature_names = result
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    mask = y >= 0
    X = X[mask]
    y = y[mask]
    
    print(f"After filtering: {X.shape[0]} samples")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # === Model 1: Random Forest ===
    print("\n=== Random Forest (Walk-Forward) ===")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_mean, rf_scores = walk_forward_validation(X, y, rf)
    print(f"RF Mean Accuracy: {rf_mean:.2%}")
    
    # === Model 2: XGBoost ===
    print("\n=== XGBoost (Walk-Forward) ===")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        random_state=42,
        eval_metric='logloss',
        reg_alpha=0.1,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb_mean, xgb_scores = walk_forward_validation(X, y, xgb)
    print(f"XGB Mean Accuracy: {xgb_mean:.2%}")
    
    # === Model 3: Ensemble ===
    print("\n=== Ensemble (RF + XGB average) ===")
    # Train on full dataset for final prediction
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    sample_weights = np.array([{c: w for c, w in zip(classes, weights)}[y] for y in y_train])
    
    rf_final = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=8, min_samples_leaf=4,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    xgb_final = XGBClassifier(
        n_estimators=500, max_depth=7, learning_rate=0.03, random_state=42,
        eval_metric='logloss', reg_alpha=0.1, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8
    )
    
    rf_final.fit(X_train, y_train)
    xgb_final.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Ensemble: average probabilities
    rf_prob = rf_final.predict_proba(X_test)
    xgb_prob = xgb_final.predict_proba(X_test)
    ensemble_prob = (rf_prob + xgb_prob) / 2
    ensemble_pred = np.argmax(ensemble_prob, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    print(f"Ensemble Accuracy: {ensemble_acc:.2%}")
    print(classification_report(y_test, ensemble_pred, target_names=['Down', 'Up']))
    
    # === Pick Winner ===
    print("\n" + "=" * 60)
    models = {
        'RandomForest': (rf_final, rf_mean),
        'XGBoost': (xgb_final, xgb_mean),
        'Ensemble': (None, ensemble_acc)
    }
    
    # Use walk-forward score for RF and XGB, test score for ensemble
    best_name = max(models.keys(), key=lambda k: models[k][1])
    best_score = models[best_name][1]
    
    if best_name == 'Ensemble':
        # Save both models for ensemble
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({
                'model': rf_final,
                'model_2': xgb_final,
                'feature_names': feature_names,
                'accuracy': best_score,
                'type': 'ensemble'
            }, f)
        print(f"🏆 WINNER: Ensemble ({best_score:.2%})")
    else:
        winner = models[best_name][0]
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({
                'model': winner,
                'feature_names': feature_names,
                'accuracy': best_score,
                'type': 'single'
            }, f)
        print(f"🏆 WINNER: {best_name} ({best_score:.2%})")
    
    # Feature importances
    if best_name != 'Ensemble':
        importances = winner.feature_importances_
    else:
        importances = rf_final.feature_importances_
    total = sum(importances)
    print(f"\nTop 10 Features:")
    for fname, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {fname:25s}: {imp/total:.2%}")
    
    print(f"\nSaved to {MODEL_FILE}")

if __name__ == "__main__":
    train()
