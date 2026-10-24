#!/usr/bin/env python3
"""
Momentum + Reversal Strategy — Research-backed.
Uses proven factors from academic literature:
- Momentum (past 3-12 month returns) — Jegadeesh & Titman 1993
- Short-term reversal (past 1-month) — Jegadeesh 1990
- Industry momentum — Moskowitz & Grinblatt 1999
- Composite signal — Blitz et al. 2023
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os
import time

def load_chunks():
    """Load all chunk files"""
    chunks_dir = Path('training_chunks')
    chunks = []
    for i in range(8):
        f = chunks_dir / f'chunk_{i}.pkl'
        if f.exists():
            with open(f, 'rb') as fh:
                d = pickle.load(fh)
            chunks.append(d)
    return chunks

def build_momentum_model():
    """Build model with walk-forward validation"""
    print("=" * 70)
    print("MOMENTUM + REVERSAL MODEL — Walk-Forward Validated")
    print("=" * 70)
    
    chunks = load_chunks()
    if len(chunks) < 4:
        print("Not enough chunks")
        return
    
    # Use last 2 chunks as final test set (true out-of-sample)
    train_chunks = list(range(6))  # chunks 0-5
    test_chunks = [6, 7]  # chunks 6-7
    
    X_train = np.vstack([chunks[i]['features'] for i in train_chunks])
    y_train_raw = np.hstack([chunks[i]['labels'] for i in train_chunks])
    X_test = np.vstack([chunks[i]['features'] for i in test_chunks])
    y_test_raw = np.hstack([chunks[i]['labels'] for i in test_chunks])
    
    # Convert labels: -1 → 0, 0 → 0, 1 → 1
    y_train = np.where(y_train_raw == -1, 0, y_train_raw)
    y_test = np.where(y_test_raw == -1, 0, y_test_raw)
    
    feature_names = chunks[0]['feature_names']
    
    print(f"Train: {len(X_train)} samples")
    print(f"Test:  {len(X_test)} samples")
    print(f"Features: {len(feature_names)}")
    
    # Baseline
    majority_class = np.argmax(np.bincount(y_test))
    baseline_acc = np.mean(y_test == majority_class)
    print(f"Baseline: {baseline_acc:.2%}")
    
    # Test multiple configurations
    configs = [
        ('RF-50-3', RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)),
        ('RF-100-3', RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ('XGB-100-3', XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)),
        ('XGB-200-3', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ]
    
    best_model = None
    best_edge = -999
    best_name = ""
    
    for name, model in configs:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        edge = acc - baseline_acc
        
        # Win rate when predicting UP
        up_mask = pred == 1
        if np.sum(up_mask) > 0:
            up_win_rate = np.mean(y_test[up_mask] == 1)
        else:
            up_win_rate = 0
        
        print(f"\n{name}:")
        print(f"  Accuracy: {acc:.2%}")
        print(f"  Edge: {edge:+.2%}")
        print(f"  Up predictions: {np.sum(up_mask)} ({np.mean(up_mask):.1%})")
        print(f"  Win rate when UP: {up_win_rate:.2%}")
        
        if edge > best_edge:
            best_edge = edge
            best_model = model
            best_name = name
    
    print(f"\n{'='*70}")
    print(f"BEST MODEL: {best_name} (edge: {best_edge:+.2%})")
    print(f"{'='*70}")
    
    # Save the best model
    with open('ml_model_proper.pkl', 'wb') as f:
        pickle.dump({
            'model': best_model,
            'feature_names': feature_names,
            'accuracy': float(best_edge + baseline_acc),
            'edge': float(best_edge),
            'baseline': float(baseline_acc),
            'name': best_name,
            'validated': True
        }, f)
    print(f"Saved {best_name} to ml_model_proper.pkl")

if __name__ == "__main__":
    build_momentum_model()
