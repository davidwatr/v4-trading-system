#!/usr/bin/env python3
"""
Train on the chunks already saved.
"""

import pickle
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings, time
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

MODEL_FILE = Path("ml_model.pkl")
CHUNK_DIR = Path("training_chunks")

def train():
    print("=" * 60)
    print("TRAINING ON SAVED CHUNKS")
    print("=" * 60)
    
    # Load all chunks
    all_X = []
    all_y = []
    feature_names = None
    
    chunk_files = sorted(CHUNK_DIR.glob("chunk_*.pkl"))
    print(f"Found {len(chunk_files)} chunks")
    
    for cf in chunk_files:
        with open(cf, 'rb') as f:
            data = pickle.load(f)
        all_X.append(data['features'])
        all_y.append(data['labels'])
        if feature_names is None:
            feature_names = data['feature_names']
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    print(f"Total: {X.shape[0]} samples, {X.shape[1]} features")
    
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
