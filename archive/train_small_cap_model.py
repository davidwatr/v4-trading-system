#!/usr/bin/env python3
"""
Train a small cap explosion model.
Target: 5%+ move in 5 days (explosive moves).
Features: Same 24 features as the main model, but trained on small cap data.
"""

import json
import sys
import pickle
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))

from production_v18 import (
    fetch_data, extract_features, calc_atr, calc_rsi,
    UNIVERSE, MODEL_FILE
)

# Small cap universe for training
SMALL_CAP_UNIVERSE = [
    "RBLX", "U", "SNAP", "PINS", "MTCH", "BMBL", "HOOD", "COIN", "SQ", "AFRM",
    "UPST", "SOFI", "LC", "RUM", "MARA", "RIOT", "CLSK", "WULF", "CORZ", "HUT",
    "BITF", "SDIG", "SMCI", "ARM", "AVGO", "MRVL", "AMAT", "LRCX", "KLAC", "TER",
    "ON", "MPWR", "SWKS", "QRVO", "MCHP", "ADI", "TXN", "IFS", "SLAB", "POWI",
    "MRNA", "BNTX", "NVAX", "SRPT", "BLUE", "CRSP", "EDIT", "NTLA", "VCEL", "BMRN",
    "FOLD", "EXEL", "HALO", "PRTA", "KPTI", "AVXL", "ARVN", "RCKT", "TNGX", "ANAB",
    "LULU", "DECK", "ONON", "BIRK", "BYND", "LMND", "PSTG", "GTLB", "AI", "PATH",
    "WIZ", "NICE", "CYBR", "QLYS", "S", "TENB", "VRNS", "ZS", "RIVN", "LCID",
    "NIO", "XPEV", "LI", "FSR", "GOEV", "WKHS", "NKLA", "RIDE", "SPWR", "ENPH",
    "SEDG", "FSLR", "CSIQ", "JKS", "PLUG", "BE", "FCEL", "BLDP", "HYZN", "SPCE",
    "RKLB", "MAXR", "IRDM", "GSAT", "LORL", "KTOS", "DRS", "OSK", "LHX", "MSTR",
    "BTG", "BITO", "GBTC", "NU", "NUVB", "MQ", "SE", "MELI", "W", "SHOP", "JD",
    "PDD", "BABA", "GPOR", "RIG", "DO", "NE", "PTEN", "WHD", "HLX", "OII", "HAL",
    "GME", "AMC", "BB", "NOK", "BBBY", "WISH", "CLOV", "SKS", "SDC", "VRAY",
    "EKSO", "CTIB", "NVOS", "GHSI", "CTXR", "EDTX", "CANF", "PTN", "OBS", "GOLD",
    "NEM", "AEM", "WPM", "PAAS", "CDE", "HL", "MUX", "AG", "EXK", "FSM", "OR",
    "SA", "BTG", "EGO", "AUY", "IAG", "BE", "BLDP", "HYZN", "SPCE", "RKLB",
]

# Remove duplicates
seen = set()
SMALL_CAP_UNIVERSE = [x for x in SMALL_CAP_UNIVERSE if not (x in seen or seen.add(x))]


def label_explosion(closes, days_forward=5, threshold=0.05):
    """
    Label: 1 if stock moves >= threshold in next days_forward days, 0 otherwise.
    """
    if len(closes) < days_forward + 1:
        return None
    
    current = closes[0]
    future_max = max(closes[1:days_forward+1])
    future_min = min(closes[1:days_forward+1])
    
    # Check if it goes up by threshold (explosion up)
    if (future_max - current) / current >= threshold:
        return 1
    # Check if it goes down by threshold (explosion down - avoid these)
    elif (current - future_min) / current >= threshold:
        return -1
    else:
        return 0


def collect_training_data():
    """Collect features and labels for small cap stocks."""
    print("Collecting training data for small cap explosion model...")
    
    # Fetch SPY for relative strength
    spy_closes, _, _, _ = fetch_data("SPY", days=730)
    if not spy_closes:
        print("ERROR: Can't fetch SPY")
        return None, None
    
    X = []
    y = []
    feature_names = None
    
    for symbol in SMALL_CAP_UNIVERSE:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=730)
            if not closes or len(closes) < 200:
                continue
            
            # For each day in the past, create a training sample
            # Use every 5th day to avoid too much overlap
            for i in range(100, len(closes) - 5, 5):
                window_closes = closes[:i+1]
                window_volumes = volumes[:i+1]
                window_highs = highs[:i+1]
                window_lows = lows[:i+1]
                
                # Extract features
                feat = extract_features(window_closes, window_volumes, window_highs, window_lows, symbol, spy_closes, spy_closes)
                if feat is None:
                    continue
                
                # Label: what happens in next 5 days?
                remaining_closes = closes[i:]
                label = label_explosion(remaining_closes, days_forward=5, threshold=0.05)
                
                if label is None:
                    continue
                
                # Only keep explosions (1) and non-explosions (0), skip crashes (-1)
                if label == -1:
                    label = 0  # Treat crashes as non-explosion (we want to avoid these)
                
                if feature_names is None:
                    feature_names = list(feat.keys())
                
                X.append([feat[name] for name in feature_names])
                y.append(label)
            
            time.sleep(0.1)
            
        except Exception as e:
            continue
    
    print(f"Collected {len(X)} samples")
    print(f"Explosions: {sum(1 for label in y if label == 1)} ({sum(1 for label in y if label == 1)/len(y)*100:.1f}%)")
    print(f"Non-explosions: {sum(1 for label in y if label == 0)} ({sum(1 for label in y if label == 0)/len(y)*100:.1f}%)")
    
    return np.array(X), np.array(y), feature_names


def train_model(X, y, feature_names):
    """Train the small cap explosion model with better handling of imbalanced data."""
    print("\nTraining small cap explosion model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Try XGBoost first (usually better for this)
    try:
        from xgboost import XGBClassifier
        
        # Calculate scale_pos_weight for imbalanced data
        neg_count = sum(1 for label in y_train if label == 0)
        pos_count = sum(1 for label in y_train if label == 1)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
    except ImportError:
        # Fallback to Random Forest with class weights
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight='balanced',  # Handle imbalance
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Explosion', 'Explosion']))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    
    # Feature importance
    importances = model.feature_importances_
    total = sum(importances)
    print(f"\nTop 10 Feature Importances:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {name:20s}: {imp/total:.4f}")
    
    return model


def main():
    print("=" * 70)
    print(f"  SMALL CAP EXPLOSION MODEL TRAINING — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Collect data
    X, y, feature_names = collect_training_data()
    
    if X is None or len(X) == 0:
        print("ERROR: No training data collected")
        return
    
    # Train model
    model = train_model(X, y, feature_names)
    
    # Save model
    output_file = Path('ml_model_small_cap.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy_score(y, model.predict(X)),
            'training_samples': len(X),
            'explosion_rate': sum(1 for label in y if label == 1) / len(y),
            'trained_at': datetime.now().isoformat()
        }, f)
    
    print(f"\nModel saved to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
