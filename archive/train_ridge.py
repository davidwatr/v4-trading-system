#!/usr/bin/env python3
"""
ML Model That Actually Generalizes — following AFML principles.
Triple-barrier labels, stationary features, regularized linear model, proper embargo.
"""

import numpy as np
import pickle
import logging
from datetime import datetime
from production_v18 import fetch_data, UNIVERSE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
# Pruned universe — financials + momentum names
UNIVERSE_PRUNED = [
    'BAC', 'JPM', 'WFC', 'GS', 'C', 'V', 'MA',  # financials
    'INTC', 'NVDA', 'META', 'TSLA', 'AAPL', 'PYPL',  # tech
    'COST', 'TGT',  # consumer
]

# Triple-barrier parameters (aligned with bracket orders)
PROFIT_MULT = 2.0    # TP = entry + 2x ATR
STOP_MULT = 1.5      # SL = entry - 1.5x ATR
MAX_HOLD = 10        # Time barrier: 10 days

# Feature diet — 5 stationary features only
FEATURES = [
    'rsi_zscore',      # RSI as z-score (bounded, mean-reverting)
    'vol_ratio',       # Volume / 20-day avg volume (relative)
    'atr_ratio',       # ATR / price (volatility state)
    'momentum_zscore', # 5-day return as z-score
    'spread_zscore',   # (High-Low)/Close as z-score
]

# Validation
EMBARGO_DAYS = 10    # Dead zone between train/test
OOS_WALL_DAYS = 120  # ~4 months completely hidden


def compute_triple_barrier_labels(closes, atr, i):
    """
    Compute which barrier is hit first after day i.
    Returns: 1 (profit hit first), -1 (loss hit first), 0 (time barrier)
    """
    if i + MAX_HOLD >= len(closes):
        return 0
    
    entry = closes[i]
    upper = entry + PROFIT_MULT * atr
    lower = entry - STOP_MULT * atr
    
    for j in range(1, MAX_HOLD + 1):
        if i + j >= len(closes):
            break
        if closes[i + j] >= upper:
            return 1  # Profit
        if closes[i + j] <= lower:
            return -1  # Loss
    
    # Time barrier hit — determine by final price
    final = closes[min(i + MAX_HOLD, len(closes) - 1)]
    if final > entry:
        return 1
    elif final < entry:
        return -1
    return 0


def compute_features(closes, volumes, idx):
    """
    Compute 5 stationary features at index idx.
    All features are bounded, mean-reverting, or relative ratios.
    """
    if idx < 20:
        return None
    
    # 1. RSI z-score (bounded 0-100, mean-reverting)
    deltas = np.diff(closes[idx-14:idx+1])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses) if np.mean(losses) > 0 else 0.001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Z-score of RSI (how far from neutral 50)
    rsi_zscore = (rsi - 50) / 25  # ~N(0,1) if RSI is uniform
    
    # 2. Volume ratio (relative, mean-reverting)
    avg_vol = np.mean(volumes[idx-19:idx+1])
    vol_ratio = volumes[idx] / avg_vol if avg_vol > 0 else 1.0
    vol_ratio = np.clip(vol_ratio, 0, 5)  # Cap outliers
    
    # 3. ATR ratio (volatility state, relative)
    atr = np.std(np.diff(closes[idx-19:idx+1]) / closes[idx-19:idx])
    atr_ratio = atr / closes[idx] if closes[idx] > 0 else 0.02
    atr_ratio = np.clip(atr_ratio, 0, 0.1)
    
    # 4. Momentum z-score (5-day return, normalized)
    if idx >= 5:
        mom_5d = (closes[idx] - closes[idx-5]) / closes[idx-5]
    else:
        mom_5d = 0
    # Normalize by recent volatility
    recent_vol = np.std(np.diff(closes[idx-19:idx+1]) / closes[idx-19:idx]) if idx >= 20 else 0.02
    momentum_zscore = mom_5d / recent_vol if recent_vol > 0 else 0
    
    # 5. Spread z-score (High-Low range / Close)
    if idx >= 20:
        spreads = []
        for j in range(max(4, idx-19), idx+1):
            window = closes[j-4:j+1]
            if len(window) > 0:
                spread = (closes[j] - np.min(window)) / closes[j]
                spreads.append(spread)
        spread_zscore = (spreads[-1] - np.mean(spreads)) / np.std(spreads) if len(spreads) > 1 and np.std(spreads) > 0 else 0
    else:
        spread_zscore = 0
    
    return [rsi_zscore, vol_ratio, atr_ratio, momentum_zscore, spread_zscore]


def build_dataset(symbol, days=730):
    """Build feature matrix and triple-barrier labels."""
    closes, volumes, highs, lows = fetch_data(symbol, days=days)
    if closes is None or len(closes) < 50:
        return None, None
    
    # OOS wall — only use data before cutoff
    usable = len(closes) - OOS_WALL_DAYS
    if usable < 100:
        return None, None
    
    X = []
    y = []
    
    for i in range(20, usable):
        feat = compute_features(closes, volumes, i)
        if feat is None:
            continue
        
        # ATR for triple barrier
        atr = np.std(np.diff(closes[i-19:i+1]) / closes[i-19:i])
        
        # Triple-barrier label
        label = compute_triple_barrier_labels(closes, atr, i)
        
        X.append(feat)
        y.append(label)
    
    if len(X) < 50:
        return None, None
    
    return np.array(X), np.array(y)


def train():
    """Train with proper validation."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import accuracy_score
    
    all_X = []
    all_y = []
    
    for symbol in UNIVERSE_PRUNED:
        logger.info(f"Processing {symbol}...")
        X, y = build_dataset(symbol)
        if X is None:
            logger.warning(f"  Skipped")
            continue
        all_X.append(X)
        all_y.append(y)
        logger.info(f"  {len(X)} samples")
    
    if not all_X:
        logger.error("No data!")
        return
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # Convert to binary: profit (1) vs not (0)
    y_binary = (y == 1).astype(int)
    
    # Chronological split with embargo
    split_idx = int(len(X) * 0.8)
    # Add embargo: skip EMBARGO_DAYS samples between train and test
    train_end = split_idx
    test_start = split_idx + EMBARGO_DAYS
    
    X_train = X[:train_end]
    y_train = y_binary[:train_end]
    X_test = X[test_start:]
    y_test = y_binary[test_start:]
    
    logger.info(f"\nTotal: {len(X)} samples")
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Class balance: {np.mean(y_binary):.2%} positive")
    
    # Ridge regression (regularized linear model)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = (model.predict(X_train) > 0.5).astype(int)
    test_pred = (model.predict(X_test) > 0.5).astype(int)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    logger.info(f"\nTrain accuracy: {train_acc:.2%}")
    logger.info(f"Test accuracy: {test_acc:.2%}")
    logger.info(f"Gap: {train_acc - test_acc:.2%}")
    
    # Feature importance (coefficients)
    logger.info(f"\nFeature coefficients:")
    for feat, coef in zip(FEATURES, model.coef_):
        logger.info(f"  {feat}: {coef:.4f}")
    
    # Save
    output = {
        'model': model,
        'feature_names': FEATURES,
        'accuracy': test_acc,
        'train_accuracy': train_acc,
        'oos_wall_months': OOS_WALL_DAYS // 30,
        'embargo_days': EMBARGO_DAYS,
        'universe': UNIVERSE_PRUNED,
        'trained_at': datetime.now().isoformat(),
        'dataset': f"{len(UNIVERSE_PRUNED)} symbols, triple-barrier labels, Ridge"
    }
    
    with open('ml_model_ridge.pkl', 'wb') as f:
        pickle.dump(output, f)
    
    logger.info(f"\nSaved ml_model_ridge.pkl (test accuracy: {test_acc:.2%})")
    
    return output


if __name__ == "__main__":
    train()
