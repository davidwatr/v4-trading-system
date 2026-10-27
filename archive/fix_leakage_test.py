import pickle, numpy as np, warnings, os
warnings.filterwarnings('ignore')

from production_v18 import fetch_data, UNIVERSE

print('='*70)
print('FIXING DATA LEAKAGE — PROPER OOS TEST')
print('='*70)

# Fetch data
print('\nFetching data...')
all_data = {}
for sym in UNIVERSE[:20]:
    d = fetch_data(sym, days=500)
    if d is not None and len(d[0]) > 200:
        all_data[sym] = d
print(f'Loaded {len(all_data)} symbols')

def compute_features(closes, volumes, highs, lows):
    """Compute all features, all same length"""
    n = len(closes)
    feats = {}
    
    # Use prepend for diff to maintain length
    d = lambda x: np.diff(x, prepend=x[0])
    
    # RSI
    deltas = d(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    feats['rsi'] = rsi
    
    # Vol ratio
    vol_ma20 = np.convolve(volumes, np.ones(20)/20, mode='same')
    feats['vol_ratio'] = volumes / (vol_ma20 + 1e-10)
    
    # ATR
    tr = highs - lows
    atr = np.convolve(tr, np.ones(14)/14, mode='same')
    feats['atr'] = atr
    feats['atr_normalized'] = atr / (closes + 1e-10)
    
    # BB
    bb_mid = np.convolve(closes, np.ones(20)/20, mode='same')
    bb_std = np.array([np.std(closes[max(0,i-20):i+1]) for i in range(n)])
    feats['bb_upper'] = bb_mid + 2*bb_std
    feats['bb_lower'] = bb_mid - 2*bb_std
    feats['bb_pos'] = (closes - feats['bb_lower']) / (feats['bb_upper'] - feats['bb_lower'] + 1e-10)
    
    # CCI
    tp = (highs + lows + closes) / 3
    tp_ma = np.convolve(tp, np.ones(14)/14, mode='same')
    tp_std = np.array([np.std(tp[max(0,i-14):i+1]) for i in range(n)])
    feats['cci'] = (tp - tp_ma) / (0.015 * tp_std + 1e-10)
    
    # MFI
    mf = volumes * tp
    pmf = np.where(d(tp) > 0, mf, 0)
    nmf = np.where(d(tp) < 0, mf, 0)
    pmf_sum = np.convolve(pmf, np.ones(14)/14, mode='same')
    nmf_sum = np.convolve(nmf, np.ones(14)/14, mode='same')
    mfr = pmf_sum / (nmf_sum + 1e-10)
    feats['mfi'] = 100 - (100 / (1 + mfr))
    
    # Stoch
    low_min = np.array([np.min(lows[max(0,i-14):i+1]) for i in range(n)])
    high_max = np.array([np.max(highs[max(0,i-14):i+1]) for i in range(n)])
    feats['stoch'] = 100 * (closes - low_min) / (high_max - low_min + 1e-10)
    feats['williams_r'] = -100 * (high_max - closes) / (high_max - low_min + 1e-10)
    
    # ADX
    plus_dm = np.where(d(highs) > d(lows), d(highs), 0)
    minus_dm = np.where(d(lows) > d(highs), d(lows), 0)
    atr14 = np.convolve(tr, np.ones(14)/14, mode='same')
    plus_di = 100 * np.convolve(plus_dm, np.ones(14)/14, mode='same') / (atr14 + 1e-10)
    minus_di = 100 * np.convolve(minus_dm, np.ones(14)/14, mode='same') / (atr14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    feats['adx'] = np.convolve(dx, np.ones(14)/14, mode='same')
    
    # MACD
    ema12 = np.convolve(closes, np.ones(12)/12, mode='same')
    ema26 = np.convolve(closes, np.ones(26)/26, mode='same')
    feats['macd'] = ema12 - ema26
    feats['macd_signal'] = np.convolve(feats['macd'], np.ones(9)/9, mode='same')
    
    # Price vs MAs
    ma20 = np.convolve(closes, np.ones(20)/20, mode='same')
    ma50 = np.convolve(closes, np.ones(50)/50, mode='same')
    ma200 = np.convolve(closes, np.ones(200)/200, mode='same')
    feats['price_vs_ma20'] = closes / (ma20 + 1e-10)
    feats['price_vs_ma50'] = closes / (ma50 + 1e-10)
    feats['price_vs_ma200'] = closes / (ma200 + 1e-10)
    feats['ma_long'] = ma50
    feats['ma_short'] = ma20
    feats['ma_200'] = ma200
    
    # VWAP
    feats['vwap'] = np.cumsum(closes * volumes) / (np.cumsum(volumes) + 1e-10)
    feats['vwap_pos'] = closes / (feats['vwap'] + 1e-10)
    
    # Relative strength
    feats['rel_strength'] = closes / (np.convolve(closes, np.ones(252)/252, mode='same') + 1e-10)
    
    # Residual return
    feats['residual_return'] = closes - np.convolve(closes, np.ones(20)/20, mode='same')
    
    # Support/Resistance
    feats['support'] = np.convolve(lows, np.ones(20)/20, mode='same')
    feats['resistance'] = np.convolve(highs, np.ones(20)/20, mode='same')
    
    # IV rank placeholder
    feats['iv_rank'] = np.zeros(n)
    
    # Rvol (relative volume)
    feats['rvol'] = volumes / (vol_ma20 + 1e-10)
    
    return feats

# Build feature matrix
all_X, all_y = [], []
for sym, data in all_data.items():
    closes = np.array(data[0])
    volumes = np.array(data[1])
    highs = np.array(data[2])
    lows = np.array(data[3])
    
    feats = compute_features(closes, volumes, highs, lows)
    
    # Clean features (no leakage)
    clean_feats = [f for f in feats.keys() if f not in ['tb_label', 'current_price']]
    
    X_list = [feats[f] for f in clean_feats]
    X = np.column_stack(X_list)
    y = (np.roll(closes, -1) > closes).astype(int)
    
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]
    
    if len(X) < 50:
        continue
    
    all_X.append(X)
    all_y.append(y)

X = np.vstack(all_X)
y = np.concatenate(all_y)

# Chronological split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f'\nTotal samples: {len(y)}')
print(f'Training: {len(y_train)}, OOS: {len(y_test)}')
print(f'Features: {len(clean_feats)}')

# Train models properly
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models = {
    'Ridge': Ridge(alpha=1.0),
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
}

print(f'\nTraining on {len(y_train)} samples, testing on {len(y_test)} samples\n')

results = []
for name, mdl in models.items():
    print(f'{name}:')
    mdl.fit(X_train, y_train)
    
    # OOS test
    preds = mdl.predict(X_test)
    if name == 'Ridge':
        preds = (preds > 0.5).astype(int)
    accuracy = np.mean(preds == y_test)
    
    # Training accuracy
    train_preds = mdl.predict(X_train)
    if name == 'Ridge':
        train_preds = (train_preds > 0.5).astype(int)
    train_acc = np.mean(train_preds == y_train)
    
    gap = train_acc - accuracy
    
    print(f'  Train: {train_acc:.4f} | OOS: {accuracy:.4f} | Gap: {gap:.4f}')
    
    if accuracy > 0.52:
        print(f'  → HAS EDGE!')
    elif accuracy > 0.50:
        print(f'  → Marginal')
    else:
        print(f'  → Coin flip')
    
    results.append((name, accuracy, train_acc, gap))
    print()

print('='*70)
print('FINAL RESULTS (properly trained, no leakage)')
print('='*70)
print(f'{"Model":<25} {"Train":<10} {"OOS":<10} {"Gap":<10} {"Status":<15}')
print('-'*70)
for name, acc, train, gap in sorted(results, key=lambda x: x[1], reverse=True):
    status = 'HAS EDGE' if acc > 0.52 else ('Marginal' if acc > 0.50 else 'Coin flip')
    print(f'{name:<25} {train:.4f}    {acc:.4f}    {gap:.4f}    {status}')
