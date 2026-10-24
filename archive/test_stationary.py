import pickle, numpy as np, warnings, os
warnings.filterwarnings('ignore')

from production_v18 import fetch_data, UNIVERSE

print('='*70)
print('CRITICAL TEST: Is the edge real or just trend/momentum?')
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
    """Compute features"""
    n = len(closes)
    feats = {}
    
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
    
    # Price vs MAs (RATIO — stationary)
    ma20 = np.convolve(closes, np.ones(20)/20, mode='same')
    ma50 = np.convolve(closes, np.ones(50)/50, mode='same')
    ma200 = np.convolve(closes, np.ones(200)/200, mode='same')
    feats['price_vs_ma20'] = closes / (ma20 + 1e-10)
    feats['price_vs_ma50'] = closes / (ma50 + 1e-10)
    feats['price_vs_ma200'] = closes / (ma200 + 1e-10)
    
    # Raw prices (NON-STATIONARY — these are the problem)
    feats['ma_long'] = ma50
    feats['ma_short'] = ma20
    feats['ma_200'] = ma200
    feats['vwap'] = np.cumsum(closes * volumes) / (np.cumsum(volumes) + 1e-10)
    feats['vwap_pos'] = closes / (feats['vwap'] + 1e-10)
    feats['support'] = np.convolve(lows, np.ones(20)/20, mode='same')
    feats['resistance'] = np.convolve(highs, np.ones(20)/20, mode='same')
    feats['rel_strength'] = closes / (np.convolve(closes, np.ones(252)/252, mode='same') + 1e-10)
    feats['residual_return'] = closes - np.convolve(closes, np.ones(20)/20, mode='same')
    feats['iv_rank'] = np.zeros(n)
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

# Test 1: All features (including non-stationary)
print('\n' + '='*70)
print('TEST 1: ALL FEATURES (including raw prices)')
print('='*70)

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

models_all = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
}

for name, mdl in models_all.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    if name == 'Ridge':
        preds = (preds > 0.5).astype(int)
    accuracy = np.mean(preds == y_test)
    print(f'{name}: OOS = {accuracy:.4f}')

# Test 2: Only stationary features (ratios, z-scores, normalized)
print('\n' + '='*70)
print('TEST 2: ONLY STATIONARY FEATURES (ratios, z-scores)')
print('='*70)

stationary_feats = [
    'rsi', 'atr_normalized', 'bb_pos', 'cci', 'mfi', 'stoch', 'williams_r',
    'adx', 'macd', 'macd_signal', 'price_vs_ma20', 'price_vs_ma50',
    'price_vs_ma200', 'vol_ratio', 'rvol', 'rel_strength', 'vwap_pos'
]

# Get indices of stationary features
stationary_idx = [clean_feats.index(f) for f in stationary_feats if f in clean_feats]
print(f'Stationary features: {len(stationary_idx)}/{len(clean_feats)}')

X_train_stat = X_train[:, stationary_idx]
X_test_stat = X_test[:, stationary_idx]

models_stat = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
}

for name, mdl in models_stat.items():
    mdl.fit(X_train_stat, y_train)
    preds = mdl.predict(X_test_stat)
    if name == 'Ridge':
        preds = (preds > 0.5).astype(int)
    accuracy = np.mean(preds == y_test)
    print(f'{name}: OOS = {accuracy:.4f}')

# Test 3: Only RSI (single feature)
print('\n' + '='*70)
print('TEST 3: ONLY RSI (single feature)')
print('='*70)

rsi_idx = clean_feats.index('rsi')
X_train_rsi = X_train[:, [rsi_idx]]
X_test_rsi = X_test[:, [rsi_idx]]

models_rsi = {
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
}

for name, mdl in models_rsi.items():
    mdl.fit(X_train_rsi, y_train)
    preds = mdl.predict(X_test_rsi)
    if name == 'Ridge':
        preds = (preds > 0.5).astype(int)
    accuracy = np.mean(preds == y_test)
    print(f'{name}: OOS = {accuracy:.4f}')

# Test 4: Random baseline
print('\n' + '='*70)
print('TEST 4: RANDOM BASELINE')
print('='*70)
random_preds = np.random.randint(0, 2, len(y_test))
print(f'Random: OOS = {np.mean(random_preds == y_test):.4f}')

# Test 5: Always predict majority class
print(f'Always UP: OOS = {np.mean(y_test == 1):.4f}')
print(f'Always DOWN: OOS = {np.mean(y_test == 0):.4f}')

print('\n' + '='*70)
print('CONCLUSION')
print('='*70)
print('If edge disappears with stationary features, the model was just')
print('learning the trend (bull market), not predictive patterns.')
