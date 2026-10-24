import pickle, numpy as np, warnings, os
warnings.filterwarnings('ignore')

from production_v18 import fetch_data, UNIVERSE

print('='*70)
print('WALK-FORWARD TEST (rolling window, multiple OOS periods)')
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
    
    # Other
    feats['support'] = np.convolve(lows, np.ones(20)/20, mode='same')
    feats['resistance'] = np.convolve(highs, np.ones(20)/20, mode='same')
    feats['rel_strength'] = closes / (np.convolve(closes, np.ones(252)/252, mode='same') + 1e-10)
    feats['residual_return'] = closes - np.convolve(closes, np.ones(20)/20, mode='same')
    feats['iv_rank'] = np.zeros(n)
    feats['rvol'] = volumes / (vol_ma20 + 1e-10)
    feats['cci'] = np.zeros(n)
    feats['mfi'] = np.zeros(n)
    feats['stoch'] = np.zeros(n)
    feats['williams_r'] = np.zeros(n)
    feats['adx'] = np.zeros(n)
    
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

print(f'\nTotal samples: {len(y)}')

# Walk-forward: train on [0, i], test on [i, i+window]
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

window_size = 100  # Test window (about 2 weeks of daily data)
min_train = 500     # Minimum training samples
step = 50           # Roll forward by 50 samples each time

folds = []
i = min_train
while i + window_size < len(y):
    train_idx = list(range(0, i))
    test_idx = list(range(i, min(i + window_size, len(y))))
    folds.append((train_idx, test_idx))
    i += step

print(f'Walk-forward folds: {len(folds)}')
print(f'Avg train size: {np.mean([len(t) for t, _ in folds]):.0f}')
print(f'Avg test size: {np.mean([len(t) for _, t in folds]):.0f}\n')

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
}

all_results = {name: [] for name in models}

for fold_i, (train_idx, test_idx) in enumerate(folds):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        accuracy = np.mean(preds == y_test)
        all_results[name].append(accuracy)

print('='*70)
print('WALK-FORWARD RESULTS')
print('='*70)
print(f'\n{"Model":<25} {"Mean":<10} {"Median":<10} {"Min":<10} {"Max":<10} {"Win Rate":<10}')
print('-'*70)
for name, accs in all_results.items():
    accs = np.array(accs)
    print(f'{name:<25} {accs.mean():.4f}    {np.median(accs):.4f}    {accs.min():.4f}    {accs.max():.4f}    {np.mean(accs > 0.5):.2f}')

# Check if any individual fold is driving the result
print('\n' + '='*70)
print('FOLD-BY-FOLD (XGBoost)')
print('='*70)
for i, acc in enumerate(all_results['XGBoost']):
    marker = '✓' if acc > 0.5 else '✗'
    print(f'  Fold {i+1}: {acc:.4f} {marker}')

print(f'\nConclusion:')
print(f'  If edge is consistent across folds → real signal')
print(f'  If edge is concentrated in 1-2 folds → overfit to regime')
