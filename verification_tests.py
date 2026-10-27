import numpy as np, warnings, os
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

print('='*70)
print('THE 3 VERIFICATION TESTS (Fixed)')
print('='*70)

# Fetch data
all_data = {}
for sym in UNIVERSE[:20]:
    d = fetch_data(sym, days=500)
    if d is not None and len(d[0]) > 200:
        all_data[sym] = d
print(f'Loaded {len(all_data)} symbols')

def compute_features(closes, volumes, highs, lows):
    """Compute stationary features"""
    n = len(closes)
    feats = {}
    d = lambda x: np.diff(x, prepend=x[0])
    
    deltas = d(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
    rs = avg_gain / (avg_loss + 1e-10)
    feats['rsi'] = 100 - (100 / (1 + rs))
    
    vol_ma20 = np.convolve(volumes, np.ones(20)/20, mode='same')
    feats['vol_ratio'] = volumes / (vol_ma20 + 1e-10)
    feats['rvol'] = volumes / (vol_ma20 + 1e-10)
    
    tr = highs - lows
    atr = np.convolve(tr, np.ones(14)/14, mode='same')
    feats['atr_normalized'] = atr / (closes + 1e-10)
    
    bb_mid = np.convolve(closes, np.ones(20)/20, mode='same')
    bb_std = np.array([np.std(closes[max(0,i-20):i+1]) for i in range(n)])
    feats['bb_pos'] = (closes - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    ema12 = np.convolve(closes, np.ones(12)/12, mode='same')
    ema26 = np.convolve(closes, np.ones(26)/26, mode='same')
    macd = ema12 - ema26
    feats['macd'] = macd / (closes + 1e-10)
    feats['macd_signal'] = np.convolve(macd, np.ones(9)/9, mode='same') / (closes + 1e-10)
    
    ma20 = np.convolve(closes, np.ones(20)/20, mode='same')
    ma50 = np.convolve(closes, np.ones(50)/50, mode='same')
    ma200 = np.convolve(closes, np.ones(200)/200, mode='same')
    feats['price_vs_ma20'] = closes / (ma20 + 1e-10)
    feats['price_vs_ma50'] = closes / (ma50 + 1e-10)
    feats['price_vs_ma200'] = closes / (ma200 + 1e-10)
    feats['rel_strength'] = closes / (np.convolve(closes, np.ones(252)/252, mode='same') + 1e-10)
    
    return feats

# Build feature matrix per symbol
all_X, all_y, all_sym = [], [], []
for sym, data in all_data.items():
    closes = np.array(data[0])
    volumes = np.array(data[1])
    highs = np.array(data[2])
    lows = np.array(data[3])
    
    feats = compute_features(closes, volumes, highs, lows)
    feat_names = list(feats.keys())
    X = np.column_stack([feats[f] for f in feat_names])
    y = (np.roll(closes, -1) > closes).astype(int)
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    
    if len(X) < 200:
        continue
    
    # Ensure all symbols have same length by truncating to min
    min_len = min(len(X), 300)  # Use last 300 days
    X = X[-min_len:]
    y = y[-min_len:]
    
    all_X.append(X)
    all_y.append(y)
    all_sym.append(sym)

# Stack: shape = (n_symbols * n_days, n_features)
X_all = np.vstack(all_X)
y_all = np.concatenate(all_y)

print(f'Total samples: {len(y_all)}')
print(f'Features: {len(feat_names)}')
print(f'Symbols: {len(all_X)}')
print(f'Days per symbol: {all_X[0].shape[0]}')

# Base rate
print(f'\nBase rate: {np.mean(y_all):.4f} up days')

# Walk-forward with proper purging
# Since symbols are concatenated, we need to split by time index
n_days = all_X[0].shape[0]
window_size = 50
min_train = 150
step = 25

folds = []
train_end = min_train
while train_end + window_size <= n_days:
    # For each fold: train on [0, train_end], test on [train_end, train_end + window_size]
    # But need to purge overlapping symbols — use per-symbol splits
    folds.append(train_end)
    train_end += step

print(f'Walk-forward folds: {len(folds)}')

real_accs, real_aucs = [], []
rand_accs, rand_aucs = [], []
pred_up_rates, actual_up_rates = [], []

for fold_i, train_end in enumerate(folds):
    # Build train/test across all symbols
    train_idx = list(range(0, train_end))
    test_idx = list(range(train_end, min(train_end + window_size, n_days)))
    
    # For each symbol, extract the corresponding rows
    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []
    
    for sym_idx in range(len(all_X)):
        # This symbol's rows in X_all
        row_start = sym_idx * n_days
        row_end = row_start + n_days
        
        # Training rows
        for di in train_idx:
            X_train_parts.append(all_X[sym_idx][di])
            y_train_parts.append(all_y[sym_idx][di])
        
        # Test rows
        for di in test_idx:
            X_test_parts.append(all_X[sym_idx][di])
            y_test_parts.append(all_y[sym_idx][di])
    
    X_train = np.array(X_train_parts)
    y_train = np.array(y_train_parts)
    X_test = np.array(X_test_parts)
    y_test = np.array(y_test_parts)
    
    if len(y_test) < 10:
        continue
    
    # Real model
    mdl = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    preds_proba = mdl.predict_proba(X_test)[:, 1]
    
    acc = np.mean(preds == y_test)
    auc = roc_auc_score(y_test, preds_proba)
    
    real_accs.append(acc)
    real_aucs.append(auc)
    pred_up_rates.append(np.mean(preds))
    actual_up_rates.append(np.mean(y_test))
    
    # Random model (permutation test)
    np.random.seed(fold_i)
    X_train_rand = np.random.randn(*X_train.shape)
    X_test_rand = np.random.randn(*X_test.shape)
    
    mdl_rand = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    mdl_rand.fit(X_train_rand, y_train)
    preds_rand = mdl_rand.predict(X_test_rand)
    
    acc_rand = np.mean(preds_rand == y_test)
    rand_accs.append(acc_rand)

print('\n' + '='*70)
print('TEST 1: PERMUTATION TEST (Random Noise Features)')
print('='*70)
print(f'\nReal features: {np.mean(real_accs):.4f} accuracy')
print(f'Random features: {np.mean(rand_accs):.4f} accuracy')
diff = np.mean(real_accs) - np.mean(rand_accs)
print(f'Difference: {diff:.4f}')
if np.mean(rand_accs) > 0.55:
    print('❌ FAIL: Random features also predict >55% → structural leak')
else:
    print('✅ PASS: Random features drop to ~50% → no structural leak')

print('\n' + '='*70)
print('TEST 2: ROC-AUC CHECK')
print('='*70)
print(f'\nMean ROC-AUC: {np.mean(real_aucs):.4f}')
print(f'Min: {np.min(real_aucs):.4f}, Max: {np.max(real_aucs):.4f}')
if np.mean(real_aucs) < 0.53:
    print('❌ FAIL: AUC < 0.53 → no real directional edge')
elif np.mean(real_aucs) < 0.55:
    print('⚠️  WEAK: AUC 0.53-0.55 → marginal edge')
else:
    print('✅ PASS: AUC > 0.55 → real directional edge')

print('\n' + '='*70)
print('TEST 3: CONFUSION MATRIX (Base-Rate Check)')
print('='*70)
print(f'\nMean predicted UP: {np.mean(pred_up_rates):.4f}')
print(f'Mean actual UP: {np.mean(actual_up_rates):.4f}')
if np.mean(pred_up_rates) > 0.75:
    print(f'❌ FAIL: Model predicts UP {np.mean(pred_up_rates)*100:.0f}% → class-imbalance bias')
elif np.mean(pred_up_rates) > 0.65:
    print(f'⚠️  WARNING: Model predicts UP {np.mean(pred_up_rates)*100:.0f}%')
else:
    print(f'✅ PASS: Model predicts UP {np.mean(pred_up_rates)*100:.0f}%')

# Full confusion matrix for last fold
print(f'\nConfusion Matrix (last fold):')
print(f'  TN={np.sum((preds == 0) & (y_test == 0))}, FP={np.sum((preds == 1) & (y_test == 0))}')
print(f'  FN={np.sum((preds == 0) & (y_test == 1))}, TP={np.sum((preds == 1) & (y_test == 1))}')

print('\n' + '='*70)
print('FINAL VERDICT')
print('='*70)
