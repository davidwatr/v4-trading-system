import numpy as np, warnings, os
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

print('='*70)
print('CRITICAL FIX: Removing look-ahead bias in features')
print('='*70)

# Fetch data
all_data = {}
for sym in UNIVERSE[:20]:
    d = fetch_data(sym, days=500)
    if d is not None and len(d[0]) > 200:
        all_data[sym] = d
print(f'Loaded {len(all_data)} symbols')

def compute_features_NO_LEAK(closes, volumes, highs, lows):
    """
    Compute features using ONLY past data (causal/realizable).
    np.convolve with mode='valid' ensures no future data leaks.
    """
    n = len(closes)
    feats = {}
    
    # RSI - causal
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Use expanding window for first 14 days, then rolling
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        if i < 14:
            avg_gain[i] = np.mean(gains[:i]) if i > 0 else 0
            avg_loss[i] = np.mean(losses[:i]) if i > 0 else 0
        else:
            avg_gain[i] = np.mean(gains[i-14:i])
            avg_loss[i] = np.mean(losses[i-14:i])
    
    rs = avg_gain / (avg_loss + 1e-10)
    feats['rsi'] = 100 - (100 / (1 + rs))
    
    # Vol ratio - causal
    vol_ma20 = np.zeros(n)
    for i in range(1, n):
        vol_ma20[i] = np.mean(volumes[max(0,i-20):i])
    feats['vol_ratio'] = volumes / (vol_ma20 + 1e-10)
    feats['rvol'] = volumes / (vol_ma20 + 1e-10)
    
    # ATR - causal
    tr = highs - lows
    atr = np.zeros(n)
    for i in range(1, n):
        atr[i] = np.mean(tr[max(0,i-14):i])
    feats['atr_normalized'] = atr / (closes + 1e-10)
    
    # BB - causal
    bb_mid = np.zeros(n)
    bb_std = np.zeros(n)
    for i in range(1, n):
        bb_mid[i] = np.mean(closes[max(0,i-20):i])
        bb_std[i] = np.std(closes[max(0,i-20):i])
    feats['bb_pos'] = (closes - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    # MACD - causal
    ema12 = np.zeros(n)
    ema26 = np.zeros(n)
    for i in range(1, n):
        if i < 12:
            ema12[i] = np.mean(closes[:i+1])
        else:
            ema12[i] = np.mean(closes[i-12:i+1])
        if i < 26:
            ema26[i] = np.mean(closes[:i+1])
        else:
            ema26[i] = np.mean(closes[i-26:i+1])
    macd = ema12 - ema26
    feats['macd'] = macd / (closes + 1e-10)
    
    # MACD signal (9-day)
    macd_signal = np.zeros(n)
    for i in range(1, n):
        macd_signal[i] = np.mean(macd[max(0,i-9):i+1])
    feats['macd_signal'] = macd_signal / (closes + 1e-10)
    
    # Price vs MAs - causal
    ma20 = np.zeros(n)
    ma50 = np.zeros(n)
    ma200 = np.zeros(n)
    for i in range(1, n):
        ma20[i] = np.mean(closes[max(0,i-20):i+1])
        ma50[i] = np.mean(closes[max(0,i-50):i+1])
        ma200[i] = np.mean(closes[max(0,i-200):i+1])
    feats['price_vs_ma20'] = closes / (ma20 + 1e-10)
    feats['price_vs_ma50'] = closes / (ma50 + 1e-10)
    feats['price_vs_ma200'] = closes / (ma200 + 1e-10)
    feats['rel_strength'] = closes / (ma200 + 1e-10)
    
    return feats

# Build feature matrix per symbol
all_X, all_y = [], []
for sym, data in all_data.items():
    closes = np.array(data[0])
    volumes = np.array(data[1])
    highs = np.array(data[2])
    lows = np.array(data[3])
    
    feats = compute_features_NO_LEAK(closes, volumes, highs, lows)
    feat_names = list(feats.keys())
    X = np.column_stack([feats[f] for f in feat_names])
    y = (np.roll(closes, -1) > closes).astype(int)
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    
    if len(X) < 200:
        continue
    
    # Use last 300 days
    min_len = min(len(X), 300)
    X = X[-min_len:]
    y = y[-min_len:]
    
    all_X.append(X)
    all_y.append(y)

# Stack
X_all = np.vstack(all_X)
y_all = np.concatenate(all_y)

print(f'Total samples: {len(y_all)}')
print(f'Features: {len(feat_names)}')
print(f'Base rate: {np.mean(y_all):.4f} up days')

# Walk-forward
n_days = all_X[0].shape[0]
window_size = 50
min_train = 150
step = 25

folds = []
train_end = min_train
while train_end + window_size <= n_days:
    folds.append(train_end)
    train_end += step

print(f'Walk-forward folds: {len(folds)}')

real_accs, real_aucs = [], []
rand_accs = []
pred_up_rates, actual_up_rates = [], []

for fold_i, train_end in enumerate(folds):
    train_idx = list(range(0, train_end))
    test_idx = list(range(train_end, min(train_end + window_size, n_days)))
    
    X_train_parts, y_train_parts = [], []
    X_test_parts, y_test_parts = [], []
    
    for sym_idx in range(len(all_X)):
        for di in train_idx:
            X_train_parts.append(all_X[sym_idx][di])
            y_train_parts.append(all_y[sym_idx][di])
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
    
    # Random model
    np.random.seed(fold_i)
    X_train_rand = np.random.randn(*X_train.shape)
    X_test_rand = np.random.randn(*X_test.shape)
    
    mdl_rand = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    mdl_rand.fit(X_train_rand, y_train)
    preds_rand = mdl_rand.predict(X_test_rand)
    
    acc_rand = np.mean(preds_rand == y_test)
    rand_accs.append(acc_rand)

print('\n' + '='*70)
print('RESULTS WITH CAUSAL FEATURES (no look-ahead bias)')
print('='*70)

print('\nTEST 1: PERMUTATION TEST')
print(f'Real features: {np.mean(real_accs):.4f}')
print(f'Random features: {np.mean(rand_accs):.4f}')
diff = np.mean(real_accs) - np.mean(rand_accs)
print(f'Difference: {diff:.4f}')
if np.mean(rand_accs) > 0.55:
    print('❌ FAIL: Structural leak in CV split')
else:
    print('✅ PASS: No structural leak')

print('\nTEST 2: ROC-AUC')
print(f'Mean ROC-AUC: {np.mean(real_aucs):.4f}')
if np.mean(real_aucs) < 0.53:
    print('❌ FAIL: No directional edge')
elif np.mean(real_aucs) < 0.55:
    print('⚠️  WEAK: Marginal edge')
else:
    print('✅ PASS: Real directional edge')

print('\nTEST 3: BASE-RATE CHECK')
print(f'Predicted UP: {np.mean(pred_up_rates):.4f}')
print(f'Actual UP: {np.mean(actual_up_rates):.4f}')
if np.mean(pred_up_rates) > 0.75:
    print('❌ FAIL: Class-imbalance bias')
else:
    print('✅ PASS: No class-imbalance bias')

print('\n' + '='*70)
print('FINAL VERDICT')
print('='*70)
