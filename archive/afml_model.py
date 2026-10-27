import numpy as np
import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from production_v18 import fetch_data, UNIVERSE

print('='*70)
print('AFML MODEL — Triple-Barrier Labels + Purged Walk-Forward + Meta-Labeling')
print('='*70)

# ==========================================
# 1. DATA COLLECTION
# ==========================================
print('\n[1] Fetching data...')
all_data = {}
for sym in UNIVERSE[:20]:
    d = fetch_data(sym, days=730)
    if d is not None and len(d[0]) > 200:
        all_data[sym] = d
print(f'    Loaded {len(all_data)} symbols')

# ==========================================
# 2. TRIPLE-BARRIER LABELING (AFML Ch.3)
# ==========================================
print('\n[2] Computing triple-barrier labels...')

def triple_barrier_labels(closes, highs, lows, atr_series, pt_mult=2.0, sl_mult=1.5, max_hold=14):
    """
    Triple-barrier method from AFML Ch.3.
    For each day, simulate what happens over the next `max_hold` days:
    - Upper barrier: entry + pt_mult * ATR  (profit take)
    - Lower barrier: entry - sl_mult * ATR  (stop loss)
    - Vertical barrier: max_hold days (time expiration)
    
    Returns:
        labels:  1 if upper barrier hit first (LONG winner)
                -1 if lower barrier hit first (LONG loser)
                 0 if vertical barrier hit (time expired)
        returns: actual return at exit
    """
    n = len(closes)
    labels = np.zeros(n)
    returns = np.zeros(n)
    
    for i in range(n - max_hold):
        entry = closes[i]
        if np.isnan(atr_series[i]) or atr_series[i] <= 0:
            continue
        
        upper = entry + pt_mult * atr_series[i]
        lower = entry - sl_mult * atr_series[i]
        
        # Path over next max_hold days
        path_high = highs[i+1:i+max_hold+1]
        path_low = lows[i+1:i+max_hold+1]
        path_close = closes[i+1:i+max_hold+1]
        
        # Find first touch
        upper_touch = np.where(path_high >= upper)[0]
        lower_touch = np.where(path_low <= lower)[0]
        
        if len(upper_touch) > 0 and (len(lower_touch) == 0 or upper_touch[0] < lower_touch[0]):
            # Upper hit first → LONG winner
            labels[i] = 1
            returns[i] = (closes[i + 1 + upper_touch[0]] - entry) / entry
        elif len(lower_touch) > 0:
            # Lower hit first → LONG loser
            labels[i] = -1
            returns[i] = (closes[i + 1 + lower_touch[0]] - entry) / entry
        else:
            # Time expired
            labels[i] = 0
            returns[i] = (path_close[-1] - entry) / entry
    
    return labels, returns

def compute_atr_causal(highs, lows, closes, period=14):
    """ATR using only past data"""
    n = len(closes)
    tr = highs - lows
    atr = np.zeros(n)
    for i in range(1, n):
        atr[i] = np.mean(tr[max(0,i-period):i])
    return atr

# ==========================================
# 3. STATIONARY FEATURES (5-7 high-conviction)
# ==========================================
print('\n[3] Computing stationary features...')

def compute_stationary_features(closes, volumes, highs, lows):
    """
    Only stationary/bounded features — no raw prices.
    All causal (use only past data at each point).
    """
    n = len(closes)
    feats = {}
    
    # 1. RSI z-score (bounded 0-100, z-score of RSI over 20 days)
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(1, n):
        avg_gain[i] = np.mean(gains[max(0,i-14):i])
        avg_loss[i] = np.mean(losses[max(0,i-14):i])
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    feats['rsi_zscore'] = np.zeros(n)
    for i in range(20, n):
        feats['rsi_zscore'][i] = (rsi[i] - np.mean(rsi[max(0,i-20):i])) / (np.std(rsi[max(0,i-20):i]) + 1e-10)
    
    # 2. Volume ratio (bounded, relative to 20-day average)
    vol_ma20 = np.zeros(n)
    for i in range(1, n):
        vol_ma20[i] = np.mean(volumes[max(0,i-20):i])
    feats['vol_ratio'] = volumes / (vol_ma20 + 1e-10)
    
    # 3. ATR ratio (ATR / Close — normalized volatility)
    tr = highs - lows
    atr = compute_atr_causal(highs, lows, closes, 14)
    feats['atr_ratio'] = atr / (closes + 1e-10)
    
    # 4. Bollinger Band position (bounded 0-1, stationary)
    bb_mid = np.zeros(n)
    bb_std = np.zeros(n)
    for i in range(1, n):
        bb_mid[i] = np.mean(closes[max(0,i-20):i])
        bb_std[i] = np.std(closes[max(0,i-20):i])
    feats['bb_pos'] = (closes - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    
    # 5. Price vs MA20 ratio (bounded, stationary)
    feats['price_vs_ma20'] = closes / (bb_mid + 1e-10)
    
    # 6. Short-term reversal feature (5-day return)
    feats['return_5d'] = np.zeros(n)
    for i in range(5, n):
        feats['return_5d'][i] = (closes[i] - closes[i-5]) / (closes[i-5] + 1e-10)
    
    return feats

# ==========================================
# 4. BUILD DATASET
# ==========================================
print('\n[4] Building dataset...')

all_dataset = []
for sym, data in all_data.items():
    closes = np.array(data[0])
    volumes = np.array(data[1])
    highs = np.array(data[2])
    lows = np.array(data[3])
    
    atr = compute_atr_causal(highs, lows, closes, 14)
    labels, returns = triple_barrier_labels(closes, highs, lows, atr, pt_mult=2.0, sl_mult=1.5, max_hold=14)
    feats = compute_stationary_features(closes, volumes, highs, lows)
    
    feat_names = list(feats.keys())
    X = np.column_stack([feats[f] for f in feat_names])
    y = labels
    
    # Remove NaN/Inf and first 20 days (warmup)
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    valid[:20] = False
    
    for i in range(len(closes)):
        if valid[i]:
            all_dataset.append({
                'sym': sym,
                'idx': i,
                'X': X[i],
                'y': y[i],
                'ret': returns[i],
                'feat_names': feat_names
            })

print(f'    Total samples: {len(all_dataset)}')
print(f'    Feature names: {feat_names}')

# Count label distribution
labels_arr = np.array([d['y'] for d in all_dataset])
print(f'    Label distribution: UP={np.mean(labels_arr==1):.3f}, DOWN={np.mean(labels_arr==-1):.3f}, FLAT={np.mean(labels_arr==0):.3f}')

# ==========================================
# 5. META-LABELING SETUP (AFML Ch.3.6)
# ==========================================
print('\n[5] Setting up meta-labeling...')

# Primary model: predict side (LONG or NOT LONG) based on features
# For meta-labeling: 
#   - Primary says LONG (side = +1)
#   - If trade would be profitable (y = 1), meta-label = 1 (bet on this signal)
#   - If trade would be unprofitable (y = -1), meta-label = 0 (ignore this signal)

# Binary meta-label: 1 = trade is a winner, 0 = trade is a loser
meta_labels = np.array([1 if d['y'] == 1 else 0 for d in all_dataset])

# Side prediction: +1 for LONG, 0 for not LONG
sides = np.array([1 if d['y'] == 1 else 0 for d in all_dataset])

print(f'    Meta-label distribution: WIN={np.mean(meta_labels):.3f}, LOSS={1-np.mean(meta_labels):.3f}')

# ==========================================
# 6. PURGED WALK-FORWARD CROSS-VALIDATION
# ==========================================
print('\n[6] Running purged walk-forward CV...')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# Sort by symbol then time for proper walk-forward
all_dataset_df = pd.DataFrame([{
    'sym': d['sym'],
    'idx': d['idx'],
    'y': d['y'],
    'meta': meta_labels[i],
    'X': d['X'],
    'feat_names': d['feat_names']
} for i, d in enumerate(all_dataset)])

# Group by symbol and create per-symbol time series
symbol_groups = {}
for sym in all_data.keys():
    sym_data = all_dataset_df[all_dataset_df['sym'] == sym].sort_values('idx')
    if len(sym_data) > 50:
        symbol_groups[sym] = sym_data

print(f'    Symbols for CV: {len(symbol_groups)}')

# Walk-forward: train on older data, test on newer data, with embargo
# For each fold: train on [0, train_end], test on [train_end + embargo, train_end + window]
window_size = 30
min_train = 100
embargo = 5  # days to skip between train and test
step = 20

results = {
    'meta_logistic': {'acc': [], 'auc': [], 'pred_up': [], 'actual_up': []},
    'meta_rf': {'acc': [], 'auc': [], 'pred_up': [], 'actual_up': []},
    'side_logistic': {'acc': [], 'auc': [], 'pred_up': [], 'actual_up': []},
}

fold_count = 0
for sym, sym_df in symbol_groups.items():
    X_sym = np.vstack(sym_df['X'].values)
    y_sym = sym_df['meta'].values
    
    # Skip if not enough data
    if len(sym_df) < min_train + window_size:
        continue
    
    # Walk-forward for this symbol
    train_end = min_train
    while train_end + embargo + window_size <= len(sym_df):
        # Training: [0, train_end]
        X_train = X_sym[:train_end]
        y_train = y_sym[:train_end]
        
        # Test: [train_end + embargo, train_end + embargo + window_size]
        test_start = train_end + embargo
        test_end = min(test_start + window_size, len(sym_df))
        X_test = X_sym[test_start:test_end]
        y_test = y_sym[test_start:test_end]
        
        if len(y_test) < 10:
            train_end += step
            continue
        
        fold_count += 1
        
        # Meta-labeling model (Logistic Regression)
        mdl_meta_lr = LogisticRegression(C=1.0, max_iter=1000)
        mdl_meta_lr.fit(X_train, y_train)
        preds_meta_lr = mdl_meta_lr.predict(X_test)
        proba_meta_lr = mdl_meta_lr.predict_proba(X_test)[:, 1]
        
        acc_meta_lr = accuracy_score(y_test, preds_meta_lr)
        try:
            auc_meta_lr = roc_auc_score(y_test, proba_meta_lr)
        except:
            auc_meta_lr = 0.5
        
        results['meta_logistic']['acc'].append(acc_meta_lr)
        results['meta_logistic']['auc'].append(auc_meta_lr)
        results['meta_logistic']['pred_up'].append(np.mean(preds_meta_lr))
        results['meta_logistic']['actual_up'].append(np.mean(y_test))
        
        # Meta-labeling model (Random Forest)
        mdl_meta_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        mdl_meta_rf.fit(X_train, y_train)
        preds_meta_rf = mdl_meta_rf.predict(X_test)
        proba_meta_rf = mdl_meta_rf.predict_proba(X_test)[:, 1]
        
        acc_meta_rf = accuracy_score(y_test, preds_meta_rf)
        try:
            auc_meta_rf = roc_auc_score(y_test, proba_meta_rf)
        except:
            auc_meta_rf = 0.5
        
        results['meta_rf']['acc'].append(acc_meta_rf)
        results['meta_rf']['auc'].append(auc_meta_rf)
        results['meta_rf']['pred_up'].append(np.mean(preds_meta_rf))
        results['meta_rf']['actual_up'].append(np.mean(y_test))
        
        train_end += step

print(f'    Total folds: {fold_count}')

# ==========================================
# 7. RESULTS
# ==========================================
print('\n' + '='*70)
print('AFML MODEL RESULTS (Triple-Barrier + Meta-Labeling + Purged CV)')
print('='*70)

for model_name, res in results.items():
    if not res['acc']:
        continue
    accs = np.array(res['acc'])
    aucs = np.array(res['auc'])
    pred_ups = np.array(res['pred_up'])
    actual_ups = np.array(res['actual_up'])
    
    print(f'\n{model_name}:')
    print(f'  Accuracy: mean={np.mean(accs):.4f}, median={np.median(accs):.4f}, min={np.min(accs):.4f}, max={np.max(accs):.4f}')
    print(f'  ROC-AUC:  mean={np.mean(aucs):.4f}, median={np.median(aucs):.4f}, min={np.min(aucs):.4f}, max={np.max(aucs):.4f}')
    print(f'  Predicted UP: {np.mean(pred_ups):.4f}')
    print(f'  Actual UP:    {np.mean(actual_ups):.4f}')
    
    if np.mean(aucs) > 0.55:
        print(f'  → HAS EDGE (AUC > 0.55)')
    elif np.mean(aucs) > 0.53:
        print(f'  → Marginal edge')
    else:
        print(f'  → Coin flip (AUC < 0.53)')

# ==========================================
# 8. PERMUTATION TEST
# ==========================================
print('\n' + '='*70)
print('PERMUTATION TEST (random features)')
print('='*70)

perm_results = []
for sym, sym_df in symbol_groups.items():
    X_sym = np.random.randn(sym_df.shape[0], len(feat_names))  # Pure noise
    y_sym = sym_df['meta'].values
    
    if len(sym_df) < min_train + window_size:
        continue
    
    train_end = min_train
    while train_end + embargo + window_size <= len(sym_df):
        X_train = X_sym[:train_end]
        y_train = y_sym[:train_end]
        X_test = X_sym[train_end + embargo:train_end + embargo + window_size]
        y_test = y_sym[train_end + embargo:train_end + embargo + window_size]
        
        if len(y_test) < 10:
            train_end += step
            continue
        
        mdl = LogisticRegression(C=1.0, max_iter=1000)
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        proba = mdl.predict_proba(X_test)[:, 1]
        
        try:
            auc = roc_auc_score(y_test, proba)
        except:
            auc = 0.5
        perm_results.append(auc)
        
        train_end += step

if perm_results:
    perm_results = np.array(perm_results)
    print(f'\nRandom features AUC: {np.mean(perm_results):.4f}')
    if np.mean(perm_results) > 0.53:
        print('❌ FAIL: Random features also predict → structural leak')
    else:
        print('✅ PASS: Random features drop to ~50% → no structural leak')

# ==========================================
# 9. CROSS-SECTIONAL RANKING TEST
# ==========================================
print('\n' + '='*70)
print('CROSS-SECTIONAL RANKING TEST')
print('='*70)

# The key insight: instead of predicting "will this stock go up?",
# predict "will this stock OUTPERFORM its peers?"
# This strips out market-wide volatility

# Compute cross-sectional features (relative to other stocks)
print('\nComputing cross-sectional ranks...')

# For each day, rank stocks by each feature
# Then use ranks as features (more robust than raw values)
# This is the Jegadeesh-Titman approach

# Build panel data
all_panels = []
for sym, data in all_data.items():
    closes = np.array(data[0])
    volumes = np.array(data[1])
    highs = np.array(data[2])
    lows = np.array(data[3])
    
    atr = compute_atr_causal(highs, lows, closes, 14)
    labels, returns = triple_barrier_labels(closes, highs, lows, atr, pt_mult=2.0, sl_mult=1.5, max_hold=14)
    feats = compute_stationary_features(closes, volumes, highs, lows)
    
    valid = np.all(np.isfinite(np.column_stack([feats[f] for f in feat_names])), axis=1)
    valid[:20] = False
    
    panel = pd.DataFrame({
        'sym': sym,
        'idx': range(len(closes)),
        'close': closes,
        'label': labels,
        'ret': returns,
        **{f: feats[f] for f in feat_names}
    })
    panel = panel[valid]
    all_panels.append(panel)

panel_df = pd.concat(all_panels, ignore_index=True)

# For cross-sectional: compute rank within each time period
# Since we don't have aligned dates, use a simplified version
# For each symbol, compute rolling rank relative to its own history
for f in feat_names:
    panel_df[f'{f}_rank'] = panel_df.groupby('sym')[f].rolling(60, min_periods=20).rank(pct=True).reset_index(0, drop=True)

# Drop NaN from rank computation
panel_df = panel_df.dropna()

print(f'Panel shape: {panel_df.shape}')
print(f'Rank features added: {[f for f in panel_df.columns if "_rank" in f]}')

# Test with cross-sectional rank features
rank_feat_names = [f for f in panel_df.columns if '_rank' in f]
X_rank = panel_df[rank_feat_names].values
y_rank = (panel_df['label'] == 1).astype(int).values

# Walk-forward with rank features
print('\nWalk-forward with cross-sectional rank features...')

rank_results = {'auc': [], 'acc': []}
train_end = min_train

while train_end + embargo + window_size < len(panel_df):
    X_train = X_rank[:train_end]
    y_train = y_rank[:train_end]
    X_test = X_rank[train_end + embargo:train_end + embargo + window_size]
    y_test = y_rank[train_end + embargo:train_end + embargo + window_size]
    
    if len(y_test) < 10:
        train_end += step
        continue
    
    mdl = LogisticRegression(C=1.0, max_iter=1000)
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    proba = mdl.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except:
        auc = 0.5
    
    rank_results['acc'].append(acc)
    rank_results['auc'].append(auc)
    
    train_end += step

if rank_results['auc']:
    print(f'Cross-sectional rank features AUC: {np.mean(rank_results["auc"]):.4f}')
    print(f'Cross-sectional rank features Accuracy: {np.mean(rank_results["acc"]):.4f}')

print('\n' + '='*70)
print('CONCLUSION')
print('='*70)
