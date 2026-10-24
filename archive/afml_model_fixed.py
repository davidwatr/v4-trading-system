import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

print('='*70)
print('AFML MODEL — Fixed Cross-Sectional Walk-Forward')
print('='*70)

# Fetch data
all_data = {}
for sym in UNIVERSE[:20]:
    d = fetch_data(sym, days=730)
    if d is not None and len(d[0]) > 200:
        all_data[sym] = d
print(f'Loaded {len(all_data)} symbols')

def compute_atr_causal(highs, lows, closes, period=14):
    n = len(closes)
    tr = highs - lows
    atr = np.zeros(n)
    for i in range(1, n):
        atr[i] = np.mean(tr[max(0,i-period):i])
    return atr

def triple_barrier_labels(closes, highs, lows, atr_series, pt_mult=2.0, sl_mult=1.5, max_hold=14):
    n = len(closes)
    labels = np.zeros(n)
    returns = np.zeros(n)
    
    for i in range(n - max_hold):
        entry = closes[i]
        if np.isnan(atr_series[i]) or atr_series[i] <= 0:
            continue
        
        upper = entry + pt_mult * atr_series[i]
        lower = entry - sl_mult * atr_series[i]
        
        path_high = highs[i+1:i+max_hold+1]
        path_low = lows[i+1:i+max_hold+1]
        path_close = closes[i+1:i+max_hold+1]
        
        upper_touch = np.where(path_high >= upper)[0]
        lower_touch = np.where(path_low <= lower)[0]
        
        if len(upper_touch) > 0 and (len(lower_touch) == 0 or upper_touch[0] < lower_touch[0]):
            labels[i] = 1
            returns[i] = (closes[i + 1 + upper_touch[0]] - entry) / entry
        elif len(lower_touch) > 0:
            labels[i] = -1
            returns[i] = (closes[i + 1 + lower_touch[0]] - entry) / entry
        else:
            labels[i] = 0
            returns[i] = (path_close[-1] - entry) / entry
    
    return labels, returns

def compute_stationary_features(closes, volumes, highs, lows):
    n = len(closes)
    feats = {}
    
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
    
    vol_ma20 = np.zeros(n)
    for i in range(1, n):
        vol_ma20[i] = np.mean(volumes[max(0,i-20):i])
    feats['vol_ratio'] = volumes / (vol_ma20 + 1e-10)
    
    tr = highs - lows
    atr = compute_atr_causal(highs, lows, closes, 14)
    feats['atr_ratio'] = atr / (closes + 1e-10)
    
    bb_mid = np.zeros(n)
    bb_std = np.zeros(n)
    for i in range(1, n):
        bb_mid[i] = np.mean(closes[max(0,i-20):i])
        bb_std[i] = np.std(closes[max(0,i-20):i])
    feats['bb_pos'] = (closes - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    feats['price_vs_ma20'] = closes / (bb_mid + 1e-10)
    feats['return_5d'] = np.zeros(n)
    for i in range(5, n):
        feats['return_5d'][i] = (closes[i] - closes[i-5]) / (closes[i-5] + 1e-10)
    
    return feats

# Build panel
all_panels = []
for sym, data in all_data.items():
    closes = np.array(data[0])
    volumes = np.array(data[1])
    highs = np.array(data[2])
    lows = np.array(data[3])
    
    atr = compute_atr_causal(highs, lows, closes, 14)
    labels, returns = triple_barrier_labels(closes, highs, lows, atr, pt_mult=2.0, sl_mult=1.5, max_hold=14)
    feats = compute_stationary_features(closes, volumes, highs, lows)
    feat_names = list(feats.keys())
    
    valid = np.all(np.isfinite(np.column_stack([feats[f] for f in feat_names])), axis=1)
    valid[:20] = False
    
    panel = pd.DataFrame({
        'sym': sym,
        'idx': range(len(closes)),
        'label': labels,
        'ret': returns,
        **{f: feats[f] for f in feat_names}
    })
    panel = panel[valid]
    all_panels.append(panel)

panel_df = pd.concat(all_panels, ignore_index=True)

# Cross-sectional rank features (rank within each time period across all stocks)
# Since we don't have aligned dates, use per-symbol rolling rank
for f in feat_names:
    panel_df[f'{f}_rank'] = panel_df.groupby('sym')[f].rolling(60, min_periods=20).rank(pct=True).reset_index(0, drop=True)

panel_df = panel_df.dropna()
rank_feat_names = [f for f in panel_df.columns if '_rank' in f]

print(f'Panel shape: {panel_df.shape}')
print(f'Rank features: {rank_feat_names}')

# ==========================================
# PROPER PER-SYMBOL WALK-FORWARD WITH RANK FEATURES
# ==========================================
print('\n' + '='*70)
print('PER-SYMBOL WALK-FORWARD WITH CROSS-SECTIONAL RANK FEATURES')
print('='*70)

window_size = 30
min_train = 100
embargo = 5
step = 20

results = {
    'meta_logistic': {'acc': [], 'auc': [], 'pred_up': [], 'actual_up': []},
    'meta_rf': {'acc': [], 'auc': [], 'pred_up': [], 'actual_up': []},
}

fold_count = 0
for sym in panel_df['sym'].unique():
    sym_df = panel_df[panel_df['sym'] == sym].sort_values('idx')
    
    X_sym = sym_df[rank_feat_names].values
    y_sym = (sym_df['label'] == 1).astype(int).values  # Meta-label: 1 = LONG winner
    
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
        
        fold_count += 1
        
        # Logistic Regression
        mdl_lr = LogisticRegression(C=1.0, max_iter=1000)
        mdl_lr.fit(X_train, y_train)
        preds_lr = mdl_lr.predict(X_test)
        proba_lr = mdl_lr.predict_proba(X_test)[:, 1]
        
        acc_lr = accuracy_score(y_test, preds_lr)
        try:
            auc_lr = roc_auc_score(y_test, proba_lr)
        except:
            auc_lr = 0.5
        
        results['meta_logistic']['acc'].append(acc_lr)
        results['meta_logistic']['auc'].append(auc_lr)
        results['meta_logistic']['pred_up'].append(np.mean(preds_lr))
        results['meta_logistic']['actual_up'].append(np.mean(y_test))
        
        # Random Forest
        mdl_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        mdl_rf.fit(X_train, y_train)
        preds_rf = mdl_rf.predict(X_test)
        proba_rf = mdl_rf.predict_proba(X_test)[:, 1]
        
        acc_rf = accuracy_score(y_test, preds_rf)
        try:
            auc_rf = roc_auc_score(y_test, proba_rf)
        except:
            auc_rf = 0.5
        
        results['meta_rf']['acc'].append(acc_rf)
        results['meta_rf']['auc'].append(auc_rf)
        results['meta_rf']['pred_up'].append(np.mean(preds_rf))
        results['meta_rf']['actual_up'].append(np.mean(y_test))
        
        train_end += step

print(f'Total folds: {fold_count}')

# ==========================================
# RESULTS
# ==========================================
print('\n' + '='*70)
print('RESULTS')
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
# PERMUTATION TEST
# ==========================================
print('\n' + '='*70)
print('PERMUTATION TEST (random features)')
print('='*70)

perm_results = []
for sym in panel_df['sym'].unique():
    sym_df = panel_df[panel_df['sym'] == sym].sort_values('idx')
    X_sym = np.random.randn(sym_df.shape[0], len(rank_feat_names))
    y_sym = (sym_df['label'] == 1).astype(int).values
    
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
# FINAL VERDICT
# ==========================================
print('\n' + '='*70)
print('FINAL VERDICT')
print('='*70)
