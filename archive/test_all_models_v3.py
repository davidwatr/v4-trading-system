import pickle, numpy as np, warnings, os
warnings.filterwarnings('ignore')

from production_v18 import fetch_data, UNIVERSE

# Load all models
models = {}
for f in sorted(os.listdir('.')):
    if f.endswith('.pkl'):
        try:
            m = pickle.load(open(f, 'rb'))
            models[f] = m
        except Exception as e:
            print(f'{f}: error loading: {e}')

print(f'Loaded {len(models)} models\n')

# Fetch data
print('Fetching data...')
all_data = {}
for sym in UNIVERSE[:20]:
    d = fetch_data(sym, days=500)
    if d is not None and len(d[0]) > 200:
        all_data[sym] = d
print(f'Loaded {len(all_data)} symbols\n')

def compute_features(closes, volumes, highs, lows):
    """Compute all features needed by any model"""
    n = len(closes)
    feats = {}
    
    # RSI
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    feats['rsi'] = rsi
    feats['rsi_zscore'] = (rsi - np.mean(rsi)) / (np.std(rsi) + 1e-10)
    
    # Vol ratio
    vol_ma20 = np.convolve(volumes, np.ones(20)/20, mode='same')
    feats['vol_ratio'] = volumes / (vol_ma20 + 1e-10)
    
    # ATR ratio
    tr = highs - lows
    atr = np.convolve(tr, np.ones(14)/14, mode='same')
    feats['atr'] = atr
    feats['atr_ratio'] = atr / (closes + 1e-10)
    feats['atr_normalized'] = atr / (closes + 1e-10)
    
    # Momentum z-score
    momentum = closes - np.roll(closes, 20)
    feats['momentum_zscore'] = (momentum - np.mean(momentum)) / (np.std(momentum) + 1e-10)
    
    # Spread z-score (high-low range / close)
    spreads = highs - lows
    feats['spread_zscore'] = (spreads - np.mean(spreads)) / (np.std(spreads) + 1e-10)
    
    # Other features
    ma20 = np.convolve(closes, np.ones(20)/20, mode='same')
    ma50 = np.convolve(closes, np.ones(50)/50, mode='same')
    ma200 = np.convolve(closes, np.ones(200)/200, mode='same')
    feats['price_vs_ma20'] = closes / (ma20 + 1e-10)
    feats['price_vs_ma50'] = closes / (ma50 + 1e-10)
    feats['price_vs_ma200'] = closes / (ma200 + 1e-10)
    feats['rvol'] = volumes / (vol_ma20 + 1e-10)
    
    # BB
    bb_mid = np.convolve(closes, np.ones(20)/20, mode='same')
    bb_std = np.array([np.std(closes[max(0,i-20):i+1]) for i in range(n)])
    feats['bb_upper'] = bb_mid + 2*bb_std
    feats['bb_lower'] = bb_mid - 2*bb_std
    feats['bb_pos'] = (closes - feats['bb_lower']) / (feats['bb_upper'] - feats['bb_lower'] + 1e-10)
    
    return feats

# Test each model
results = []
print('='*70)
print('PROPER OOS TEST (chronological 80/20 split)')
print('='*70 + '\n')

for name, m in sorted(models.items()):
    if not isinstance(m, dict) or 'model' not in m:
        continue
    
    model = m['model']
    features = m.get('feature_names', [])
    stored_acc = m.get('accuracy', None)
    
    # Check for leakage
    has_leaky = 'tb_label' in features or 'current_price' in features
    
    print(f'{name}:')
    print(f'  Type: {type(model).__name__}')
    print(f'  Stored accuracy: {stored_acc:.4f}' if stored_acc else '  Stored accuracy: N/A')
    print(f'  Features: {len(features)}')
    
    if has_leaky:
        leaky_list = [f for f in features if f in ['tb_label', 'current_price']]
        print(f'  ⚠️  LEAKY FEATURES: {leaky_list}')
        clean_feats = [f for f in features if f not in ['tb_label', 'current_price']]
        print(f'  Testing WITHOUT leakage ({len(clean_feats)} features)...')
    else:
        clean_feats = features
        print(f'  Clean features: {len(clean_feats)}')
    
    # Build feature matrix
    all_X, all_y = [], []
    
    for sym, data in all_data.items():
        closes = np.array(data[0])
        volumes = np.array(data[1])
        highs = np.array(data[2])
        lows = np.array(data[3])
        
        feats = compute_features(closes, volumes, highs, lows)
        
        X_list = []
        for f in clean_feats:
            if f in feats:
                X_list.append(feats[f])
            else:
                X_list.append(np.zeros_like(closes))
        
        X = np.column_stack(X_list)
        y = (np.roll(closes, -1) > closes).astype(int)
        
        # Remove NaN/Inf
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
        X = X[valid]
        y = y[valid]
        
        if len(X) < 50:
            continue
        
        all_X.append(X)
        all_y.append(y)
    
    if not all_X:
        print(f'  SKIP: no valid data\n')
        continue
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # Chronological split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    try:
        preds = model.predict(X_test)
        accuracy = np.mean(preds == y_test)
        print(f'  OOS Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
        print(f'  Samples: {len(y_test)}')
        results.append((name, accuracy, has_leaky, stored_acc or 0.5))
    except Exception as e:
        print(f'  Error: {e}')
    print()

print('\n' + '='*70)
print('FINAL SUMMARY')
print('='*70)
print(f'{"Model":<28} {"Claimed":<10} {"Actual OOS":<12} {"Status":<20}')
print('-'*70)
for name, acc, leaky, claimed in sorted(results, key=lambda x: x[1] if not x[2] else 0, reverse=True):
    status = 'LEAKY - WORTHLESS' if leaky else ('COIN FLIP' if abs(acc - 0.5) < 0.03 else 'HAS EDGE')
    claimed_str = f'{claimed*100:.1f}%'
    oos_str = f'{acc*100:.2f}%' if not leaky else '~50%'
    print(f'{name:<28} {claimed_str:<10} {oos_str:<12} {status}')
