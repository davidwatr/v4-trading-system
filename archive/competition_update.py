#!/usr/bin/env python3
"""Daily Competition Update for Paper Trading Tournament — 5 STOCKS PER MODEL"""

import json
import pickle
import time
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, '.')
from production_v18 import fetch_data, extract_features, get_market_regime, UNIVERSE

# ============================================
# CONFIGURATION
# ============================================
MODELS = {
    'v3_ml': 'ml_model.pkl',
    'v4_xgb': 'ml_model_xgb.pkl',
    'v5_proper': 'ml_model_proper.pkl',
    'v6_ridge': 'ml_model_ridge.pkl',
}

STARTING_CAPITAL = 20.0
POSITIONS_PER_MODEL = 5
CAPITAL_PER_POSITION = STARTING_CAPITAL / POSITIONS_PER_MODEL  # $4 each
STOP_LOSS_PCT = -0.08
TAKE_PROFIT_PCT = 0.15
MAX_HOLD_DAYS = 14

# Current competition state — 5 slots per model
# Slots: 0-4, each with holding, entry, days_held
competition_state = {
    'v3_ml': [
        {'holding': 'PYPL', 'entry': 52.67, 'days_held': 3},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
    ],
    'v4_xgb': [
        {'holding': 'PYPL', 'entry': 52.67, 'days_held': 3},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
    ],
    'v5_proper': [
        {'holding': 'PYPL', 'entry': 52.67, 'days_held': 3},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
    ],
    'v6_ridge': [
        {'holding': 'PYPL', 'entry': 52.67, 'days_held': 3},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
    ],
    'v18_heuristic': [
        {'holding': 'INTC', 'entry': 104.47, 'days_held': 1},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
        {'holding': None, 'entry': None, 'days_held': 0},
    ],
}

# Load all models
print("=" * 70)
print(f"  COMPETITION UPDATE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

loaded_models = {}
for name, path in MODELS.items():
    with open(path, 'rb') as f:
        d = pickle.load(f)
    loaded_models[name] = d
    leaky = 'tb_label' in d['feature_names'] or 'current_price' in d['feature_names']
    print(f"  {name}: {len(d['feature_names'])} features {'⚠️ LEAKY' if leaky else '✓ clean'}")

# Fetch market data
spy_closes, _, spy_highs, spy_lows = fetch_data('SPY', days=730)
regime, *_ = get_market_regime(spy_closes, spy_highs, spy_lows)
print(f"\n  Market Regime: {regime.upper()}")
print(f"  SPY: ${spy_closes[-1]:.2f}")

# Get all prices
print("\n  Fetching prices...")
prices = {}
for symbol in UNIVERSE:
    try:
        data = fetch_data(symbol, days=300)
        if data and data[0]:
            prices[symbol] = data[0][-1]
    except:
        pass
    time.sleep(0.05)
print(f"  Got {len(prices)} prices")

# ============================================
# RUN ALL MODELS
# ============================================
def run_ml_model(model_name, model_dict, universe):
    """Run an ML model and return sorted results"""
    model = model_dict['model']
    features = model_dict['feature_names']
    results = []
    
    for symbol in universe:
        if symbol not in prices:
            continue
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=730)
            if not closes or len(closes) < 200:
                continue
            sector_etf = {'AAPL':'XLK','MSFT':'XLK','GOOGL':'XLK','NVDA':'XLK','META':'XLK',
                          'INTC':'XLK','AMD':'XLK','CSCO':'XLK','ORCL':'XLK','CRM':'XLK','ADBE':'XLK',
                          'JPM':'XLF','BAC':'XLF','GS':'XLF','MS':'XLF','WFC':'XLF','C':'XLF',
                          'BRK-B':'XLF','V':'XLF','MA':'XLF','PYPL':'XLF',
                          'AMZN':'XLY','TSLA':'XLY','DIS':'XLY','NFLX':'XLY'}.get(symbol, 'SPY')
            sector_closes, _, _, _ = fetch_data(sector_etf, days=730)
            if not sector_closes:
                sector_closes = spy_closes
            feat = extract_features(closes, volumes, highs, lows, symbol, spy_closes, sector_closes)
            if feat is None:
                continue
            X = [[feat[name] for name in features]]
            prob = float(model.predict_proba(X)[0][1])
            results.append({
                'symbol': symbol,
                'price': round(float(feat['current_price']), 2),
                'probability': round(float(prob), 4),
                'score': round(float(prob - 0.5), 4)
            })
        except Exception as e:
            continue
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results


def run_ridge_model(model_dict, universe):
    """Run the Ridge model with its custom features"""
    model = model_dict['model']
    features = model_dict['feature_names']
    results = []
    
    for symbol in universe:
        if symbol not in prices:
            continue
        try:
            data = fetch_data(symbol, days=500)
            if data is None or len(data[0]) < 200:
                continue
            closes = np.array(data[0])
            volumes = np.array(data[1])
            highs = np.array(data[2])
            lows = np.array(data[3])
            
            n = len(closes)
            deltas = np.zeros(n)
            deltas[1:] = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.convolve(gains, np.ones(14)/14, mode='same')
            avg_loss = np.convolve(losses, np.ones(14)/14, mode='same')
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            vol_ma20 = np.convolve(volumes, np.ones(20)/20, mode='same')
            tr = highs - lows
            atr = np.convolve(tr, np.ones(14)/14, mode='same')
            momentum = closes - np.roll(closes, 20)
            spreads = highs - lows
            
            feats = {
                'rsi_zscore': (rsi[-1] - np.mean(rsi)) / (np.std(rsi) + 1e-10),
                'vol_ratio': volumes[-1] / (vol_ma20[-1] + 1e-10),
                'atr_ratio': atr[-1] / (closes[-1] + 1e-10),
                'momentum_zscore': (momentum[-1] - np.mean(momentum)) / (np.std(momentum) + 1e-10),
                'spread_zscore': (spreads[-1] - np.mean(spreads)) / (np.std(spreads) + 1e-10),
            }
            
            X = np.array([[feats[fn] for fn in features]])
            raw_pred = float(model.predict(X)[0])
            prob = 1 / (1 + np.exp(-raw_pred))
            
            results.append({
                'symbol': symbol,
                'price': round(float(closes[-1]), 2),
                'probability': round(float(prob), 4),
                'score': round(float(prob - 0.5), 4)
            })
        except Exception as e:
            continue
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results


# Run all models
all_results = {}
for name, model_dict in loaded_models.items():
    print(f"\n  Running {name}...")
    if name == 'v6_ridge':
        results = run_ridge_model(model_dict, UNIVERSE)
    else:
        results = run_ml_model(name, model_dict, UNIVERSE)
    all_results[name] = results
    if results:
        top = results[0]
        print(f"    Top: {top['symbol']} ${top['price']:.2f} | P={top['probability']:.2%}")

# ============================================
# CHECK CURRENT HOLDINGS (SL/TP) — 5 per model
# ============================================
print("\n" + "=" * 70)
print("  CHECKING CURRENT HOLDINGS (5 PER MODEL)")
print("=" * 70)

def get_held_symbols(slots):
    """Get set of currently held symbols"""
    return {s['holding'] for s in slots if s['holding']}

def check_sl_tp(slot, current_price):
    """Check if a single position hits SL/TP/max-hold"""
    entry = slot['entry']
    if not entry:
        return 'HOLD'
    return_pct = (current_price - entry) / entry
    
    if return_pct <= STOP_LOSS_PCT:
        return 'SELL_SL'
    elif return_pct >= TAKE_PROFIT_PCT:
        return 'SELL_TP'
    elif slot['days_held'] >= MAX_HOLD_DAYS:
        return 'SELL_MAXHOLD'
    return 'HOLD'

for model_name, slots in competition_state.items():
    held = get_held_symbols(slots)
    print(f"\n  {model_name}:")
    
    for i, slot in enumerate(slots):
        if not slot['holding']:
            print(f"    Slot {i+1}: EMPTY")
            continue
            
        symbol = slot['holding']
        if symbol not in prices:
            print(f"    Slot {i+1}: {symbol} — no price data")
            continue
            
        current_price = prices[symbol]
        entry = slot['entry']
        return_pct = (current_price - entry) / entry
        action = check_sl_tp(slot, current_price)
        
        status_emoji = {'HOLD': '✅', 'SELL_SL': '🛑', 'SELL_TP': '🎯', 'SELL_MAXHOLD': '⏰'}
        print(f"    Slot {i+1}: {symbol} @ ${entry:.2f} → ${current_price:.2f} ({return_pct:+.2%}) {status_emoji.get(action, '')} {action}")
        
        slot['current_price'] = current_price
        slot['return_pct'] = return_pct
        slot['action'] = action

# ============================================
# GENERATE NEW SIGNALS & FILL EMPTY SLOTS
# ============================================
print("\n" + "=" * 70)
print("  GENERATING NEW SIGNALS & REPLACEMENTS")
print("=" * 70)

for model_name, slots in competition_state.items():
    held = get_held_symbols(slots)
    
    # Find slots that need replacement
    slots_to_fill = []
    for i, slot in enumerate(slots):
        action = slot.get('action', 'HOLD')
        if action.startswith('SELL') or not slot['holding']:
            slots_to_fill.append(i)
    
    if not slots_to_fill:
        print(f"\n  {model_name}: All slots holding, no changes")
        continue
    
    print(f"\n  {model_name}: {len(slots_to_fill)} slot(s) to fill")
    
    # Get buy signals (sorted by probability, excluding already held)
    if model_name == 'v18_heuristic':
        # Heuristic: pick top momentum stocks not already held
        candidates = []
        for symbol in UNIVERSE:
            if symbol in held or symbol not in prices:
                continue
            try:
                data = fetch_data(symbol, days=100)
                if data and len(data[0]) > 50:
                    ret_1m = (data[0][-1] - data[0][-20]) / data[0][-20]
                    candidates.append((symbol, ret_1m, prices[symbol]))
            except:
                pass
        candidates.sort(key=lambda x: x[1], reverse=True)
    else:
        # ML models: use model results
        results = all_results.get(model_name, [])
        candidates = []
        for r in results:
            if r['symbol'] not in held and r['probability'] > 0.55:
                candidates.append((r['symbol'], r['probability'], r['price']))
    
    # Fill slots
    for slot_idx in slots_to_fill:
        slot = slots[slot_idx]
        
        # First, close old position if any
        if slot['holding']:
            old_symbol = slot['holding']
            old_return = slot.get('return_pct', 0)
            print(f"    Slot {slot_idx+1}: CLOSED {old_symbol} ({old_return:+.2%})")
        
        # Find new buy signal
        if candidates:
            # Pick best candidate not already in held
            for symbol, score, price in candidates:
                if symbol not in get_held_symbols(slots):
                    slot['holding'] = symbol
                    slot['entry'] = price
                    slot['days_held'] = 0
                    slot['current_price'] = price
                    slot['return_pct'] = 0.0
                    slot['action'] = 'HOLD'
                    held.add(symbol)
                    score_label = f"mom={score:.2%}" if model_name == 'v18_heuristic' else f"P={score:.2%}"
                    print(f"    Slot {slot_idx+1}: BUY {symbol} @ ${price:.2f} ({score_label})")
                    break
            else:
                # All candidates already held
                slot['holding'] = None
                slot['entry'] = None
                slot['days_held'] = 0
                print(f"    Slot {slot_idx+1}: NO SIGNAL → CASH")
        else:
            slot['holding'] = None
            slot['entry'] = None
            slot['days_held'] = 0
            print(f"    Slot {slot_idx+1}: NO SIGNAL → CASH")

# ============================================
# UPDATE PRICES & CALCULATE PORTFOLIO VALUES
# ============================================
for model_name, slots in competition_state.items():
    for slot in slots:
        if slot['holding'] and slot['holding'] in prices:
            slot['current_price'] = prices[slot['holding']]
            slot['return_pct'] = (prices[slot['holding']] - slot['entry']) / slot['entry'] if slot['entry'] else 0
            slot['days_held'] = slot.get('days_held', 0) + 1

# ============================================
# WRITE COMPETITION FILE
# ============================================
md_content = f"""# 🏆 Paper Trading Tournament — {datetime.now().strftime('%Y-%m-%d')}

**Market Regime:** {regime.upper()} | **SPY:** ${spy_closes[-1]:.2f}
**Rules:** {POSITIONS_PER_MODEL} positions per model | ${CAPITAL_PER_POSITION:.0f} each | SL {STOP_LOSS_PCT:.0%} / TP {TAKE_PROFIT_PCT:.0%} / Max {MAX_HOLD_DAYS} days

## 📊 Leaderboard

| Model | Slot 1 | Slot 2 | Slot 3 | Slot 4 | Slot 5 | Total | Return |
|-------|--------|--------|--------|--------|--------|-------|--------|
"""

# Sort by total portfolio value
model_totals = {}
for model_name, slots in competition_state.items():
    total = 0
    for slot in slots:
        if slot['holding'] and slot['entry']:
            ret = slot.get('return_pct', 0)
            total += CAPITAL_PER_POSITION * (1 + ret)
        else:
            total += CAPITAL_PER_POSITION
    model_totals[model_name] = total

sorted_models = sorted(competition_state.items(), key=lambda x: model_totals[x[0]], reverse=True)

for model_name, slots in sorted_models:
    total = model_totals[model_name]
    total_ret = (total - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    
    slot_strs = []
    for slot in slots:
        if slot['holding']:
            ret = slot.get('return_pct', 0) * 100
            slot_strs.append(f"{slot['holding']} {ret:+.1f}%")
        else:
            slot_strs.append("CASH")
    
    md_content += f"| {model_name} | {' | '.join(slot_strs)} | ${total:.2f} | {total_ret:+.2f}% |\n"

md_content += f"""
## 📈 Today's Top Signals (All Models)

| Model | #1 | #2 | #3 | #4 | #5 |
|-------|----|----|----|----|----
"""

for name, results in all_results.items():
    top5 = results[:5]
    signals = []
    for r in top5:
        signals.append(f"{r['symbol']} {r['probability']:.1%}")
    while len(signals) < 5:
        signals.append("—")
    md_content += f"| {name} | {' | '.join(signals)} |\n"

md_content += f"""
## 📋 Rules
- Starting capital: ${STARTING_CAPITAL} (${CAPITAL_PER_POSITION:.0f} per position × {POSITIONS_PER_MODEL})
- Max {POSITIONS_PER_MODEL} concurrent positions per model
- Max hold: {MAX_HOLD_DAYS} days per position
- Stop loss: {STOP_LOSS_PCT:.0%}
- Take profit: {TAKE_PROFIT_PCT:.0%}

---
*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

# Write file
comp_path = Path('C:/Users/david_wncs0ps/OneDrive/Documents/hermes and i/Trading/competition.md')
comp_path.parent.mkdir(parents=True, exist_ok=True)
with open(comp_path, 'w') as f:
    f.write(md_content)

print(f"\n✅ Competition file written to {comp_path}")

# ============================================
# LOG TO JSONL
# ============================================
log_entry = {
    'date': datetime.now().strftime('%Y-%m-%d'),
    'time': datetime.now().strftime('%H:%M'),
    'prices': {s: round(p, 2) for s, p in prices.items()},
    'competition': {}
}

for model_name, slots in competition_state.items():
    total = model_totals[model_name]
    log_entry['competition'][model_name] = {
        'slots': [
            {
                'holding': s.get('holding'),
                'entry': s.get('entry'),
                'current': s.get('current_price'),
                'return': round(s.get('return_pct', 0) * 100, 2) if s.get('return_pct') else 0,
            }
            for s in slots
        ],
        'total_value': round(total, 2),
        'total_return': round((total - STARTING_CAPITAL) / STARTING_CAPITAL * 100, 2)
    }

with open('competition_log.jsonl', 'a') as f:
    f.write(json.dumps(log_entry) + '\n')

print("✅ Log entry appended to competition_log.jsonl")
print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)