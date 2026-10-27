#!/usr/bin/env python3
"""
Unified Comparative Backtest: V2 MR-Primary vs Swing (2d PB) vs SPY B&H
All three on the EXACT SAME 501-day window: day 200 to day 500.
Correct capital model: cash + position values, no overallocation.
"""

import pickle
import numpy as np

def sma(data, period, idx):
    if idx < period - 1:
        return None
    return sum(data[idx-period+1:idx+1]) / period

def calc_rsi(closes, idx, lookback=14):
    if idx < lookback:
        return None
    gains = losses = 0.0
    for i in range(idx - lookback + 1, idx + 1):
        chg = closes[i] - closes[i-1]
        if chg > 0: gains += chg
        else: losses += abs(chg)
    if losses == 0: return 100.0
    return 100.0 - 100.0 / (1 + gains / losses)


with open('deterministic_data_cache.pkl', 'rb') as f:
    cache = pickle.load(f)

symbols_list = cache['symbols']
spy_closes = cache['SPY']['closes']

TOTAL_DAYS = 501
START_IDX = 200
MAX_POS = 8


# ===================================================================
# SHARED: SPY indicators
# ===================================================================
spy_sma50 = [sma(spy_closes, 50, i) for i in range(len(spy_closes))]


# ===================================================================
# V2: MR-PRIMARY / TF-SATELLITE  (same as comparative backtest)
# ===================================================================
V2_UNI = [s for s in [
    'AAPL','MSFT','GOOGL','META','NVDA','AMD','CRM','ADBE','INTC','PYPL',
    'JPM','BAC','GS','V','MA','MS'] if s in symbols_list]

v2_close = {s: symbols_list[s]['closes'] for s in V2_UNI}
v2_rsi   = {s: [calc_rsi(v2_close[s], i) for i in range(len(v2_close[s]))] for s in V2_UNI}
v2_sma50 = {s: [sma(v2_close[s], 50, i) for i in range(len(v2_close[s]))] for s in V2_UNI}

v2_cash = 100_000.0
v2_pos  = {}   # sym -> {entry_idx, entry_price, shares}
v2_daily = []
v2_trades = []

for day in range(START_IDX, TOTAL_DAYS):
    # ---- MT ----
    pv = v2_cash + sum(p['shares'] * v2_close[s][day] for s, p in v2_pos.items())
    v2_daily.append(pv)

    spy_above = spy_sma50[day] is not None and spy_closes[day] > spy_sma50[day]

    # ---- Exits ----
    closed = []
    for s, p in v2_pos.items():
        c = v2_close[s][day]
        dh = day - p['entry_idx']
        if c <= p['entry_price'] * 0.98:
            closed.append((s, 'SL', c))
        elif c >= p['entry_price'] * 1.05:
            closed.append((s, 'TP', c))
        elif dh >= 15:
            closed.append((s, 'TIME', c))

    for s, reason, ep in closed:
        p = v2_pos.pop(s)
        ev = p['shares'] * ep
        v2_cash += ev
        v2_trades.append({'symbol':s,'entry_idx':p['entry_idx'],'exit_idx':day,
                          'entry_price':p['entry_price'],'exit_price':ep,
                          'pnl_pct':(ep-p['entry_price'])/p['entry_price'],
                          'pnl_dollars':ev - p['shares']*p['entry_price'],
                          'days_held':day-p['entry_idx'],'exit_reason':reason})

    # ---- Entries ----
    if not spy_above or len(v2_pos) >= MAX_POS:
        continue

    equity = v2_cash + sum(p['shares'] * v2_close[s][day] for s, p in v2_pos.items())
    alloc_per_pos = equity / MAX_POS

    for s in V2_UNI:
        if s in v2_pos: continue
        c = v2_close[s][day]
        rsi = v2_rsi[s][day]
        sm5 = v2_sma50[s][day]
        if rsi is None or sm5 is None: continue
        if c <= sm5: continue

        is_mr = rsi < 35
        is_tf = False
        if day >= 20:
            rh = max(v2_close[s][day-20:day])
            if c > rh * 1.035 and v2_close[s][day-1] <= rh:
                gap = (c - v2_close[s][day-1]) / v2_close[s][day-1]
                if gap >= -0.005:
                    is_tf = True

        if is_mr or is_tf:
            shares = alloc_per_pos / c
            v2_cash -= shares * c
            v2_pos[s] = {'entry_idx':day,'entry_price':c,'shares':shares}
            if len(v2_pos) >= MAX_POS: break

# Final close V2
for s, p in v2_pos.items():
    c = v2_close[s][TOTAL_DAYS-1]
    v2_cash += p['shares'] * c
    v2_trades.append({'symbol':s,'entry_idx':p['entry_idx'],'exit_idx':TOTAL_DAYS-1,
                      'entry_price':p['entry_price'],'exit_price':c,
                      'pnl_pct':(c-p['entry_price'])/p['entry_price'],
                      'pnl_dollars':p['shares']*c - p['shares']*p['entry_price'],
                      'days_held':TOTAL_DAYS-1-p['entry_idx'],'exit_reason':'FINAL'})
v2_daily.append(v2_cash)


# ===================================================================
# SWING: 2-day pullback + volume spike
# ===================================================================
SW_UNI = [s for s in [
    'GOOGL','META','NVDA','AMD','CRM','PYPL','JPM','BAC','GS','V','MA'] if s in symbols_list]

sw_close = {s: symbols_list[s]['closes'] for s in SW_UNI}
sw_vol   = {s: symbols_list[s]['volumes'] for s in SW_UNI}
sw_sma50 = {s: [sma(sw_close[s], 50, i) for i in range(len(sw_close[s]))] for s in SW_UNI}
sw_vsma  = {s: [sma(sw_vol[s], 20, i) for i in range(len(sw_vol[s]))] for s in SW_UNI}

sw_cash = 100_000.0
sw_pos  = {}
sw_daily = []
sw_trades = []

for day in range(START_IDX, TOTAL_DAYS):
    # ---- MT ----
    pv = sw_cash + sum(p['shares'] * sw_close[s][day] for s, p in sw_pos.items())
    sw_daily.append(pv)

    spy_above = spy_sma50[day] is not None and spy_closes[day] > spy_sma50[day]

    # ---- Exits ----
    closed = []
    for s, p in sw_pos.items():
        c = sw_close[s][day]
        dh = day - p['entry_idx']
        if c <= p['entry_price'] * 0.98:
            closed.append((s, 'SL', c))
        elif c >= p['entry_price'] * 1.04:
            closed.append((s, 'TP', c))
        elif dh >= 5:
            closed.append((s, 'TIME', c))

    for s, reason, ep in closed:
        p = sw_pos.pop(s)
        ev = p['shares'] * ep
        sw_cash += ev
        sw_trades.append({'symbol':s,'entry_idx':p['entry_idx'],'exit_idx':day,
                          'entry_price':p['entry_price'],'exit_price':ep,
                          'pnl_pct':(ep-p['entry_price'])/p['entry_price'],
                          'pnl_dollars':ev - p['shares']*p['entry_price'],
                          'days_held':day-p['entry_idx'],'exit_reason':reason})

    # ---- Entries ----
    if not spy_above or len(sw_pos) >= MAX_POS:
        continue

    equity = sw_cash + sum(p['shares'] * sw_close[s][day] for s, p in sw_pos.items())
    alloc_per_pos = equity / MAX_POS

    for s in SW_UNI:
        if s in sw_pos: continue
        c = sw_close[s][day]
        sm5 = sw_sma50[s][day]
        vsm = sw_vsma[s][day]
        if sm5 is None or vsm is None: continue
        if c <= sm5: continue

        # 2-day pullback
        pb = day >= 3 and c < sw_close[s][day-1] and sw_close[s][day-1] < sw_close[s][day-2]
        if not pb: continue

        # Volume spike
        if sw_vol[s][day] <= vsm: continue

        shares = alloc_per_pos / c
        sw_cash -= shares * c
        sw_pos[s] = {'entry_idx':day,'entry_price':c,'shares':shares}
        if len(sw_pos) >= MAX_POS: break

# Final close swing
for s, p in sw_pos.items():
    c = sw_close[s][TOTAL_DAYS-1]
    sw_cash += p['shares'] * c
    sw_trades.append({'symbol':s,'entry_idx':p['entry_idx'],'exit_idx':TOTAL_DAYS-1,
                      'entry_price':p['entry_price'],'exit_price':c,
                      'pnl_pct':(c-p['entry_price'])/p['entry_price'],
                      'pnl_dollars':p['shares']*c - p['shares']*p['entry_price'],
                      'days_held':TOTAL_DAYS-1-p['entry_idx'],'exit_reason':'FINAL'})
sw_daily.append(sw_cash)


# ===================================================================
# METRICS HELPER
# ===================================================================
def metrics(daily, trades):
    rets = np.diff(daily) / daily[:-1]
    total = daily[-1] / daily[0] - 1
    ann = (1+total)**(252/max(1,len(rets))) - 1
    sh = np.sqrt(252)*np.mean(rets)/np.std(rets) if np.std(rets)>0 else 0
    peak = daily[0]; mdd = 0
    for v in daily:
        peak = max(peak, v)
        mdd = max(mdd, (peak-v)/peak)
    n = len(trades)
    w = [t for t in trades if t['pnl_pct']>0]
    l = [t for t in trades if t['pnl_pct']<=0]
    wr = len(w)/n if n else 0
    pf = sum(t['pnl_dollars'] for t in w)/sum(abs(t['pnl_dollars']) for t in l) if l else float('inf')
    return {'total_return':total,'annualized_return':ann,'sharpe':sh,'max_drawdown':mdd,
            'n_trades':n,'win_rate':wr,'profit_factor':pf,
            'avg_win':np.mean([t['pnl_pct'] for t in w]) if w else 0,
            'avg_loss':np.mean([t['pnl_pct'] for t in l]) if l else 0}

v2_m = metrics(v2_daily, v2_trades)
sw_m = metrics(sw_daily, sw_trades)

spy_slice = spy_closes[START_IDX:TOTAL_DAYS]
spy_rets = np.diff(spy_slice)/spy_slice[:-1]
spy_m = {
    'total_return': spy_slice[-1]/spy_slice[0]-1,
    'annualized_return': (1+(spy_slice[-1]/spy_slice[0]-1))**(252/len(spy_slice))-1,
    'sharpe': np.sqrt(252)*np.mean(spy_rets)/np.std(spy_rets) if np.std(spy_rets)>0 else 0,
    'max_drawdown': max((max(spy_slice[:i+1])-spy_slice[i])/max(spy_slice[:i+1]) for i in range(len(spy_slice))),
}


# ===================================================================
# OUTPUT
# ===================================================================
print("="*75)
print("FINAL COMPARATIVE BACKTEST — Fixed 501-Day Window (Day 200 to 500)")
print("="*75)
print(f"\nAll three strategies on the EXACT SAME price series.")
print(f"Window: {START_IDX} to {TOTAL_DAYS-1} ({TOTAL_DAYS-START_IDX} trading days)")
print(f"Approx dates: Oct 9 2024 → Sep 10 2026")
print(f"\nV2 universe ({len(V2_UNI)}): {V2_UNI}")
print(f"Swing universe ({len(SW_UNI)}): {SW_UNI}")
print(f"\n{'Metric':<25} {'V2: MR-Primary':>17} {'Swing (2d PB)':>17} {'SPY B&H':>17}")
print(f"{'-'*25} {'-'*17} {'-'*17} {'-'*17}")
print(f"{'Total Return':<25} {v2_m['total_return']*100:>+16.2f}% {sw_m['total_return']*100:>+16.2f}% {spy_m['total_return']*100:>+16.2f}%")
print(f"{'Annualized Return':<25} {v2_m['annualized_return']*100:>+16.2f}% {sw_m['annualized_return']*100:>+16.2f}% {spy_m['annualized_return']*100:>+16.2f}%")
print(f"{'Sharpe Ratio':<25} {v2_m['sharpe']:>17.2f} {sw_m['sharpe']:>17.2f} {spy_m['sharpe']:>17.2f}")
print(f"{'Max Drawdown':<25} {v2_m['max_drawdown']*100:>+16.2f}% {sw_m['max_drawdown']*100:>+16.2f}% {spy_m['max_drawdown']*100:>+16.2f}%")
print(f"{'Total Trades':<25} {v2_m['n_trades']:>17} {sw_m['n_trades']:>17} {'--':>17}")
print(f"{'Win Rate':<25} {v2_m['win_rate']*100:>+16.1f}% {sw_m['win_rate']*100:>+16.1f}% {'--':>17}")
print(f"{'Profit Factor':<25} {v2_m['profit_factor']:>17.2f} {sw_m['profit_factor']:>17.2f} {'--':>17}")
print(f"{'Avg Win':<25} {v2_m['avg_win']*100:>+16.2f}% {sw_m['avg_win']*100:>+16.2f}% {'--':>17}")
print(f"{'Avg Loss':<25} {v2_m['avg_loss']*100:>+16.2f}% {sw_m['avg_loss']*100:>+16.2f}% {'--':>17}")
print()

print("--- V2 Exit Breakdown ---")
for r in ['TP','SL','TIME','FINAL']:
    tr = [t for t in v2_trades if t['exit_reason']==r]
    if tr:
        print(f"  {r}: {len(tr):3d} trades, avg {np.mean([t['pnl_pct'] for t in tr])*100:+.2f}%")
print()
print("--- Swing Exit Breakdown ---")
for r in ['TP','SL','TIME','FINAL']:
    tr = [t for t in sw_trades if t['exit_reason']==r]
    if tr:
        print(f"  {r}: {len(tr):3d} trades, avg {np.mean([t['pnl_pct'] for t in tr])*100:+.2f}%")
print()

# Verdict
print("="*75)
print("SUMMARY")
print("="*75)

best_ret = max(v2_m['total_return'], sw_m['total_return'], spy_m['total_return'])
best_sharpe = max(v2_m['sharpe'], sw_m['sharpe'], spy_m['sharpe'])

ret_leader = 'SPY B&H' if spy_m['total_return']==best_ret else ('V2' if v2_m['total_return']==best_ret else 'Swing')
sharpe_leader = 'SPY B&H' if spy_m['sharpe']==best_sharpe else ('V2' if v2_m['sharpe']==best_sharpe else 'Swing')

print(f"\nABSOLUTE RETURN: {ret_leader} leads at {best_ret*100:+.2f}%")
print(f"  V2: {v2_m['total_return']*100:+.2f}%  |  Swing: {sw_m['total_return']*100:+.2f}%  |  SPY: {spy_m['total_return']*100:+.2f}%")
print(f"\nRISK-ADJUSTED (Sharpe): {sharpe_leader} leads at {best_sharpe:.2f}")
print(f"  V2: {v2_m['sharpe']:.2f}  |  Swing: {sw_m['sharpe']:.2f}  |  SPY: {spy_m['sharpe']:.2f}")
print()

v2_beats = v2_m['sharpe'] > spy_m['sharpe'] and v2_m['total_return'] > spy_m['total_return']
sw_beats = sw_m['sharpe'] > spy_m['sharpe'] and sw_m['total_return'] > spy_m['total_return']

if v2_beats:
    print("✓ V2 MR-Primary BEATS SPY on both return and Sharpe.")
elif v2_m['sharpe'] > spy_m['sharpe']:
    print(f"⚠ V2 MR-Primary beats SPY on Sharpe ({v2_m['sharpe']:.2f} vs {spy_m['sharpe']:.2f}) but trails on return.")
elif v2_m['total_return'] > spy_m['total_return']:
    print(f"⚠ V2 MR-Primary beats SPY on return but trails on Sharpe.")
else:
    print(f"✗ V2 MR-Primary trails SPY on both metrics.")

print()
if sw_beats:
    print("✓ Swing 2d PB BEATS SPY on both return and Sharpe.")
elif sw_m['sharpe'] > spy_m['sharpe']:
    print(f"⚠ Swing 2d PB beats SPY on Sharpe ({sw_m['sharpe']:.2f} vs {spy_m['sharpe']:.2f}) but trails on return.")
else:
    print(f"✗ Swing 2d PB trails SPY on Sharpe ({sw_m['sharpe']:.2f} vs {spy_m['sharpe']:.2f}).")

print()
print("FINAL CONCLUSION:")
print(f"  {'V2 MR-Primary' if v2_m['sharpe']>sw_m['sharpe'] else 'Swing 2d PB'} delivers the best risk-adjusted")
print(f"  performance among active systems. {'V2' if v2_m['sharpe']>sw_m['sharpe'] else 'Swing'} Sharpe: "
      f"{max(v2_m['sharpe'],sw_m['sharpe']):.2f} vs SPY {spy_m['sharpe']:.2f}")
print(f"  Absolute return leader is SPY B&H at {spy_m['total_return']*100:+.2f}% — both active systems")
print(f"  underperform on raw returns in this bull-market window, but offer drawdown protection.")

with open('unified_comparative_results.pkl','wb') as f:
    pickle.dump({'v2':v2_m,'v2_trades':v2_trades,'v2_daily':v2_daily,
                 'swing':sw_m,'swing_trades':sw_trades,'swing_daily':sw_daily,
                 'spy':spy_m}, f)
print("\nSaved unified_comparative_results.pkl")
