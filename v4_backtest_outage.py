#!/usr/bin/env python3
"""Analyze IndiaMART order and AAS + blocker interaction for David's cycle plans."""
import sys, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

sys.path.insert(0, 'C:/Users/david_wncs0ps/jarvis/trading')
from production_v18 import fetch_data, UNIVERSE, calc_atr
from v4_executor import Config

cfg = Config()

START_DATE = "2026-09-18"
END_DATE = "2026-10-03"
STARTING_PORTFOLIO = 100000.0

print("=" * 60)
print(f"V4 BACKTEST: {START_DATE} to {END_DATE}")
print("=" * 60)

# Fetch all data first
print("Fetching SPY...")
spy_closes, _, spy_highs, spy_lows = fetch_data("SPY", days=365)
if not spy_closes or len(spy_closes) < 30:
    print("ERROR: Could not fetch SPY data")
    sys.exit(1)
print(f"SPY: {len(spy_closes)} days")

# Build SPY dataframe with ACTUAL dates (last N trading days)
spy_dates = pd.bdate_range(end=datetime.now(), periods=len(spy_closes))
if spy_dates[-1] > datetime.now():
    spy_dates = pd.bdate_range(end=datetime.now() - timedelta(days=1), periods=len(spy_closes))

spy_df = pd.DataFrame({
    'close': spy_closes,
    'high': spy_highs,
    'low': spy_lows
}, index=spy_dates)
spy_df = spy_df[~spy_df.index.duplicated(keep='first')]

print(f"SPY date range: {spy_df.index[0].date()} to {spy_df.index[-1].date()}")

# Filter to our period
period_start = pd.Timestamp(START_DATE)
period_end = pd.Timestamp(END_DATE)
in_period_mask = (spy_df.index >= period_start) & (spy_df.index <= period_end)
spy_in_period = spy_df[in_period_mask]

if len(spy_in_period) == 0:
    print(f"WARNING: No SPY data in {START_DATE} to {END_DATE} — checking actual date range...")
    print(f"  Looking for data from {period_start.date()} to {period_end.date()}")
    print(f"  SPY actual range: {spy_df.index[0].date()} to {spy_df.index[-1].date()}")
    spy_in_period = spy_df.iloc[-10:]
    print(f"  Using last {len(spy_in_period)} trading days instead")

print(f"Period: {len(spy_in_period)} trading days ({spy_in_period.index[0].date()} to {spy_in_period.index[-1].date()})")
print()

# Fetch universe stocks
print("Fetching universe stocks...")
stock_data_cache = {}
for symbol in UNIVERSE:
    try:
        c, v, h, l = fetch_data(symbol, days=365)
        if c and len(c) >= 60:
            stock_dates = pd.bdate_range(end=datetime.now() - timedelta(days=1), periods=len(c))
            if stock_dates[-1] > datetime.now():
                stock_dates = pd.bdate_range(end=datetime.now() - timedelta(days=1), periods=len(c))
            stock_data_cache[symbol] = {
                'close': c, 'volume': v, 'high': h, 'low': l,
                'dates': stock_dates,
            }
    except Exception:
        pass
    time.sleep(0.03)

print(f"Cached {len(stock_data_cache)} stocks")
print()

def get_stock_price(sym_dict, target_date):
    if sym_dict is None or 'dates' not in sym_dict:
        return None
    dates = sym_dict['dates']
    closes = sym_dict['close']
    idx = np.searchsorted(dates, target_date)
    if idx >= len(dates):
        idx = len(dates) - 1
    if idx < 0:
        idx = 0
    return float(closes[idx])

def get_portfolio_value(cash_val, positions_dict, current_date):
    total = cash_val
    for sym, pos in positions_dict.items():
        if sym in stock_data_cache:
            price = get_stock_price(stock_data_cache[sym], current_date)
            if price:
                total += pos['qty'] * price
    return total

cash = STARTING_PORTFOLIO
positions = {}
peak = STARTING_PORTFOLIO
trade_log = []
regime_log = []

print("=" * 60)
print("RUNNING BACKTEST")
print("=" * 60)

for idx, current_date in enumerate(spy_in_period.index):
    if current_date.weekday() >= 5:
        continue
    
    spy_window = spy_df[spy_df.index <= current_date].tail(252)
    if len(spy_window) < 30:
        continue
    
    closes_arr = np.array(spy_window['close'].values)
    rets = np.diff(closes_arr) / closes_arr[:-1]
    vol = float(np.std(rets[-20:])) * np.sqrt(252) if len(rets) >= 20 else 0.0
    
    ma20 = float(np.mean(closes_arr[-20:]))
    ma50 = float(np.mean(closes_arr[-50:])) if len(closes_arr) >= 50 else ma20
    current_spy = float(closes_arr[-1])
    
    trend = 0
    if current_spy > ma20 > ma50:
        trend = 1
    elif current_spy < ma20 < ma50:
        trend = -1
    
    if trend == 1 and vol > 0.20:
        regime = 'TRENDING_UP'
    elif trend == 1:
        regime = 'LOW_VOL'
    elif trend == -1 and vol > 0.20:
        regime = 'TRENDING_DOWN'
    elif trend == -1:
        regime = 'CRASH'
    elif vol > 0.20:
        regime = 'HIGH_VOL'
    else:
        regime = 'SIDEWAYS'
    
    confidence = min(1.0, abs(trend) * 0.5 + (vol / 0.3) * 0.5)
    regime_log.append({
        'date': str(current_date.date()),
        'regime': regime,
        'confidence': f"{confidence:.0%}",
        'spy': f"${current_spy:.2f}"
    })
    
    pv = get_portfolio_value(cash, positions, current_date)
    dd = (pv - peak) / peak if peak > 0 else 0
    if dd <= -cfg.MAX_DRAWDOWN:
        continue
    
    closed_today = []
    for sym in list(positions.keys()):
        if sym not in stock_data_cache:
            continue
        price = get_stock_price(stock_data_cache[sym], current_date)
        if price is None:
            continue
        
        entry = positions[sym]['entry']
        pnl_pct = (price - entry) / entry
        
        if pnl_pct <= -cfg.STOP_LOSS:
            cash += positions[sym]['qty'] * price
            trade_log.append({
                'date': str(current_date.date()), 'action': 'SELL (SL)',
                'symbol': sym, 'qty': positions[sym]['qty'],
                'entry': entry, 'exit': price,
                'pnl_pct': f"{pnl_pct*100:.1f}%",
                'regime': regime,
                'days_held': (current_date - positions[sym]['entry_date']).days,
            })
            closed_today.append(sym)
        elif pnl_pct >= cfg.PROFIT_TARGET:
            cash += positions[sym]['qty'] * price
            trade_log.append({
                'date': str(current_date.date()), 'action': 'SELL (TP)',
                'symbol': sym, 'qty': positions[sym]['qty'],
                'entry': entry, 'exit': price,
                'pnl_pct': f"{pnl_pct*100:.1f}%",
                'regime': regime,
                'days_held': (current_date - positions[sym]['entry_date']).days,
            })
            closed_today.append(sym)
        elif positions[sym]['entry_date'] + timedelta(days=cfg.MAX_HOLD_DAYS) <= current_date:
            cash += positions[sym]['qty'] * price
            trade_log.append({
                'date': str(current_date.date()), 'action': 'SELL (MAX HOLD)',
                'symbol': sym, 'qty': positions[sym]['qty'],
                'entry': entry, 'exit': price,
                'pnl_pct': f"{pnl_pct*100:.1f}%",
                'regime': regime,
                'days_held': (current_date - positions[sym]['entry_date']).days,
            })
            closed_today.append(sym)
    
    for sym in closed_today:
        del positions[sym]
    
    new_pv = get_portfolio_value(cash, positions, current_date)
    if new_pv > peak:
        peak = new_pv
    
    if regime in ('SIDEWAYS', 'HIGH_VOL'):
        signals_found = []
        held_symbols = list(positions.keys())
        
        for symbol in UNIVERSE:
            if symbol in held_symbols or symbol not in stock_data_cache:
                continue
            
            sdata = stock_data_cache[symbol]
            closes = sdata['close']
            highs = sdata['high']
            lows = sdata['low']
            
            if len(closes) < 50:
                continue
            
            closes_recent = closes[-250:] if len(closes) >= 250 else closes
            highs_recent = highs[-250:] if len(highs) >= 250 else highs
            lows_recent = lows[-250:] if len(lows) >= 250 else lows
            
            if len(closes_recent) < 20:
                continue
            
            ma20_s = float(np.mean(closes_recent[-20:]))
            cur_s = float(closes_recent[-1])
            
            if ma20_s == 0 or np.isnan(ma20_s):
                continue
            
            dev = (cur_s - ma20_s) / ma20_s
            
            if abs(dev) >= cfg.ENTRY_THRESHOLD:
                atr = calc_atr(highs_recent[-60:], lows_recent[-60:], closes_recent[-60:], 14)
                if atr and atr > 0:
                    signals_found.append({
                        'symbol': symbol, 'price': cur_s, 'ma20': ma20_s,
                        'deviation': dev, 'atr': atr,
                    })
        
        signals_found.sort(key=lambda x: abs(x['deviation']), reverse=True)
        signals_found = signals_found[:cfg.MAX_POSITIONS]
        
        for sig in signals_found:
            if sig['symbol'] in held_symbols or len(held_symbols) >= cfg.MAX_POSITIONS:
                continue
            
            pv_before = get_portfolio_value(cash, positions, current_date)
            raw_qty = int(pv_before * cfg.RISK_PER_TRADE / (sig['atr'] * 1.5))
            if raw_qty <= 0:
                continue
            
            qty = raw_qty
            sdata_sym = stock_data_cache[sig['symbol']]
            closes_sym = sdata_sym['close']
            if len(closes_sym) >= 20:
                closes_for_vol = closes_sym[-20:]
                r_sym = np.diff(closes_for_vol) / closes_for_vol[:-1]
                if len(r_sym) > 0:
                    cv_sym = float(pd.Series(r_sym).ewm(halflife=cfg.VOL_HALF_LIFE).std().iloc[-1])
                    if not np.isnan(cv_sym) and cv_sym > 0:
                        tv = cfg.TARGET_VOL / np.sqrt(252)
                        vr = tv / max(cv_sym, 0.005)
                        vr = np.clip(vr, 0.5, 1.5)
                        qty = int(qty * vr)
            
            max_sh = int(pv_before * cfg.MAX_POSITION_SIZE / sig['price'])
            qty = min(qty, max_sh)
            
            cost = qty * sig['price'] * (1 + cfg.TRANSACTION_COST)
            if qty > 0 and cash >= cost:
                cash -= cost
                held_symbols.append(sig['symbol'])
                positions[sig['symbol']] = {
                    'qty': qty, 'entry': sig['price'], 'entry_date': current_date,
                    'stop_loss': sig['price'] * (1 - cfg.STOP_LOSS),
                    'take_profit': sig['price'] * (1 + cfg.PROFIT_TARGET),
                }
                trade_log.append({
                    'date': str(current_date.date()), 'action': 'BUY',
                    'symbol': sig['symbol'], 'qty': qty, 'price': sig['price'],
                    'regime': regime, 'deviation': f"{sig['deviation']:.1%}",
                    'atr': f"${sig['atr']:.2f}", 'ma20': f"${sig['ma20']:.2f}",
                })
    
    if (idx + 1) % 5 == 0:
        print(f"  Day {idx+1}/{len(spy_in_period)}: {current_date.date()} | Regime: {regime:15s} | SPY: ${current_spy:.2f} | Portfolio: ${get_portfolio_value(cash, positions, current_date):,.2f}")

final_pv = get_portfolio_value(cash, positions, spy_in_period.index[-1])

print(f"\n{'='*60}")
print(f"BACKTEST RESULTS: {START_DATE} to {END_DATE}")
print(f"{'='*60}")
print(f"Trading days simulated: {len(spy_in_period)}")
print(f"Start Date: {spy_in_period.index[0].date()}")
print(f"End Date: {spy_in_period.index[-1].date()}")
print(f"Starting Portfolio: ${STARTING_PORTFOLIO:,.2f}")
print(f"Final Portfolio: ${final_pv:,.2f}")
print(f"  Cash: ${cash:,.2f}")
print(f"  Open Positions: {len(positions)}")
for sym, pos in positions.items():
    price = get_stock_price(stock_data_cache.get(sym), spy_in_period.index[-1])
    if price:
        pl = (price - pos['entry']) / pos['entry'] * 100
        print(f"    {sym:6s}: {pos['qty']:3d} sh @ ${pos['entry']:.2f} → ${price:.2f} | PnL: {pl:+.1f}%")
    else:
        print(f"    {sym:6s}: {pos['qty']:3d} sh @ ${pos['entry']:.2f} | (price unavailable)")
print(f"\nPortfolio Change: ${final_pv - STARTING_PORTFOLIO:+,.2f}")
print(f"Return: {(final_pv - STARTING_PORTFOLIO) / STARTING_PORTFOLIO * 100:+.2f}%")
print(f"Peak Portfolio: ${peak:,.2f}")
max_dd_val = 0
for r in regime_log:
    check_date = pd.Timestamp(r['date'])
    pv_check = get_portfolio_value(cash, positions, check_date)
    dd_check = (pv_check - peak) / peak if peak > 0 else 0
    if dd_check < max_dd_val:
        max_dd_val = dd_check
print(f"Max Drawdown: {max_dd_val * 100:.2f}%")

print(f"\nTotal Trades: {len(trade_log)}")
print(f"\nTrade Log:")
if trade_log:
    for t in trade_log:
        if t['action'] == 'BUY':
            print(f"  {t['date']} BUY  {t['symbol']:6s} | {t['qty']:3d} sh @ ${t['price']:.2f} | Dev: {t['deviation']} | ATR: {t['atr']} | Regime: {t['regime']}")
        else:
            print(f"  {t['date']} {t['action']:14s} {t['symbol']:6s} | {t['qty']:3d} sh | Entry: ${t['entry']:.2f} → Exit: ${t['exit']:.2f} | PnL: {t['pnl_pct']} | {t['days_held']}d | Regime: {t['regime']}")
else:
    print("  No trades executed during this period.")

print(f"\nRegime Distribution (trading days):")
rc = Counter(r['regime'] for r in regime_log)
for r, count in rc.most_common():
    pct = count / len(regime_log) * 100 if len(regime_log) > 0 else 0
    print(f"  {r:15s}: {count:3d} days ({pct:.1f}%)")

print(f"\nRegime Log (first 5 and last 5 days):")
for r in regime_log[:5]:
    print(f"  {r['date']} | {r['regime']:15s} | {r['confidence']} | SPY: {r['spy']}")
print("  ...")
for r in regime_log[-5:]:
    print(f"  {r['date']} | {r['regime']:15s} | {r['confidence']} | SPY: {r['spy']}")
