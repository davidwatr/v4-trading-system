#!/usr/bin/env python3
"""
Regime-Switching Meta-Controller Backtest
Runs Strategy A (Trend Following) + Strategy B (Mean Reversion) + Short TF (bear regime)
with dynamic regime-based allocation.
Uses fixed cached data window (same as deterministic_backtest.py).
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from production_v18 import fetch_data, calc_rsi, calc_bollinger, calc_macd

# Local SMA (not in production_v18)
def calc_sma(closes, period):
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period
from regime_meta_controller import (
    extract_regime_features, classify_regime, get_strategy_allocation,
    check_volatility_override, get_volatility_override_action,
)
from strategy_signals import (
    tf_signal, tf_signal_short, mr_signal_with_sector,
    calculate_trailing_stop, check_trailing_stop_exit,
    MR_UNIVERSE,
)

# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_CAPITAL = 100000
TP_PCT_MR = 0.03       # Mean Reversion profit target
SL_PCT = 0.02           # Hard stop loss (all strategies)
HORIZON_DAYS_MR = 5     # Mean Reversion max hold
HORIZON_DAYS_TF = 15    # Trend Following max hold (longer to let trends develop)
TRAIL_PCT = 0.05        # Trailing stop percentage for TF (widened: let winners run)
TRAIL_ACTIVATION = 0.03 # Activation threshold for trailing stop (3% above entry)

DATA_CACHE = Path('deterministic_data_cache.pkl')
TRADING_DAYS = 501
WARMUP_DAYS = 200

UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC',
            'JPM', 'BAC', 'GS', 'MS', 'PYPL', 'TSLA', 'V', 'MA', 'CRM', 'ADBE']


# ============================================================
# DATA LOADING
# ============================================================
def load_cache():
    if not DATA_CACHE.exists():
        logger.error("No data cache. Run deterministic_backtest.py first.")
        return None
    with open(DATA_CACHE, 'rb') as f:
        return pickle.load(f)


def get_backtest_window(cache):
    spy_data = cache['SPY']
    spy_closes = spy_data['closes']
    total_needed = WARMUP_DAYS + TRADING_DAYS
    if len(spy_closes) < total_needed:
        return None
    
    warmup_end = len(spy_closes) - TRADING_DAYS
    return {
        'spy_closes': spy_closes,
        'backtest_start': warmup_end,
        'backtest_end': len(spy_closes),
        'symbols': {},
    }


def get_symbol_window(cache, sym, backtest_start, warmup_days):
    data = cache['symbols'].get(sym)
    if data is None:
        return None
    start = backtest_start - warmup_days
    end = backtest_start + TRADING_DAYS
    if start < 0 or end > len(data['closes']):
        return None
    return {
        'closes': data['closes'][start:end],
        'volumes': data['volumes'][start:end],
        'highs': data['highs'][start:end],
        'lows': data['lows'][start:end],
    }


# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest():
    logger.info(f"\n{'='*70}")
    logger.info(f"REGIME-SWITCHING META-CONTROLLER BACKTEST")
    logger.info(f"{'='*70}")
    
    cache = load_cache()
    if cache is None:
        return None
    
    window = get_backtest_window(cache)
    if window is None:
        return None
    
    spy_closes = window['spy_closes']
    backtest_start = window['backtest_start']
    
    # Load symbol windows
    symbol_windows = {}
    for sym in UNIVERSE:
        sw = get_symbol_window(cache, sym, backtest_start, WARMUP_DAYS)
        if sw:
            symbol_windows[sym] = sw
    
    logger.info(f"Backtest: {TRADING_DAYS} days, {len(symbol_windows)} symbols")
    logger.info(f"Date range: day {backtest_start} to {backtest_start + TRADING_DAYS - 1}")
    
    # Precompute SPY features for regime classification
    spy_features_history = []
    for day_offset in range(TRADING_DAYS):
        day = backtest_start + day_offset
        features = extract_regime_features(spy_closes, None, None, day)
        spy_features_history.append(features)
    
    # Backtest state
    capital = INITIAL_CAPITAL
    positions = {}  # symbol -> position dict
    portfolio_values = [capital]
    trades = []
    
    # Per-symbol cooldown tracker
    last_exit_per_symbol = {}
    
    # For trailing stop tracking: track highest price per position
    position_highs = {}  # symbol -> highest_price_since_entry
    
    # Regime tracking
    regime_counts = {'trend_bull': 0, 'trend_bear': 0, 'range': 0, 'high_vol': 0, 'transition': 0}
    
    # ================================================================
    # MAIN BACKTEST LOOP
    # ================================================================
    for day_offset in range(TRADING_DAYS):
        day = backtest_start + day_offset
        
        # ---- DAILY MARK-TO-MARKET (every day, before any trading) ----
        port_val = capital
        for sym, pos in positions.items():
            sw = symbol_windows.get(sym)
            if sw is not None and day_offset < len(sw['closes']):
                cur_price = sw['closes'][day_offset]
                if pos['direction'] == 1:
                    port_val += pos['shares'] * cur_price
                else:
                    port_val += pos['shares'] * (pos['entry_price'] - cur_price)
        portfolio_values.append(port_val)
        
        # ---- 1. Determine Regime ----
        features = spy_features_history[day_offset]
        regime, regime_reason = classify_regime(features)
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        alloc = get_strategy_allocation(regime, regime_reason)
        
        # ---- 2. Volatility Override Check (for active shorts) ----
        vol_override = get_volatility_override_action(features)
        shorts_to_close = []
        if vol_override['override']:
            # Force-close all short positions
            for sym, pos in positions.items():
                if pos['direction'] == -1:  # Short position
                    shorts_to_close.append(sym)
        
        # ---- 3. Check Exits for All Positions ----
        to_close = []
        
        for sym, pos in positions.items():
            sw = symbol_windows.get(sym)
            if sw is None:
                continue
            closes_sym = sw['closes']
            if day_offset >= len(closes_sym):
                continue
            
            current_price = closes_sym[day_offset]
            entry = pos['entry_price']
            
            # Update highest price for trailing stop
            if sym in position_highs:
                position_highs[sym] = max(position_highs[sym], current_price)
            else:
                position_highs[sym] = max(current_price, entry)
            
            highest = position_highs[sym]
            
            exit_reason = None
            pnl_pct = 0
            
            if pos['direction'] == 1:  # Long position
                if pos['strategy'] == 'tf':
                    # Trend Following: trailing stop exit
                    if current_price >= entry * (1 + 0.05):  # Also check 5% hard take-profit as safety
                        # Check trailing stop
                        if check_trailing_stop_exit(entry, highest, current_price, TRAIL_PCT):
                            exit_reason = 'trailing_stop'
                            pnl_pct = (current_price - entry) / entry
                        elif current_price >= entry * 1.10:  # Safety: 10% hard cap
                            exit_reason = 'hard_tp_10pct'
                            pnl_pct = (current_price - entry) / entry
                        elif (day_offset - pos['entry_idx']) >= HORIZON_DAYS_TF:
                            exit_reason = 'time_exit'
                            pnl_pct = (current_price - entry) / entry
                    else:
                        # Not yet at 5% — check trailing or time
                        if check_trailing_stop_exit(entry, highest, current_price, TRAIL_PCT):
                            exit_reason = 'trailing_stop'
                            pnl_pct = (current_price - entry) / entry
                        elif (day_offset - pos['entry_idx']) >= HORIZON_DAYS_TF:
                            exit_reason = 'time_exit'
                            pnl_pct = (current_price - entry) / entry
                else:  # Mean Reversion
                    if current_price >= entry * (1 + TP_PCT_MR):
                        exit_reason = 'tp'
                        pnl_pct = TP_PCT_MR
                    elif current_price <= entry * (1 - SL_PCT):
                        exit_reason = 'sl'
                        pnl_pct = -SL_PCT
                    elif (day_offset - pos['entry_idx']) >= HORIZON_DAYS_MR:
                        exit_reason = 'time_exit'
                        pnl_pct = (current_price - entry) / entry
            
            elif pos['direction'] == -1:  # Short position
                if current_price <= entry * (1 - 0.05):  # 5% profit on short
                    if check_trailing_stop_exit(entry, highest, current_price, TRAIL_PCT):
                        exit_reason = 'trailing_stop_short'
                        pnl_pct = (entry - current_price) / entry  # Positive = profit
                    elif current_price <= entry * 0.90:  # Safety: 10% hard cap
                        exit_reason = 'hard_tp_10pct_short'
                        pnl_pct = (entry - current_price) / entry
                    elif (day_offset - pos['entry_idx']) >= HORIZON_DAYS_TF:
                        exit_reason = 'time_exit'
                        pnl_pct = (entry - current_price) / entry
                else:
                    if check_trailing_stop_exit(entry, highest, current_price, TRAIL_PCT):
                        exit_reason = 'trailing_stop_short'
                        pnl_pct = (entry - current_price) / entry
                    elif (day_offset - pos['entry_idx']) >= HORIZON_DAYS_TF:
                        exit_reason = 'time_exit'
                        pnl_pct = (entry - current_price) / entry
            
            # Also check hard stop (2%) for all positions
            if not exit_reason:
                if pos['direction'] == 1 and current_price <= entry * (1 - SL_PCT):
                    exit_reason = 'hard_sl'
                    pnl_pct = -SL_PCT
                elif pos['direction'] == -1 and current_price >= entry * (1 + SL_PCT):
                    exit_reason = 'hard_sl_short'
                    pnl_pct = -SL_PCT
            
            if exit_reason or sym in shorts_to_close:
                if exit_reason or (sym in shorts_to_close and not exit_reason):
                    if not exit_reason and sym in shorts_to_close:
                        exit_reason = 'volatility_override'
                        pnl_pct = (entry - current_price) / entry
                    
                    to_close.append(sym)
                    trades.append({
                        'symbol': sym,
                        'pnl_pct': pnl_pct,
                        'strategy': pos['strategy'],
                        'direction': pos['direction'],
                        'regime': pos['regime'],
                        'days': day_offset - pos['entry_idx'],
                        'entry_idx': pos['entry_idx'],
                        'exit_reason': exit_reason,
                        'rsi': pos.get('rsi', 50),
                        'strength': pos.get('strength', 0),
                    })
        
        # Close positions
        for sym in to_close:
            pos = positions.pop(sym, None)
            if pos is None:
                continue
            sw = symbol_windows.get(sym)
            if sw is None:
                continue
            exit_price = sw['closes'][min(day_offset, len(sw['closes'])-1)]
            slippage_bps = pos.get('slippage_bps', 10)
            
            if pos['direction'] == 1:
                capital += pos['shares'] * exit_price * (1 - slippage_bps / 10000)
            else:  # Short
                capital += pos['shares'] * (pos['entry_price'] - exit_price) * (1 - slippage_bps / 10000)
            
            position_highs.pop(sym, None)
            # Track last exit for cooldown (store before popping)
            last_exit_per_symbol[sym] = day_offset
        
        # Initialize cooldown tracker
        if 'last_exit_per_symbol' not in dir():
            last_exit_per_symbol = {}
        
        # ---- 4. Generate New Signals ----
        new_signals = []
        
        # --- Trend Following Long Signals ---
        if alloc['tf_enabled'] and alloc['tf_weight'] > 0:
            for sym, sw in symbol_windows.items():
                if sym in positions:
                    continue
                # Per-symbol cooldown: skip if exited within last 5 days
                if sym in last_exit_per_symbol and (day_offset - last_exit_per_symbol[sym]) < 5:
                    continue
                closes = sw['closes']
                if day_offset >= len(closes) or day_offset < 50:
                    continue
                
                strength, direction, rsi, entry_type = tf_signal(closes, day_offset, 
                                                                 sw['highs'], sw['lows'])
                if direction == 1 and strength > 0:
                    new_signals.append({
                        'symbol': sym,
                        'strength': strength,
                        'direction': 1,
                        'strategy': 'tf',
                        'regime': regime,
                        'entry_type': entry_type,
                        'rsi': rsi,
                        'price': closes[day_offset],
                    })
        
        # --- Trend Following Short Signals ---
        if alloc['short_enabled'] and alloc['short_weight'] > 0:
            for sym, sw in symbol_windows.items():
                if sym in positions:
                    continue
                if sym in last_exit_per_symbol and (day_offset - last_exit_per_symbol[sym]) < 5:
                    continue
                closes = sw['closes']
                if day_offset >= len(closes) or day_offset < 50:
                    continue
                
                strength, direction, rsi, entry_type = tf_signal_short(closes, day_offset,
                                                                        sw['highs'], sw['lows'])
                if direction == -1 and strength > 0:
                    new_signals.append({
                        'symbol': sym,
                        'strength': strength,
                        'direction': -1,
                        'strategy': 'tf_short',
                        'regime': regime,
                        'entry_type': entry_type,
                        'rsi': rsi,
                        'price': closes[day_offset],
                    })
        
        # --- Mean Reversion Signals ---
        if alloc['mr_enabled'] and alloc['mr_weight'] > 0:
            for sym, sw in symbol_windows.items():
                if sym in positions:
                    continue
                if sym in last_exit_per_symbol and (day_offset - last_exit_per_symbol[sym]) < 5:
                    continue
                closes = sw['closes']
                if day_offset >= len(closes) or day_offset < 20:
                    continue
                
                strength, direction, rsi, bb_pos = mr_signal_with_sector(sym, closes, day_offset)
                if direction == 1 and strength > 0:
                    new_signals.append({
                        'symbol': sym,
                        'strength': strength,
                        'direction': 1,
                        'strategy': 'mr',
                        'regime': regime,
                        'entry_type': 'mr_oversold',
                        'rsi': rsi,
                        'price': closes[day_offset],
                    })
        
        # ---- 5. Sort Signals & Execute ----
        # Sort by strength (highest first), then by strategy priority
        strategy_priority = {'tf': 0, 'tf_short': 0, 'mr': 1}
        new_signals.sort(key=lambda x: (-x['strength'], strategy_priority.get(x['strategy'], 2)))
        
        # Count existing positions by strategy
        existing_tf = sum(1 for p in positions.values() if p['strategy'] == 'tf' and p['direction'] == 1)
        existing_mr = sum(1 for p in positions.values() if p['strategy'] == 'mr')
        existing_short = sum(1 for p in positions.values() if p['direction'] == -1)
        
        # Calculate max positions per strategy based on allocation
        max_tf = int(alloc['max_positions'] * alloc['tf_weight'])
        max_mr = int(alloc['max_positions'] * alloc['mr_weight'])
        max_short = int(alloc['max_positions'] * alloc['short_weight'])
        
        for sig in new_signals:
            sym = sig['symbol']
            if sym in positions:
                continue
            
            # Check position limits per strategy
            if sig['strategy'] == 'tf' and existing_tf >= max_tf:
                continue
            if sig['strategy'] == 'tf_short' and existing_short >= max_short:
                continue
            if sig['strategy'] == 'mr' and existing_mr >= max_mr:
                continue
            
            # Calculate position size (Kelly-based)
            kelly_f = min(sig['strength'], alloc['max_position_kelly']) * alloc['kelly_multiplier']
            kelly_f = max(kelly_f, 0.01)  # Minimum 1% per trade
            
            position_value = capital * kelly_f
            entry_price = sig['price']
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            # Slippage cost
            slippage_bps = alloc['slippage_bps']
            if sig['direction'] == 1:
                cost = shares * entry_price * (1 + slippage_bps / 10000)
            else:
                cost = shares * entry_price * (1 + slippage_bps / 10000)  # Same cost for shorts
            
            if capital - cost < 0:
                continue
            
            # Execute
            capital -= cost
            
            positions[sym] = {
                'entry_price': entry_price,
                'entry_idx': day_offset,
                'shares': shares,
                'strategy': sig['strategy'],
                'direction': sig['direction'],
                'regime': regime,
                'rsi': sig['rsi'],
                'strength': sig['strength'],
                'slippage_bps': slippage_bps,
                'entry_type': sig.get('entry_type', 'unknown'),
            }
            
            if sig['strategy'] == 'tf':
                existing_tf += 1
            elif sig['strategy'] == 'mr':
                existing_mr += 1
            elif sig['direction'] == -1:
                existing_short += 1
    
    # ================================================================
    # FINAL VALUATION — close remaining open positions at last price
    # ================================================================
    for sym, pos in positions.items():
        sw = symbol_windows.get(sym)
        if sw is not None and len(sw['closes']) > 0:
            last_price = sw['closes'][-1]
            if pos['direction'] == 1:
                capital += pos['shares'] * last_price * (1 - pos.get('slippage_bps', 10) / 10000)
            else:
                capital += pos['shares'] * (pos['entry_price'] - last_price) * (1 - pos.get('slippage_bps', 10) / 10000)
    
    # Append final portfolio value after closing everything
    portfolio_values.append(capital)
    
    # ================================================================
    # METRICS
    # ================================================================
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)
    
    total_return = capital / INITIAL_CAPITAL - 1
    n_trading_days = max(1, len(portfolio_values) - 1)  # number of daily return observations
    ann_return = (1 + total_return) ** (252 / n_trading_days) - 1
    # Handle the case where total_return is very negative (annualized = negative large number)
    if ann_return < -1:
        ann_return = -1.0  # Floor at -100%
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (running_max - portfolio_values) / running_max
    max_dd = np.max(drawdown)
    
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    win_loss_ratio = abs(avg_win / avg_loss) if losses and avg_loss != 0 else 0
    gross_wins = sum(t['pnl_pct'] for t in wins)
    gross_losses = abs(sum(t['pnl_pct'] for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    # Breakdown by strategy
    tf_trades = [t for t in trades if t['strategy'] == 'tf']
    tf_short_trades = [t for t in trades if t['strategy'] == 'tf_short']
    mr_trades = [t for t in trades if t['strategy'] == 'mr']
    
    # Breakdown by regime
    regime_breakdown = {}
    for r in ['trend_bull', 'trend_bear', 'range', 'high_vol', 'transition']:
        r_trades = [t for t in trades if t['regime'] == r]
        r_wins = [t for t in r_trades if t['pnl_pct'] > 0]
        regime_breakdown[r] = {
            'trades': len(r_trades),
            'wins': len(r_wins),
            'win_rate': len(r_wins)/len(r_trades) if r_trades else 0,
            'avg_pnl': np.mean([t['pnl_pct'] for t in r_trades]) if r_trades else 0,
        }
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': len(trades),
        'n_wins': len(wins),
        'n_losses': len(losses),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio,
        'profit_factor': profit_factor,
        'final_capital': capital,
        'trades': trades,
        'regime_counts': regime_counts,
        'regime_breakdown': regime_breakdown,
        'strategy_breakdown': {
            'tf_long': {'trades': len(tf_trades), 'wins': len([t for t in tf_trades if t['pnl_pct'] > 0])},
            'tf_short': {'trades': len(tf_short_trades), 'wins': len([t for t in tf_short_trades if t['pnl_pct'] > 0])},
            'mr': {'trades': len(mr_trades), 'wins': len([t for t in mr_trades if t['pnl_pct'] > 0])},
        },
        'final_positions': {sym: {
            'shares': p['shares'],
            'entry_price': p['entry_price'],
            'strategy': p['strategy'],
            'direction': p['direction'],
            'current_price': symbol_windows.get(sym, {}).get('closes', [0])[-1] if symbol_windows.get(sym) else 0,
        } for sym, p in positions.items()},
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
    }


# ============================================================
# SPY BENCHMARK
# ============================================================
def run_spy_benchmark(cache):
    spy_closes = cache['SPY']['closes']
    total_needed = WARMUP_DAYS + TRADING_DAYS
    if len(spy_closes) < total_needed:
        return None
    
    warmup_end = len(spy_closes) - TRADING_DAYS
    spy_window = spy_closes[warmup_end:]
    
    initial_price = spy_window[0]
    final_price = spy_window[-1]
    
    total_return = (final_price / initial_price) - 1
    ann_return = (1 + total_return) ** (252 / len(spy_window)) - 1
    
    daily_returns = np.diff(spy_window) / spy_window[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    running_max = np.maximum.accumulate(spy_window)
    max_dd = np.max((running_max - spy_window) / running_max)
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'initial_price': initial_price,
        'final_price': final_price,
        'n_days': len(spy_window),
    }


# ============================================================
# PRINT RESULTS
# ============================================================
def print_results(result, spy_result, label="Meta-Controller"):
    trades = result['trades']
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {label}")
    print(f"{'='*70}")
    
    print(f"\n  Portfolio Performance:")
    print(f"  {'Total Return:':<25s} {result['total_return']:>+8.2%}")
    print(f"  {'Annualized Return:':<25s} {result['annualized_return']:>+8.2%}")
    print(f"  {'Sharpe Ratio:':<25s} {result['sharpe']:>8.2f}")
    print(f"  {'Max Drawdown:':<25s} {result['max_drawdown']:>8.2%}")
    print(f"  {'Final Capital:':<25s} ${result['final_capital']:>10,.2f}")
    
    print(f"\n  Trade Statistics:")
    print(f"  {'Total Trades:':<25s} {result['n_trades']:>6d}")
    print(f"  {'Win Rate:':<25s} {result['win_rate']:>8.2%} ({result['n_wins']}W / {result['n_losses']}L)")
    print(f"  {'Avg Win:':<25s} {result['avg_win']:>+8.2%}")
    print(f"  {'Avg Loss:':<25s} {result['avg_loss']:>+8.2%}")
    print(f"  {'Win/Loss Ratio:':<25s} {result['win_loss_ratio']:>8.2f}")
    print(f"  {'Profit Factor:':<25s} {result['profit_factor']:>8.2f}")
    
    print(f"\n  Strategy Breakdown:")
    sb = result['strategy_breakdown']
    print(f"  {'Trend Following (Long):':<30s} {sb['tf_long']['trades']:>4d} trades | {sb['tf_long']['wins']:>3d} wins")
    print(f"  {'Trend Following (Short):':<30s} {sb['tf_short']['trades']:>4d} trades | {sb['tf_short']['wins']:>3d} wins")
    print(f"  {'Mean Reversion:':<30s} {sb['mr']['trades']:>4d} trades | {sb['mr']['wins']:>3d} wins")
    
    print(f"\n  Regime Breakdown:")
    rb = result['regime_breakdown']
    rc = result['regime_counts']
    for regime in ['trend_bull', 'trend_bear', 'range', 'high_vol', 'transition']:
        rb_data = rb.get(regime, {})
        print(f"  {regime:<20s}: {rc.get(regime, 0):>3d} days | {rb_data.get('trades', 0):>3d} trades | "
              f"WR {rb_data.get('win_rate', 0):>6.2%} | AvgPnL {rb_data.get('avg_pnl', 0):>+6.2%}")
    
    print(f"\n  Exit Reason Breakdown:")
    exit_counts = {}
    for t in trades:
        reason = t.get('exit_reason', 'unknown')
        exit_counts[reason] = exit_counts.get(reason, 0) + 1
    for reason, count in sorted(exit_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason:<25s}: {count:>4d} trades")
    
    # Compare to SPY
    print(f"\n  {'='*50}")
    print(f"  COMPARISON: Meta-Controller vs SPY Buy-and-Hold")
    print(f"  {'='*50}")
    print(f"  {'Metric':<25s} | {'Meta-Controller':>15s} | {'SPY B&H':>10s} | {'Difference':>10s}")
    print(f"  {'-'*65}")
    print(f"  {'Annualized Return':<25s} | {result['annualized_return']:>+14.2%} | {spy_result['annualized_return']:>+9.2%} | "
          f"{result['annualized_return'] - spy_result['annualized_return']:>+9.2%}")
    print(f"  {'Sharpe Ratio':<25s} | {result['sharpe']:>15.2f} | {spy_result['sharpe']:>10.2f} | "
          f"{result['sharpe'] - spy_result['sharpe']:>+10.2f}")
    print(f"  {'Max Drawdown':<25s} | {result['max_drawdown']:>15.2%} | {spy_result['max_drawdown']:>10.2%} | "
          f"{result['max_drawdown'] - spy_result['max_drawdown']:>+10.2%}")
    print(f"  {'Total Return':<25s} | {result['total_return']:>15.2%} | {spy_result['total_return']:>10.2%} | "
          f"{result['total_return'] - spy_result['total_return']:>+10.2%}")
    
    return result


def main():
    logger.info(f"Started: {datetime.now()}")
    
    cache = load_cache()
    if cache is None:
        return
    
    # SPY benchmark
    spy_result = run_spy_benchmark(cache)
    if spy_result:
        print(f"\nSPY Buy-and-Hold Benchmark ({spy_result['n_days']} days):")
        print(f"  Initial:  ${spy_result['initial_price']:.2f}")
        print(f"  Final:    ${spy_result['final_price']:.2f}")
        print(f"  Total Return:   {spy_result['total_return']:+.2%}")
        print(f"  Annualized:     {spy_result['annualized_return']:+.2%}")
        print(f"  Sharpe:         {spy_result['sharpe']:.2f}")
        print(f"  Max Drawdown:   {spy_result['max_drawdown']:.2%}")
    
    # Run meta-controller backtest
    result = run_backtest()
    if result:
        print_results(result, spy_result, "Regime-Switching Meta-Controller")
        
        # Save
        with open('meta_controller_results.pkl', 'wb') as f:
            pickle.dump({'meta_controller': result, 'spy': spy_result}, f)
        logger.info(f"\nResults saved to: meta_controller_results.pkl")
    else:
        logger.error("Backtest failed.")


if __name__ == '__main__':
    main()
