#!/usr/bin/env python3
"""
Comparative Backtest: Baseline vs Variant 1 (Momentum Hold) vs Variant 2 (MR-Primary)
All run on the same fixed 501-day cached data window.
Outputs side-by-side comparison table vs SPY B&H.
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from production_v18 import fetch_data, calc_rsi, calc_bollinger, calc_adx, calc_atr

# Local SMA — production_v18 doesn't export it
def calc_sma(closes, period):
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period
from regime_meta_controller import extract_regime_features, classify_regime, get_strategy_allocation
from strategy_signals import check_trailing_stop_exit

# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_CAPITAL = 100000
SL_PCT = 0.02           # Hard stop loss (all variants)
TP_PCT_MR = 0.03        # Mean Reversion profit target
HORIZON_DAYS_MR = 5     # MR max hold
DATA_CACHE = Path('deterministic_data_cache.pkl')
TRADING_DAYS = 501
WARMUP_DAYS = 200

UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC',
            'JPM', 'BAC', 'GS', 'MS', 'PYPL', 'TSLA', 'V', 'MA', 'CRM', 'ADBE']

MR_UNIVERSE = {'JPM', 'BAC', 'GS', 'MS', 'V', 'MA',  # Financials
               'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE'}  # Tech


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
# COMMON HELPERS
# ============================================================
def calc_sma_local(closes, period):
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def portfolio_value(capital, positions, symbol_windows, day_offset):
    """Calculate current portfolio value (cash + open positions MTM)."""
    val = capital
    for sym, pos in positions.items():
        sw = symbol_windows.get(sym)
        if sw is not None and day_offset < len(sw['closes']):
            cp = sw['closes'][day_offset]
            if pos['direction'] == 1:
                val += pos['shares'] * cp
            elif pos['direction'] == -1:
                val += pos['shares'] * (pos['entry_price'] - cp)
    return val


def exit_position(pos, symbol_windows, day_offset, slippage_bps):
    """Close a position and return capital change."""
    sw = symbol_windows.get(pos['symbol'])
    if sw is None:
        return 0
    exit_price = sw['closes'][min(day_offset, len(sw['closes'])-1)]
    if pos['direction'] == 1:
        return pos['shares'] * exit_price * (1 - slippage_bps / 10000)
    else:
        return pos['shares'] * (pos['entry_price'] - exit_price) * (1 - slippage_bps / 10000)


# ============================================================
# VARIANT 1: PURE MOMENTUM HOLD (SMA Crossover)
# ============================================================
def run_variant1_momentum_hold(cache):
    """
    Variant 1: Enter when 20-SMA crosses above 50-SMA.
    Exit when 20-SMA crosses below 50-SMA OR 2% hard stop.
    No time limit, no trailing stop.
    """
    logger.info("Running Variant 1: Pure Momentum Hold (SMA Crossover)...")
    
    window = get_backtest_window(cache)
    if window is None:
        return None
    
    spy_closes = window['spy_closes']
    backtest_start = window['backtest_start']
    
    symbol_windows = {}
    for sym in UNIVERSE:
        sw = get_symbol_window(cache, sym, backtest_start, WARMUP_DAYS)
        if sw:
            symbol_windows[sym] = sw
    
    capital = INITIAL_CAPITAL
    positions = {}
    portfolio_values = [capital]
    trades = []
    
    for day_offset in range(TRADING_DAYS):
        # Daily mark-to-market
        pv = portfolio_value(capital, positions, symbol_windows, day_offset)
        portfolio_values.append(pv)
        
        # Check exits: SMA crossover reverse OR hard stop
        to_close = []
        for sym, pos in positions.items():
            sw = symbol_windows.get(sym)
            if sw is None or day_offset >= len(sw['closes']):
                continue
            closes_sym = sw['closes'][:day_offset+1]
            current_price = closes_sym[-1]
            
            # SMA crossover exit: 20-SMA crosses below 50-SMA
            sma20_now = calc_sma_local(closes_sym, 20)
            sma50_now = calc_sma_local(closes_sym, 50)
            
            if sma20_now is not None and sma50_now is not None:
                # Check if crossover happened today
                if len(closes_sym) >= 52:  # Need enough history for previous SMAs
                    closes_prev = closes_sym[:-1]
                    sma20_prev = calc_sma_local(closes_prev, 20)
                    sma50_prev = calc_sma_local(closes_prev, 50)
                    if sma20_prev is not None and sma50_prev is not None:
                        # Bullish crossover was in effect (20>50 before, now 20<50)
                        was_bullish = sma20_prev > sma50_prev
                        is_bearish = sma20_now < sma50_now
                        if was_bullish and is_bearish:
                            # Reverse crossover — exit
                            to_close.append((sym, 'sma_crossover_exit', 
                                             (current_price - pos['entry_price']) / pos['entry_price']))
            
            # Hard stop check
            if sym not in [t[0] for t in to_close]:
                if current_price <= pos['entry_price'] * (1 - SL_PCT):
                    to_close.append((sym, 'hard_stop', -SL_PCT))
        
        # Close positions
        for sym, reason, pnl_pct in to_close:
            pos = positions.pop(sym, None)
            if pos is None:
                continue
            sw = symbol_windows.get(sym)
            if sw is None:
                continue
            exit_price = sw['closes'][min(day_offset, len(sw['closes'])-1)]
            slippage_bps = 10
            capital += pos['shares'] * exit_price * (1 - slippage_bps / 10000)
            
            trades.append({
                'symbol': sym,
                'pnl_pct': pnl_pct,
                'exit_reason': reason,
                'days': day_offset - pos['entry_idx'],
                'entry_idx': pos['entry_idx'],
                'strategy': 'momentum_hold',
                'regime': 'n/a',
                'rsi': calc_rsi(sw['closes'][:day_offset+1], 14) or 50,
            })
        
        # Generate new signals: SMA crossover entry
        if len(positions) < 8:  # Max 8 concurrent momentum positions
            for sym, sw in symbol_windows.items():
                if sym in positions:
                    continue
                
                closes = sw['closes'][:day_offset+1]
                if len(closes) < 52:  # Need enough history
                    continue
                
                sma20 = calc_sma_local(closes, 20)
                sma50 = calc_sma_local(closes, 50)
                
                if sma20 is None or sma50 is None:
                    continue
                
                # Check for bullish crossover today
                if len(closes) >= 52:
                    closes_prev = closes[:-1]
                    sma20_prev = calc_sma_local(closes_prev, 20)
                    sma50_prev = calc_sma_local(closes_prev, 50)
                    
                    if sma20_prev is not None and sma50_prev is not None:
                        was_bearish = sma20_prev < sma50_prev
                        is_bullish = sma20 > sma50
                        
                        if was_bearish and is_bullish:
                            # Bullish crossover — enter
                            entry_price = closes[-1]
                            kelly_f = 0.08  # Fixed 8% per position
                            shares = int(capital * kelly_f / entry_price)
                            
                            if shares > 0:
                                cost = shares * entry_price * (1 + 10 / 10000)
                                if capital - cost >= 0:
                                    capital -= cost
                                    positions[sym] = {
                                        'symbol': sym,
                                        'entry_price': entry_price,
                                        'entry_idx': day_offset,
                                        'shares': shares,
                                        'direction': 1,
                                        'strategy': 'momentum_hold',
                                        'slippage_bps': 10,
                                    }
        
        # Check if we've exited all positions (end of loop)
        # Already handled above via to_close
        
    # Final valuation: close remaining positions
    for sym, pos in positions.items():
        sw = symbol_windows.get(sym)
        if sw is not None and len(sw['closes']) > 0:
            last_price = sw['closes'][-1]
            capital += pos['shares'] * last_price * (1 - 10 / 10000)
    
    portfolio_values.append(capital)
    
    # Compute metrics
    pv_arr = np.array(portfolio_values)
    daily_returns = np.diff(pv_arr) / pv_arr[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)
    
    total_return = capital / INITIAL_CAPITAL - 1
    n_obs = max(1, len(pv_arr) - 1)
    ann_return = (1 + total_return) ** (252 / n_obs) - 1
    if ann_return < -1:
        ann_return = -1.0
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    running_max = np.maximum.accumulate(pv_arr)
    max_dd = np.max((running_max - pv_arr) / running_max)
    
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    win_loss_ratio = abs(avg_win / avg_loss) if losses and avg_loss != 0 else 0
    profit_factor = abs(sum(t['pnl_pct'] for t in wins)) / abs(sum(t['pnl_pct'] for t in losses)) if losses and sum(t['pnl_pct'] for t in losses) != 0 else float('inf')
    
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
        'portfolio_values': pv_arr,
        'trades': trades,
    }


# ============================================================
# VARIANT 2: MR-PRIMARY / TF-SATELLITE
# ============================================================
def run_variant2_mr_primary(cache):
    """
    Variant 2: MR is primary (RSI < 35, no BB requirement).
    TF acts as satellite — only top 5% conviction signals.
    """
    logger.info("Running Variant 2: MR-Primary / TF-Satellite...")
    
    # Use the same regime meta-controller for regime detection
    window = get_backtest_window(cache)
    if window is None:
        return None
    
    spy_closes = window['spy_closes']
    backtest_start = window['backtest_start']
    
    symbol_windows = {}
    for sym in UNIVERSE:
        sw = get_symbol_window(cache, sym, backtest_start, WARMUP_DAYS)
        if sw:
            symbol_windows[sym] = sw
    
    # Precompute SPY features for regime
    spy_features_history = []
    for day_offset in range(TRADING_DAYS):
        day = backtest_start + day_offset
        features = extract_regime_features(spy_closes, None, None, day)
        spy_features_history.append(features)
    
    capital = INITIAL_CAPITAL
    positions = {}
    portfolio_values = [capital]
    trades = []
    
    # For TF satellite: track all TF signal strengths to determine top 5% cutoff
    tf_strength_history = []
    
    for day_offset in range(TRADING_DAYS):
        # Regime determination
        features = spy_features_history[day_offset]
        regime, regime_reason = classify_regime(features)
        alloc = get_strategy_allocation(regime, regime_reason)
        
        # Daily mark-to-market
        pv = portfolio_value(capital, positions, symbol_windows, day_offset)
        portfolio_values.append(pv)
        
        # ---- Check Exits ----
        to_close = []
        position_highs = {}
        
        for sym, pos in positions.items():
            sw = symbol_windows.get(sym)
            if sw is None or day_offset >= len(sw['closes']):
                continue
            closes_sym = sw['closes'][:day_offset+1]
            current_price = closes_sym[-1]
            
            # Update highest price for trailing stop
            if sym not in position_highs:
                position_highs[sym] = max(current_price, pos['entry_price'])
            else:
                position_highs[sym] = max(position_highs[sym], current_price)
            highest = position_highs[sym]
            
            exit_reason = None
            pnl_pct = 0
            
            if pos['strategy'] == 'mr':
                # MR: fixed 3% TP, 2% SL, 5-day horizon
                if current_price >= pos['entry_price'] * (1 + TP_PCT_MR):
                    exit_reason = 'tp'
                    pnl_pct = TP_PCT_MR
                elif current_price <= pos['entry_price'] * (1 - SL_PCT):
                    exit_reason = 'sl'
                    pnl_pct = -SL_PCT
                elif (day_offset - pos['entry_idx']) >= HORIZON_DAYS_MR:
                    exit_reason = 'time_exit'
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            elif pos['strategy'] == 'tf_sat':
                # TF Satellite: trailing stop exit (5% trail, 3% activation)
                if check_trailing_stop_exit(pos['entry_price'], highest, current_price, 0.05):
                    exit_reason = 'trailing_stop'
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                elif current_price >= pos['entry_price'] * 1.10:
                    exit_reason = 'hard_tp_10pct'
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
            
            # Hard stop for all
            if not exit_reason and current_price <= pos['entry_price'] * (1 - SL_PCT):
                exit_reason = 'hard_stop'
                pnl_pct = -SL_PCT
            
            if exit_reason:
                to_close.append((sym, exit_reason, pnl_pct))
        
        for sym, reason, pnl_pct in to_close:
            pos = positions.pop(sym, None)
            if pos is None:
                continue
            sw = symbol_windows.get(sym)
            if sw is None:
                continue
            exit_price = sw['closes'][min(day_offset, len(sw['closes'])-1)]
            capital += pos['shares'] * exit_price * (1 - pos.get('slippage_bps', 10) / 10000)
            
            trades.append({
                'symbol': sym,
                'pnl_pct': pnl_pct,
                'exit_reason': reason,
                'days': day_offset - pos['entry_idx'],
                'entry_idx': pos['entry_idx'],
                'strategy': pos['strategy'],
                'regime': regime,
                'rsi': pos.get('rsi', 50),
                'strength': pos.get('strength', 0),
            })
        
        # ---- Generate New Signals ----
        new_mr_signals = []
        new_tf_signals = []
        
        # MR signals: RSI < 35, no BB requirement (relaxed primary)
        for sym, sw in symbol_windows.items():
            if sym in positions:
                continue
            closes = sw['closes'][:day_offset+1]
            if len(closes) < 20:
                continue
            
            rsi_val = calc_rsi(closes, 14)
            if rsi_val is None:
                continue
            
            # Relaxed MR: RSI < 35, no BB filter
            if rsi_val < 35:
                strength = min(0.10, (35 - rsi_val) / 35 * 0.08)
                if strength > 0.02:
                    new_mr_signals.append({
                        'symbol': sym,
                        'strength': strength,
                        'rsi': rsi_val,
                        'price': closes[-1],
                        'strategy': 'mr',
                        'regime': regime,
                    })
            
            # Also trigger on RSI < 30 (classic oversold) regardless of BB
            if rsi_val < 30:
                strength2 = min(0.12, (30 - rsi_val) / 30 * 0.10)
                if strength2 > 0.03:
                    new_mr_signals.append({
                        'symbol': sym,
                        'strength': strength2,
                        'rsi': rsi_val,
                        'price': closes[-1],
                        'strategy': 'mr',
                        'regime': regime,
                    })
        
        # TF satellite signals: collect all, filter to top 5% later
        if alloc['tf_enabled'] and alloc['tf_weight'] > 0:
            for sym, sw in symbol_windows.items():
                if sym in positions:
                    continue
                closes = sw['closes'][:day_offset+1]
                if len(closes) < 50:
                    continue
                
                # Use breakout signal from strategy_signals
                from strategy_signals import tf_signal
                strength, direction, rsi, entry_type = tf_signal(closes, day_offset, sw['highs'], sw['lows'])
                if direction == 1 and strength > 0:
                    new_tf_signals.append({
                        'symbol': sym,
                        'strength': strength,
                        'rsi': rsi,
                        'price': closes[-1],
                        'strategy': 'tf_sat',
                        'regime': regime,
                        'entry_type': entry_type,
                    })
        
        # Determine TF satellite threshold: top 5% of all signals today
        tf_sat_enabled = len(new_tf_signals) > 0
        tf_threshold = 0
        if tf_sat_enabled:
            sorted_strengths = sorted([s['strength'] for s in new_tf_signals], reverse=True)
            idx_95 = max(1, int(len(sorted_strengths) * 0.95))
            tf_threshold = sorted_strengths[idx_95 - 1] if idx_95 <= len(sorted_strengths) else sorted_strengths[-1]
        
        # Filter TF to top 5%
        tf_sat_signals = [s for s in new_tf_signals if s['strength'] >= tf_threshold] if tf_sat_enabled else []
        
        # Combine and sort: MR first (primary), then TF satellite
        all_signals = new_mr_signals + tf_sat_signals
        all_signals.sort(key=lambda x: (-x['strength'], 0 if x['strategy'] == 'mr' else 1))
        
        # Position limits
        max_mr_pos = int(alloc['max_positions'] * alloc['mr_weight'])
        max_tf_pos = int(alloc['max_positions'] * alloc['tf_weight'])
        
        existing_mr = sum(1 for p in positions.values() if p['strategy'] == 'mr')
        existing_tf = sum(1 for p in positions.values() if p['strategy'] == 'tf_sat')
        
        for sig in all_signals:
            sym = sig['symbol']
            if sym in positions:
                continue
            
            if sig['strategy'] == 'mr' and existing_mr >= max_mr_pos:
                continue
            if sig['strategy'] == 'tf_sat' and existing_tf >= max_tf_pos:
                continue
            
            kelly_f = min(sig['strength'], alloc['max_position_kelly']) * alloc['kelly_multiplier']
            kelly_f = max(kelly_f, 0.01)
            
            position_value = capital * kelly_f
            entry_price = sig['price']
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            cost = shares * entry_price * (1 + alloc['slippage_bps'] / 10000)
            if capital - cost < 0:
                continue
            
            capital -= cost
            positions[sym] = {
                'symbol': sym,
                'entry_price': entry_price,
                'entry_idx': day_offset,
                'shares': shares,
                'direction': 1,
                'strategy': sig['strategy'],
                'regime': regime,
                'rsi': sig['rsi'],
                'strength': sig['strength'],
                'slippage_bps': alloc['slippage_bps'],
            }
            
            if sig['strategy'] == 'mr':
                existing_mr += 1
            else:
                existing_tf += 1
    
    # Final valuation
    for sym, pos in positions.items():
        sw = symbol_windows.get(sym)
        if sw is not None and len(sw['closes']) > 0:
            last_price = sw['closes'][-1]
            capital += pos['shares'] * last_price * (1 - pos.get('slippage_bps', 10) / 10000)
    
    portfolio_values.append(capital)
    
    # Compute metrics
    pv_arr = np.array(portfolio_values)
    daily_returns = np.diff(pv_arr) / pv_arr[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)
    
    total_return = capital / INITIAL_CAPITAL - 1
    n_obs = max(1, len(pv_arr) - 1)
    ann_return = (1 + total_return) ** (252 / n_obs) - 1
    if ann_return < -1:
        ann_return = -1.0
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    running_max = np.maximum.accumulate(pv_arr)
    max_dd = np.max((running_max - pv_arr) / running_max)
    
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    win_loss_ratio = abs(avg_win / avg_loss) if losses and avg_loss != 0 else 0
    profit_factor = abs(sum(t['pnl_pct'] for t in wins)) / abs(sum(t['pnl_pct'] for t in losses)) if losses and sum(t['pnl_pct'] for t in losses) != 0 else float('inf')
    
    # Strategy breakdown
    mr_trades = [t for t in trades if t['strategy'] == 'mr']
    tf_trades = [t for t in trades if t['strategy'] == 'tf_sat']
    
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
        'portfolio_values': pv_arr,
        'trades': trades,
        'strategy_breakdown': {
            'mr': {'trades': len(mr_trades), 'wins': len([t for t in mr_trades if t['pnl_pct'] > 0])},
            'tf_sat': {'trades': len(tf_trades), 'wins': len([t for t in tf_trades if t['pnl_pct'] > 0])},
        },
        'tf_threshold_used': tf_threshold,
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
    
    total_return = (spy_window[-1] / spy_window[0]) - 1
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
        'initial_price': spy_window[0],
        'final_price': spy_window[-1],
        'n_days': len(spy_window),
    }


# ============================================================
# PRINT COMPARISON TABLE
# ============================================================
def print_comparison_table(v1, v2, spy):
    print(f"\n{'='*100}")
    print(f"COMPARATIVE BACKTEST RESULTS — Fixed 501-Day Window")
    print(f"{'='*100}")
    
    print(f"\n{'Metric':<25s} | {'Variant 1 (Momentum Hold)':>22s} | {'Variant 2 (MR-Primary)':>20s} | {'SPY B&H':>12s} | {'V1 vs SPY':>12s} | {'V2 vs SPY':>12s}")
    print(f"{'-'*100}")
    
    metrics = [
        ('Total Return', 'total_return', True),
        ('Annualized Return', 'annualized_return', True),
        ('Sharpe Ratio', 'sharpe', False),
        ('Max Drawdown', 'max_drawdown', True),
        ('Total Trades', 'n_trades', False),
        ('Win Rate', 'win_rate', True),
        ('Profit Factor', 'profit_factor', False),
    ]
    
    for label, key, is_pct in metrics:
        v1_val = v1.get(key, 0)
        v2_val = v2.get(key, 0)
        s_val = spy.get(key, 0)
        
        if is_pct:
            v1_str = f"{v1_val:>+21.2%}"
            v2_str = f"{v2_val:>+19.2%}"
            s_str = f"{s_val:>+11.2%}"
        elif key == 'sharpe' or key == 'profit_factor':
            v1_str = f"{v1_val:>21.2f}"
            v2_str = f"{v2_val:>19.2f}"
            s_str = f"{s_val:>11.2f}"
        else:
            v1_str = f"{v1_val:>21d}"
            v2_str = f"{v2_val:>19d}"
            s_str = f"{s_val:>11d}"
        
        diff1 = v1_val - s_val
        diff2 = v2_val - s_val
        
        if is_pct:
            d1_str = f"{diff1:>+11.2%}"
            d2_str = f"{diff2:>+11.2%}"
        elif key == 'sharpe' or key == 'profit_factor':
            d1_str = f"{diff1:>+11.2f}"
            d2_str = f"{diff2:>+11.2f}"
        else:
            d1_str = f"{diff1:>+11d}"
            d2_str = f"{diff2:>+11d}"
        
        print(f"  {label:<23s} | {v1_str} | {v2_str} | {s_str} | {d1_str} | {d2_str}")
    
    # Additional detail
    print(f"\n{'='*100}")
    print(f"ADDITIONAL DETAILS")
    print(f"{'='*100}")
    
    print(f"\n  Variant 1 (Momentum Hold):")
    print(f"    Avg Win:  {v1['avg_win']:+.2%}")
    print(f"    Avg Loss: {v1['avg_loss']:+.2%}")
    print(f"    Win/Loss Ratio: {v1['win_loss_ratio']:.2f}")
    
    print(f"\n  Variant 2 (MR-Primary / TF-Satellite):")
    sb2 = v2.get('strategy_breakdown', {})
    mr = sb2.get('mr', {})
    tf = sb2.get('tf_sat', {})
    print(f"    MR Trades: {mr['trades']} ({mr['wins']} wins)")
    print(f"    TF Satellite Trades: {tf['trades']} ({tf['wins']} wins)")
    print(f"    TF Activation Threshold: top 5% (min strength {v2.get('tf_threshold_used', 0):.4f})")
    print(f"    Avg Win:  {v2['avg_win']:+.2%}")
    print(f"    Avg Loss: {v2['avg_loss']:+.2%}")
    print(f"    Win/Loss Ratio: {v2['win_loss_ratio']:.2f}")
    
    # Verdict
    print(f"\n{'='*100}")
    print(f"VERDICT")
    print(f"{'='*100}")
    
    v1_sharpe = v1['sharpe']
    v2_sharpe = v2['sharpe']
    spy_sharpe = spy['sharpe']
    
    v1_ann = v1['annualized_return']
    v2_ann = v2['annualized_return']
    spy_ann = spy['annualized_return']
    
    print(f"\n  SPY B&H Baseline: Sharpe {spy_sharpe:.2f}, Ann Return {spy_ann:+.2%}")
    
    v1_beats = v1_sharpe >= spy_sharpe and v1_ann >= spy_ann
    v2_beats = v2_sharpe >= spy_sharpe and v2_ann >= spy_ann
    
    if v1_beats and v2_beats:
        print(f"  ✅ BOTH variants match or beat SPY on risk-adjusted basis")
    elif v1_beats:
        print(f"  ✅ Variant 1 (Momentum Hold) matches/beats SPY. Variant 2 does not.")
    elif v2_beats:
        print(f"  ✅ Variant 2 (MR-Primary) matches/beats SPY. Variant 1 does not.")
    else:
        print(f"  ❌ Neither variant matches/beats SPY on risk-adjusted basis.")
        if v1_sharpe > v2_sharpe:
            print(f"     Variant 1 is closer (Sharpe {v1_sharpe:.2f} vs {v2_sharpe:.2f})")
        else:
            print(f"     Variant 2 is closer (Sharpe {v2_sharpe:.2f} vs {v1_sharpe:.2f})")
    
    # Which has better absolute return
    if v1_ann > v2_ann:
        print(f"     Higher absolute return: Variant 1 ({v1_ann:+.2%} vs {v2_ann:+.2%})")
    else:
        print(f"     Higher absolute return: Variant 2 ({v2_ann:+.2%} vs {v1_ann:+.2%})")
    
    return v1, v2, spy


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Comparative backtest on fixed {TRADING_DAYS}-day window")
    
    cache = load_cache()
    if cache is None:
        return
    
    # SPY benchmark
    spy = run_spy_benchmark(cache)
    print(f"\nSPY Buy-and-Hold Benchmark ({spy['n_days']} days):")
    print(f"  Initial:  ${spy['initial_price']:.2f}")
    print(f"  Final:    ${spy['final_price']:.2f}")
    print(f"  Total Return:   {spy['total_return']:+.2%}")
    print(f"  Annualized:     {spy['annualized_return']:+.2%}")
    print(f"  Sharpe:         {spy['sharpe']:.2f}")
    print(f"  Max Drawdown:   {spy['max_drawdown']:.2%}")
    
    # Variant 1: Momentum Hold
    v1 = run_variant1_momentum_hold(cache)
    
    # Variant 2: MR-Primary
    v2 = run_variant2_mr_primary(cache)
    
    # Print comparison
    print_comparison_table(v1, v2, spy)
    
    # Save results
    results = {
        'variant1_momentum_hold': v1,
        'variant2_mr_primary': v2,
        'spy_benchmark': spy,
    }
    with open('comparative_backtest_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"\nResults saved to: comparative_backtest_results.pkl")


if __name__ == '__main__':
    main()
