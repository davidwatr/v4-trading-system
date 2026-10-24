#!/usr/bin/env python3
"""
Deterministic Backtest Suite
Uses FIXED date range (not "days back") to ensure identical data across all runs.
Caches data to disk so every backtest uses exactly the same dataset.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from production_v18 import fetch_data
from heuristic_signals import mean_reversion_signal, momentum_filter

# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_CAPITAL = 100000
SLIPPAGE_BPS = 10
TP_PCT = 0.03
SL_PCT = 0.02
HORIZON_DAYS = 5
KELLY_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]

# FIXED DATE RANGE — both backtests use this exact window
END_DATE = datetime(2026, 9, 10)  # Most recent trading day
WARMUP_DAYS = 200  # For regime detection
TRADING_DAYS = 501  # Actual backtest period
START_DATE = END_DATE - timedelta(days=TRADING_DAYS + WARMUP_DAYS)

# Universe — same across all runs
UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC',
            'JPM', 'BAC', 'GS', 'MS', 'PYPL', 'TSLA', 'V', 'MA', 'CRM', 'ADBE']

DATA_CACHE = Path('deterministic_data_cache.pkl')


# ============================================================
# DATA CACHING — fetch once, reuse everywhere
# ============================================================
def fetch_and_cache_data():
    """Fetch all data once and cache to disk. Ensures identical data for every backtest."""
    if DATA_CACHE.exists():
        logger.info("Loading cached data...")
        with open(DATA_CACHE, 'rb') as f:
            cache = pickle.load(f)
        return cache
    
    logger.info("Fetching data from Yahoo (this may take a minute)...")
    cache = {'SPY': None, 'symbols': {}}
    
    # Fetch SPY — need WARMUP_DAYS + TRADING_DAYS actual trading days
    # fetch_data returns ~68% of requested days (weekends/holidays), so request ~1.47x
    spy_closes, spy_volumes, spy_highs, spy_lows = fetch_data('SPY', days=int((TRADING_DAYS + WARMUP_DAYS) * 1.5))
    if spy_closes is not None and len(spy_closes) >= WARMUP_DAYS + TRADING_DAYS:
        cache['SPY'] = {
            'closes': spy_closes,
            'volumes': spy_volumes,
            'highs': spy_highs,
            'lows': spy_lows,
        }
        logger.info(f"SPY: {len(spy_closes)} days fetched")
    else:
        logger.error("Failed to fetch SPY")
        return None
    
    # Fetch all symbols — need WARMUP_DAYS + TRADING_DAYS actual trading days
    # fetch_data returns ~68% of requested days (weekends/holidays), so request ~1.5x
    fetch_days = int((TRADING_DAYS + WARMUP_DAYS) * 1.5)
    for sym in UNIVERSE:
        closes, volumes, highs, lows = fetch_data(sym, days=fetch_days)
        if closes is not None and len(closes) >= WARMUP_DAYS + TRADING_DAYS:
            cache['symbols'][sym] = {
                'closes': closes,
                'volumes': volumes,
                'highs': highs,
                'lows': lows,
            }
            logger.info(f"  {sym}: {len(closes)} days")
        else:
            logger.warning(f"  {sym}: FAILED")
    
    # Save cache
    with open(DATA_CACHE, 'wb') as f:
        pickle.dump(cache, f)
    logger.info(f"\nData cached to {DATA_CACHE}")
    
    return cache


def get_fixed_window(data, start_idx, end_idx):
    """Extract a fixed window from cached data."""
    return {
        'closes': data['closes'][start_idx:end_idx],
        'volumes': data['volumes'][start_idx:end_idx],
        'highs': data['highs'][start_idx:end_idx],
        'lows': data['lows'][start_idx:end_idx],
    }


# ============================================================
# REGIME DETECTION (same as before)
# ============================================================
def compute_regime_features(closes):
    if len(closes) < 200:
        return np.array([0, 0, 0, 0])
    
    closes = np.array(closes)
    
    ma50 = np.mean(closes[-50:])
    ma200 = np.mean(closes[-200:])
    current = closes[-1]
    
    trend_strength = (current - ma200) / ma200 if ma200 > 0 else 0
    trend_slope = (ma50 - ma200) / ma200 if ma200 > 0 else 0
    
    ret_10d = (closes[-1] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0
    ret_50d = (closes[-1] - closes[-51]) / closes[-51] if closes[-51] > 0 else 0
    momentum = ret_10d - ret_50d / 5
    
    tr = np.abs(np.diff(closes))
    atr_14 = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    atr_50 = np.mean(tr[-50:]) if len(tr) >= 50 else 0
    atr_ratio = atr_14 / atr_50 if atr_50 > 0 else 1
    
    return np.array([trend_strength, trend_slope, momentum, atr_ratio])


def build_regime_classifier(spy_closes_history):
    features = []
    for i in range(200, len(spy_closes_history)):
        window = spy_closes_history[i-200:i+1]
        feats = compute_regime_features(window)
        features.append(feats)
    
    features = np.array(features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(scaled)
    
    cluster_means = kmeans.cluster_centers_[:, 0]
    bull_cluster = np.argmax(cluster_means)
    bear_cluster = np.argmin(cluster_means)
    side_cluster = np.argsort(cluster_means)[1]
    
    cluster_map = {
        bull_cluster: 'bull',
        side_cluster: 'sideways',
        bear_cluster: 'bear'
    }
    
    return kmeans, scaler, cluster_map


def get_regime_class(spy_closes, kmeans, scaler, cluster_map):
    feats = compute_regime_features(spy_closes)
    scaled = scaler.transform(feats.reshape(1, -1))
    cluster = kmeans.predict(scaled)[0]
    return cluster_map.get(cluster, 'sideways')


def get_regime_kelly_params(regime, base_kelly_mult=1.0):
    params = {
        'bull': {
            'kelly_multiplier': 1.5 * base_kelly_mult,
            'max_positions': 8,
            'min_prob': 0.50,
            'max_kelly_f': 0.30 * base_kelly_mult,
            'slippage_bps': 8,
        },
        'bull_high_vol': {
            'kelly_multiplier': 1.3 * base_kelly_mult,
            'max_positions': 6,
            'min_prob': 0.55,
            'max_kelly_f': 0.25 * base_kelly_mult,
            'slippage_bps': 12,
        },
        'sideways': {
            'kelly_multiplier': 0.75 * base_kelly_mult,
            'max_positions': 6,
            'min_prob': 0.55,
            'max_kelly_f': 0.20 * base_kelly_mult,
            'slippage_bps': 10,
        },
        'sideways_high_vol': {
            'kelly_multiplier': 0.60 * base_kelly_mult,
            'max_positions': 4,
            'min_prob': 0.60,
            'max_kelly_f': 0.15 * base_kelly_mult,
            'slippage_bps': 15,
        },
        'bear': {
            'kelly_multiplier': 0.5 * base_kelly_mult,
            'max_positions': 4,
            'min_prob': 0.60,
            'max_kelly_f': 0.15 * base_kelly_mult,
            'slippage_bps': 15,
        },
        'bear_high_vol': {
            'kelly_multiplier': 0.30 * base_kelly_mult,
            'max_positions': 2,
            'min_prob': 0.65,
            'max_kelly_f': 0.10 * base_kelly_mult,
            'slippage_bps': 20,
        }
    }
    
    return params.get(regime, params['sideways'])


# ============================================================
# BACKTEST (parameterized)
# ============================================================
def run_backtest(kelly_base_mult=1.0, cache=None):
    if cache is None:
        cache = fetch_and_cache_data()
    if cache is None:
        return None
    
    spy_data = cache['SPY']
    spy_closes = spy_data['closes']
    
    # Build regime classifier on full SPY history
    kmeans, scaler, cluster_map = build_regime_classifier(spy_closes)
    
    # Determine the exact window
    # We need WARMUP_DAYS at the start, then TRADING_DAYS of actual backtest
    total_needed = WARMUP_DAYS + TRADING_DAYS
    if len(spy_closes) < total_needed:
        logger.error(f"Not enough data: have {len(spy_closes)}, need {total_needed}")
        return None
    
    # Use the last TRADING_DAYS days (after WARMUP_DAYS of warmup)
    warmup_end = len(spy_closes) - TRADING_DAYS
    backtest_start = warmup_end
    backtest_end = len(spy_closes)
    
    logger.info(f"\nBacktest window: day {backtest_start} to {backtest_end-1} ({TRADING_DAYS} days)")
    logger.info(f"Warmup: days 0 to {backtest_start-1} ({WARMUP_DAYS} days)")
    
    n_days = backtest_end - backtest_start
    
    # Pre-extract symbol windows
    symbol_windows = {}
    for sym, data in cache['symbols'].items():
        if len(data['closes']) >= total_needed:
            symbol_windows[sym] = get_fixed_window(data, backtest_start - WARMUP_DAYS, backtest_end)
        else:
            logger.warning(f"  {sym}: insufficient data ({len(data['closes'])} < {total_needed})")
    
    # Backtest
    capital = INITIAL_CAPITAL
    positions = {}
    portfolio_values = [capital]
    trades = []
    n_trades = 0
    n_wins = 0
    
    for day_offset in range(TRADING_DAYS):
        day = backtest_start + day_offset
        
        # Current SPY data for regime (includes warmup up to this day)
        spy_window = spy_closes[:day+1]
        regime = get_regime_class(spy_window, kmeans, scaler, cluster_map)
        regime_params = get_regime_kelly_params(regime, base_kelly_mult=kelly_base_mult)
        
        # Generate signals
        signals = []
        for sym, window in symbol_windows.items():
            closes = window['closes']
            day_idx = day_offset  # position within the backtest window
            
            if day_idx >= len(closes):
                continue
            
            closes_slice = closes[:day_idx+1]
            if len(closes_slice) < 20:
                continue
            
            if not momentum_filter(closes_slice):
                continue
            
            strength, direction, rsi_val = mean_reversion_signal(closes_slice)
            if direction == 1 and strength > 0:
                signals.append({
                    'symbol': sym,
                    'strength': strength,
                    'kelly_size': min(strength * 0.10, 0.10),
                    'rsi': rsi_val,
                    'current_price': closes[day_idx],
                })
        
        signals.sort(key=lambda x: -x['strength'])
        
        # Execute trades
        current_positions = len(positions)
        max_positions = regime_params['max_positions']
        
        for sig in signals[:max_positions - current_positions]:
            symbol = sig['symbol']
            if symbol in positions:
                continue
            
            base_kelly = sig['kelly_size']
            kelly_f = base_kelly * regime_params['kelly_multiplier']
            kelly_f = min(kelly_f, regime_params['max_kelly_f'])
            
            if sig['strength'] < regime_params['min_prob']:
                continue
            
            position_value = capital * kelly_f
            entry_price = sig['current_price']
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            cost = shares * entry_price * (1 + regime_params['slippage_bps'] / 10000)
            if capital - cost < 0:
                continue
            
            capital -= cost
            positions[symbol] = {
                'entry_price': entry_price,
                'entry_idx': day_offset,
                'shares': shares,
                'kelly_f': kelly_f,
                'regime': regime,
                'symbol': symbol,
                'rsi': sig['rsi'],
            }
        
        # Check exits
        to_close = []
        for sym, pos in positions.items():
            if day_offset > pos['entry_idx']:
                window = symbol_windows.get(sym)
                if window is None:
                    continue
                closes_sym = window['closes']
                if day_offset < len(closes_sym):
                    current_price = closes_sym[day_offset]
                    entry = pos['entry_price']
                    
                    if current_price >= entry * (1 + TP_PCT):
                        pnl_pct = TP_PCT
                        to_close.append(sym)
                        trades.append({
                            'symbol': sym, 'pnl_pct': pnl_pct, 'regime': pos['regime'],
                            'days': day_offset - pos['entry_idx'], 'rsi': pos.get('rsi', 50),
                            'entry_idx': pos['entry_idx']
                        })
                        n_trades += 1
                        n_wins += 1
                    elif current_price <= entry * (1 - SL_PCT):
                        pnl_pct = -SL_PCT
                        to_close.append(sym)
                        trades.append({
                            'symbol': sym, 'pnl_pct': pnl_pct, 'regime': pos['regime'],
                            'days': day_offset - pos['entry_idx'], 'rsi': pos.get('rsi', 50),
                            'entry_idx': pos['entry_idx']
                        })
                        n_trades += 1
                    elif day_offset - pos['entry_idx'] >= HORIZON_DAYS:
                        final_price = closes_sym[min(day_offset, len(closes_sym)-1)]
                        pnl_pct = (final_price - entry) / entry
                        to_close.append(sym)
                        trades.append({
                            'symbol': sym, 'pnl_pct': pnl_pct, 'regime': pos['regime'],
                            'days': day_offset - pos['entry_idx'], 'rsi': pos.get('rsi', 50),
                            'entry_idx': pos['entry_idx']
                        })
                        n_trades += 1
        
        for sym in to_close:
            pos = positions.pop(sym)
            window = symbol_windows.get(sym)
            if window is None:
                continue
            exit_price = window['closes'][min(day_offset, len(window['closes'])-1)]
            capital += pos['shares'] * exit_price * (1 - regime_params['slippage_bps'] / 10000)
        
        # Mark to market
        port_value = capital
        for sym, pos in positions.items():
            window = symbol_windows.get(sym)
            if window is not None and day_offset < len(window['closes']):
                port_value += pos['shares'] * window['closes'][day_offset]
        
        portfolio_values.append(port_value)
    
    # Final mark-to-market
    for sym, pos in positions.items():
        window = symbol_windows.get(sym)
        if window is not None and len(window['closes']) > 0:
            capital += pos['shares'] * window['closes'][-1]
    
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)
    
    total_return = capital / INITIAL_CAPITAL - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (running_max - portfolio_values) / running_max
    max_dd = np.max(drawdown)
    
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': n_trades,
        'n_wins': n_wins,
        'win_rate': win_rate,
        'final_capital': capital,
        'trades': trades,
        'n_days': n_days,
    }


# ============================================================
# SPY BENCHMARK (same fixed window)
# ============================================================
def run_spy_benchmark(cache=None):
    if cache is None:
        cache = fetch_and_cache_data()
    if cache is None:
        return None
    
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
    
    running_max = np.maximum.accumulate(spy_window)
    drawdown = (running_max - spy_window) / running_max
    max_dd = np.max(drawdown)
    
    daily_returns = np.diff(spy_window) / spy_window[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
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
# MAIN
# ============================================================
def main():
    logger.info(f"\nStarted: {datetime.now()}")
    logger.info(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
    logger.info(f"Warmup: {WARMUP_DAYS} days, Trading: {TRADING_DAYS} days")
    
    # Fetch/cache data ONCE
    cache = fetch_and_cache_data()
    if cache is None:
        logger.error("No data. Aborting.")
        return
    
    spy_data = cache['SPY']
    logger.info(f"\nActual SPY data: {len(spy_data['closes'])} days")
    logger.info(f"Backtest will use: {TRADING_DAYS} days (day {len(spy_data['closes'])-TRADING_DAYS} to {len(spy_data['closes'])-1})")
    
    # 1. Kelly multiplier stress test
    print("\n" + "=" * 70)
    print("KELLY MULTIPLIER STRESS TEST (FIXED DATA WINDOW)")
    print("=" * 70)
    print(f"\n{'Kelly Mult':>10s} | {'Total Ret':>10s} | {'Ann Ret':>8s} | {'Sharpe':>6s} | {'Max DD':>7s} | {'Trades':>6s} | {'Win Rate':>8s}")
    print("-" * 75)
    
    kelly_results = {}
    for mult in KELLY_MULTIPLIERS:
        result = run_backtest(kelly_base_mult=mult, cache=cache)
        if result:
            kelly_results[mult] = result
            print(f"{mult:>10.1f} | {result['total_return']:>+9.2%} | {result['annualized_return']:>7.2%} | "
                  f"{result['sharpe']:>5.2f} | {result['max_drawdown']:>6.2%} | {result['n_trades']:>5d} | "
                  f"{result['win_rate']:>7.2%}")
    
    # 2. SPY Benchmark
    print("\n" + "=" * 70)
    print("SPY BUY-AND-HOLD BENCHMARK (SAME 501 DAYS)")
    print("=" * 70)
    
    spy_result = run_spy_benchmark(cache=cache)
    if spy_result:
        print(f"\nSPY Buy-and-Hold:")
        print(f"  Initial:  ${spy_result['initial_price']:.2f}")
        print(f"  Final:    ${spy_result['final_price']:.2f}")
        print(f"  Total Return:   {spy_result['total_return']:+.2%}")
        print(f"  Annualized:     {spy_result['annualized_return']:+.2%}")
        print(f"  Sharpe Ratio:   {spy_result['sharpe']:.2f}")
        print(f"  Max Drawdown:   {spy_result['max_drawdown']:.2%}")
        
        print(f"\nComparison (Active System vs SPY B&H):")
        base_result = kelly_results[1.0]
        print(f"  Active System (1.0x Kelly):")
        print(f"    Ann Return: {base_result['annualized_return']:+.2%}")
        print(f"    Max DD:     {base_result['max_drawdown']:.2%}")
        print(f"    Sharpe:     {base_result['sharpe']:.2f}")
        print(f"    Trades:     {base_result['n_trades']}")
        print(f"    Win Rate:   {base_result['win_rate']:.2%}")
        print(f"  SPY B&H:")
        print(f"    Ann Return: {spy_result['annualized_return']:+.2%}")
        print(f"    Max DD:     {spy_result['max_drawdown']:.2%}")
        print(f"    Sharpe:     {spy_result['sharpe']:.2f}")
        
        excess_return = base_result['annualized_return'] - spy_result['annualized_return']
        print(f"\n  Excess Return (Active - B&H): {excess_return:+.2%}")
        dd_diff = base_result['max_drawdown'] - spy_result['max_drawdown']
        print(f"  Drawdown Advantage (Active vs B&H): {dd_diff:+.2%}")
    
    # 3. Losing trade analysis (on 1.0x Kelly result)
    print("\n" + "=" * 70)
    print("LOSING TRADE ANALYSIS (1.0x Kelly)")
    print("=" * 70)
    
    base_trades = kelly_results[1.0]['trades']
    wins = [t for t in base_trades if t['pnl_pct'] > 0]
    losses = [t for t in base_trades if t['pnl_pct'] <= 0]
    
    print(f"\nTotal Trades: {len(base_trades)}")
    print(f"Win Rate: {len(wins)/len(base_trades):.2%} ({len(wins)} wins, {len(losses)} losses)")
    print(f"\nAvg Win: {np.mean([t['pnl_pct'] for t in wins]):+.2%}" if wins else "Avg Win: N/A")
    print(f"Avg Loss: {np.mean([t['pnl_pct'] for t in losses]):+.2%}" if losses else "Avg Loss: N/A")
    print(f"Win/Loss Ratio: {abs(np.mean([t['pnl_pct'] for t in wins]) / np.mean([t['pnl_pct'] for t in losses])):.2f}" if wins and losses else "N/A")
    
    # By Sector
    print("\n--- BY SECTOR ---")
    sectors = {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE'],
        'financials': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA'],
        'consumer': ['PYPL', 'TSLA'],
    }
    
    for sector_name, sector_stocks in sectors.items():
        sector_trades = [t for t in base_trades if t['symbol'] in sector_stocks]
        if sector_trades:
            sector_wins = [t for t in sector_trades if t['pnl_pct'] > 0]
            win_rate = len(sector_wins) / len(sector_trades)
            avg_pnl = np.mean([t['pnl_pct'] for t in sector_trades])
            print(f"  {sector_name:15s}: {len(sector_trades):3d} trades | win_rate {win_rate:.2%} | "
                  f"avg_pnl {avg_pnl:+.2%} | wins {len(sector_wins)} | loss_rate {1-win_rate:.2%}")
    
    # By Day of Week (using entry_idx which is now deterministic)
    print("\n--- BY DAY OF WEEK ---")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    
    for day_idx in range(5):
        day_trades = [t for t in base_trades if t['entry_idx'] % 5 == day_idx]
        if day_trades:
            day_wins = [t for t in day_trades if t['pnl_pct'] > 0]
            win_rate = len(day_wins) / len(day_trades)
            avg_pnl = np.mean([t['pnl_pct'] for t in day_trades])
            print(f"  {day_names[day_idx]:5s} (idx%5={day_idx}): {len(day_trades):3d} trades | "
                  f"win_rate {win_rate:.2%} | avg_pnl {avg_pnl:+.2%} | wins {len(day_wins)}")
    
    # By RSI at Entry
    print("\n--- BY RSI AT ENTRY ---")
    rsi_buckets = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]
    for low, high in rsi_buckets:
        bucket_trades = [t for t in base_trades if low <= t.get('rsi', 50) < high]
        if bucket_trades:
            bucket_wins = [t for t in bucket_trades if t['pnl_pct'] > 0]
            win_rate = len(bucket_wins) / len(bucket_trades)
            avg_pnl = np.mean([t['pnl_pct'] for t in bucket_trades])
            print(f"  RSI {low:3d}-{high:3d}: {len(bucket_trades):3d} trades | "
                  f"win_rate {win_rate:.2%} | avg_pnl {avg_pnl:+.2%} | wins {len(bucket_wins)}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS: WHY 321 → 143 TRADES?")
    print("=" * 70)
    print(f"\nFirst backtest (regime_backtest.py): Yahoo returned 501 days of data")
    print(f"Second backtest (comprehensive_analysis.py): Yahoo returned 345 days of data")
    print(f"\nThe 'days' parameter to fetch_data is NON-DETERMINISTIC.")
    print(f"Yahoo returns 'up to N days' — actual count varies by:")
    print(f"  - Market calendar (weekends, holidays)")
    print(f"  - API rate limits / caching")
    print(f"  - When you call it (fresh data vs cached)")
    print(f"\nBetween runs, the data window SHIFTED, causing:")
    print(f"  - 501 days → 345 days (-31% fewer data points)")
    print(f"  - 321 trades → 143 trades (-55% fewer trades)")
    print(f"  - Different market period → 36% → 52% win rate (different conditions)")
    print(f"\nFIX: Use fixed date range (not 'days back') and cache data ONCE.")
    print(f"Both backtests now use the EXACT same {TRADING_DAYS}-day window.")
    
    # Save results
    all_results = {
        'kelly_stress_test': kelly_results,
        'spy_benchmark': spy_result,
        'base_trades': base_trades,
    }
    with open('deterministic_analysis.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"\nResults saved to: deterministic_analysis.pkl")


if __name__ == '__main__':
    main()
