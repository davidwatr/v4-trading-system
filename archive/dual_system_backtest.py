#!/usr/bin/env python3
"""
Dual-System Regime-Switching Backtest
Runs Mean Reversion + Trend Following in parallel.
Meta-regime detector picks the best system each day.
Uses the SAME fixed 501-day window as deterministic_backtest.py.
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

from production_v18 import calc_rsi, calc_atr, calc_adx, calc_macd, fetch_data

# Local SMA (production_v18 has no calc_sma)
def calc_sma(closes, period):
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period

# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_CAPITAL = 100000
TP_PCT = 0.03
SL_PCT = 0.02
HORIZON_DAYS = 5

# Use the SAME fixed date range as deterministic_backtest
DATA_CACHE = Path('deterministic_data_cache.pkl')
TRADING_DAYS = 501
WARMUP_DAYS = 200

UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC',
            'JPM', 'BAC', 'GS', 'MS', 'PYPL', 'TSLA', 'V', 'MA', 'CRM', 'ADBE']


# ============================================================
# LOAD CACHED DATA
# ============================================================
def load_cache():
    if not DATA_CACHE.exists():
        logger.error("No data cache. Run deterministic_backtest.py first.")
        return None
    with open(DATA_CACHE, 'rb') as f:
        return pickle.load(f)


def get_backtest_window(cache):
    """Extract the exact same 501-day backtest window used by deterministic_backtest."""
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
    """Get the symbol data aligned to the backtest window."""
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
# REGIME DETECTION — Trend Strength Classifier
# ============================================================
def compute_regime_features(spy_closes, day_idx):
    """
    Compute features that distinguish trending vs mean-reverting markets.
    Returns dict of features used for regime classification.
    """
    if day_idx < 50:
        return None
    
    closes = spy_closes[max(0, day_idx-200):day_idx+1]
    if len(closes) < 50:
        return None
    
    # Trend strength features
    sma20 = calc_sma(closes, 20)
    sma50 = calc_sma(closes, 50)
    sma200 = calc_sma(closes, 200)
    adx_val = calc_adx(closes, 14)
    atr_val = calc_atr(closes, 14)
    
    price = closes[-1]
    
    # Trend score: how strong and consistent is the trend?
    trend_strength = 0.0
    if sma20 and sma50 and sma200:
        # Price position relative to MAs
        above_sma20 = (price - sma20) / sma20
        above_sma50 = (price - sma50) / sma50
        above_sma200 = (price - sma200) / sma200
        
        # MA alignment (bullish: SMA20 > SMA50 > SMA200)
        ma_alignment = 0.0
        if sma20 > sma50:
            ma_alignment += 1.0
        if sma50 > sma200:
            ma_alignment += 1.0
        
        # SMA20 slope (is the short-term trend rising?)
        sma20_slope = (sma20 - calc_sma(closes[:-10], 20)) / (calc_sma(closes[:-10], 20) + 1e-10) if len(closes) > 30 else 0
        
        trend_strength = (above_sma200 * 0.4 + ma_alignment * 0.3 + max(0, sma20_slope) * 0.3)
    
    # Volatility regime
    vol_regime = atr_val / price if atr_val and price > 0 else 0.02
    
    return {
        'price': price,
        'sma20': sma20,
        'sma50': sma50,
        'sma200': sma200,
        'adx': adx_val,
        'atr': atr_val,
        'trend_strength': trend_strength,
        'vol_regime': vol_regime,
        'above_sma50': (price > sma50) if sma50 else False,
        'above_sma200': (price > sma200) if sma200 else False,
    }


def classify_regime(features):
    """
    Classify market regime into one of three modes:
    - 'trend': Strong trend — use trend following
    - 'mean_reversion': Ranging/oversold — use mean reversion
    - 'neutral': No clear edge — reduce size
    
    Decision logic:
    - ADX > 25 + price above SMA50 = trending
    - ADX < 20 or RSI extreme = mean reversion opportunity
    - Otherwise = neutral
    """
    if features is None:
        return 'neutral'
    
    adx = features['adx'] or 20
    price = features['price']
    sma50 = features['sma50']
    sma200 = features['sma200']
    trend_strength = features['trend_strength']
    
    # Trending regime
    if adx > 25 and price > (sma50 or 0) and price > (sma200 or 0):
        return 'trend'
    
    # Downtrend — mean reversion can still work on dips
    if adx > 25 and price < (sma200 or 999999):
        return 'mean_reversion'
    
    # Low volatility / ranging — mean reversion
    if adx < 20:
        return 'mean_reversion'
    
    # Moderate — default to trend following if above SMA200, else mean reversion
    if price > (sma200 or 0):
        return 'trend'
    else:
        return 'mean_reversion'


# ============================================================
# TREND FOLLOWING SIGNALS
# ============================================================
def trend_following_signal(closes, day_idx):
    """
    Trend following entry signal.
    Enter when:
    - Price above 20-day SMA AND 50-day SMA (confirmed uptrend)
    - OR price makes fresh 20-day high (momentum breakout)
    
    Returns (strength, direction, rsi) or (0, 0, 0) if no signal.
    """
    if day_idx < 50:
        return 0, 0, 0
    
    closes_slice = closes[:day_idx+1]
    if len(closes_slice) < 50:
        return 0, 0, 0
    
    current_price = closes_slice[-1]
    
    # Moving averages
    sma20 = calc_sma(closes_slice, 20)
    sma50 = calc_sma(closes_slice, 50)
    
    if sma20 is None or sma50 is None:
        return 0, 0, 0
    
    # Trend condition: price above both MAs
    above_sma20 = current_price > sma20
    above_sma50 = current_price > sma50
    
    # Momentum: 20-day high breakout
    lookback_20d_high = max(closes_slice[-20:]) if len(closes_slice) >= 20 else current_price
    near_20d_high = current_price >= lookback_20d_high * 0.98  # Within 2% of 20-day high
    
    # RSI for confirmation (not for entry trigger — trend following doesn't need oversold)
    rsi_val = calc_rsi(closes_slice, 14)
    
    # Strength based on trend alignment
    strength = 0.0
    if above_sma20 and above_sma50:
        # How far above SMA50?
        dist_from_sma50 = (current_price - sma50) / sma50
        strength = min(0.15, 0.05 + dist_from_sma50 * 0.5)
    elif near_20d_high and above_sma20:
        strength = 0.08
    elif above_sma20 and not above_sma50:
        # Pullback in uptrend — buy the dip
        strength = 0.04
    
    if strength > 0:
        return strength, 1, rsi_val or 50
    
    return 0, 0, rsi_val or 50


def trend_following_signal_v2(closes, day_idx):
    """
    Alternative trend following: Donchian channel breakout.
    Enter when price breaks above 20-day high.
    """
    if day_idx < 20:
        return 0, 0, 0
    
    closes_slice = closes[:day_idx+1]
    current_price = closes_slice[-1]
    
    # 20-day Donchian high
    donchian_high = max(closes_slice[-20:])
    donchian_low = min(closes_slice[-20:])
    
    # Breakout above 20-day high
    if current_price > donchian_high and len(closes_slice) >= 20:
        # How strong is the breakout?
        breakout_pct = (current_price - donchian_high) / donchian_high
        strength = min(0.12, 0.03 + breakout_pct * 2)
        rsi_val = calc_rsi(closes_slice, 14) or 50
        return strength, 1, rsi_val
    
    return 0, 0, calc_rsi(closes_slice, 14) or 50


# ============================================================
# MEAN REVERSION SIGNALS (existing, slightly modified)
# ============================================================
def mean_reversion_signal_v2(closes, day_idx):
    """
    Mean reversion entry: RSI < 40 (oversold bounce).
    Same as existing but with explicit day_idx.
    """
    if day_idx < 20:
        return 0, 0, 0
    
    closes_slice = closes[:day_idx+1]
    if len(closes_slice) < 20:
        return 0, 0, 0
    
    if not momentum_filter(closes_slice):
        return 0, 0, 0
    
    strength, direction, rsi_val = mean_reversion_signal(closes_slice)
    return strength, direction, rsi_val


# ============================================================
# BACKTEST
# ============================================================
def run_dual_system_backtest(cache=None, use_v2_trend=False):
    if cache is None:
        cache = load_cache()
    if cache is None:
        return None
    
    window = get_backtest_window(cache)
    if window is None:
        return None
    
    spy_closes = window['spy_closes']
    backtest_start = window['backtest_start']
    
    # Pre-load symbol windows
    symbol_windows = {}
    for sym in UNIVERSE:
        sw = get_symbol_window(cache, sym, backtest_start, WARMUP_DAYS)
        if sw:
            symbol_windows[sym] = sw
    
    logger.info(f"Backtest: {TRADING_DAYS} days, {len(symbol_windows)} symbols")
    
    # Backtest state
    capital = INITIAL_CAPITAL
    positions = {}  # symbol -> {entry_price, entry_idx, shares, system, regime, ...}
    portfolio_values = [capital]
    trades = []
    
    # For each day in the backtest
    for day_offset in range(TRADING_DAYS):
        day = backtest_start + day_offset
        
        # 1. Determine regime
        spy_window = spy_closes[:day+1]
        features = compute_regime_features(spy_window, len(spy_window)-1)
        regime = classify_regime(features)
        
        # 2. Generate signals from BOTH systems
        mr_signals = []  # Mean reversion signals
        tf_signals = []  # Trend following signals
        
        for sym, sw in symbol_windows.items():
            closes = sw['closes']
            if day_offset >= len(closes):
                continue
            if day_offset < 20:
                continue
            
            closes_slice = closes[:day_offset+1]
            
            # Mean reversion signal
            mr_strength, mr_dir, mr_rsi = mean_reversion_signal_v2(closes, day_offset)
            if mr_dir == 1 and mr_strength > 0:
                mr_signals.append({
                    'symbol': sym,
                    'strength': mr_strength,
                    'rsi': mr_rsi,
                    'price': closes[day_offset],
                    'system': 'mean_reversion',
                })
            
            # Trend following signal
            if use_v2_trend:
                tf_strength, tf_dir, tf_rsi = trend_following_signal_v2(closes, day_offset)
            else:
                tf_strength, tf_dir, tf_rsi = trend_following_signal(closes, day_offset)
            
            if tf_dir == 1 and tf_strength > 0:
                tf_signals.append({
                    'symbol': sym,
                    'strength': tf_strength,
                    'rsi': tf_rsi,
                    'price': closes[day_offset],
                    'system': 'trend_following',
                })
        
        # 3. Select signals based on regime
        active_signals = []
        
        if regime == 'trend':
            # Use trend following ONLY
            active_signals = sorted(tf_signals, key=lambda x: -x['strength'])
            logger.debug(f"Day {day_offset}: TREND regime — {len(tf_signals)} TF signals, {len(mr_signals)} MR signals ignored")
        elif regime == 'mean_reversion':
            # Use mean reversion ONLY
            active_signals = sorted(mr_signals, key=lambda x: -x['strength'])
            logger.debug(f"Day {day_offset}: MR regime — {len(mr_signals)} MR signals, {len(tf_signals)} TF signals ignored")
        else:
            # Neutral — use BOTH but with reduced size
            combined = mr_signals + tf_signals
            active_signals = sorted(combined, key=lambda x: -x['strength'])
            logger.debug(f"Day {day_offset}: NEUTRAL regime — {len(mr_signals)} MR + {len(tf_signals)} TF signals, reduced size")
        
        # 4. Execute trades
        max_positions = 8 if regime == 'trend' else (6 if regime == 'mean_reversion' else 4)
        kelly_base = 0.08 if regime == 'trend' else (0.06 if regime == 'mean_reversion' else 0.04)
        
        current_positions = len(positions)
        slots_available = max(0, max_positions - current_positions)
        
        for sig in active_signals[:slots_available]:
            sym = sig['symbol']
            if sym in positions:
                continue
            
            # Kelly sizing
            kelly_f = min(sig['strength'], kelly_base)
            if regime == 'neutral':
                kelly_f *= 0.5  # Reduce in neutral
            
            position_value = capital * kelly_f
            entry_price = sig['price']
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            # Slippage
            slippage_bps = 10 if regime == 'trend' else (12 if regime == 'mean_reversion' else 15)
            cost = shares * entry_price * (1 + slippage_bps / 10000)
            
            if capital - cost < 0:
                continue
            
            # Determine exit parameters
            if sig['system'] == 'trend_following':
                exit_tp = 0.05  # Wider target for trend
                exit_sl = 0.025  # Wider stop for trend
                exit_horizon = 10  # Longer hold for trend
            else:
                exit_tp = TP_PCT
                exit_sl = SL_PCT
                exit_horizon = HORIZON_DAYS
            
            capital -= cost
            positions[sym] = {
                'entry_price': entry_price,
                'entry_idx': day_offset,
                'shares': shares,
                'system': sig['system'],
                'regime': regime,
                'tp_pct': exit_tp,
                'sl_pct': exit_sl,
                'horizon': exit_horizon,
                'rsi': sig['rsi'],
            }
        
        # 5. Check exits
        to_close = []
        for sym, pos in positions.items():
            if day_offset <= pos['entry_idx']:
                continue
            
            sw = symbol_windows.get(sym)
            if sw is None:
                continue
            closes_sym = sw['closes']
            if day_offset >= len(closes_sym):
                continue
            
            current_price = closes_sym[day_offset]
            entry = pos['entry_price']
            
            hit_tp = current_price >= entry * (1 + pos['tp_pct'])
            hit_sl = current_price <= entry * (1 - pos['sl_pct'])
            hit_time = (day_offset - pos['entry_idx']) >= pos['horizon']
            
            if hit_tp or hit_sl or hit_time:
                if hit_tp:
                    pnl_pct = pos['tp_pct']
                elif hit_sl:
                    pnl_pct = -pos['sl_pct']
                else:
                    final_price = closes_sym[min(day_offset, len(closes_sym)-1)]
                    pnl_pct = (final_price - entry) / entry if entry > 0 else 0
                
                to_close.append(sym)
                trades.append({
                    'symbol': sym,
                    'pnl_pct': pnl_pct,
                    'system': pos['system'],
                    'regime': pos['regime'],
                    'days': day_offset - pos['entry_idx'],
                    'rsi': pos.get('rsi', 50),
                    'entry_idx': pos['entry_idx'],
                })
        
        for sym in to_close:
            pos = positions.pop(sym)
            sw = symbol_windows.get(sym)
            if sw is None:
                continue
            exit_price = sw['closes'][min(day_offset, len(sw['closes'])-1)]
            slippage_bps = 10
            capital += pos['shares'] * exit_price * (1 - slippage_bps / 10000)
        
        # Mark to market
        port_value = capital
        for sym, pos in positions.items():
            sw = symbol_windows.get(sym)
            if sw is not None and day_offset < len(sw['closes']):
                port_value += pos['shares'] * sw['closes'][day_offset]
        
        portfolio_values.append(port_value)
    
    # Final valuation
    for sym, pos in positions.items():
        sw = symbol_windows.get(sym)
        if sw is not None and len(sw['closes']) > 0:
            capital += pos['shares'] * sw['closes'][-1]
    
    # Compute metrics
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)
    
    total_return = capital / INITIAL_CAPITAL - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (running_max - portfolio_values) / running_max
    max_dd = np.max(drawdown)
    
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': len(trades),
        'n_wins': len(wins),
        'n_losses': len(losses),
        'win_rate': win_rate,
        'avg_win': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
        'avg_loss': np.mean([t['pnl_pct'] for t in losses]) if losses else 0,
        'final_capital': capital,
        'trades': trades,
        'regime_counts': {},
        'system_counts': {},
    }


def analyze_results(result):
    """Analyze backtest results by system and regime."""
    if result is None:
        return
    
    trades = result['trades']
    
    # Count by system
    mr_trades = [t for t in trades if t['system'] == 'mean_reversion']
    tf_trades = [t for t in trades if t['system'] == 'trend_following']
    
    mr_wins = [t for t in mr_trades if t['pnl_pct'] > 0]
    tf_wins = [t for t in tf_trades if t['pnl_pct'] > 0]
    
    # Count by regime
    trend_trades = [t for t in trades if t['regime'] == 'trend']
    mr_regime_trades = [t for t in trades if t['regime'] == 'mean_reversion']
    neutral_trades = [t for t in trades if t['regime'] == 'neutral']
    
    result['system_counts'] = {
        'mean_reversion': {'trades': len(mr_trades), 'wins': len(mr_wins), 
                          'win_rate': len(mr_wins)/len(mr_trades) if mr_trades else 0},
        'trend_following': {'trades': len(tf_trades), 'wins': len(tf_wins),
                          'win_rate': len(tf_wins)/len(tf_trades) if tf_trades else 0},
    }
    
    result['regime_counts'] = {
        'trend': len(trend_trades),
        'mean_reversion': len(mr_regime_trades),
        'neutral': len(neutral_trades),
    }
    
    return result


# ============================================================
# RUN ALL VARIANTS
# ============================================================
def main():
    logger.info(f"\n{'='*70}")
    logger.info(f"DUAL SYSTEM REGIME-SWITCHING BACKTEST")
    logger.info(f"{'='*70}")
    
    cache = load_cache()
    if cache is None:
        return
    
    window = get_backtest_window(cache)
    spy_closes = window['spy_closes']
    backtest_start = window['backtest_start']
    
    # SPY benchmark
    spy_window = spy_closes[backtest_start:]
    spy_initial = spy_window[0]
    spy_final = spy_window[-1]
    spy_total_ret = (spy_final / spy_initial) - 1
    spy_ann_ret = (1 + spy_total_ret) ** (252 / len(spy_window)) - 1
    spy_daily = np.diff(spy_window) / spy_window[:-1]
    spy_sharpe = np.mean(spy_daily) / np.std(spy_daily) * np.sqrt(252) if np.std(spy_daily) > 0 else 0
    spy_running_max = np.maximum.accumulate(spy_window)
    spy_max_dd = np.max((spy_running_max - spy_window) / spy_running_max)
    
    logger.info(f"\nSPY Benchmark ({len(spy_window)} days):")
    logger.info(f"  Total Return: {spy_total_ret:+.2%}")
    logger.info(f"  Annualized:   {spy_ann_ret:+.2%}")
    logger.info(f"  Sharpe:       {spy_sharpe:.2f}")
    logger.info(f"  Max DD:       {spy_max_dd:.2%}")
    
    # Run both variants
    print(f"\n{'='*70}")
    print("VARIANT 1: Trend Following (MA crossover) + Mean Reversion")
    print(f"{'='*70}")
    
    result1 = run_dual_system_backtest(cache, use_v2_trend=False)
    if result1:
        result1 = analyze_results(result1)
        print_results(result1, spy_total_ret, spy_ann_ret, spy_sharpe, spy_max_dd, "V1 (MA crossover TF)")
    
    print(f"\n{'='*70}")
    print("VARIANT 2: Trend Following (Donchian breakout) + Mean Reversion")
    print(f"{'='*70}")
    
    result2 = run_dual_system_backtest(cache, use_v2_trend=True)
    if result2:
        result2 = analyze_results(result2)
        print_results(result2, spy_total_ret, spy_ann_ret, spy_sharpe, spy_max_dd, "V2 (Donchian breakout TF)")
    
    # Compare
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    if result1 and result2:
        print(f"\n{'System':<35s} | {'AnnRet':>7s} | {'Sharpe':>6s} | {'MaxDD':>7s} | {'Trades':>6s} | {'WinRt':>7s} | {'AvgWin':>7s} | {'AvgLoss':>8s}")
        print("-" * 95)
        print(f"{'V1: MA crossover TF + MR':<35s} | {result1['annualized_return']:>+6.2%} | {result1['sharpe']:>5.2f} | {result1['max_drawdown']:>6.2%} | {result1['n_trades']:>5d} | {result1['win_rate']:>6.2%} | {result1['avg_win']:>+6.2%} | {result1['avg_loss']:>+7.2%}")
        print(f"{'V2: Donchian breakout TF + MR':<35s} | {result2['annualized_return']:>+6.2%} | {result2['sharpe']:>5.2f} | {result2['max_drawdown']:>6.2%} | {result2['n_trades']:>5d} | {result2['win_rate']:>6.2%} | {result2['avg_win']:>+6.2%} | {result2['avg_loss']:>+7.2%}")
        print(f"{'SPY Buy-and-Hold':<35s} | {spy_ann_ret:>+6.2%} | {spy_sharpe:>5.2f} | {spy_max_dd:>6.2%} | {'—':>5s} | {'—':>6s} | {'—':>6s} | {'—':>7s}")
        
        # Best system breakdown
        print(f"\n--- System Breakdown (Best Variant) ---")
        best = result2 if result2['annualized_return'] > result1['annualized_return'] else result1
        label = "V2 (Donchian)" if best == result2 else "V1 (MA crossover)"
        print(f"  Mean Reversion trades: {best['system_counts']['mean_reversion']['trades']} | "
              f"Win Rate: {best['system_counts']['mean_reversion']['win_rate']:.2%}")
        print(f"  Trend Following trades: {best['system_counts']['trend_following']['trades']} | "
              f"Win Rate: {best['system_counts']['trend_following']['win_rate']:.2%}")
        
        mr_sw = best['system_counts']['mean_reversion']
        tf_sw = best['system_counts']['trend_following']
        mr_pct = mr_sw['trades'] / best['n_trades'] * 100 if best['n_trades'] > 0 else 0
        tf_pct = tf_sw['trades'] / best['n_trades'] * 100 if best['n_trades'] > 0 else 0
        print(f"  MR: {mr_sw['trades']} trades ({mr_pct:.0f}%) | TF: {tf_sw['trades']} trades ({tf_pct:.0f}%)")
    
    # Save
    all_results = {
        'variant1': result1,
        'variant2': result2,
        'spy_benchmark': {
            'total_return': spy_total_ret,
            'annualized_return': spy_ann_ret,
            'sharpe': spy_sharpe,
            'max_drawdown': spy_max_dd,
        }
    }
    with open('dual_system_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    logger.info(f"\nResults saved to: dual_system_results.pkl")


def print_results(result, spy_total, spy_ann, spy_sharpe, spy_max_dd, label):
    trades = result['trades']
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    
    mr = result['system_counts']['mean_reversion']
    tf = result['system_counts']['trend_following']
    
    print(f"\n  Total Return:   {result['total_return']:+.2%}")
    print(f"  Annualized:     {result['annualized_return']:+.2%}")
    print(f"  Sharpe Ratio:   {result['sharpe']:.2f}")
    print(f"  Max Drawdown:   {result['max_drawdown']:.2%}")
    print(f"  Total Trades:   {result['n_trades']}")
    print(f"  Win Rate:       {result['win_rate']:.2%} ({result['n_wins']}W / {result['n_losses']}L)")
    print(f"  Avg Win:        {result['avg_win']:+.2%}")
    print(f"  Avg Loss:       {result['avg_loss']:+.2%}")
    print(f"  Win/Loss Ratio: {abs(result['avg_win'] / result['avg_loss']):.2f}" if losses else "  Win/Loss Ratio: N/A")
    
    print(f"\n  --- By System ---")
    print(f"  Mean Reversion: {mr['trades']} trades | Win Rate: {mr['win_rate']:.2%} ({mr['wins']} wins)")
    print(f"  Trend Following: {tf['trades']} trades | Win Rate: {tf['win_rate']:.2%} ({tf['wins']} wins)")
    
    mr_pct = mr['trades'] / result['n_trades'] * 100 if result['n_trades'] > 0 else 0
    tf_pct = tf['trades'] / result['n_trades'] * 100 if result['n_trades'] > 0 else 0
    print(f"  MR: {mr['trades']} ({mr_pct:.0f}%) | TF: {tf['trades']} ({tf_pct:.0f}%)")
    
    print(f"\n  --- By Regime (Day Classification) ---")
    print(f"  Trend days:         {result['regime_counts'].get('trend', 0)}")
    print(f"  Mean Reversion days: {result['regime_counts'].get('mean_reversion', 0)}")
    print(f"  Neutral days:       {result['regime_counts'].get('neutral', 0)}")
    
    excess = result['annualized_return'] - spy_ann
    dd_diff = result['max_drawdown'] - spy_max_dd
    print(f"\n  vs SPY: Excess Return {excess:+.2%} | DD diff {dd_diff:+.2%}")


if __name__ == '__main__':
    main()
