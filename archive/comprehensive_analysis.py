#!/usr/bin/env python3
"""
Comprehensive Analysis:
1. Kelly multiplier stress test (1.5x, 2x, 2.5x, 3x)
2. SPY buy-and-hold benchmark over same 501 days
3. Losing trade pattern analysis (sector, day of week, volatility)
"""

import numpy as np
import pandas as pd
import pickle
import logging
from datetime import datetime
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

# Kelly multiplier stress test values
KELLY_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]

# ============================================================
# REGIME DETECTION (same as before)
# ============================================================
def compute_regime_features(spy_closes):
    if len(spy_closes) < 200:
        return np.array([0, 0, 0, 0])
    
    closes = np.array(spy_closes)
    
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
    """Kelly params with configurable base multiplier."""
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
# BACKTEST (parameterized by Kelly multiplier)
# ============================================================
def run_backtest_with_kelly_mult(kelly_base_mult=1.0, date_range_days=501, symbols=None):
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC',
                   'JPM', 'BAC', 'GS', 'MS', 'PYPL', 'TSLA', 'V', 'MA', 'CRM', 'ADBE']
    
    # Fetch data
    all_data = {}
    for sym in symbols:
        closes, volumes, highs, lows = fetch_data(sym, days=date_range_days)
        if closes is not None and len(closes) >= 100:
            all_data[sym] = {'closes': closes, 'volumes': volumes, 'highs': highs, 'lows': lows}
    
    spy_closes, _, spy_highs, spy_lows = fetch_data('SPY', days=date_range_days)
    if spy_closes is None or len(spy_closes) < 200:
        return None
    
    # Build regime classifier
    kmeans, scaler, cluster_map = build_regime_classifier(spy_closes)
    
    # Backtest
    capital = INITIAL_CAPITAL
    positions = {}
    portfolio_values = [capital]
    trades = []
    n_trades = 0
    n_wins = 0
    
    n_days = min(len(v['closes']) for v in all_data.values())
    n_days = min(n_days, len(spy_closes))
    
    for day in range(200, n_days):
        regime = get_regime_class(spy_closes[:day+1], kmeans, scaler, cluster_map)
        regime_params = get_regime_kelly_params(regime, base_kelly_mult=kelly_base_mult)
        
        # Generate signals
        signals = []
        for sym, data in all_data.items():
            closes = data['closes']
            if day >= len(closes):
                continue
            if not momentum_filter(closes[:day+1]):
                continue
            strength, direction, rsi_val = mean_reversion_signal(closes[:day+1])
            if direction == 1 and strength > 0:
                signals.append({
                    'symbol': sym,
                    'strength': strength,
                    'kelly_size': min(strength * 0.10, 0.10),
                    'rsi': rsi_val,
                    'current_price': closes[day],
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
                'entry_idx': day,
                'shares': shares,
                'kelly_f': kelly_f,
                'regime': regime,
                'symbol': symbol,
            }
        
        # Check exits
        to_close = []
        for sym, pos in positions.items():
            if day > pos['entry_idx']:
                closes_sym = all_data[sym]['closes']
                if day < len(closes_sym):
                    current_price = closes_sym[day]
                    entry = pos['entry_price']
                    
                    if current_price >= entry * (1 + TP_PCT):
                        pnl_pct = TP_PCT
                        to_close.append(sym)
                        trades.append({'symbol': sym, 'pnl_pct': pnl_pct, 'regime': pos['regime'],
                                       'days': day - pos['entry_idx'], 'rsi': pos.get('rsi', 50)})
                        n_trades += 1
                        n_wins += 1
                    elif current_price <= entry * (1 - SL_PCT):
                        pnl_pct = -SL_PCT
                        to_close.append(sym)
                        trades.append({'symbol': sym, 'pnl_pct': pnl_pct, 'regime': pos['regime'],
                                       'days': day - pos['entry_idx'], 'rsi': pos.get('rsi', 50)})
                        n_trades += 1
                    elif day - pos['entry_idx'] >= HORIZON_DAYS:
                        final_price = closes_sym[min(day, len(closes_sym)-1)]
                        pnl_pct = (final_price - entry) / entry
                        to_close.append(sym)
                        trades.append({'symbol': sym, 'pnl_pct': pnl_pct, 'regime': pos['regime'],
                                       'days': day - pos['entry_idx'], 'rsi': pos.get('rsi', 50)})
                        n_trades += 1
        
        for sym in to_close:
            pos = positions.pop(sym)
            exit_price = all_data[sym]['closes'][min(day, len(all_data[sym]['closes'])-1)]
            capital += pos['shares'] * exit_price * (1 - regime_params['slippage_bps'] / 10000)
        
        # Mark to market
        port_value = capital
        for sym, pos in positions.items():
            if day < len(all_data[sym]['closes']):
                port_value += pos['shares'] * all_data[sym]['closes'][day]
        
        portfolio_values.append(port_value)
    
    # Final mark-to-market
    for sym, pos in positions.items():
        if len(all_data[sym]['closes']) > 0:
            port_value = capital + sum(
                p['shares'] * all_data[s]['closes'][-1]
                for s, p in positions.items() if len(all_data[s]['closes']) > 0
            )
    
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)
    
    total_return = portfolio_values[-1] / INITIAL_CAPITAL - 1
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
        'final_capital': portfolio_values[-1],
        'trades': trades,
    }


# ============================================================
# SPY BENCHMARK
# ============================================================
def run_spy_benchmark(days=501):
    """Buy-and-hold SPY over the same period."""
    spy_closes, _, _, _ = fetch_data('SPY', days=days + 200)  # Extra for warm-up
    if spy_closes is None or len(spy_closes) < 200:
        return None
    
    # Use same 501-day window as backtest (starting after 200-day warm-up)
    spy_closes = spy_closes[-days:]
    
    initial_price = spy_closes[0]
    final_price = spy_closes[-1]
    
    total_return = (final_price / initial_price) - 1
    ann_return = (1 + total_return) ** (252 / len(spy_closes)) - 1
    
    # Max drawdown
    running_max = np.maximum.accumulate(spy_closes)
    drawdown = (running_max - spy_closes) / running_max
    max_dd = np.max(drawdown)
    
    # Daily returns for Sharpe
    daily_returns = np.diff(spy_closes) / spy_closes[:-1]
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    # Price history for context
    prices = spy_closes
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'initial_price': initial_price,
        'final_price': final_price,
        'prices': prices,
        'n_days': len(spy_closes),
    }


# ============================================================
# LOSING TRADE ANALYSIS
# ============================================================
def analyze_losing_trades(trades, all_data):
    """Analyze patterns in losing trades."""
    losing = [t for t in trades if t['pnl_pct'] < 0]
    winning = [t for t in trades if t['pnl_pct'] > 0]
    
    print("=" * 60)
    print("LOSING TRADE ANALYSIS")
    print("=" * 60)
    
    # Win/Loss ratio
    wins = [t['pnl_pct'] for t in winning]
    losses = [t['pnl_pct'] for t in losing]
    
    print(f"\nTotal Trades: {len(trades)}")
    print(f"Win Rate: {len(winning)/len(trades):.2%} ({len(winning)} wins, {len(losing)} losses)")
    print(f"\nAvg Win: {np.mean(wins):+.2%}" if wins else "Avg Win: N/A")
    print(f"Avg Loss: {np.mean(losses):+.2%}" if losses else "Avg Loss: N/A")
    print(f"Win/Loss Ratio: {abs(np.mean(wins) / np.mean(losses)):.2f}" if wins and losses else "Win/Loss Ratio: N/A")
    
    # By Sector
    print("\n--- BY SECTOR ---")
    sectors = {
        'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE'],
        'financials': ['JPM', 'BAC', 'GS', 'MS', 'V', 'MA'],
        'consumer': ['PYPL', 'TSLA'],
    }
    
    for sector_name, sector_stocks in sectors.items():
        sector_trades = [t for t in trades if t['symbol'] in sector_stocks]
        sector_losses = [t for t in sector_trades if t['pnl_pct'] < 0]
        if sector_trades:
            win_rate = len([t for t in sector_trades if t['pnl_pct'] > 0]) / len(sector_trades)
            avg_pnl = np.mean([t['pnl_pct'] for t in sector_trades])
            print(f"  {sector_name:15s}: {len(sector_trades):3d} trades | win_rate {win_rate:.2%} | "
                  f"avg_pnl {avg_pnl:+.2%} | losses {len(sector_losses)} | "
                  f"loss_rate {len(sector_losses)/len(sector_trades):.2%}")
    
    # By Day of Week
    print("\n--- BY DAY OF WEEK ---")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    # We need to map entry_idx to actual day of week
    # Since we don't have the actual dates, use index modulo 5 as proxy
    for day_idx in range(5):
        day_trades = [t for t in trades if t['entry_idx'] % 5 == day_idx]
        day_losses = [t for t in day_trades if t['pnl_pct'] < 0]
        if day_trades:
            win_rate = len([t for t in day_trades if t['pnl_pct'] > 0]) / len(day_trades)
            avg_pnl = np.mean([t['pnl_pct'] for t in day_trades])
            print(f"  Day {day_idx} ({day_names[day_idx]}): {len(day_trades):3d} trades | "
                  f"win_rate {win_rate:.2%} | avg_pnl {avg_pnl:+.2%} | losses {len(day_losses)}")
    
    # By RSI at Entry
    print("\n--- BY RSI AT ENTRY ---")
    rsi_buckets = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]
    for low, high in rsi_buckets:
        bucket_trades = [t for t in trades if low <= t.get('rsi', 50) < high]
        bucket_losses = [t for t in bucket_trades if t['pnl_pct'] < 0]
        if bucket_trades:
            win_rate = len([t for t in bucket_trades if t['pnl_pct'] > 0]) / len(bucket_trades)
            avg_pnl = np.mean([t['pnl_pct'] for t in bucket_trades])
            print(f"  RSI {low:3d}-{high:3d}: {len(bucket_trades):3d} trades | "
                  f"win_rate {win_rate:.2%} | avg_pnl {avg_pnl:+.2%} | losses {len(bucket_losses)}")
    
    # Conclusion
    print("\n--- SUMMARY ---")
    if losses:
        avg_loss = np.mean(losses)
        print(f"  Average loss: {avg_loss:+.2%}")
        print(f"  Largest loss: {min(losses):+.2%}")
        print(f"  Smallest loss: {max(losses):+.2%}")
        
        # Are losses mostly stop-losses?
        stop_losses = [t for t in losing if t['pnl_pct'] <= -SL_PCT + 0.001]
        time_exits = [t for t in losing if t['pnl_pct'] > -SL_PCT + 0.001 and t['pnl_pct'] < 0]
        print(f"\n  Loss Breakdown:")
        print(f"    Stop-loss hits: {len(stop_losses)} ({len(stop_losses)/len(losing):.2%})")
        print(f"    Time/other exits: {len(time_exits)} ({len(time_exits)/len(losing):.2%})")


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info(f"\nStarted: {datetime.now()}")
    
    # 1. Kelly multiplier stress test
    print("\n" + "=" * 60)
    print("KELLY MULTIPLIER STRESS TEST")
    print("=" * 60)
    print(f"\n{'Kelly Mult':>10s} | {'Total Ret':>10s} | {'Ann Ret':>8s} | {'Sharpe':>6s} | {'Max DD':>7s} | {'Trades':>6s} | {'Win Rate':>8s}")
    print("-" * 75)
    
    kelly_results = {}
    for mult in KELLY_MULTIPLIERS:
        result = run_backtest_with_kelly_mult(kelly_base_mult=mult)
        if result:
            kelly_results[mult] = result
            print(f"{mult:>10.1f} | {result['total_return']:>+9.2%} | {result['annualized_return']:>7.2%} | "
                  f"{result['sharpe']:>5.2f} | {result['max_drawdown']:>6.2%} | {result['n_trades']:>5d} | "
                  f"{result['win_rate']:>7.2%}")
    
    # Find best Kelly that meets criteria (>10% ann return, <12% max DD)
    print("\n--- Kelly Multipliers Meeting Criteria (>10% Ann Return, <12% Max DD) ---")
    for mult, result in kelly_results.items():
        meets_return = result['annualized_return'] > 0.10
        meets_dd = result['max_drawdown'] < 0.12
        if meets_return and meets_dd:
            print(f"  ✅ {mult:.1f}x: Ann={result['annualized_return']:.2%}, MaxDD={result['max_drawdown']:.2%}, Sharpe={result['sharpe']:.2f}")
        elif meets_return:
            print(f"  ❌ {mult:.1f}x: Meets return ({result['annualized_return']:.2%}) but DD too high ({result['max_drawdown']:.2%})")
        elif meets_dd:
            print(f"  ⚠️  {mult:.1f}x: Meets DD ({result['max_drawdown']:.2%}) but return too low ({result['annualized_return']:.2%})")
    
    # 2. SPY Benchmark
    print("\n" + "=" * 60)
    print("SPY BUY-AND-HOLD BENCHMARK (same 501 days)")
    print("=" * 60)
    
    spy_result = run_spy_benchmark(days=501)
    if spy_result:
        print(f"\nSPY Buy-and-Hold:")
        print(f"  Initial:  ${spy_result['initial_price']:.2f}")
        print(f"  Final:    ${spy_result['final_price']:.2f}")
        print(f"  Total Return:   {spy_result['total_return']:+.2%}")
        print(f"  Annualized:     {spy_result['annualized_return']:+.2%}")
        print(f"  Sharpe Ratio:   {spy_result['sharpe']:.2f}")
        print(f"  Max Drawdown:   {spy_result['max_drawdown']:.2%}")
        
        print(f"\nComparison (Active vs SPY B&H):")
        base_result = kelly_results[1.0]
        print(f"  Active System (1.0x Kelly):")
        print(f"    Ann Return: {base_result['annualized_return']:+.2%}")
        print(f"    Max DD:     {base_result['max_drawdown']:.2%}")
        print(f"    Sharpe:     {base_result['sharpe']:.2f}")
        print(f"  SPY B&H:")
        print(f"    Ann Return: {spy_result['annualized_return']:+.2%}")
        print(f"    Max DD:     {spy_result['max_drawdown']:.2%}")
        print(f"    Sharpe:     {spy_result['sharpe']:.2f}")
        
        excess_return = base_result['annualized_return'] - spy_result['annualized_return']
        print(f"\n  Excess Return (Active - B&H): {excess_return:+.2%}")
        if excess_return > 0:
            print(f"  ✅ Active system BEAT the benchmark by {excess_return:.2%}")
        else:
            print(f"  ❌ Active system UNDERPERFORMED benchmark by {abs(excess_return):.2%}")
    
    # 3. Losing Trade Analysis
    print("\n" + "=" * 60)
    print("LOSING TRADE PATTERN ANALYSIS")
    print("=" * 60)
    
    base_trades = kelly_results[1.0]['trades']
    
    # Rebuild all_data for sector mapping
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC',
               'JPM', 'BAC', 'GS', 'MS', 'PYPL', 'TSLA', 'V', 'MA', 'CRM', 'ADBE']
    all_data = {}
    for sym in symbols:
        closes, _, _, _ = fetch_data(sym, days=501)
        if closes is not None:
            all_data[sym] = closes
    
    analyze_losing_trades(base_trades, all_data)
    
    # Save all results
    all_results = {
        'kelly_stress_test': kelly_results,
        'spy_benchmark': spy_result,
        'base_trades': base_trades,
    }
    with open('comprehensive_analysis.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    logger.info("\nAll results saved to: comprehensive_analysis.pkl")


if __name__ == '__main__':
    main()
