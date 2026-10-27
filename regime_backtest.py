#!/usr/bin/env python3
"""
Regime-Aware Heuristic Backtest
Tests the heuristic signals with regime-based Kelly scaling.
Regime detection: trend (SPY MA50/MA200) + volatility (ATR ratio) + K-Means cluster.
Kelly scaling: bull=1.5x, sideways=0.75x, bear=0.5x, high_vol=0.7x penalty.
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

# Import from existing modules
from production_v18 import fetch_data, get_market_regime, UNIVERSE
from heuristic_signals import generate_heuristic_signals, mean_reversion_signal, momentum_filter

# ============================================================
# CONFIGURATION
# ============================================================
INITIAL_CAPITAL = 100000
SLIPPAGE_BPS = 10        # 0.1% slippage
FEE_PER_SHARE = 0.005    # $0.005 per share
MAX_KELLY = 0.25         # Cap at 25% per position
MIN_KELLY_PROB = 0.55    # Minimum probability to trade
TP_PCT = 0.03            # 3% profit target
SL_PCT = 0.02            # 2% stop loss
HORIZON_DAYS = 5         # 5-day holding period for backtest


# ============================================================
# ADVANCED REGIME DETECTION — K-MEANS + TREND/VOLATILITY
# ============================================================
def compute_regime_features(spy_closes):
    """
    Compute regime feature vector from SPY price data.
    Uses: trend strength, momentum, volatility state, volume confirmation.
    """
    if len(spy_closes) < 200:
        return np.array([0, 0, 0, 0])
    
    closes = np.array(spy_closes)
    N = len(closes)
    
    # Trend features
    ma50 = np.mean(closes[-50:])
    ma200 = np.mean(closes[-200:])
    current = closes[-1]
    
    trend_strength = (current - ma200) / ma200 if ma200 > 0 else 0
    trend_slope = (ma50 - ma200) / ma200 if ma200 > 0 else 0
    
    # Momentum (10-day vs 50-day)
    ret_10d = (closes[-1] - closes[-11]) / closes[-11] if closes[-11] > 0 else 0
    ret_50d = (closes[-1] - closes[-51]) / closes[-51] if closes[-51] > 0 else 0
    momentum = ret_10d - ret_50d / 5  # Relative momentum
    
    # Volatility (ATR ratio)
    highs = np.array([current] * N)  # simplified
    lows = np.array([current * 0.98] * N)
    tr = np.abs(np.diff(closes))
    atr_14 = np.mean(tr[-14:]) if len(tr) >= 14 else 0
    atr_50 = np.mean(tr[-50:]) if len(tr) >= 50 else 0
    atr_ratio = atr_14 / atr_50 if atr_50 > 0 else 1
    
    return np.array([trend_strength, trend_slope, momentum, atr_ratio])


def build_regime_classifier(spy_closes_history):
    """
    Build K-Means regime classifier from historical SPY data.
    Returns fitted K-Means model + scaler + cluster interpretation.
    """
    features = []
    for i in range(200, len(spy_closes_history)):
        window = spy_closes_history[i-200:i+1]
        feats = compute_regime_features(window)
        features.append(feats)
    
    features = np.array(features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    # 3 clusters: bull, sideways, bear
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(scaled)
    
    # Label clusters by mean trend strength
    cluster_means = kmeans.cluster_centers_[:, 0]  # trend_strength is first feature
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
    """Classify current regime using K-Means."""
    feats = compute_regime_features(spy_closes)
    scaled = scaler.transform(feats.reshape(1, -1))
    cluster = kmeans.predict(scaled)[0]
    return cluster_map.get(cluster, 'sideways')


# ============================================================
# REGIME-BASED KELLY SCALING
# ============================================================
def get_regime_kelly_params(regime):
    """
    Regime-specific Kelly multipliers and constraints.
    Conservative in bad regimes, aggressive in good ones.
    """
    params = {
        'bull': {
            'kelly_multiplier': 1.5,
            'max_positions': 8,
            'min_prob': 0.50,
            'max_kelly_f': 0.30,
            'slippage_bps': 8,
            'description': 'Strong uptrend — aggressive sizing'
        },
        'bull_high_vol': {
            'kelly_multiplier': 1.3,
            'max_positions': 6,
            'min_prob': 0.55,
            'max_kelly_f': 0.25,
            'slippage_bps': 12,
            'description': 'Uptrend but volatile — moderate aggression'
        },
        'sideways': {
            'kelly_multiplier': 0.75,
            'max_positions': 6,
            'min_prob': 0.55,
            'max_kelly_f': 0.20,
            'slippage_bps': 10,
            'description': 'Choppy — reduced sizing'
        },
        'sideways_high_vol': {
            'kelly_multiplier': 0.60,
            'max_positions': 4,
            'min_prob': 0.60,
            'max_kelly_f': 0.15,
            'slippage_bps': 15,
            'description': 'Choppy + volatile — defensive'
        },
        'bear': {
            'kelly_multiplier': 0.5,
            'max_positions': 4,
            'min_prob': 0.60,
            'max_kelly_f': 0.15,
            'slippage_bps': 15,
            'description': 'Confirmed downtrend — very defensive'
        },
        'bear_high_vol': {
            'kelly_multiplier': 0.30,
            'max_positions': 2,
            'min_prob': 0.65,
            'max_kelly_f': 0.10,
            'slippage_bps': 20,
            'description': 'Panic — minimal exposure'
        }
    }
    
    return params.get(regime, params['sideways'])


# ============================================================
# HEURISTIC SIGNAL GENERATION (for backtest)
# ============================================================
def generate_signals_for_symbol(closes, volumes, highs, lows, symbol):
    """
    Generate heuristic signals for backtesting.
    Uses the same logic as heuristic_signals.py.
    """
    if closes is None or len(closes) < 20:
        return None
    
    # Check trend
    if not momentum_filter(closes):
        return None
    
    # Mean reversion signal
    strength, direction, rsi_val = mean_reversion_signal(closes)
    
    if direction == 1 and strength > 0:
        # Kelly sizing
        kelly_size = min(strength * 0.10, 0.10)
        
        # ATR
        atr = np.std(np.diff(closes[-20:]) / closes[-20:-1]) * closes[-1] if len(closes) >= 21 else closes[-1] * 0.02
        if atr == 0:
            atr = closes[-1] * 0.02
        
        return {
            'symbol': symbol,
            'strength': strength,
            'kelly_size': kelly_size,
            'rsi': rsi_val,
            'atr': atr,
            'current_price': closes[-1],
            'upper_mult': TP_PCT / atr if atr > 0 else 2.0,
            'lower_mult': SL_PCT / atr if atr > 0 else 1.5,
        }
    
    return None


# ============================================================
# EVENT-DRIVEN BACKTEST
# ============================================================
def run_regime_backtest(date_range_days=730, symbols=None, regime_kmeans=None, regime_scaler=None, regime_map=None):
    """
    Backtest heuristic signals with regime-based Kelly scaling.
    """
    if symbols is None:
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 
                   'JPM', 'BAC', 'GS', 'MS', 'PYPL', 'TSLA']
    
    logger.info("=" * 60)
    logger.info("REGIME-AWARE HEURISTIC BACKTEST")
    logger.info("=" * 60)
    
    # Pre-fetch all data
    logger.info("\nFetching data...")
    all_data = {}
    for sym in symbols:
        closes, volumes, highs, lows = fetch_data(sym, days=date_range_days)
        if closes is not None and len(closes) >= 100:
            all_data[sym] = {
                'closes': closes,
                'volumes': volumes,
                'highs': highs,
                'lows': lows
            }
            logger.info(f"  {sym}: {len(closes)} days")
    
    if not all_data:
        logger.error("No data fetched. Aborting.")
        return None
    
    # Fetch SPY for regime
    spy_closes, _, spy_highs, spy_lows = fetch_data('SPY', days=date_range_days)
    if spy_closes is None or len(spy_closes) < 200:
        logger.error("Failed to fetch SPY. Aborting.")
        return None
    
    # Build regime classifier if not provided
    if regime_kmeans is None:
        logger.info("\nBuilding regime classifier...")
        regime_kmeans, regime_scaler, regime_map = build_regime_classifier(spy_closes)
    
    # Backtest
    capital = INITIAL_CAPITAL
    positions = {}  # symbol -> {entry_price, entry_idx, shares, kelly_f, regime_at_entry}
    portfolio_values = [capital]
    trades = []
    n_trades = 0
    n_wins = 0
    
    n_days = min(len(v['closes']) for v in all_data.values())
    n_days = min(n_days, len(spy_closes))
    
    logger.info(f"\nBacktesting {n_days} days, {len(all_data)} symbols...")
    
    for day in range(200, n_days):  # Start after enough data for regime
        # Update regime
        current_spy = spy_closes[day]
        regime = get_regime_class(spy_closes[:day+1], regime_kmeans, regime_scaler, regime_map)
        regime_params = get_regime_kelly_params(regime)
        
        # Generate signals for each symbol
        signals = []
        for sym, data in all_data.items():
            closes = data['closes']
            if day >= len(closes):
                continue
            
            signal = generate_signals_for_symbol(
                closes[:day+1], 
                data['volumes'][:day+1] if day+1 <= len(data['volumes']) else data['volumes'],
                data['highs'][:day+1] if day+1 <= len(data['highs']) else data['highs'],
                data['lows'][:day+1] if day+1 <= len(data['lows']) else data['lows'],
                sym
            )
            if signal:
                signals.append(signal)
        
        # Sort by strength
        signals.sort(key=lambda x: -x['strength'])
        
        # Execute trades
        current_positions = len(positions)
        max_positions = regime_params['max_positions']
        
        for sig in signals[:max_positions - current_positions]:
            symbol = sig['symbol']
            
            # Skip if already in position
            if symbol in positions:
                continue
            
            # Kelly sizing with regime multiplier
            base_kelly = sig['kelly_size']
            kelly_f = base_kelly * regime_params['kelly_multiplier']
            kelly_f = min(kelly_f, regime_params['max_kelly_f'])
            
            # Minimum probability threshold
            if sig['strength'] < regime_params['min_prob']:
                continue
            
            # Calculate position
            position_value = capital * kelly_f
            entry_price = sig['current_price']
            shares = int(position_value / entry_price)
            
            if shares <= 0:
                continue
            
            # Check buying power
            cost = shares * entry_price * (1 + regime_params['slippage_bps'] / 10000)
            if capital - cost < 0:
                continue
            
            # Enter position
            capital -= cost
            positions[symbol] = {
                'entry_price': entry_price,
                'entry_idx': day,
                'shares': shares,
                'kelly_f': kelly_f,
                'regime': regime,
                'tp_pct': TP_PCT,
                'sl_pct': SL_PCT,
            }
        
        # Check exits
        to_close = []
        for sym, pos in positions.items():
            # Check TP/SL hit
            if day > pos['entry_idx']:
                closes_sym = all_data[sym]['closes']
                if day < len(closes_sym):
                    current_price = closes_sym[day]
                    entry = pos['entry_price']
                    
                    if current_price >= entry * (1 + pos['tp_pct']):
                        # Hit profit target
                        pnl_pct = pos['tp_pct']
                        to_close.append(sym)
                        trades.append({'symbol': sym, 'pnl_pct': pnl_pct, 'regime': pos['regime'], 'days': day - pos['entry_idx']})
                        n_trades += 1
                        n_wins += 1
                    elif current_price <= entry * (1 - pos['sl_pct']):
                        # Hit stop loss
                        pnl_pct = -pos['sl_pct']
                        to_close.append(sym)
                        trades.append({'symbol': sym, 'pnl_pct': pnl_pct, 'regime': pos['regime'], 'days': day - pos['entry_idx']})
                        n_trades += 1
                    elif day - pos['entry_idx'] >= HORIZON_DAYS:
                        # Time exit — close at market
                        final_price = closes_sym[min(day, len(closes_sym)-1)]
                        pnl_pct = (final_price - entry) / entry
                        to_close.append(sym)
                        trades.append({'symbol': sym, 'pnl_pct': pnl_pct, 'regime': pos['regime'], 'days': day - pos['entry_idx']})
                        n_trades += 1
            
            # Also close if regime turned bearish while in position
            if pos['regime'] == 'bull' and regime in ['bear', 'bear_high_vol']:
                # Exit on regime change
                if sym not in to_close:
                    to_close.append(sym)
                    # Mark as regime exit
                    if sym in trades:
                        trades[-1]['regime_exit'] = True
        
        for sym in to_close:
            pos = positions.pop(sym)
            exit_price = all_data[sym]['closes'][min(day, len(all_data[sym]['closes'])-1)]
            capital += pos['shares'] * exit_price * (1 - regime_params['slippage_bps'] / 10000)
        
        # Track portfolio value (positions marked to market)
        port_value = capital
        for sym, pos in positions.items():
            if day < len(all_data[sym]['closes']):
                port_value += pos['shares'] * all_data[sym]['closes'][day]
        
        portfolio_values.append(port_value)
        
        if day % 60 == 0:
            logger.info(f"  Day {day}: Portfolio=${port_value:,.0f}, Positions={len(positions)}, Regime={regime}")
    
    # Final mark-to-market
    for sym, pos in positions.items():
        if len(all_data[sym]['closes']) > 0:
            port_value = capital + sum(
                p['shares'] * all_data[s]['closes'][-1] 
                for s, p in positions.items() if len(all_data[s]['closes']) > 0
            )
    
    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)
    
    total_return = portfolio_values[-1] / INITIAL_CAPITAL - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    # Max drawdown
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (running_max - portfolio_values) / running_max
    max_dd = np.max(drawdown)
    
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    
    # Regime distribution
    regime_counts = {}
    for t in trades:
        r = t.get('regime', 'unknown')
        regime_counts[r] = regime_counts.get(r, 0) + 1
    
    results = {
        'total_return': total_return,
        'annualized_return': ann_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': n_trades,
        'n_wins': n_wins,
        'win_rate': win_rate,
        'final_capital': portfolio_values[-1],
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
        'trades': trades,
        'regime_distribution': regime_counts,
        'regime_params': regime_params,
    }
    
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    logger.info(f"\nStarted: {datetime.now()}")
    
    # Run backtest
    results = run_regime_backtest(
        date_range_days=730,
        symbols=['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC', 
                 'JPM', 'BAC', 'GS', 'MS', 'PYPL', 'TSLA', 'V', 'MA', 'CRM', 'ADBE']
    )
    
    if results is None:
        logger.error("Backtest failed.")
        return
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return:       {results['total_return']:+.2%}")
    logger.info(f"Annualized Return:  {results['annualized_return']:+.2%}")
    logger.info(f"Sharpe Ratio:       {results['sharpe']:.2f}")
    logger.info(f"Max Drawdown:       {results['max_drawdown']:.2%}")
    logger.info(f"Total Trades:       {results['n_trades']}")
    logger.info(f"Win Rate:           {results['win_rate']:.2%}")
    logger.info(f"Final Capital:      ${results['final_capital']:,.2f}")
    
    logger.info("\nTrades by Regime:")
    for regime, count in sorted(results['regime_distribution'].items()):
        pct = count / results['n_trades'] * 100 if results['n_trades'] > 0 else 0
        logger.info(f"  {regime}: {count} trades ({pct:.1f}%)")
    
    # Regime params used
    logger.info("\nRegime Parameters:")
    logger.info(f"  bull: kelly_mult={get_regime_kelly_params('bull')['kelly_multiplier']}, "
                f"max_pos={get_regime_kelly_params('bull')['max_positions']}, "
                f"min_prob={get_regime_kelly_params('bull')['min_prob']}")
    logger.info(f"  sideways: kelly_mult={get_regime_kelly_params('sideways')['kelly_multiplier']}, "
                f"max_pos={get_regime_kelly_params('sideways')['max_positions']}, "
                f"min_prob={get_regime_kelly_params('sideways')['min_prob']}")
    logger.info(f"  bear: kelly_mult={get_regime_kelly_params('bear')['kelly_multiplier']}, "
                f"max_pos={get_regime_kelly_params('bear')['max_positions']}, "
                f"min_prob={get_regime_kelly_params('bear')['min_prob']}")
    
    # Save results
    with open('regime_backtest_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    logger.info("\nResults saved to: regime_backtest_results.pkl")


if __name__ == '__main__':
    main()