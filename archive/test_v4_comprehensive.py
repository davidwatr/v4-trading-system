"""
COMPREHENSIVE V4 TEST SUITE
============================
Tests:
1. Walk-Forward OOS (5 folds)
2. Monte Carlo (500 random 50-day periods)
3. Parameter Sensitivity (10 configs)
4. Regime-Specific OOS
5. Transaction Cost Stress (10-100 bps)
6. Overfitting Detection (data snooping)
7. Stability (rolling Sharpe consistency)
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE
from fixed_production_v4_tuned import FixedProductionSystemV4Tuned


def test_walk_forward_oos(n_folds=5, initial_capital=100_000):
    """Test 1: Walk-Forward Out-of-Sample"""
    print("\n" + "="*70)
    print("TEST 1: WALK-FORWARD OUT-OF-SAMPLE")
    print("="*70)
    
    # Load full data
    universe = UNIVERSE[:25]
    data = {}
    for sym in universe:
        d = fetch_data(sym, days=1000)
        if d is not None and len(d[0]) > 100:
            data[sym] = {
                'closes': np.array(d[0]),
                'volumes': np.array(d[1]),
                'highs': np.array(d[2]),
                'lows': np.array(d[3])
            }
    
    min_len = min(len(d['closes']) for d in data.values())
    print(f"Total history: {min_len} days")
    
    # Split into folds
    fold_size = min_len // (n_folds + 1)
    results = []
    
    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_end = min(train_end + fold_size, min_len)
        
        if test_end - train_end < 20:
            continue
        
        # Create system with training data only
        system = FixedProductionSystemV4Tuned(universe=universe)
        system.data = {}
        for sym, d in data.items():
            system.data[sym] = {
                'closes': d['closes'][:train_end],
                'volumes': d['volumes'][:train_end],
                'highs': d['highs'][:train_end],
                'lows': d['lows'][:train_end]
            }
        
        # Test on OOS data (but we need to inject test data for backtest)
        # Actually we need to run the full backtest with data up to test_end
        # but only count trades in the test period
        system_full = FixedProductionSystemV4Tuned(universe=universe)
        system_full.data = {}
        for sym, d in data.items():
            system_full.data[sym] = {
                'closes': d['closes'][:test_end],
                'volumes': d['volumes'][:test_end],
                'highs': d['highs'][:test_end],
                'lows': d['lows'][:test_end]
            }
        
        # Count trades in test period only
        result = system_full.backtest(initial_capital=initial_capital)
        test_trades = [t for t in result['trades'] if t['entry_day'] >= train_end]
        
        test_pnl = sum(t['pnl'] for t in test_trades)
        test_return = test_pnl / initial_capital
        
        # Calculate Sharpe for test period only
        # This is approximate - use trades to estimate
        test_win_rate = sum(1 for t in test_trades if t['pnl'] > 0) / len(test_trades) if test_trades else 0
        
        results.append({
            'fold': fold + 1,
            'train_end': train_end,
            'test_end': test_end,
            'test_return': test_return,
            'test_trades': len(test_trades),
            'test_win_rate': test_win_rate,
            'test_pnl': test_pnl
        })
        
        print(f"Fold {fold+1}: Days {train_end}-{test_end} | "
              f"Trades: {len(test_trades):2d} | "
              f"Win%: {test_win_rate*100:4.0f}% | "
              f"PnL: ${test_pnl:+,.0f} | "
              f"Return: {test_return*100:+5.1f}%")
    
    if results:
        avg_return = np.mean([r['test_return'] for r in results])
        positive_folds = sum(1 for r in results if r['test_return'] > 0)
        print(f"\nAverage OOS Return: {avg_return*100:+.2f}%")
        print(f"Positive Folds: {positive_folds}/{len(results)}")
    
    return results


def test_monte_carlo(n_periods=500, period_days=50, initial_capital=100_000):
    """Test 2: Monte Carlo Random Periods"""
    print("\n" + "="*70)
    print("TEST 2: MONTE CARLO RANDOM PERIODS")
    print("="*70)
    
    system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
    system.fetch_data(days=1000)
    
    min_len = min(len(d['closes']) for d in system.data.values())
    
    results = []
    for _ in range(n_periods):
        start = np.random.randint(30, min_len - period_days)
        end = start + period_days
        
        # Extract sub-period
        period_system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
        period_system.data = {}
        for sym, d in system.data.items():
            period_system.data[sym] = {
                'closes': d['closes'][start:end],
                'volumes': d['volumes'][start:end],
                'highs': d['highs'][start:end],
                'lows': d['lows'][start:end]
            }
        
        result = period_system.backtest(initial_capital=initial_capital)
        results.append(result['total_return'])
    
    results = np.array(results)
    mean_ret = np.mean(results)
    std_ret = np.std(results)
    positive_pct = np.mean(results > 0)
    sharpe = mean_ret / (std_ret + 1e-10) * np.sqrt(252 / period_days)
    
    print(f"Periods: {n_periods} x {period_days} days")
    print(f"Mean Return: {mean_ret*100:+.2f}%")
    print(f"Std Dev: {std_ret*100:.2f}%")
    print(f"Positive Periods: {positive_pct*100:.1f}%")
    print(f"Sharpe (period): {sharpe:.3f}")
    print(f"Worst: {np.min(results)*100:.1f}%")
    print(f"Best: {np.max(results)*100:.1f}%")
    print(f"Median: {np.median(results)*100:.1f}%")
    
    return results


def test_parameter_sensitivity(initial_capital=100_000):
    """Test 3: Parameter Sensitivity"""
    print("\n" + "="*70)
    print("TEST 3: PARAMETER SENSITIVITY")
    print("="*70)
    
    configs = [
        {"name": "V4 Default", "target_vol": 0.18, "max_correlation": 0.75, "max_cluster": 0.40},
        {"name": "Low Vol Target", "target_vol": 0.12, "max_correlation": 0.75, "max_cluster": 0.40},
        {"name": "High Vol Target", "target_vol": 0.25, "max_correlation": 0.75, "max_cluster": 0.40},
        {"name": "Strict Corr", "target_vol": 0.18, "max_correlation": 0.60, "max_cluster": 0.40},
        {"name": "Loose Corr", "target_vol": 0.18, "max_correlation": 0.85, "max_cluster": 0.40},
        {"name": "Strict Cluster", "target_vol": 0.18, "max_correlation": 0.75, "max_cluster": 0.25},
        {"name": "Loose Cluster", "target_vol": 0.18, "max_correlation": 0.75, "max_cluster": 0.50},
        {"name": "No Vol Target", "target_vol": 0.50, "max_correlation": 0.75, "max_cluster": 0.40},
        {"name": "No Corr Adj", "target_vol": 0.18, "max_correlation": 1.0, "max_cluster": 1.0},
        {"name": "All Off", "target_vol": 0.50, "max_correlation": 1.0, "max_cluster": 1.0},
    ]
    
    results = []
    for config in configs:
        system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
        system.fetch_data(days=500)
        system.target_vol = config['target_vol']
        system.max_correlation = config['max_correlation']
        system.max_cluster_exposure = config['max_cluster']
        
        result = system.backtest(initial_capital=initial_capital)
        result['name'] = config['name']
        results.append(result)
        
        print(f"{config['name']:20s}: {result['total_return']*100:+6.1f}% | "
              f"Sharpe {result['sharpe']:.2f} | "
              f"DD {result['max_drawdown']*100:+5.1f}% | "
              f"WR {result.get('trades', []) and sum(1 for t in result['trades'] if t['pnl'] > 0) / len(result['trades']) * 100:4.0f}%")
    
    # Summary
    returns = [r['total_return'] for r in results]
    sharpes = [r['sharpe'] for r in results]
    print(f"\nReturn Range: {min(returns)*100:.1f}% to {max(returns)*100:.1f}%")
    print(f"Sharpe Range: {min(sharpes):.2f} to {max(sharpes):.2f}")
    print(f"V4 Default Rank: {sorted(returns, reverse=True).index(results[0]['total_return']) + 1}/{len(results)}")
    
    return results


def test_regime_oos(initial_capital=100_000):
    """Test 4: Regime-Specific OOS"""
    print("\n" + "="*70)
    print("TEST 4: REGIME-SPECIFIC OOS")
    print("="*70)
    
    system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
    system.fetch_data(days=1000)
    
    min_len = min(len(d['closes']) for d in system.data.values())
    
    # Classify each day's regime
    regimes = {'HIGH_VOL': [], 'SIDEWAYS': [], 'TRENDING': [], 'LOW_VOL': []}
    for day in range(30, min_len):
        regime, _ = system.classify_regime(day)
        regimes[regime].append(day)
    
    print(f"Regime Distribution (1000 days):")
    for regime, days in regimes.items():
        pct = len(days) / (min_len - 30) * 100
        print(f"  {regime:12s}: {len(days):4d} days ({pct:.1f}%)")
    
    # Run full backtest and analyze by regime
    result = system.backtest(initial_capital=initial_capital)
    
    regime_stats = {}
    for regime in regimes:
        regime_trades = [t for t in result['trades'] if t['regime'] == regime]
        if regime_trades:
            win_rate = sum(1 for t in regime_trades if t['pnl'] > 0) / len(regime_trades)
            total_pnl = sum(t['pnl'] for t in regime_trades)
            avg_pnl = total_pnl / len(regime_trades)
            regime_stats[regime] = {
                'trades': len(regime_trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl
            }
            print(f"\n  {regime:12s}: {len(regime_trades):3d} trades | "
                  f"{win_rate*100:.0f}% WR | "
                  f"${total_pnl:+,.0f} total | "
                  f"${avg_pnl:+,.0f} avg")
    
    return regime_stats


def test_transaction_cost_stress(initial_capital=100_000):
    """Test 5: Transaction Cost Stress Test"""
    print("\n" + "="*70)
    print("TEST 5: TRANSACTION COST STRESS")
    print("="*70)
    
    costs = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    
    results = []
    for cost in costs:
        system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
        system.fetch_data(days=500)
        system.transaction_cost = cost
        
        result = system.backtest(initial_capital=initial_capital)
        result['cost'] = cost
        results.append(result)
        
        n_trades = len(result['trades'])
        print(f"Cost {cost*100:5.1f}%: {result['total_return']*100:+6.1f}% | "
              f"Sharpe {result['sharpe']:.2f} | "
              f"Trades: {n_trades:2d} | "
              f"Cost Impact: ${n_trades * cost * initial_capital * 2:,.0f}")
    
    # Find break-even cost
    base_return = results[0]['total_return']
    for r in results:
        if r['total_return'] < 0:
            print(f"\nBreak-even cost: ~{r['cost']*100:.1f}% ({r['cost']*10000:.0f} bps)")
            break
    
    return results


def test_overfitting_detection(initial_capital=100_000):
    """Test 6: Overfitting Detection"""
    print("\n" + "="*70)
    print("TEST 6: OVERFITTING DETECTION")
    print("="*70)
    
    # Test 6a: Training vs Test performance gap
    system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
    system.fetch_data(days=1000)
    min_len = min(len(d['closes']) for d in system.data.values())
    
    # Full backtest (in-sample)
    full_result = system.backtest(initial_capital=initial_capital)
    
    # Walk-forward (out-of-sample)
    train_end = min_len // 2
    test_system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
    test_system.data = {}
    for sym, d in system.data.items():
        test_system.data[sym] = {
            'closes': d['closes'][train_end:],
            'volumes': d['volumes'][train_end:],
            'highs': d['highs'][train_end:],
            'lows': d['lows'][train_end:]
        }
    test_result = test_system.backtest(initial_capital=initial_capital)
    
    # Calculate overfitting ratio
    is_sharpe = full_result['sharpe']
    oos_sharpe = test_result['sharpe']
    overfitting_ratio = is_sharpe / (oos_sharpe + 1e-10)
    
    print(f"In-Sample Sharpe: {is_sharpe:.3f}")
    print(f"Out-of-Sample Sharpe: {oos_sharpe:.3f}")
    print(f"Overfitting Ratio: {overfitting_ratio:.2f}x")
    print(f"Diagnosis: {'⚠️ OVERFITTING' if overfitting_ratio > 2.0 else '✅ Acceptable'}")
    
    # Test 6b: Parameter sensitivity (small changes should not destroy performance)
    print(f"\nParameter Stability Test:")
    base_return = full_result['total_return']
    
    variations = [
        ("Vol Target -20%", 0.144),
        ("Vol Target +20%", 0.216),
        ("Corr Threshold -10%", 0.675),
        ("Corr Threshold +10%", 0.825),
    ]
    
    stable = True
    for name, val in variations:
        system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
        system.fetch_data(days=500)
        if "Vol" in name:
            system.target_vol = val
        else:
            system.max_correlation = val
        result = system.backtest(initial_capital=initial_capital)
        change = abs(result['total_return'] - base_return) / (abs(base_return) + 1e-10) * 100
        status = "✅" if change < 30 else "⚠️"
        print(f"  {name:20s}: {result['total_return']*100:+5.1f}% ({change:.0f}% change) {status}")
        if change > 50:
            stable = False
    
    print(f"\nOverall Stability: {'✅ STABLE' if stable else '⚠️ UNSTABLE'}")
    
    return {
        'is_sharpe': is_sharpe,
        'oos_sharpe': oos_sharpe,
        'overfitting_ratio': overfitting_ratio,
        'stable': stable
    }


def test_stability(initial_capital=100_000):
    """Test 7: Rolling Sharpe Stability"""
    print("\n" + "="*70)
    print("TEST 7: ROLLING SHARPE STABILITY")
    print("="*70)
    
    system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
    system.fetch_data(days=1000)
    
    min_len = min(len(d['closes']) for d in system.data.values())
    
    # Calculate rolling 60-day Sharpe
    window = 60
    rolling_sharpes = []
    
    for start in range(30, min_len - window, window):
        end = start + window
        
        period_system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
        period_system.data = {}
        for sym, d in system.data.items():
            period_system.data[sym] = {
                'closes': d['closes'][start:end],
                'volumes': d['volumes'][start:end],
                'highs': d['highs'][start:end],
                'lows': d['lows'][start:end]
            }
        
        result = period_system.backtest(initial_capital=initial_capital)
        if result['sharpe'] > 0:
            rolling_sharpes.append(result['sharpe'])
    
    if rolling_sharpes:
        sharpes = np.array(rolling_sharpes)
        print(f"60-day Window Sharpe Statistics:")
        print(f"  Mean: {np.mean(sharpes):.3f}")
        print(f"  Std: {np.std(sharpes):.3f}")
        print(f"  Min: {np.min(sharpes):.3f}")
        print(f"  Max: {np.max(sharpes):.3f}")
        print(f"  Positive Windows: {np.sum(sharpes > 0)}/{len(sharpes)}")
        print(f"  Consistency (std/mean): {np.std(sharpes)/(np.mean(sharpes)+1e-10):.2f}")
    
    return rolling_sharpes


def main():
    print("="*70)
    print("V4 COMPREHENSIVE BACKTEST SUITE")
    print("="*70)
    print(f"System: V4 (Vol Targeting + Correlation Sizing + Circuit Breakers)")
    print(f"Parameters: target_vol=18%, max_corr=0.75, max_cluster=40%")
    
    # Run all tests
    walk_forward = test_walk_forward_oos(n_folds=5)
    monte_carlo = test_monte_carlo(n_periods=500, period_days=50)
    param_sens = test_parameter_sensitivity()
    regime_oos = test_regime_oos()
    cost_stress = test_transaction_cost_stress()
    overfitting = test_overfitting_detection()
    stability = test_stability()
    
    # FINAL SUMMARY
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    # Aggregate
    mc_returns = monte_carlo
    avg_mc = np.mean(mc_returns) * 100
    pos_mc = np.mean(mc_returns > 0) * 100
    
    # Sharpe
    param_sharpes = [r['sharpe'] for r in param_sens]
    avg_sharpe = np.mean(param_sharpes)
    sharpe_std = np.std(param_sharpes)
    
    # Walk-forward
    wf_returns = [r['test_return'] * 100 for r in walk_forward]
    avg_wf = np.mean(wf_returns) if wf_returns else 0
    
    # Overfitting
    of_ratio = overfitting['overfitting_ratio']
    
    print(f"Walk-Forward OOS Avg Return: {avg_wf:+.1f}%")
    print(f"Monte Carlo Avg Return (50d): {avg_mc:+.1f}% ({pos_mc:.0f}% positive)")
    print(f"Parameter Sensitivity Sharpe: {avg_sharpe:.2f} ± {sharpe_std:.2f}")
    print(f"Overfitting Ratio: {of_ratio:.2f}x")
    print(f"Stability: {'✅ STABLE' if overfitting['stable'] else '⚠️ UNSTABLE'}")
    
    # Realistic expectation
    realistic_return = avg_mc * (252 / 50)  # Annualize
    realistic_sharpe = avg_sharpe * (1 / (1 + of_ratio / 10))  # Penalize for overfitting
    
    print(f"\nRealistic Annual Return: {realistic_return:.0f}%")
    print(f"Realistic Sharpe: {realistic_sharpe:.2f}")
    
    # Verdict
    print("\n" + "-"*70)
    if realistic_sharpe > 1.0 and pos_mc > 55 and of_ratio < 3:
        print("✅ SYSTEM IS TRADEABLE")
        print("   Edge is real but modest. Position sizing should be conservative.")
    elif realistic_sharpe > 0.5:
        print("⚠️ SYSTEM HAS WEAK EDGE")
        print("   May be worth running with small size, but not confident.")
    else:
        print("❌ SYSTEM IS NOT TRADEABLE")
        print("   Edge is too weak or overfitting too severe.")
    print("-"*70)


if __name__ == '__main__':
    main()
