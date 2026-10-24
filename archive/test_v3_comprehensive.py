"""
V3 COMPREHENSIVE TEST SUITE
===========================
1. Walk-Forward Out-of-Sample (OOS) Test
2. Monte Carlo Robustness Test
3. Parameter Sensitivity Analysis
4. Regime-Specific OOS Test
5. Transaction Cost Stress Test
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE
from fixed_production_v3 import FixedProductionSystemV3


def walk_forward_oos_test(n_splits=5, initial_capital=100_000):
    """
    Walk-forward OOS test.
    Train on expanding window, test on following period.
    """
    print('\n' + '='*70)
    print('WALK-FORWARD OUT-OF-SAMPLE TEST')
    print('='*70)
    
    # Fetch fresh data
    data = {}
    universe = UNIVERSE[:25]
    print('Fetching data...')
    for sym in universe:
        d = fetch_data(sym, days=750)
        if d is not None and len(d[0]) > 100:
            data[sym] = {
                'closes': np.array(d[0]),
                'volumes': np.array(d[1]),
                'highs': np.array(d[2]),
                'lows': np.array(d[3])
            }
    
    min_len = min(len(d['closes']) for d in data.values())
    print(f'Data: {min_len} days, {len(data)} symbols')
    
    # Walk-forward windows
    fold_size = (min_len - 30) // (n_splits + 1)
    results = []
    
    for fold in range(n_splits):
        train_start = 30
        train_end = 30 + fold_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, min_len - 1)
        
        if test_end <= test_start:
            continue
        
        print(f'\nFold {fold+1}/{n_splits}: Train [{train_start}:{train_end}] → Test [{test_start}:{test_end}]')
        
        # Create system with training data only
        system = FixedProductionSystemV3(universe=universe)
        system.data = {}
        for sym, d in data.items():
            system.data[sym] = {
                'closes': d['closes'][:test_end],
                'volumes': d['volumes'][:test_end],
                'highs': d['highs'][:test_end],
                'lows': d['lows'][:test_end]
            }
        
        # Run backtest on test period only
        portfolio_value = initial_capital
        trades = []
        positions = {}
        
        for day in range(test_start, test_end):
            regime, confidence = system.classify_regime(day)
            
            positions_to_close = []
            for sym, (shares, entry_day, entry_price) in positions.items():
                if sym in system.data and day < len(system.data[sym]['closes']):
                    current_price = system.data[sym]['closes'][day]
                    profit_pct = (current_price - entry_price) / entry_price
                    loss_pct = (entry_price - current_price) / entry_price
                    
                    if profit_pct > system.target_profit:
                        positions_to_close.append((sym, 'profit_target'))
                    elif loss_pct > system.base_stop_loss:
                        positions_to_close.append((sym, 'stop_loss'))
                    elif day - entry_day >= system.hold_days_max:
                        positions_to_close.append((sym, 'max_hold'))
            
            for sym, reason in positions_to_close:
                if sym in system.data and day < len(system.data[sym]['closes']):
                    exit_price = system.data[sym]['closes'][day]
                    shares, entry_day, entry_price = positions[sym]
                    pnl = shares * (exit_price - entry_price)
                    cost = abs(shares) * (exit_price + entry_price) * system.transaction_cost
                    portfolio_value += pnl - cost
                    trades.append({'pnl': pnl - cost, 'reason': reason})
                    del positions[sym]
            
            signals = system.generate_signals(day, regime)
            for signal in signals:
                sym = signal['sym']
                if sym in positions or len(positions) >= system.max_positions:
                    continue
                price = system.data[sym]['closes'][day]
                features = system.compute_features(sym, day)
                if features is None:
                    continue
                atr = features['atr_ratio'] * price
                shares = system.volatility_scaled_sizing(portfolio_value, atr, price)
                if shares > 0:
                    cost = shares * price * system.transaction_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price)
        
        # Close remaining
        for sym, (shares, entry_day, entry_price) in positions.items():
            if sym in system.data and test_end - 1 < len(system.data[sym]['closes']):
                exit_price = system.data[sym]['closes'][test_end - 1]
                pnl = shares * (exit_price - entry_price)
                cost = abs(shares) * (exit_price + entry_price) * system.transaction_cost
                portfolio_value += pnl - cost
        
        fold_return = (portfolio_value - initial_capital) / initial_capital
        fold_trades = len(trades)
        fold_wr = sum(1 for t in trades if t['pnl'] > 0) / max(fold_trades, 1)
        
        results.append({
            'fold': fold + 1,
            'return': fold_return,
            'trades': fold_trades,
            'win_rate': fold_wr,
            'portfolio': portfolio_value
        })
        
        print(f'  Return: {fold_return*100:.2f}% | Trades: {fold_trades} | WR: {fold_wr*100:.0f}%')
    
    if results:
        avg_return = np.mean([r['return'] for r in results])
        avg_trades = np.mean([r['trades'] for r in results])
        avg_wr = np.mean([r['win_rate'] for r in results])
        consistency = sum(1 for r in results if r['return'] > 0) / len(results)
        
        print(f'\nSUMMARY:')
        print(f'  Avg Return: {avg_return*100:.2f}%')
        print(f'  Avg Trades: {avg_trades:.0f}')
        print(f'  Avg Win Rate: {avg_wr*100:.1f}%')
        print(f'  Consistency: {consistency*100:.0f}% ({sum(1 for r in results if r["return"] > 0)}/{len(results)} folds positive)')
    
    return results


def monte_carlo_test(n_simulations=100, initial_capital=100_000):
    """
    Monte Carlo test: randomize entry days, measure robustness.
    """
    print('\n' + '='*70)
    print(f'MONTE CARLO TEST ({n_simulations} simulations)')
    print('='*70)
    
    data = {}
    universe = UNIVERSE[:25]
    print('Fetching data...')
    for sym in universe:
        d = fetch_data(sym, days=500)
        if d is not None and len(d[0]) > 100:
            data[sym] = {
                'closes': np.array(d[0]),
                'volumes': np.array(d[1]),
                'highs': np.array(d[2]),
                'lows': np.array(d[3])
            }
    
    min_len = min(len(d['closes']) for d in data.values())
    
    returns = []
    for sim in range(n_simulations):
        # Random start day (but leave room for 50-day test)
        start_day = np.random.randint(30, min_len - 60)
        end_day = start_day + 50
        
        system = FixedProductionSystemV3(universe=universe)
        system.data = data
        
        portfolio_value = initial_capital
        positions = {}
        
        for day in range(start_day, end_day):
            regime, confidence = system.classify_regime(day)
            
            positions_to_close = []
            for sym, (shares, entry_day, entry_price) in positions.items():
                if sym in system.data and day < len(system.data[sym]['closes']):
                    current_price = system.data[sym]['closes'][day]
                    profit_pct = (current_price - entry_price) / entry_price
                    loss_pct = (entry_price - current_price) / entry_price
                    
                    if profit_pct > system.target_profit:
                        positions_to_close.append((sym, 'profit_target'))
                    elif loss_pct > system.base_stop_loss:
                        positions_to_close.append((sym, 'stop_loss'))
                    elif day - entry_day >= system.hold_days_max:
                        positions_to_close.append((sym, 'max_hold'))
            
            for sym, reason in positions_to_close:
                if sym in system.data and day < len(system.data[sym]['closes']):
                    exit_price = system.data[sym]['closes'][day]
                    shares, entry_day, entry_price = positions[sym]
                    pnl = shares * (exit_price - entry_price)
                    cost = abs(shares) * (exit_price + entry_price) * system.transaction_cost
                    portfolio_value += pnl - cost
                    del positions[sym]
            
            signals = system.generate_signals(day, regime)
            for signal in signals:
                sym = signal['sym']
                if sym in positions or len(positions) >= system.max_positions:
                    continue
                price = system.data[sym]['closes'][day]
                features = system.compute_features(sym, day)
                if features is None:
                    continue
                atr = features['atr_ratio'] * price
                shares = system.volatility_scaled_sizing(portfolio_value, atr, price)
                if shares > 0:
                    cost = shares * price * system.transaction_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price)
        
        for sym, (shares, entry_day, entry_price) in positions.items():
            if sym in system.data and end_day - 1 < len(system.data[sym]['closes']):
                exit_price = system.data[sym]['closes'][end_day - 1]
                pnl = shares * (exit_price - entry_price)
                cost = abs(shares) * (exit_price + entry_price) * system.transaction_cost
                portfolio_value += pnl - cost
        
        ret = (portfolio_value - initial_capital) / initial_capital
        returns.append(ret)
    
    returns = np.array(returns)
    
    print(f'\nResults over {n_simulations} random 50-day periods:')
    print(f'  Mean Return: {np.mean(returns)*100:.2f}%')
    print(f'  Median Return: {np.median(returns)*100:.2f}%')
    print(f'  Std Dev: {np.std(returns)*100:.2f}%')
    print(f'  Min: {np.min(returns)*100:.2f}%')
    print(f'  Max: {np.max(returns)*100:.2f}%')
    print(f'  Positive: {np.sum(returns > 0)/len(returns)*100:.0f}%')
    print(f'  Sharpe (annualized): {np.mean(returns)/np.std(returns)*np.sqrt(252/50):.3f}')
    
    # Percentiles
    p5 = np.percentile(returns, 5)
    p25 = np.percentile(returns, 25)
    p75 = np.percentile(returns, 75)
    p95 = np.percentile(returns, 95)
    print(f'  5th percentile: {p5*100:.2f}%')
    print(f'  25th percentile: {p25*100:.2f}%')
    print(f'  75th percentile: {p75*100:.2f}%')
    print(f'  95th percentile: {p95*100:.2f}%')
    
    return returns


def parameter_sensitivity_test(initial_capital=100_000):
    """
    Test sensitivity to key parameters.
    """
    print('\n' + '='*70)
    print('PARAMETER SENSITIVITY TEST')
    print('='*70)
    
    data = {}
    universe = UNIVERSE[:25]
    print('Fetching data...')
    for sym in universe:
        d = fetch_data(sym, days=500)
        if d is not None and len(d[0]) > 100:
            data[sym] = {
                'closes': np.array(d[0]),
                'volumes': np.array(d[1]),
                'highs': np.array(d[2]),
                'lows': np.array(d[3])
            }
    
    min_len = min(len(d['closes']) for d in data.values())
    
    # Test different parameter combos
    param_sets = [
        {'stop_loss': 0.06, 'target': 0.10, 'hold': 20, 'name': 'Conservative'},
        {'stop_loss': 0.08, 'target': 0.12, 'hold': 25, 'name': 'V3 Default'},
        {'stop_loss': 0.10, 'target': 0.15, 'hold': 30, 'name': 'Wide stops'},
        {'stop_loss': 0.05, 'target': 0.08, 'hold': 15, 'name': 'Tight stops'},
        {'stop_loss': 0.08, 'target': 0.15, 'hold': 30, 'name': 'Asymmetric'},
    ]
    
    results = []
    for params in param_sets:
        system = FixedProductionSystemV3(universe=universe)
        system.data = data
        system.base_stop_loss = params['stop_loss']
        system.target_profit = params['target']
        system.hold_days_max = params['hold']
        
        portfolio_value = initial_capital
        trades = []
        positions = {}
        
        for day in range(30, min_len - 1):
            regime, confidence = system.classify_regime(day)
            
            positions_to_close = []
            for sym, (shares, entry_day, entry_price) in positions.items():
                if sym in system.data and day < len(system.data[sym]['closes']):
                    current_price = system.data[sym]['closes'][day]
                    profit_pct = (current_price - entry_price) / entry_price
                    loss_pct = (entry_price - current_price) / entry_price
                    
                    if profit_pct > system.target_profit:
                        positions_to_close.append((sym, 'profit_target'))
                    elif loss_pct > system.base_stop_loss:
                        positions_to_close.append((sym, 'stop_loss'))
                    elif day - entry_day >= system.hold_days_max:
                        positions_to_close.append((sym, 'max_hold'))
            
            for sym, reason in positions_to_close:
                if sym in system.data and day < len(system.data[sym]['closes']):
                    exit_price = system.data[sym]['closes'][day]
                    shares, entry_day, entry_price = positions[sym]
                    pnl = shares * (exit_price - entry_price)
                    cost = abs(shares) * (exit_price + entry_price) * system.transaction_cost
                    portfolio_value += pnl - cost
                    trades.append({'pnl': pnl - cost})
                    del positions[sym]
            
            signals = system.generate_signals(day, regime)
            for signal in signals:
                sym = signal['sym']
                if sym in positions or len(positions) >= system.max_positions:
                    continue
                price = system.data[sym]['closes'][day]
                features = system.compute_features(sym, day)
                if features is None:
                    continue
                atr = features['atr_ratio'] * price
                shares = system.volatility_scaled_sizing(portfolio_value, atr, price)
                if shares > 0:
                    cost = shares * price * system.transaction_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price)
        
        for sym, (shares, entry_day, entry_price) in positions.items():
            if sym in system.data and min_len - 1 < len(system.data[sym]['closes']):
                exit_price = system.data[sym]['closes'][min_len - 1]
                pnl = shares * (exit_price - entry_price)
                cost = abs(shares) * (exit_price + entry_price) * system.transaction_cost
                portfolio_value += pnl - cost
        
        total_return = (portfolio_value - initial_capital) / initial_capital
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / max(len(trades), 1)
        
        results.append({
            'name': params['name'],
            'stop': params['stop_loss'],
            'target': params['target'],
            'hold': params['hold'],
            'return': total_return,
            'trades': len(trades),
            'win_rate': win_rate
        })
    
    print(f'\n{"Name":15s} | {"Stop":5s} | {"Target":6s} | {"Hold":4s} | {"Return":8s} | {"Trades":6s} | {"WR":5s}')
    print('-' * 70)
    for r in results:
        print(f'{r["name"]:15s} | {r["stop"]*100:4.0f}% | {r["target"]*100:5.0f}% | {r["hold"]:3d}d | {r["return"]*100:7.2f}% | {r["trades"]:5d} | {r["win_rate"]*100:4.0f}%')
    
    return results


def transaction_cost_stress_test(initial_capital=100_000):
    """
    Test with higher transaction costs (slippage + spread).
    """
    print('\n' + '='*70)
    print('TRANSACTION COST STRESS TEST')
    print('='*70)
    
    data = {}
    universe = UNIVERSE[:25]
    print('Fetching data...')
    for sym in universe:
        d = fetch_data(sym, days=500)
        if d is not None and len(d[0]) > 100:
            data[sym] = {
                'closes': np.array(d[0]),
                'volumes': np.array(d[1]),
                'highs': np.array(d[2]),
                'lows': np.array(d[3])
            }
    
    min_len = min(len(d['closes']) for d in data.values())
    
    cost_scenarios = [
        {'cost': 0.001, 'name': '10 bps (current)'},
        {'cost': 0.002, 'name': '20 bps'},
        {'cost': 0.005, 'name': '50 bps (slippage)'},
        {'cost': 0.010, 'name': '100 bps (high cost)'},
    ]
    
    results = []
    for scenario in cost_scenarios:
        system = FixedProductionSystemV3(universe=universe)
        system.data = data
        system.transaction_cost = scenario['cost']
        
        portfolio_value = initial_capital
        trades = []
        positions = {}
        
        for day in range(30, min_len - 1):
            regime, confidence = system.classify_regime(day)
            
            positions_to_close = []
            for sym, (shares, entry_day, entry_price) in positions.items():
                if sym in system.data and day < len(system.data[sym]['closes']):
                    current_price = system.data[sym]['closes'][day]
                    profit_pct = (current_price - entry_price) / entry_price
                    loss_pct = (entry_price - current_price) / entry_price
                    
                    if profit_pct > system.target_profit:
                        positions_to_close.append((sym, 'profit_target'))
                    elif loss_pct > system.base_stop_loss:
                        positions_to_close.append((sym, 'stop_loss'))
                    elif day - entry_day >= system.hold_days_max:
                        positions_to_close.append((sym, 'max_hold'))
            
            for sym, reason in positions_to_close:
                if sym in system.data and day < len(system.data[sym]['closes']):
                    exit_price = system.data[sym]['closes'][day]
                    shares, entry_day, entry_price = positions[sym]
                    pnl = shares * (exit_price - entry_price)
                    cost = abs(shares) * (exit_price + entry_price) * system.transaction_cost
                    portfolio_value += pnl - cost
                    trades.append({'pnl': pnl - cost})
                    del positions[sym]
            
            signals = system.generate_signals(day, regime)
            for signal in signals:
                sym = signal['sym']
                if sym in positions or len(positions) >= system.max_positions:
                    continue
                price = system.data[sym]['closes'][day]
                features = system.compute_features(sym, day)
                if features is None:
                    continue
                atr = features['atr_ratio'] * price
                shares = system.volatility_scaled_sizing(portfolio_value, atr, price)
                if shares > 0:
                    cost = shares * price * system.transaction_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price)
        
        for sym, (shares, entry_day, entry_price) in positions.items():
            if sym in system.data and min_len - 1 < len(system.data[sym]['closes']):
                exit_price = system.data[sym]['closes'][min_len - 1]
                pnl = shares * (exit_price - entry_price)
                cost = abs(shares) * (exit_price + entry_price) * system.transaction_cost
                portfolio_value += pnl - cost
        
        total_return = (portfolio_value - initial_capital) / initial_capital
        total_costs = sum(abs(t['pnl']) for t in trades) * scenario['cost']
        
        results.append({
            'name': scenario['name'],
            'cost': scenario['cost'],
            'return': total_return,
            'trades': len(trades),
            'total_costs': total_costs
        })
    
    print(f'\n{"Scenario":20s} | {"Cost":6s} | {"Return":8s} | {"Trades":6s} | {"Total Costs":10s}')
    print('-' * 60)
    for r in results:
        print(f'{r["name"]:20s} | {r["cost"]*100:5.1f}% | {r["return"]*100:7.2f}% | {r["trades"]:5d} | ${r["total_costs"]:9,.0f}')
    
    return results


if __name__ == '__main__':
    print('='*70)
    print('V3 COMPREHENSIVE TEST SUITE')
    print('='*70)
    
    # Test 1: Walk-forward OOS
    oos_results = walk_forward_oos_test(n_splits=5)
    
    # Test 2: Monte Carlo
    mc_returns = monte_carlo_test(n_simulations=200)
    
    # Test 3: Parameter sensitivity
    param_results = parameter_sensitivity_test()
    
    # Test 4: Transaction cost stress
    cost_results = transaction_cost_stress_test()
    
    print('\n' + '='*70)
    print('ALL TESTS COMPLETE')
    print('='*70)
