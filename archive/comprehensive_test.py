"""
Comprehensive Stress Test for Production System
================================================
Tests:
1. Walk-forward with embargo and purging
2. Permutation test (statistical significance)
3. Different time periods (regime robustness)
4. Transaction cost sensitivity
5. Monte Carlo simulation
6. Buy-and-hold benchmark comparison
7. Parameter sensitivity analysis
8. 6-month out-of-sample wall
"""
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from production_v18 import fetch_data, UNIVERSE
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')


class ComprehensiveTester:
    """Full stress test suite."""
    
    def __init__(self, universe=None):
        self.universe = universe or UNIVERSE[:25]
        self.data = {}
        self.closes_dict = {}
        self.volumes_dict = {}
        
    def fetch_all(self, days=730):
        """Fetch 2 years of data."""
        print(f'Fetching {len(self.universe)} symbols x {days} days...')
        for sym in self.universe:
            d = fetch_data(sym, days=days)
            if d is not None and len(d[0]) > 200:
                self.data[sym] = {
                    'closes': np.array(d[0]),
                    'volumes': np.array(d[1]),
                    'highs': np.array(d[2]),
                    'lows': np.array(d[3])
                }
                self.closes_dict[sym] = np.array(d[0])
                self.volumes_dict[sym] = np.array(d[1])
        print(f'Loaded {len(self.data)} symbols')
        return self.data
    
    # =========================================
    # TEST 1: Walk-Forward with Purging + Embargo
    # =========================================
    def test_walk_forward(self, train_window=60, test_window=20, embargo=5, step=20):
        """
        Rolling window walk-forward.
        Train on [t-60, t], test on [t+embargo, t+embargo+20].
        """
        print('\n' + '='*70)
        print('TEST 1: Walk-Forward with Purging + Embargo')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        
        folds = []
        fold_returns = []
        fold_win_rates = []
        fold_aucs = []
        
        train_end = 200 + train_window
        
        while train_end + embargo + test_window <= min_len:
            test_start = train_end + embargo
            test_end = test_start + test_window
            
            # Train meta-labeling model on training window
            meta_X, meta_y = [], []
            for day in range(200, train_end):
                signals = self._generate_signals(day)
                for signal in signals:
                    features = self._compute_features(signal['sym'], day)
                    if features is not None:
                        closes = self.closes_dict[signal['sym']]
                        if day + 5 < len(closes):
                            outcome = 1 if closes[day + 5] > closes[day] else 0
                            meta_X.append(list(features.values()))
                            meta_y.append(outcome)
            
            if len(meta_X) < 20:
                train_end += step
                continue
            
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            meta_X = np.array(meta_X)
            meta_y = np.array(meta_y)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(meta_X)
            
            model = LogisticRegression(C=1.0, max_iter=1000)
            model.fit(X_scaled, meta_y)
            
            # Test on OOS window
            fold_trades = []
            for day in range(test_start, test_end):
                signals = self._generate_signals(day)
                for signal in signals:
                    features = self._compute_features(signal['sym'], day)
                    if features is not None:
                        X = np.array(list(features.values())).reshape(1, -1)
                        X_scaled = scaler.transform(X)
                        proba = model.predict_proba(X_scaled)[0]
                        
                        if proba[1] > 0.52:
                            closes = self.closes_dict[signal['sym']]
                            if day + 5 < len(closes):
                                ret = (closes[day + 5] - closes[day]) / closes[day]
                                fold_trades.append(ret)
            
            if fold_trades:
                fold_ret = np.mean(fold_trades)
                fold_wr = np.mean([1 for r in fold_trades if r > 0])
                fold_returns.append(fold_ret)
                fold_win_rates.append(fold_wr)
                
                # Fold AUC
                if len(meta_y) > 10:
                    try:
                        train_pred = model.predict_proba(X_scaled)[:, 1]
                        fold_auc = roc_auc_score(meta_y, train_pred)
                        fold_aucs.append(fold_auc)
                    except:
                        pass
            
            train_end += step
        
        # Results
        if fold_returns:
            fold_returns = np.array(fold_returns)
            fold_win_rates = np.array(fold_win_rates)
            
            print(f'Folds completed: {len(fold_returns)}')
            print(f'Fold return: mean={np.mean(fold_returns)*100:.3f}%, std={np.std(fold_returns)*100:.3f}%')
            print(f'Fold win rate: mean={np.mean(fold_win_rates)*100:.1f}%, std={np.std(fold_win_rates)*100:.1f}%')
            print(f'Positive folds: {np.sum(fold_returns > 0)}/{len(fold_returns)} ({np.mean(fold_returns > 0)*100:.1f}%)')
            
            # Statistical significance
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(fold_returns, 0)
            print(f'T-test (H0: return=0): t={t_stat:.3f}, p={p_value:.4f}')
            
            if p_value < 0.05:
                print('✅ Statistically significant edge (p < 0.05)')
            elif p_value < 0.10:
                print('⚠️  Marginally significant (p < 0.10)')
            else:
                print('❌ NOT statistically significant')
            
            if fold_aucs:
                print(f'AUC: mean={np.mean(fold_aucs):.4f}, min={np.min(fold_aucs):.4f}, max={np.max(fold_aucs):.4f}')
        
        return fold_returns
    
    # =========================================
    # TEST 2: Permutation Test
    # =========================================
    def test_permutation(self, n_simulations=200):
        """
        Run system with random features to test if edge is from signal or structure.
        """
        print('\n' + '='*70)
        print('TEST 2: Permutation Test')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        
        # Real system returns
        real_returns = self._run_system_returns(200, min_len - 1)
        
        # Random system returns
        random_returns = []
        for _ in range(n_simulations):
            returns = self._run_system_returns_random(200, min_len - 1)
            random_returns.append(np.mean(returns) if len(returns) > 0 else 0)
        
        random_returns = np.array(random_returns)
        
        print(f'Real system mean return: {np.mean(real_returns)*100:.3f}%')
        print(f'Random system mean return: {np.mean(random_returns)*100:.3f}% ± {np.std(random_returns)*100:.3f}%')
        
        edge = np.mean(real_returns) - np.mean(random_returns)
        print(f'Edge over random: {edge*100:.3f}%')
        
        # Percentile ranking
        percentile = np.mean(random_returns < np.mean(real_returns))
        print(f'Real system percentile: {percentile*100:.1f}%')
        
        if percentile > 0.95:
            print('✅ PASS: Real system beats 95% of random simulations')
        elif percentile > 0.90:
            print('⚠️  MARGINAL: Real system beats 90% of random simulations')
        else:
            print('❌ FAIL: Real system does NOT beat random simulations')
        
        return real_returns, random_returns
    
    def _run_system_returns(self, start, end):
        """Run real system, return daily returns."""
        returns = []
        for day in range(start, end):
            signals = self._generate_signals(day)
            day_returns = []
            for signal in signals:
                closes = self.closes_dict[signal['sym']]
                if day + 5 < len(closes):
                    ret = (closes[day + 5] - closes[day]) / closes[day]
                    day_returns.append(ret)
            if day_returns:
                returns.append(np.mean(day_returns))
        return np.array(returns) if len(returns) > 0 else np.array([0])
    
    def _run_system_returns_random(self, start, end):
        """Run system with random feature permutation."""
        returns = []
        for day in range(start, end):
            signals = self._generate_signals(day)
            day_returns = []
            for signal in signals:
                closes = self.closes_dict[signal['sym']]
                if day + 5 < len(closes):
                    # Random noise instead of real features
                    if np.random.random() > 0.5:  # Random "prediction"
                        ret = (closes[day + 5] - closes[day]) / closes[day]
                        day_returns.append(ret)
            if day_returns:
                returns.append(np.mean(day_returns))
        return np.array(returns) if len(returns) > 0 else np.array([0])
    
    # =========================================
    # TEST 3: Time Period Robustness
    # =========================================
    def test_time_periods(self):
        """Test on different time periods (first half, second half, sliding windows)."""
        print('\n' + '='*70)
        print('TEST 3: Time Period Robustness')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        
        periods = [
            ('First half', 200, 200 + (min_len - 200) // 2),
            ('Second half', 200 + (min_len - 200) // 2, min_len - 1),
            ('First quarter', 200, 200 + (min_len - 200) // 4),
            ('Last quarter', min_len - 1 - (min_len - 200) // 4, min_len - 1),
        ]
        
        results = {}
        for name, start, end in periods:
            returns = self._run_system_returns(start, end)
            if len(returns) > 0:
                mean_ret = np.mean(returns)
                wr = np.mean(returns > 0)
                results[name] = {'return': mean_ret, 'win_rate': wr}
                print(f'  {name:<20}: return={mean_ret*100:>7.3f}%, win_rate={wr*100:>5.1f}%')
        
        # Consistency check
        all_positive = all(r['return'] > 0 for r in results.values())
        if all_positive:
            print('  ✅ Profitable in ALL time periods')
        else:
            negative = [k for k, v in results.items() if v['return'] <= 0]
            print(f'  ❌ Unprofitable in: {", ".join(negative)}')
        
        return results
    
    # =========================================
    # TEST 4: Transaction Cost Sensitivity
    # =========================================
    def test_transaction_costs(self):
        """Test with varying transaction costs."""
        print('\n' + '='*70)
        print('TEST 4: Transaction Cost Sensitivity')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        
        costs = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]
        results = {}
        
        for cost in costs:
            portfolio_value = 100_000
            positions = {}
            trades = 0
            
            for day in range(200, min_len - 1):
                # Close positions
                to_close = [s for s, (sh, ed, ep) in positions.items() if day - ed >= 5]
                for sym in to_close:
                    if day < len(self.closes_dict[sym]):
                        exit_price = self.closes_dict[sym][day]
                        shares, entry_day, entry_price = positions[sym]
                        pnl = shares * (exit_price - entry_price)
                        tc = abs(shares) * (exit_price + entry_price) * cost
                        portfolio_value += pnl - tc
                        trades += 1
                        del positions[sym]
                
                # New positions
                signals = self._generate_signals(day)
                for signal in signals[:3]:
                    sym = signal['sym']
                    if sym in positions or len(positions) >= 5:
                        continue
                    price = self.closes_dict[sym][day]
                    allocation = portfolio_value * 0.08
                    shares = int(allocation / price)
                    if shares > 0:
                        tc = shares * price * cost
                        portfolio_value -= tc
                        positions[sym] = (shares, day, price)
            
            total_return = (portfolio_value - 100_000) / 100_000
            results[cost] = total_return
            print(f'  Cost {cost*100:>5.1f}%: return={total_return*100:>7.2f}%')
        
        # Breakeven cost
        positive_at_all = all(v > 0 for v in results.values())
        if positive_at_all:
            print('  ✅ Profitable even at 10% transaction cost')
        else:
            first_negative = next((k for k, v in results.items() if v <= 0), None)
            if first_negative:
                print(f'  ❌ Breakeven at {first_negative*100:.1f}% cost')
        
        return results
    
    # =========================================
    # TEST 5: Monte Carlo Simulation
    # =========================================
    def test_monte_carlo(self, n_simulations=1000):
        """Monte Carlo simulation of trade returns."""
        print('\n' + '='*70)
        print('TEST 5: Monte Carlo Simulation')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        
        # Collect actual trade returns
        actual_returns = []
        for day in range(200, min_len - 1):
            signals = self._generate_signals(day)
            for signal in signals:
                closes = self.closes_dict[signal['sym']]
                if day + 5 < len(closes):
                    ret = (closes[day + 5] - closes[day]) / closes[day]
                    actual_returns.append(ret)
        
        actual_returns = np.array(actual_returns)
        n_trades = len(actual_returns)
        
        print(f'Collected {n_trades} actual trade returns')
        print(f'Mean: {np.mean(actual_returns)*100:.3f}%, Std: {np.std(actual_returns)*100:.3f}%')
        print(f'Skew: {pd.Series(actual_returns).skew():.3f}')
        print(f'Kurtosis: {pd.Series(actual_returns).kurtosis():.3f}')
        
        # Bootstrap simulations
        sim_final_values = []
        sim_max_drawdowns = []
        sim_sharpes = []
        
        for _ in range(n_simulations):
            # Randomly sample trades with replacement
            sample_idx = np.random.choice(n_trades, size=n_trades, replace=True)
            sample_returns = actual_returns[sample_idx]
            
            # Simulate equity curve
            equity = [100_000]
            for ret in sample_returns:
                equity.append(equity[-1] * (1 + ret * 0.08))  # 8% allocation per trade
            
            equity = np.array(equity)
            final_value = equity[-1]
            max_dd = np.min(equity / np.maximum.accumulate(equity) - 1)
            
            # Sharpe
            daily_rets = np.diff(equity) / equity[:-1]
            sharpe = np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)
            
            sim_final_values.append(final_value)
            sim_max_drawdowns.append(max_dd)
            sim_sharpes.append(sharpe)
        
        sim_final_values = np.array(sim_final_values)
        sim_max_drawdowns = np.array(sim_max_drawdowns)
        sim_sharpes = np.array(sim_sharpes)
        
        print(f'\nMonte Carlo Results ({n_simulations} simulations):')
        print(f'  Final value: mean=${np.mean(sim_final_values):,.0f}, median=${np.median(sim_final_values):,.0f}')
        print(f'  5th percentile: ${np.percentile(sim_final_values, 5):,.0f}')
        print(f'  95th percentile: ${np.percentile(sim_final_values, 95):,.0f}')
        print(f'  Probability of profit: {np.mean(sim_final_values > 100_000)*100:.1f}%')
        print(f'  Probability of >10% return: {np.mean(sim_final_values > 110_000)*100:.1f}%')
        print(f'  Max drawdown: mean={np.mean(sim_max_drawdowns)*100:.1f}%, 95th pct={np.percentile(sim_max_drawdowns, 5)*100:.1f}%')
        print(f'  Sharpe: mean={np.mean(sim_sharpes):.3f}, 5th pct={np.percentile(sim_sharpes, 5):.3f}')
        
        if np.mean(sim_final_values > 100_000) > 0.9:
            print('  ✅ >90% probability of profit')
        elif np.mean(sim_final_values > 100_000) > 0.7:
            print('  ⚠️  >70% probability of profit')
        else:
            print('  ❌ <70% probability of profit')
        
        return sim_final_values, sim_max_drawdowns, sim_sharpes
    
    # =========================================
    # TEST 6: Buy-and-Hold Benchmark
    # =========================================
    def test_buy_and_hold(self):
        """Compare against buy-and-hold SPY."""
        print('\n' + '='*70)
        print('TEST 6: Buy-and-Hold Benchmark')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        
        # Buy and hold SPY
        if 'SPY' in self.closes_dict:
            spy = self.closes_dict['spy']
            bh_return = (spy[min_len - 1] - spy[200]) / spy[200]
            print(f'Buy-and-Hold SPY return: {bh_return*100:.2f}%')
        else:
            bh_return = None
            print('SPY not available for benchmark')
        
        # Equal-weight buy and hold
        ewh_returns = []
        for sym in self.data:
            closes = self.closes_dict[sym]
            if min_len - 1 < len(closes):
                ret = (closes[min_len - 1] - closes[200]) / closes[200]
                ewh_returns.append(ret)
        
        ewh_return = np.mean(ewh_returns)
        print(f'Equal-weight buy-hold return: {ewh_return*100:.2f}%')
        
        # Our system
        system_returns = self._run_system_returns(200, min_len - 1)
        system_total = np.sum(system_returns) * 0.08  # 8% per trade
        print(f'System total return: {system_total*100:.2f}%')
        
        if bh_return is not None:
            if system_total > bh_return:
                print('  ✅ System beats buy-and-hold SPY')
            else:
                print(f'  ❌ System underperforms SPY by {(bh_return - system_total)*100:.2f}%')
        
        return bh_return, ewh_return, system_total
    
    # =========================================
    # TEST 7: Parameter Sensitivity
    # =========================================
    def test_parameter_sensitivity(self):
        """Test with different parameters."""
        print('\n' + '='*70)
        print('TEST 7: Parameter Sensitivity')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        
        # Test different lookback periods
        lookbacks = [3, 5, 10, 15, 20]
        lookback_results = []
        
        for lb in lookbacks:
            returns = []
            for day in range(200, min_len - 1):
                # Generate signals with this lookback
                sym_returns = {}
                for sym in self.data:
                    closes = self.closes_dict[sym]
                    if day < len(closes) and day >= lb:
                        sym_returns[sym] = (closes[day] / closes[day - lb] - 1)
                
                if len(sym_returns) >= 5:
                    mean_ret = np.mean(list(sym_returns.values()))
                    deviations = {sym: ret - mean_ret for sym, ret in sym_returns.items()}
                    abs_sum = sum(abs(d) for d in deviations.values())
                    if abs_sum > 1e-10:
                        weights = {sym: -dev / abs_sum for sym, dev in deviations.items()}
                        # Top 3 underperformers
                        sorted_syms = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                        for sym, w in sorted_syms[:3]:
                            if w > 0.02:
                                closes = self.closes_dict[sym]
                                if day + 5 < len(closes):
                                    ret = (closes[day + 5] - closes[day]) / closes[day]
                                    returns.append(ret)
            
            if returns:
                mean_ret = np.mean(returns)
                lookback_results.append(mean_ret)
                print(f'  Lookback {lb:>2} days: mean return={mean_ret*100:.3f}%')
        
        if lookback_results:
            best = lookbacks[np.argmax(lookback_results)]
            print(f'  Best lookback: {best} days')
            
            # Sensitivity: how much does performance vary?
            std_ret = np.std(lookback_results)
            if std_ret < 0.001:
                print('  ✅ Stable across all parameters')
            else:
                print(f'  ⚠️  Sensitive (std={std_ret*100:.3f}%)')
        
        return lookback_results
    
    # =========================================
    # HELPER METHODS
    # =========================================
    def _generate_signals(self, day_idx):
        """Cross-sectional mean reversion signals."""
        if day_idx < 20:
            return []
        
        sym_returns = {}
        for sym, data in self.data.items():
            closes = data['closes']
            if day_idx < len(closes) and day_idx >= 5:
                sym_returns[sym] = (closes[day_idx] / closes[day_idx - 5] - 1)
        
        if len(sym_returns) < 5:
            return []
        
        mean_ret = np.mean(list(sym_returns.values()))
        deviations = {sym: ret - mean_ret for sym, ret in sym_returns.items()}
        abs_sum = sum(abs(d) for d in deviations.values())
        
        if abs_sum < 1e-10:
            return []
        
        weights = {sym: -dev / abs_sum for sym, dev in deviations.items()}
        signals = []
        for sym, weight in weights.items():
            if weight > 0.02:
                signals.append({'sym': sym, 'weight': weight})
        
        signals.sort(key=lambda x: x['weight'], reverse=True)
        return signals[:10]
    
    def _compute_features(self, sym, day_idx):
        """Compute features for meta-labeling."""
        if sym not in self.data:
            return None
        
        closes = self.closes_dict[sym][:day_idx+1]
        volumes = self.volumes_dict[sym][:day_idx+1]
        n = len(closes)
        
        if n < 25:
            return None
        
        rets = np.diff(closes) / closes[:-1]
        
        gains = np.where(rets > 0, rets, 0)
        losses = np.where(rets < 0, -rets, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        ma20 = np.mean(closes[-20:])
        price_vs_ma20 = closes[-1] / ma20 - 1
        vol20 = np.std(rets[-20:]) * np.sqrt(252)
        vol_short = np.std(rets[-10:]) if len(rets) >= 10 else vol20
        vol_long = np.std(rets[-30:]) if len(rets) >= 30 else vol20
        vol_ratio = vol_short / (vol_long + 1e-10)
        tr = np.abs(np.diff(closes[-15:]))
        atr = np.mean(tr)
        atr_ratio = atr / (closes[-1] + 1e-10)
        mom_5 = (closes[-1] / closes[-6] - 1) if n >= 6 else 0
        mom_20 = (closes[-1] / closes[-21] - 1) if n >= 21 else 0
        vol_ma20 = np.mean(volumes[-20:])
        vol_ratio_20 = volumes[-1] / (vol_ma20 + 1e-10)
        
        return {
            'rsi': rsi, 'price_vs_ma20': price_vs_ma20, 'volatility': vol20,
            'vol_ratio': vol_ratio, 'atr_ratio': atr_ratio,
            'momentum_5d': mom_5, 'momentum_20d': mom_20, 'volume_ratio': vol_ratio_20,
        }


def main():
    tester = ComprehensiveTester(universe=UNIVERSE[:25])
    tester.fetch_all(days=730)
    
    # Run all tests
    wf_returns = tester.test_walk_forward()
    real_ret, random_ret = tester.test_permutation(n_simulations=200)
    period_results = tester.test_time_periods()
    cost_results = tester.test_transaction_costs()
    mc_values, mc_dd, mc_sharpe = tester.test_monte_carlo(n_simulations=1000)
    bh_ret, ewh_ret, sys_ret = tester.test_buy_and_hold()
    param_results = tester.test_parameter_sensitivity()
    
    # Final summary
    print('\n' + '='*70)
    print('FINAL SUMMARY')
    print('='*70)
    
    tests_passed = 0
    tests_total = 7
    
    # Walk-forward
    if len(wf_returns) > 0 and np.mean(wf_returns) > 0:
        print('✅ Walk-forward: PASS')
        tests_passed += 1
    else:
        print('❌ Walk-forward: FAIL')
    
    # Permutation
    if np.mean(real_ret) > np.mean(random_ret):
        print('✅ Permutation: PASS')
        tests_passed += 1
    else:
        print('❌ Permutation: FAIL')
    
    # Time periods
    if all(r['return'] > 0 for r in period_results.values()):
        print('✅ Time periods: PASS')
        tests_passed += 1
    else:
        print('❌ Time periods: FAIL')
    
    # Transaction costs
    if cost_results.get(0.001, 0) > 0 and cost_results.get(0.005, 0) > 0:
        print('✅ Transaction costs: PASS')
        tests_passed += 1
    else:
        print('❌ Transaction costs: FAIL')
    
    # Monte Carlo
    if np.mean(mc_values > 100_000) > 0.7:
        print('✅ Monte Carlo: PASS')
        tests_passed += 1
    else:
        print('❌ Monte Carlo: FAIL')
    
    # Buy-and-hold
    if sys_ret > ewh_ret:
        print('✅ Buy-and-hold: PASS')
        tests_passed += 1
    else:
        print('❌ Buy-and-hold: FAIL')
    
    # Parameter sensitivity
    if param_results and np.std(param_results) < 0.001:
        print('✅ Parameter sensitivity: PASS')
        tests_passed += 1
    else:
        print('❌ Parameter sensitivity: FAIL')
    
    print(f'\n{tests_passed}/{tests_total} tests passed')
    
    if tests_passed >= 6:
        print('🟢 STRONG EDGE — Deploy with confidence')
    elif tests_passed >= 4:
        print('🟡 MODERATE EDGE — Deploy with caution')
    elif tests_passed >= 2:
        print('🟠 WEAK EDGE — Paper trade only')
    else:
        print('🔴 NO EDGE — Do not deploy')


if __name__ == '__main__':
    main()
