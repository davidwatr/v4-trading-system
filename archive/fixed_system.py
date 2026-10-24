"""
Fix-the-Edge Trading System
===========================
Fixes the 5 failures from comprehensive_test.py:

FAIL 1: Random beats real (mean reversion wrong in bull markets)
  → FIX: Add momentum overlay. Buy pullbacks in uptrends, not just underperformers.

FAIL 2: Transaction cost sensitivity (breakeven at 0.5%)
  → FIX: Reduce trade frequency. Only top 2 signals per week, 10-day holds.

FAIL 3: Underperforms buy-and-hold by 24.79%
  → FIX: Regime-aware. In strong bull markets, weight momentum > mean reversion.

FAIL 4: Walk-forward not significant (p=0.42)
  → FIX: Stronger signal threshold + regime filter = fewer, higher-quality trades.

FAIL 5: Parameter sensitive
  → FIX: Ensemble of lookbacks (3, 5, 10 days) instead of single parameter.
"""
import numpy as np
import pandas as pd
import warnings
from production_v18 import fetch_data, UNIVERSE
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')


class FixedTradingSystem:
    """
    Fixed system addressing all 5 failures.
    
    Key changes:
    1. Momentum + Mean Reversion combo (not just MR)
    2. Regime-aware weighting (bull=momentum, choppy=MR)
    3. Reduced trade frequency (top 2 per week, 10-day holds)
    4. Higher confidence threshold (weight > 0.05)
    5. Ensemble lookbacks (3, 5, 10 days)
    """
    
    def __init__(self, universe=None):
        self.universe = universe or UNIVERSE[:25]
        self.data = {}
        
    def fetch_data(self, days=730):
        print(f'Fetching {len(self.universe)} symbols...')
        for sym in self.universe:
            d = fetch_data(sym, days=days)
            if d is not None and len(d[0]) > 200:
                self.data[sym] = {
                    'closes': np.array(d[0]),
                    'volumes': np.array(d[1]),
                    'highs': np.array(d[2]),
                    'lows': np.array(d[3])
                }
        print(f'Loaded {len(self.data)} symbols')
    
    def detect_regime(self, day_idx):
        """
        Detect market regime to weight momentum vs mean reversion.
        
        Returns: 'bull', 'bear', 'choppy'
        """
        # Use available market proxy
        for spy_sym in ['SPY', 'QQQ', 'AAPL', 'MSFT']:
            if spy_sym in self.data:
                closes = self.data[spy_sym]['closes'][:day_idx+1]
                if len(closes) < 30:
                    continue
                
                rets = np.diff(closes) / closes[:-1]
                vol = np.std(rets[-20:]) * np.sqrt(252)
                
                # 20-day trend
                trend_20 = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0
                
                # 5-day trend
                trend_5 = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
                
                if trend_20 > 0.02 and vol < 0.25:
                    return 'bull'
                elif trend_20 < -0.02 and vol > 0.20:
                    return 'bear'
                else:
                    return 'choppy'
        
        return 'choppy'
    
    def compute_combined_score(self, day_idx):
        """
        Combined momentum + mean reversion score.
        
        In bull markets: weight momentum higher (0.7) than MR (0.3)
        In bear markets: weight MR higher (0.6) than momentum (0.4)
        In choppy: equal weight (0.5/0.5)
        
        Also uses ensemble of lookbacks to reduce parameter sensitivity.
        """
        if day_idx < 20:
            return []
        
        regime = self.detect_regime(day_idx)
        
        # Regime-dependent weights
        if regime == 'bull':
            mr_weight, mom_weight = 0.3, 0.7
        elif regime == 'bear':
            mr_weight, mom_weight = 0.6, 0.4
        else:
            mr_weight, mom_weight = 0.5, 0.5
        
        # Compute returns for each stock with ensemble lookbacks
        sym_scores = {}
        for sym, data in self.data.items():
            closes = data['closes']
            if day_idx >= len(closes):
                continue
            
            # Mean reversion component (ensemble of 3, 5, 10 days)
            mr_signals = []
            for lb in [3, 5, 10]:
                if day_idx >= lb:
                    ret = (closes[day_idx] / closes[day_idx - lb] - 1)
                    mr_signals.append(-ret)  # Negative because we want underperformers
            
            mr_score = np.mean(mr_signals) if mr_signals else 0
            
            # Momentum component (20-day trend)
            if day_idx >= 20:
                mom_score = (closes[day_idx] / closes[day_idx - 20] - 1)
            else:
                mom_score = 0
            
            # Combined score
            combined = mr_weight * mr_score + mom_weight * mom_score
            
            # Only include if momentum is positive (stock is in uptrend)
            # This avoids value traps
            if mom_score > -0.01:  # Allow slightly negative for bear markets
                sym_scores[sym] = combined
        
        if len(sym_scores) < 5:
            return []
        
        # Cross-sectional ranking
        mean_score = np.mean(list(sym_scores.values()))
        deviations = {sym: score - mean_score for sym, score in sym_scores.items()}
        abs_sum = sum(abs(d) for d in deviations.values())
        
        if abs_sum < 1e-10:
            return []
        
        # Normalize to weights
        weights = {sym: dev / abs_sum for sym, dev in deviations.items()}
        
        # Select top signals with high confidence threshold
        signals = []
        for sym, weight in weights.items():
            if weight > 0.05:  # Higher threshold = fewer, better signals
                signals.append({
                    'sym': sym,
                    'weight': weight,
                    'regime': regime
                })
        
        # Sort by weight and take top 2 only (reduces trade frequency)
        signals.sort(key=lambda x: x['weight'], reverse=True)
        return signals[:2]
    
    def backtest(self, initial_capital=100_000, trans_cost=0.001, hold_days=10):
        """
        Backtest with reduced frequency and longer holds.
        """
        min_len = min(len(d['closes']) for d in self.data.values())
        
        portfolio_value = initial_capital
        daily_values = []
        trades = []
        positions = {}
        
        for day in range(200, min_len - 1):
            # Close expired positions
            to_close = [s for s, (sh, ed, ep) in positions.items() if day - ed >= hold_days]
            for sym in to_close:
                if day < len(self.data[sym]['closes']):
                    exit_price = self.data[sym]['closes'][day]
                    shares, entry_day, entry_price = positions[sym]
                    pnl = shares * (exit_price - entry_price)
                    cost = abs(shares) * (exit_price + entry_price) * trans_cost
                    portfolio_value += pnl - cost
                    trades.append({'sym': sym, 'pnl': pnl - cost, 'regime': positions[sym][2] if len(positions[sym]) > 2 else 'unknown'})
                    del positions[sym]
            
            # Generate new signals (only 2 per day max)
            signals = self.compute_combined_score(day)
            
            for signal in signals:
                sym = signal['sym']
                if sym in positions:
                    continue
                if len(positions) >= 3:  # Max 3 positions (reduced from 5)
                    break
                
                price = self.data[sym]['closes'][day]
                allocation = portfolio_value * min(signal['weight'] * 1.5, 0.15)
                shares = int(allocation / price)
                if shares > 0:
                    cost = shares * price * trans_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price)
            
            daily_values.append(portfolio_value)
        
        # Close remaining
        for sym, (shares, entry_day, entry_price) in positions.items():
            if min_len - 1 < len(self.data[sym]['closes']):
                exit_price = self.data[sym]['closes'][min_len - 1]
                pnl = shares * (exit_price - entry_price)
                cost = abs(shares) * (exit_price + entry_price) * trans_cost
                portfolio_value += pnl - cost
        
        daily_values.append(portfolio_value)
        daily_values = np.array(daily_values)
        daily_rets = np.diff(daily_values) / daily_values[:-1]
        
        total_return = (portfolio_value - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(daily_values)) - 1
        sharpe = np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)
        max_dd = np.min(daily_values / np.maximum.accumulate(daily_values) - 1)
        
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
        avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'daily_values': daily_values,
            'trades': trades
        }


def run_comparison():
    """Compare fixed system vs original vs buy-and-hold."""
    print('='*70)
    print('FIXED SYSTEM vs ORIGINAL vs BUY-AND-HOLD')
    print('='*70)
    
    universe = UNIVERSE[:25]
    
    # Fetch data
    print('\nFetching data...')
    system = FixedTradingSystem(universe)
    system.fetch_data(days=730)
    
    min_len = min(len(d['closes']) for d in system.data.values())
    
    # Test 1: Fixed system
    print('\n' + '-'*70)
    print('TEST 1: FIXED SYSTEM (Momentum + MR + Regime)')
    print('-'*70)
    fixed_results = system.backtest()
    
    print(f'Total Return: {fixed_results["total_return"]*100:.2f}%')
    print(f'Annual Return: {fixed_results["annual_return"]*100:.2f}%')
    print(f'Sharpe Ratio: {fixed_results["sharpe"]:.3f}')
    print(f'Max Drawdown: {fixed_results["max_drawdown"]*100:.2f}%')
    print(f'Win Rate: {fixed_results["win_rate"]*100:.1f}%')
    print(f'Avg Win: ${fixed_results["avg_win"]:.2f}')
    print(f'Avg Loss: ${fixed_results["avg_loss"]:.2f}')
    print(f'Total Trades: {fixed_results["total_trades"]}')
    
    # Test 2: Buy-and-hold benchmark
    print('\n' + '-'*70)
    print('TEST 2: BUY-AND-HOLD BENCHMARK')
    print('-'*70)
    
    ewh_returns = []
    for sym in system.data:
        closes = system.data[sym]['closes']
        if min_len - 1 < len(closes):
            ret = (closes[min_len - 1] - closes[200]) / closes[200]
            ewh_returns.append(ret)
    
    ewh_return = np.mean(ewh_returns)
    print(f'Equal-weight buy-hold return: {ewh_return*100:.2f}%')
    
    # Test 3: Permutation test on fixed system
    print('\n' + '-'*70)
    print('TEST 3: PERMUTATION TEST (Fixed System)')
    print('-'*70)
    
    # Real system returns
    real_returns = []
    for day in range(200, min_len - 6):
        signals = system.compute_combined_score(day)
        for signal in signals:
            closes = system.data[signal['sym']]['closes']
            if day + 10 < len(closes):
                ret = (closes[day + 10] - closes[day]) / closes[day]
                real_returns.append(ret)
    
    real_returns = np.array(real_returns)
    
    # Random returns
    random_returns = []
    for _ in range(200):
        sample_idx = np.random.choice(len(real_returns), size=len(real_returns), replace=True)
        random_returns.append(np.mean(real_returns[sample_idx]))
    
    random_returns = np.array(random_returns)
    
    # But random should pick random stocks, not random days of our stocks
    # Better permutation: pick random stocks from universe
    perm_returns = []
    for _ in range(200):
        perm_day_returns = []
        for day in range(200, min_len - 6, 3):  # Every 3rd day to match frequency
            # Pick 2 random stocks
            random_syms = np.random.choice(list(system.data.keys()), size=2, replace=False)
            for sym in random_syms:
                closes = system.data[sym]['closes']
                if day + 10 < len(closes):
                    ret = (closes[day + 10] - closes[day]) / closes[day]
                    perm_day_returns.append(ret)
        if perm_day_returns:
            perm_returns.append(np.mean(perm_day_returns))
    
    perm_returns = np.array(perm_returns)
    
    print(f'Real system mean return: {np.mean(real_returns)*100:.3f}%')
    print(f'Random stock mean return: {np.mean(perm_returns)*100:.3f}% ± {np.std(perm_returns)*100:.3f}%')
    
    edge = np.mean(real_returns) - np.mean(perm_returns)
    print(f'Edge over random: {edge*100:.3f}%')
    
    percentile = np.mean(perm_returns < np.mean(real_returns))
    print(f'Real system percentile: {percentile*100:.1f}%')
    
    if percentile > 0.90:
        print('✅ PASS: Real system beats 90% of random')
    elif percentile > 0.70:
        print('⚠️  MARGINAL: Real system beats 70% of random')
    else:
        print('❌ FAIL: Real system does NOT beat random')
    
    # Test 4: Walk-forward with purging
    print('\n' + '-'*70)
    print('TEST 4: WALK-FORWARD WITH PURGING')
    print('-'*70)
    
    from scipy import stats
    
    fold_returns = []
    train_window = 60
    test_window = 20
    embargo = 5
    step = 20
    
    train_end = 200 + train_window
    while train_end + embargo + test_window <= min_len:
        test_start = train_end + embargo
        test_end = test_start + test_window
        
        # Test on OOS window
        fold_trades = []
        for day in range(test_start, test_end):
            signals = system.compute_combined_score(day)
            for signal in signals:
                closes = system.data[signal['sym']]['closes']
                if day + 10 < len(closes):
                    ret = (closes[day + 10] - closes[day]) / closes[day]
                    fold_trades.append(ret)
        
        if fold_trades:
            fold_returns.append(np.mean(fold_trades))
        
        train_end += step
    
    if fold_returns:
        fold_returns = np.array(fold_returns)
        print(f'Folds: {len(fold_returns)}')
        print(f'Fold return: mean={np.mean(fold_returns)*100:.3f}%, std={np.std(fold_returns)*100:.3f}%')
        print(f'Positive folds: {np.sum(fold_returns > 0)}/{len(fold_returns)} ({np.mean(fold_returns > 0)*100:.1f}%)')
        
        t_stat, p_value = stats.ttest_1samp(fold_returns, 0)
        print(f'T-test: t={t_stat:.3f}, p={p_value:.4f}')
        
        if p_value < 0.05:
            print('✅ Statistically significant')
        elif p_value < 0.10:
            print('⚠️  Marginally significant')
        else:
            print('❌ NOT statistically significant')
    
    # Test 5: Transaction cost sensitivity
    print('\n' + '-'*70)
    print('TEST 5: TRANSACTION COST SENSITIVITY')
    print('-'*70)
    
    costs = [0.0, 0.001, 0.002, 0.005, 0.01]
    for cost in costs:
        result = system.backtest(trans_cost=cost)
        print(f'  Cost {cost*100:>5.1f}%: return={result["total_return"]*100:>7.2f}%')
    
    # Final verdict
    print('\n' + '='*70)
    print('FINAL VERDICT')
    print('='*70)
    
    print(f'\nFixed System:')
    print(f'  Return: {fixed_results["total_return"]*100:.2f}%')
    print(f'  Sharpe: {fixed_results["sharpe"]:.3f}')
    print(f'  Win Rate: {fixed_results["win_rate"]*100:.1f}%')
    print(f'  Trades: {fixed_results["total_trades"]}')
    
    print(f'\nBuy-and-Hold:')
    print(f'  Return: {ewh_return*100:.2f}%')
    
    if fixed_results['total_return'] > ewh_return:
        print('\n  ✅ FIXED SYSTEM BEATS BUY-AND-HOLD')
    else:
        print(f'\n  ❌ Fixed system underperforms by {(ewh_return - fixed_results["total_return"])*100:.2f}%')
        print(f'  → Buy-and-hold is still better')


if __name__ == '__main__':
    run_comparison()
