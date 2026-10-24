#!/usr/bin/env python3
"""
V4 TEST SUITE — Matches production format (tuple-based)
=======================================================
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, 'C:/Users/david_wncs0ps/jarvis/trading')
from production_v18 import fetch_data, UNIVERSE


class V4Tester:
    def __init__(self):
        self.data = {}
        self.closes_cache = {}
        self.config = {
            'target_vol': 0.18,
            'vol_half_life': 10,
            'corr_threshold': 0.75,
            'max_cluster_exposure': 0.30,
            'max_positions': 5,
            'risk_per_trade': 0.02,
            'stop_loss': 0.08,
            'profit_target': 0.12,
            'max_hold_days': 25,
            'entry_threshold': 0.03,
        }
        self.max_position_size = 0.15
        self.transaction_cost = 0.001

    def fetch_all(self, days=500):
        for sym in UNIVERSE[:25]:
            d = fetch_data(sym, days=days)
            if d is not None and len(d[0]) > 50:
                self.data[sym] = {
                    'closes': np.array(d[0]),
                    'volumes': np.array(d[1]),
                    'highs': np.array(d[2]),
                    'lows': np.array(d[3])
                }

    def classify_regime(self, day_idx):
        for spy_sym in ['SPY', 'QQQ', 'AAPL']:
            if spy_sym in self.data:
                closes = self.data[spy_sym]['closes'][:day_idx+1]
                if len(closes) < 30:
                    continue
                rets = np.diff(closes) / closes[:-1]
                vol = np.std(rets[-20:]) * np.sqrt(252)
                if len(rets) > 21:
                    autocorr = np.corrcoef(rets[-21:-1], rets[-20:])[0, 1]
                    if np.isnan(autocorr): autocorr = 0.0
                else:
                    autocorr = 0.0
                if vol > 0.30:
                    return 'HIGH_VOL', 0.7
                elif autocorr > 0.15 and abs((closes[-1] - closes[-20]) / closes[-20]) > 0.02:
                    return 'TRENDING', 0.7
                elif vol < 0.12:
                    return 'LOW_VOL', 0.5
                else:
                    return 'SIDEWAYS', 0.5
        return 'UNKNOWN', 0.0

    def compute_atr(self, sym, day_idx):
        if sym not in self.data: return None
        highs = self.data[sym]['highs'][:day_idx+1]
        lows = self.data[sym]['lows'][:day_idx+1]
        closes = self.data[sym]['closes'][:day_idx+1]
        if len(closes) < 15: return None
        trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(closes))]
        return np.mean(trs[-14:])

    def volatility_target_size(self, sym, capital, held_symbols):
        """Returns (qty, atr)"""
        risk_amount = capital * self.config['risk_per_trade']
        atr = self.compute_atr(sym, len(self.data[sym]['closes'])-1)
        if atr is None or atr == 0:
            return 0, None
        
        raw_qty = int(risk_amount / (atr * 1.5))
        if raw_qty <= 0:
            return 0, None

        # Vol targeting
        closes = self.data[sym]['closes']
        if len(closes) >= 20:
            rets = np.diff(closes) / closes[:-1]
            current_vol = float(pd.Series(rets).ewm(halflife=self.config['vol_half_life']).std().iloc[-1])
            if not np.isnan(current_vol) and current_vol > 0:
                target_vol = self.config['target_vol'] / np.sqrt(252)
                vol_ratio = target_vol / max(current_vol, 0.005)
                vol_ratio = np.clip(vol_ratio, 0.5, 1.5)
                raw_qty = int(raw_qty * vol_ratio)

        # Correlation sizing
        if held_symbols and sym in self.data:
            target_closes = self.data[sym]['closes']
            if len(target_closes) >= 30:
                target_rets = np.diff(target_closes) / target_closes[:-1]
                high_corr_count = 0
                total_corr_count = 0
                for hsym in held_symbols:
                    if hsym in self.data:
                        held_closes = self.data[hsym]['closes']
                        if len(held_closes) >= 30:
                            held_rets = np.diff(held_closes) / held_closes[:-1]
                            min_len = min(len(target_rets), len(held_rets))
                            if min_len > 10:
                                corr = np.corrcoef(target_rets[:min_len], held_rets[:min_len])[0, 1]
                                if not np.isnan(corr):
                                    total_corr_count += 1
                                    if corr > self.config['corr_threshold']:
                                        high_corr_count += 1
                if high_corr_count > 0 and total_corr_count > 0:
                    cluster_ratio = high_corr_count / total_corr_count
                    if cluster_ratio > self.config['max_cluster_exposure']:
                        raw_qty = int(raw_qty * 0.5)
                    else:
                        raw_qty = int(raw_qty * (1 - cluster_ratio * 0.3))

        return raw_qty, atr

    def run_backtest(self, initial_capital=100_000):
        if not self.data:
            self.fetch_all()
        
        min_len = min(len(d['closes']) for d in self.data.values())
        portfolio_value = initial_capital
        daily_values = []
        trades = []
        positions = {}  # sym -> (shares, entry_day, entry_price, atr)

        for day in range(30, min_len - 1):
            regime, confidence = self.classify_regime(day)
            
            # Check exits
            to_close = []
            for sym, (shares, entry_day, entry_price, atr) in positions.items():
                if day >= len(self.data[sym]['closes']): continue
                current_price = self.data[sym]['closes'][day]
                profit_pct = (current_price - entry_price) / entry_price
                loss_pct = (entry_price - current_price) / entry_price
                
                if profit_pct >= self.config['profit_target']:
                    to_close.append((sym, 'TARGET'))
                elif loss_pct >= self.config['stop_loss']:
                    to_close.append((sym, 'STOP'))
                elif day - entry_day >= self.config['max_hold_days']:
                    to_close.append((sym, 'HOLD'))

            for sym, reason in to_close:
                if sym in positions:
                    shares, entry_day, entry_price, atr = positions[sym]
                    exit_price = self.data[sym]['closes'][day]
                    pnl = shares * (exit_price - entry_price)
                    cost = abs(shares) * (exit_price + entry_price) * self.transaction_cost
                    portfolio_value += pnl - cost
                    trades.append({
                        'sym': sym, 'pnl': pnl - cost, 'reason': reason,
                        'days_held': day - entry_day
                    })
                    del positions[sym]

            # Entries
            if regime in ('SIDEWAYS', 'HIGH_VOL') and len(positions) < self.config['max_positions']:
                held = list(positions.keys())
                signals = []
                for sym in self.data:
                    if sym in held: continue
                    closes = self.data[sym]['closes']
                    if day < 25 or day >= len(closes): continue
                    ma20 = np.mean(closes[day-20:day+1])
                    current_price = closes[day]
                    deviation = (current_price - ma20) / ma20
                    if abs(deviation) >= self.config['entry_threshold']:
                        signals.append((sym, abs(deviation), deviation))
                
                signals.sort(key=lambda x: x[1], reverse=True)
                
                for sym, _, dev in signals[:self.config['max_positions'] - len(positions)]:
                    if sym in held or len(positions) >= self.config['max_positions']:
                        break
                    price = self.data[sym]['closes'][day]
                    qty, atr = self.volatility_target_size(sym, portfolio_value, list(positions.keys()))
                    if qty > 0 and qty * price < portfolio_value * self.max_position_size:
                        cost = qty * price * self.transaction_cost
                        portfolio_value -= cost
                        positions[sym] = (qty, day, price, atr)

            daily_values.append(portfolio_value)

        # Close remaining
        for sym, (shares, entry_day, entry_price, atr) in positions.items():
            exit_price = self.data[sym]['closes'][min_len - 1]
            pnl = shares * (exit_price - entry_price)
            cost = abs(shares) * (exit_price + entry_price) * self.transaction_cost
            portfolio_value += pnl - cost
            trades.append({'sym': sym, 'pnl': pnl - cost, 'reason': 'FINAL', 'days_held': min_len - entry_day})

        daily_values.append(portfolio_value)
        daily_values = np.array(daily_values)
        daily_rets = np.diff(daily_values) / daily_values[:-1]
        
        total_return = (portfolio_value - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(daily_values)) - 1
        sharpe = np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)
        max_dd = np.min(daily_values / np.maximum.accumulate(daily_values) - 1)
        
        return {
            'total_return': total_return,
            'ann_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(trades),
            'final_value': portfolio_value,
            'daily_values': daily_values,
        }


def main():
    print("=" * 70)
    print("V4 COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    # === Test 1: Full Backtest ===
    print("\n[1/7] Full Backtest (In-Sample)...")
    tester = V4Tester()
    tester.fetch_all(500)
    r = tester.run_backtest()
    if r:
        print(f"  Return:    {r['total_return']:.2%}")
        print(f"  Sharpe:    {r['sharpe']:.3f}")
        print(f"  Max DD:    {r['max_drawdown']:.2%}")
        print(f"  Trades:    {r['num_trades']}")
        print(f"  Final:     ${r['final_value']:,.2f}")

    # === Test 2: Walk-Forward OOS (5 folds) ===
    print("\n[2/7] Walk-Forward OOS (5 folds)...")
    tester = V4Tester()
    tester.fetch_all(500)
    min_len = min(len(d['closes']) for d in tester.data.values())
    fold_size = (min_len - 30) // 5
    
    oos_results = []
    for fold in range(5):
        start_day = 30 + fold * fold_size
        end_day = start_day + fold_size
        
        # Temporarily truncate data
        for sym in tester.data:
            tester.data[sym]['closes'] = tester.data[sym]['closes'][:end_day+1]
            tester.data[sym]['volumes'] = tester.data[sym]['volumes'][:end_day+1]
            tester.data[sym]['highs'] = tester.data[sym]['highs'][:end_day+1]
            tester.data[sym]['lows'] = tester.data[sym]['lows'][:end_day+1]
        
        r = tester.run_backtest()
        if r:
            oos_results.append(r)
            print(f"  Fold {fold+1}: Return={r['total_return']:.2%}, Sharpe={r['sharpe']:.3f}, DD={r['max_drawdown']:.2%}")
        
        # Reset for next fold
        tester.data = {}
        tester.fetch_all(500)

    if oos_results:
        avg_ret = np.mean([r['total_return'] for r in oos_results])
        avg_sharpe = np.mean([r['sharpe'] for r in oos_results])
        pos_folds = sum(1 for r in oos_results if r['total_return'] > 0)
        print(f"  Average: Return={avg_ret:.2%}, Sharpe={avg_sharpe:.3f}")
        print(f"  Positive: {pos_folds}/{len(oos_results)}")

    # === Test 3: Monte Carlo ===
    print("\n[3/7] Monte Carlo (500 random 50-day periods)...")
    if r and len(r['daily_values']) > 50:
        dv = r['daily_values']
        periods = []
        for _ in range(500):
            start = np.random.randint(0, len(dv) - 50)
            period_ret = (dv[start+50] - dv[start]) / dv[start]
            periods.append(period_ret)
        
        pos_count = sum(1 for p in periods if p > 0)
        print(f"  Positive periods: {pos_count}/500 ({pos_count/500:.1%})")
        print(f"  Avg return: {np.mean(periods):.4%}")
        print(f"  Median: {np.median(periods):.4%}")
        print(f"  Worst: {np.min(periods):.2%}")
        print(f"  Best: {np.max(periods):.2%}")

    # === Test 4: Parameter Sensitivity ===
    print("\n[4/7] Parameter Sensitivity...")
    configs = [
        ('Conservative', 0.12, 0.6, 0.25),
        ('Default', 0.18, 0.75, 0.30),
        ('Aggressive', 0.25, 0.85, 0.40),
    ]
    for name, vol, corr, cluster in configs:
        tester = V4Tester()
        tester.config['target_vol'] = vol
        tester.config['corr_threshold'] = corr
        tester.config['max_cluster_exposure'] = cluster
        tester.fetch_all(500)
        r = tester.run_backtest()
        if r:
            print(f"  {name:15s}: Return={r['total_return']:+.2%}, Sharpe={r['sharpe']:.3f}, DD={r['max_drawdown']:.2%}")

    # === Test 5: Stability (Rolling Sharpe) ===
    print("\n[5/7] Stability (Rolling Sharpe)...")
    tester = V4Tester()
    tester.fetch_all(500)
    r = tester.run_backtest()
    if r and len(r['daily_values']) > 100:
        dv = r['daily_values']
        window = len(dv) // 4
        sharpes = []
        for i in range(0, len(dv) - window, max(1, window // 2)):
            period = dv[i:i+window]
            rets = np.diff(period) / period[:-1]
            if np.std(rets) > 0:
                s = np.mean(rets) / np.std(rets) * np.sqrt(252)
                sharpes.append(s)
        if sharpes:
            print(f"  Windows: {len(sharpes)}")
            print(f"  Avg: {np.mean(sharpes):.3f}, Min: {np.min(sharpes):.3f}, Max: {np.max(sharpes):.3f}")
            print(f"  All positive: {sum(1 for s in sharpes if s > 0)}/{len(sharpes)}")

    # === Test 6: Transaction Cost ===
    print("\n[6/7] Transaction Cost Robustness...")
    if r:
        num_trades = r['num_trades']
        total_cost_rt = num_trades * 0.001 * 2  # round-trip
        adj_return = r['total_return'] - total_cost_rt
        print(f"  Trades: {num_trades}")
        print(f"  Cost (10bps RT): {total_cost_rt:.2%}")
        print(f"  Original: {r['total_return']:.2%}")
        print(f"  After costs: {adj_return:.2%}")
        if num_trades > 0:
            print(f"  Break-even: {r['total_return'] / num_trades / 2:.3%} per RT")

    # === Test 7: V3 vs V4 Comparison ===
    print("\n[7/7] V3 vs V4 Comparison...")
    from fixed_production_v3 import FixedProductionSystemV3
    
    v3 = FixedProductionSystemV3(universe=UNIVERSE[:25])
    v3.fetch_data(500)
    v3_results = v3.backtest()
    
    v4 = V4Tester()
    v4.fetch_all(500)
    v4_results = v4.run_backtest()
    
    print(f"  {'Metric':<20} {'V3':>12} {'V4':>12} {'Delta':>12}")
    print(f"  {'-'*56}")
    print(f"  {'Return':<20} {v3_results['total_return']:>11.2%} {v4_results['total_return']:>11.2%} {v4_results['total_return']-v3_results['total_return']:>+11.2%}")
    print(f"  {'Sharpe':<20} {v3_results['sharpe']:>12.3f} {v4_results['sharpe']:>12.3f} {v4_results['sharpe']-v3_results['sharpe']:>+12.3f}")
    print(f"  {'Max DD':<20} {v3_results['max_drawdown']:>11.2%} {v4_results['max_drawdown']:>11.2%} {v4_results['max_drawdown']-v3_results['max_drawdown']:>+11.2%}")
    print(f"  {'Trades':<20} {len(v3_results['trades']):>12} {v4_results['num_trades']:>12}")

    print("\n" + "=" * 70)
    print("V4 TEST SUITE COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
