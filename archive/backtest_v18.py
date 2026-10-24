#!/usr/bin/env python3
"""
Backtest v18 — Bull Market Cruise Control
Market regime-aware position sizing and trade filtering.
"""

import requests
import json
import time
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

# Import production functions
from production_v18 import (
    fetch_data, extract_features, select_features, get_market_regime,
    dynamic_triple_barriers, kelly_criterion, triple_barrier_labels,
    calc_atr, calc_rsi, calc_bollinger, calc_vwap, calc_iv_rank,
    calc_beta, calc_residual_returns, calc_relative_strength, calc_rvol,
    UNIVERSE, SECTOR_ETFS, MODEL_FILE
)

# ============================================
# PORTFOLIO TRACKER
# ============================================
class Portfolio:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.closed_trades = []
        self.equity_curve = []
    
    def total_value(self, current_prices):
        position_value = 0
        for sym, pos in self.positions.items():
            if sym in current_prices and isinstance(current_prices[sym], dict):
                price = current_prices[sym].get('close', pos['entry'])
            else:
                price = pos['entry']
            position_value += pos['shares'] * price
        return self.cash + position_value
    
    def open_position(self, symbol, entry_price, stop_loss, take_profit, shares, max_hold, date):
        cost = shares * entry_price
        if cost > self.cash:
            shares = int(self.cash / entry_price)
            cost = shares * entry_price
        if shares <= 0: return False
        self.cash -= cost
        self.positions[symbol] = {
            'entry': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
            'shares': shares, 'entry_date': date, 'max_hold': max_hold, 'current_hold': 0
        }
        return True
    
    def close_position(self, symbol, exit_price, exit_reason, date):
        if symbol not in self.positions: return
        pos = self.positions[symbol]
        proceeds = pos['shares'] * exit_price
        pnl = proceeds - (pos['shares'] * pos['entry'])
        pnl_pct = (exit_price - pos['entry']) / pos['entry']
        self.cash += proceeds
        self.closed_trades.append({
            'symbol': symbol, 'entry_date': pos['entry_date'], 'exit_date': date,
            'entry_price': pos['entry'], 'exit_price': exit_price,
            'shares': pos['shares'], 'pnl': pnl, 'pnl_pct': pnl_pct,
            'exit_reason': exit_reason, 'hold_days': pos['current_hold']
        })
        del self.positions[symbol]
    
    def update_positions(self, date, current_prices):
        to_close = []
        for symbol, pos in self.positions.items():
            pos['current_hold'] += 1
            if symbol not in current_prices: continue
            high = current_prices[symbol]['high']
            low = current_prices[symbol]['low']
            close = current_prices[symbol]['close']
            if high >= pos['take_profit']: to_close.append((symbol, pos['take_profit'], 'take_profit'))
            elif low <= pos['stop_loss']: to_close.append((symbol, pos['stop_loss'], 'stop_loss'))
            elif pos['current_hold'] >= pos['max_hold']: to_close.append((symbol, close, 'time_expiry'))
        for symbol, exit_price, reason in to_close:
            self.close_position(symbol, exit_price, reason, date)

# ============================================
# BACKTEST ENGINE
# ============================================
class BacktestEngine:
    def __init__(self, start_date='2024-01-01', end_date='2026-08-31', initial_capital=10000):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.portfolio = Portfolio(initial_capital)
        self.spy_benchmark = initial_capital
        self.trade_log = []
        self.skipped_trades = 0
        
        if not MODEL_FILE.exists(): raise FileNotFoundError("XGBoost model not found.")
        with open(MODEL_FILE, 'rb') as f: saved = pickle.load(f)
        self.model = saved['model']
        self.model_features = saved['feature_names']
        self.selected_features, _ = select_features(self.model, self.model_features, threshold=0.02)
    
    def run(self):
        print("=" * 70)
        print("  BACKTEST v18 — Bull Market Cruise Control")
        print("=" * 70)
        print(f"  Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}")
        print(f"  Initial Capital: ${self.portfolio.initial_capital:,.2f}")
        print(f"  Universe: {len(UNIVERSE)} stocks")
        print("=" * 70)
        
        print("\nFetching historical data...")
        all_data = {}
        for symbol in UNIVERSE + ['SPY']:
            data = fetch_data(symbol, days=730)
            if data[0]: all_data[symbol] = {'closes': data[0], 'volumes': data[1], 'highs': data[2], 'lows': data[3]}
            time.sleep(0.3)
        
        if 'SPY' not in all_data: print("ERROR: Failed to fetch SPY"); return
        
        spy_closes = all_data['SPY']['closes']
        spy_highs = all_data['SPY']['highs']
        spy_lows = all_data['SPY']['lows']
        
        total_days = len(spy_closes)
        print(f"\nSimulating {total_days} trading days...")
        
        for day_idx in range(200, total_days - 14):
            current_prices = {}
            for symbol in UNIVERSE:
                if symbol in all_data and day_idx < len(all_data[symbol]['closes']):
                    current_prices[symbol] = {
                        'close': all_data[symbol]['closes'][day_idx],
                        'high': all_data[symbol]['highs'][day_idx],
                        'low': all_data[symbol]['lows'][day_idx]
                    }
            
            self.portfolio.update_positions(day_idx, current_prices)
            
            if day_idx % 5 != 0: continue
            
            # Market Regime
            regime, regime_penalty, big_cap_max, small_cap_max, kelly_mult = get_market_regime(
                spy_closes[:day_idx+1], spy_highs[:day_idx+1], spy_lows[:day_idx+1]
            )
            
            scores = []
            for symbol in UNIVERSE:
                if symbol not in all_data or day_idx >= len(all_data[symbol]['closes']): continue
                
                closes = all_data[symbol]['closes'][:day_idx+1]
                volumes = all_data[symbol]['volumes'][:day_idx+1]
                highs = all_data[symbol]['highs'][:day_idx+1]
                lows = all_data[symbol]['lows'][:day_idx+1]
                
                sector_etf = SECTOR_ETFS.get(symbol, 'SPY')
                sector_closes = all_data.get(sector_etf, {}).get('closes', spy_closes)[:day_idx+1]
                
                features = extract_features(closes, volumes, highs, lows, symbol, spy_closes[:day_idx+1], sector_closes)
                if features is None: continue
                
                X = [[features[name] for name in self.selected_features]]
                raw_prob = self.model.predict_proba(X)[0][1]
                win_prob = raw_prob * regime_penalty
                
                upper_mult, lower_mult, max_hold = dynamic_triple_barriers(features['atr'], closes, features['iv_rank'])
                
                payout_ratio = upper_mult / lower_mult
                base_kelly = kelly_criterion(win_prob, payout_ratio, max_risk=0.1)
                kelly_size = min(base_kelly * kelly_multiplier, 0.25)
                
                scores.append({
                    'symbol': symbol, 'features': features, 'win_prob': win_prob,
                    'kelly_size': kelly_size, 'upper_mult': upper_mult, 'lower_mult': lower_mult, 'max_hold': max_hold
                })
            
            scores.sort(key=lambda x: x['features']['residual_return'], reverse=True)
            
            # Take top trades up to max_positions
            for score in scores[:max_positions]:
                if score['kelly_size'] > 0:
                    symbol = score['symbol']
                    if symbol not in self.portfolio.positions:
                        entry_price = score['features']['current_price']
                        atr = score['features']['atr']
                        stop_loss = entry_price - (score['lower_mult'] * atr)
                        take_profit = entry_price + (score['upper_mult'] * atr)
                        
                        portfolio_value = self.portfolio.total_value(current_prices)
                        risk_amount = portfolio_value * score['kelly_size']
                        shares = int(risk_amount / entry_price)
                        
                        if shares > 0:
                            opened = self.portfolio.open_position(
                                symbol, entry_price, stop_loss, take_profit, shares, score['max_hold'], day_idx
                            )
                            if opened:
                                self.trade_log.append({
                                    'date': day_idx, 'symbol': symbol, 'entry': entry_price,
                                    'stop_loss': stop_loss, 'take_profit': take_profit,
                                    'shares': shares, 'kelly': score['kelly_size'], 'win_prob': score['win_prob']
                                })
            
            portfolio_value = self.portfolio.total_value(current_prices)
            self.portfolio.equity_curve.append(portfolio_value)
        
        # Close remaining positions
        final_prices = {}
        for symbol in UNIVERSE:
            if symbol in all_data and total_days - 1 < len(all_data[symbol]['closes']):
                final_prices[symbol] = {
                    'close': all_data[symbol]['closes'][-1],
                    'high': all_data[symbol]['highs'][-1],
                    'low': all_data[symbol]['lows'][-1]
                }
        
        for symbol in list(self.portfolio.positions.keys()):
            if symbol in final_prices:
                self.portfolio.close_position(symbol, final_prices[symbol]['close'], 'end_of_backtest', total_days - 1)
        
        spy_start = spy_closes[200]
        spy_end = spy_closes[-1]
        self.spy_benchmark = self.portfolio.initial_capital * (spy_end / spy_start)
        
        self.generate_report()
    
    def generate_report(self):
        print("\n" + "=" * 70)
        print("  BACKTEST RESULTS — Bull Market Cruise Control")
        print("=" * 70)
        
        final_equity = self.portfolio.equity_curve[-1] if self.portfolio.equity_curve else self.portfolio.initial_capital
        total_return = (final_equity - self.portfolio.initial_capital) / self.portfolio.initial_capital * 100
        spy_return = (self.spy_benchmark - self.portfolio.initial_capital) / self.portfolio.initial_capital * 100
        
        print(f"\n  PORTFOLIO PERFORMANCE")
        print(f"  {'=' * 40}")
        print(f"  Initial Capital:     ${self.portfolio.initial_capital:,.2f}")
        print(f"  Final Equity:        ${final_equity:,.2f}")
        print(f"  Total Return:        {total_return:+.2f}%")
        print(f"  vs SPY Benchmark:    {spy_return:+.2f}%")
        print(f"  Alpha Generated:     {total_return - spy_return:+.2f}%")
        
        total_trades = len(self.portfolio.closed_trades)
        winning_trades = [t for t in self.portfolio.closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in self.portfolio.closed_trades if t['pnl'] <= 0]
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        print(f"\n  TRADE STATISTICS")
        print(f"  {'=' * 40}")
        print(f"  Total Trades:        {total_trades}")
        print(f"  Winning Trades:      {len(winning_trades)}")
        print(f"  Losing Trades:       {len(losing_trades)}")
        print(f"  Win Rate:            {win_rate:.1f}%")
        
        exit_reasons = defaultdict(int)
        for trade in self.portfolio.closed_trades: exit_reasons[trade['exit_reason']] += 1
        
        print(f"\n  EXIT REASON BREAKDOWN")
        print(f"  {'=' * 40}")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_trades * 100 if total_trades > 0 else 0
            print(f"  {reason:20s}: {count:3d} ({pct:.1f}%)")
        
        if winning_trades:
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades])
            max_win = max([t['pnl_pct'] for t in winning_trades])
        else: avg_win = max_win = 0
        
        if losing_trades:
            avg_loss = np.mean([t['pnl_pct'] for t in losing_trades])
            max_loss = min([t['pnl_pct'] for t in losing_trades])
        else: avg_loss = max_loss = 0
        
        print(f"\n  P&L STATISTICS")
        print(f"  {'=' * 40}")
        print(f"  Avg Win:             {avg_win:+.2%}")
        print(f"  Avg Loss:            {avg_loss:+.2%}")
        print(f"  Max Win:             {max_win:+.2%}")
        print(f"  Max Loss:            {max_loss:+.2%}")
        print(f"  Avg Payout Ratio:    {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Avg Payout Ratio:    N/A")
        
        if self.portfolio.equity_curve:
            peak = self.portfolio.equity_curve[0]
            max_dd = 0
            dd_start = dd_end = 0
            current_dd_start = 0
            
            for i, equity in enumerate(self.portfolio.equity_curve):
                if equity > peak: peak = equity; current_dd_start = i
                dd = (peak - equity) / peak
                if dd > max_dd: max_dd = dd; dd_start = current_dd_start; dd_end = i
            
            print(f"\n  DRAWDOWN ANALYSIS")
            print(f"  {'=' * 40}")
            print(f"  Max Drawdown:        {max_dd:.2%}")
            print(f"  Drawdown Duration:   {dd_end - dd_start} days")
        
        if len(self.portfolio.equity_curve) > 1:
            returns = np.diff(self.portfolio.equity_curve) / self.portfolio.equity_curve[:-1]
            returns = returns[~np.isnan(returns)]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                downside_returns = returns[returns < 0]
                sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 and np.std(downside_returns) > 0 else 0
                print(f"\n  RISK-ADJUSTED RETURNS")
                print(f"  {'=' * 40}")
                print(f"  Sharpe Ratio:        {sharpe:.2f}")
                print(f"  Sortino Ratio:       {sortino:.2f}")
        
        print(f"\n  TRADE LOG (Last 10)")
        print(f"  {'=' * 70}")
        print(f"  {'Date':>6s} {'Symbol':>6s} {'Entry':>10s} {'Exit':>10s} {'PnL':>10s} {'Reason':>15s}")
        print(f"  {'-' * 70}")
        for trade in self.portfolio.closed_trades[-10:]:
            print(f"  {trade['entry_date']:6d} {trade['symbol']:>6s} ${trade['entry_price']:>9.2f} ${trade['exit_price']:>9.2f} ${trade['pnl']:>9.2f} {trade['exit_reason']:>15s}")
        
        print(f"\n{'=' * 70}")
        print(f"  BACKTEST COMPLETE")
        print(f"{'=' * 70}")

if __name__ == "__main__":
    engine = BacktestEngine(start_date='2024-01-01', end_date='2026-08-31', initial_capital=10000)
    engine.run()
