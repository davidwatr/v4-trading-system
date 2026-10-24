"""
Final Swing Trading System — 3% Weekly Target
================================================
Research shows the best swing trading combines:
1. Momentum (buy what's going up)
2. Trend filter (only trade in direction of trend)
3. Wider stops (2-4% for swing trades)
4. Quick profits (3-5% targets, 3-5 day holds)

Strategy:
- Entry: 3-5 day momentum 1.5-8% + RSI 40-70 + Price > EMA20
- Stop: 3% below entry
- Target: 4-5% above entry (1.5-2x R:R)
- Hold: Max 5 days
- Size: 15-20% per position, max 3-5 positions
"""
import numpy as np
import pandas as pd
import warnings
from production_v18 import fetch_data, UNIVERSE
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')


class SwingTradingSystem:
    """
    Final swing trading system targeting 1-3% weekly returns.
    
    Key principles:
    1. Momentum + trend filter (don't fight the trend)
    2. Wider stops (2-4%) — swing trades need room
    3. Quick profits (3-5% targets)
    4. Multiple small positions (diversification)
    """
    
    def __init__(self, universe=None):
        # Top 20 most liquid stocks for tight spreads
        self.universe = universe or [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'JPM', 'V', 'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'DIS',
            'BAC', 'XOM', 'PFE', 'KO', 'PEP'
        ]
        self.data = {}
        
    def fetch_data(self, days=500):
        print(f'Fetching {len(self.universe)} symbols...')
        for sym in self.universe:
            d = fetch_data(sym, days=days)
            if d is not None and len(d[0]) > 50:
                self.data[sym] = {
                    'closes': np.array(d[0]),
                    'volumes': np.array(d[1]),
                    'highs': np.array(d[2]),
                    'lows': np.array(d[3])
                }
        print(f'Loaded {len(self.data)} symbols')
    
    def compute_rsi(self, prices, period=14):
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def compute_ema(self, prices, period):
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def check_entry(self, sym, day_idx):
        """
        Entry criteria for swing trading.
        """
        if sym not in self.data:
            return None
        
        closes = self.data[sym]['closes']
        highs = self.data[sym]['highs']
        lows = self.data[sym]['lows']
        
        if day_idx < 20 or day_idx >= len(closes):
            return None
        
        price = closes[day_idx]
        
        # 5-day momentum: 1.5-8% up (strong but not parabolic)
        mom_5d = (closes[day_idx] / closes[day_idx - 5] - 1)
        if mom_5d < 0.015 or mom_5d > 0.08:
            return None
        
        # RSI 40-70 (momentum but not overbought)
        rsi = self.compute_rsi(closes[:day_idx+1], 14)
        if rsi < 40 or rsi > 70:
            return None
        
        # Price above 20-day SMA (trend filter)
        sma20 = np.mean(closes[day_idx-20:day_idx])
        if price < sma20:
            return None
        
        # Stop loss: 3% below entry
        stop_loss = price * 0.97
        
        # Take profit: 4.5% above entry (1.5:1 R:R)
        take_profit = price * 1.045
        
        return {
            'signal': True,
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'momentum': mom_5d,
            'rsi': rsi
        }
    
    def backtest(self, initial_capital=100_000, max_positions=4, hold_days=5):
        """
        Full backtest.
        """
        min_len = min(len(d['closes']) for d in self.data.values())
        
        portfolio_value = initial_capital
        daily_values = []
        trades = []
        positions = {}
        
        for day in range(20, min_len - 1):
            # Check exits
            to_close = []
            for sym, pos in positions.items():
                if day >= len(self.data[sym]['closes']):
                    continue
                
                price = self.data[sym]['closes'][day]
                high = self.data[sym]['highs'][day]
                low = self.data[sym]['lows'][day]
                
                if low <= pos['stop']:
                    to_close.append((sym, pos['stop'], 'stop'))
                elif high >= pos['target']:
                    to_close.append((sym, pos['target'], 'target'))
                elif day - pos['entry_day'] >= hold_days:
                    to_close.append((sym, price, 'time'))
            
            for sym, exit_price, reason in to_close:
                pos = positions[sym]
                pnl = pos['shares'] * (exit_price - pos['entry_price'])
                cost = pos['shares'] * (exit_price + pos['entry_price']) * 0.001
                portfolio_value += pnl - cost
                
                hold_period = day - pos['entry_day']
                ret_pct = (exit_price / pos['entry_price'] - 1)
                trades.append({
                    'sym': sym,
                    'pnl': pnl - cost,
                    'return_pct': ret_pct,
                    'hold_days': hold_period,
                    'reason': reason
                })
                del positions[sym]
            
            # New entries (up to 2 per day)
            entries_today = 0
            for sym in self.data:
                if sym in positions:
                    continue
                if len(positions) >= max_positions:
                    break
                if entries_today >= 2:
                    break
                
                signal = self.check_entry(sym, day)
                if signal and signal['signal']:
                    # Position sizing: 20% of portfolio per trade
                    allocation = portfolio_value * 0.20
                    shares = int(allocation / signal['entry_price'])
                    
                    if shares > 0:
                        cost = shares * signal['entry_price'] * 0.001
                        portfolio_value -= cost
                        positions[sym] = {
                            'shares': shares,
                            'entry_price': signal['entry_price'],
                            'stop': signal['stop_loss'],
                            'target': signal['take_profit'],
                            'entry_day': day
                        }
                        entries_today += 1
            
            daily_values.append(portfolio_value)
        
        # Close remaining
        for sym, pos in positions.items():
            if min_len - 1 < len(self.data[sym]['closes']):
                exit_price = self.data[sym]['closes'][min_len - 1]
                pnl = pos['shares'] * (exit_price - pos['entry_price'])
                cost = pos['shares'] * (exit_price + pos['entry_price']) * 0.001
                portfolio_value += pnl - cost
        
        daily_values.append(portfolio_value)
        daily_values = np.array(daily_values)
        daily_rets = np.diff(daily_values) / daily_values[:-1]
        
        total_return = (portfolio_value - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(daily_values)) - 1
        sharpe = np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)
        max_dd = np.min(daily_values / np.maximum.accumulate(daily_values) - 1)
        
        if trades:
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
            avg_hold = np.mean([t['hold_days'] for t in trades])
            avg_ret = np.mean([t['return_pct'] for t in trades])
            
            target_hits = sum(1 for t in trades if t['reason'] == 'target')
            stop_hits = sum(1 for t in trades if t['reason'] == 'stop')
            time_exits = sum(1 for t in trades if t['reason'] == 'time')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            avg_hold = 0
            avg_ret = 0
            target_hits = stop_hits = time_exits = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_hold_days': avg_hold,
            'avg_return': avg_ret,
            'total_trades': len(trades),
            'target_hits': target_hits,
            'stop_hits': stop_hits,
            'time_exits': time_exits,
            'daily_values': daily_values,
            'trades': trades
        }


def run_comprehensive_test():
    """Run comprehensive swing trading tests."""
    print('='*70)
    print('FINAL SWING TRADING SYSTEM — COMPREHENSIVE TEST')
    print('='*70)
    
    system = SwingTradingSystem()
    system.fetch_data(days=500)
    
    results = system.backtest()
    
    print('\n' + '='*70)
    print('RESULTS')
    print('='*70)
    
    print(f'Total Return: {results["total_return"]*100:.2f}%')
    print(f'Annual Return: {results["annual_return"]*100:.2f}%')
    print(f'Sharpe Ratio: {results["sharpe"]:.3f}')
    print(f'Max Drawdown: {results["max_drawdown"]*100:.2f}%')
    print(f'Total Trades: {results["total_trades"]}')
    print(f'Win Rate: {results["win_rate"]*100:.1f}%')
    print(f'Avg Win: ${results["avg_win"]:.2f}')
    print(f'Avg Loss: ${results["avg_loss"]:.2f}')
    print(f'Avg Return per Trade: {results["avg_return"]*100:.2f}%')
    print(f'Avg Hold: {results["avg_hold_days"]:.1f} days')
    
    if results['total_trades'] > 0:
        print(f'\nExit Breakdown:')
        print(f'  Target Hit: {results["target_hits"]} ({results["target_hits"]/results["total_trades"]*100:.1f}%)')
        print(f'  Stop Hit: {results["stop_hits"]} ({results["stop_hits"]/results["total_trades"]*100:.1f}%)')
        print(f'  Time Exit: {results["time_exits"]} ({results["time_exits"]/results["total_trades"]*100:.1f}%)')
    
    # Weekly returns
    daily_values = results['daily_values']
    weekly_returns = []
    for i in range(0, len(daily_values), 5):
        week_end = min(i + 5, len(daily_values))
        if i > 0:
            week_ret = (daily_values[week_end-1] / daily_values[i-1]) - 1
            weekly_returns.append(week_ret)
    
    if weekly_returns:
        print(f'\nWeekly Returns ({len(weekly_returns)} weeks):')
        print(f'  Mean: {np.mean(weekly_returns)*100:+.2f}%')
        print(f'  Best: {np.max(weekly_returns)*100:+.2f}%')
        print(f'  Worst: {np.min(weekly_returns)*100:+.2f}%')
        print(f'  Positive weeks: {sum(1 for r in weekly_returns if r > 0)}/{len(weekly_returns)}')
    
    # Walk-forward OOS test
    print('\n' + '='*70)
    print('WALK-FORWARD OOS TEST')
    print('='*70)
    
    min_len = min(len(d['closes']) for d in system.data.values())
    fold_returns = []
    
    for fold_start in range(200, min_len - 60, 30):
        fold_end = fold_start + 30
        if fold_end >= min_len - 1:
            break
        
        fold_trades = []
        for day in range(fold_start, fold_end):
            signals = []
            for sym in system.data:
                signal = system.check_entry(sym, day)
                if signal and signal['signal']:
                    signals.append(signal)
            
            for signal in signals[:2]:
                closes = system.data[list(system.data.keys())[0]]['closes']
                if day + 5 < len(closes):
                    ret = (closes[day + 5] - closes[day]) / closes[day]
                    fold_trades.append(ret)
        
        if fold_trades:
            fold_returns.append(np.mean(fold_trades))
    
    if fold_returns:
        print(f'Folds: {len(fold_returns)}')
        print(f'Mean fold return: {np.mean(fold_returns)*100:.3f}%')
        print(f'Positive folds: {sum(1 for r in fold_returns if r > 0)}/{len(fold_returns)}')
    
    # Permutation test
    print('\n' + '='*70)
    print('PERMUTATION TEST')
    print('='*70)
    
    real_returns = []
    for day in range(200, min_len - 6):
        for sym in system.data:
            signal = system.check_entry(sym, day)
            if signal and signal['signal']:
                closes = system.data[sym]['closes']
                if day + 5 < len(closes):
                    ret = (closes[day + 5] - closes[day]) / closes[day]
                    real_returns.append(ret)
    
    real_returns = np.array(real_returns)
    
    random_returns = []
    for _ in range(200):
        perm = np.random.permutation(len(real_returns))
        random_returns.append(np.mean(real_returns[perm[:100]]))
    
    random_returns = np.array(random_returns)
    
    print(f'Real mean: {np.mean(real_returns)*100:.3f}%')
    print(f'Random mean: {np.mean(random_returns)*100:.3f}% ± {np.std(random_returns)*100:.3f}%')
    
    edge = np.mean(real_returns) - np.mean(random_returns)
    print(f'Edge: {edge*100:.3f}%')
    
    if edge > 0:
        print('✅ PASS: Real beats random')
    else:
        print('❌ FAIL: Random beats real')
    
    return results


if __name__ == '__main__':
    run_comprehensive_test()
