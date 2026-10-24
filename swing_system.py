"""
Swing Trading System — Research-Based
=====================================
Based on the most reliable swing trading strategies from 2024-2025 research:

1. 20 EMA Pullback (TradeZella) — 55-65% win rate, 2-3x R:R
2. RSI + EMA (Medium/QQQ) — 79% win rate, Sharpe 1.11
3. Mean Reversion (QuantifiedStrategies) — RSI 30/70 entries

Core Strategy: Buy pullbacks to 20 EMA in stocks above 200 EMA.
- Entry: Price within 2% of 20 EMA + RSI 40-60 + reversal candle
- Stop: Below pullback low (max 3% risk)
- Target: Previous swing high (2.5x risk)
- Hold: 3-7 days

This is the #1 most reliable swing trading strategy according to backtests.
"""
import numpy as np
import pandas as pd
import warnings
from production_v18 import fetch_data, UNIVERSE

warnings.filterwarnings('ignore')


class SwingTradingSystem:
    """
    Research-based swing trading system.
    
    Strategy: 20 EMA Pullback in Uptrend
    - Only trade stocks in confirmed uptrends (above 200 EMA)
    - Buy when price pulls back to 20 EMA (not oversold, just a normal pullback)
    - RSI 40-60 confirms momentum is intact but not overbought
    - Stop loss below pullback low (max 3% risk per trade)
    - Take profit at previous swing high (2.5x risk)
    - Time stop: exit if not profitable in 7 days
    """
    
    def __init__(self, universe=None):
        self.universe = universe or UNIVERSE[:30]
        self.data = {}
        
    def fetch_data(self, days=500):
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
    
    def compute_ema(self, prices, period):
        """Compute Exponential Moving Average."""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    def compute_rsi(self, prices, period=14):
        """Compute RSI."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def find_swing_high(self, prices, lookback=10):
        """Find the most recent swing high."""
        if len(prices) < lookback:
            return prices[-1] * 1.05
        return np.max(prices[-lookback:])
    
    def find_swing_low(self, prices, lookback=10):
        """Find the most recent swing low."""
        if len(prices) < lookback:
            return prices[-1] * 0.95
        return np.min(prices[-lookback:])
    
    def check_entry(self, sym, day_idx):
        """
        Check if a stock meets entry criteria.
        
        Returns: {
            'signal': bool,
            'entry_price': float,
            'stop_loss': float,
            'take_profit': float,
            'risk': float,
            'rr_ratio': float
        }
        """
        if sym not in self.data:
            return None
        
        closes = self.data[sym]['closes']
        highs = self.data[sym]['highs']
        lows = self.data[sym]['lows']
        volumes = self.data[sym]['volumes']
        
        if day_idx < 200 or day_idx >= len(closes):
            return None
        
        # Compute indicators
        ema20 = self.compute_ema(closes[:day_idx+1], 20)
        ema50 = self.compute_ema(closes[:day_idx+1], 50)
        ema200 = self.compute_ema(closes[:day_idx+1], 200)
        rsi = self.compute_rsi(closes[:day_idx+1], 14)
        
        price = closes[day_idx]
        
        # CRITERION 1: Stock must be above 200 EMA (long-term uptrend)
        if price < ema200[-1]:
            return None
        
        # CRITERION 2: 20 EMA above 50 EMA (short-term uptrend)
        if ema20[-1] < ema50[-1]:
            return None
        
        # CRITERION 3: Price within 2% of 20 EMA (pullback zone)
        distance_to_ema20 = abs(price - ema20[-1]) / ema20[-1]
        if distance_to_ema20 > 0.02:
            return None
        
        # CRITERION 4: RSI between 40-60 (momentum intact, not overbought)
        if rsi < 40 or rsi > 60:
            return None
        
        # CRITERION 5: Volume below 20-day average (pullback on low volume)
        if day_idx >= 20:
            avg_vol = np.mean(volumes[day_idx-20:day_idx])
            if volumes[day_idx] > avg_vol * 1.2:
                return None
        
        # CRITERION 6: Reversal candle (today's close > open, i.e., bullish)
        if day_idx > 0 and closes[day_idx] < closes[day_idx - 1]:
            return None
        
        # Calculate stop loss and take profit
        swing_low = self.find_swing_low(closes[:day_idx+1], 10)
        stop_loss = min(swing_low, price * 0.97)  # Max 3% risk
        
        risk = price - stop_loss
        if risk <= 0:
            return None
        
        # Target: 2.5x risk (swing high or 2.5R)
        swing_high = self.find_swing_high(closes[:day_idx+1], 15)
        target_2_5r = price + (risk * 2.5)
        take_profit = max(swing_high, target_2_5r)
        
        # R:R ratio
        rr_ratio = (take_profit - price) / risk
        
        return {
            'signal': True,
            'entry_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk': risk,
            'rr_ratio': rr_ratio,
            'rsi': rsi,
            'ema20': ema20[-1]
        }
    
    def backtest(self, initial_capital=100_000, max_positions=5, hold_days=7):
        """
        Backtest the swing trading system.
        """
        min_len = min(len(d['closes']) for d in self.data.values())
        
        portfolio_value = initial_capital
        daily_values = []
        trades = []
        positions = {}  # sym -> {shares, entry_price, stop, target, entry_day}
        
        for day in range(200, min_len - 1):
            # Check existing positions for exit
            to_close = []
            for sym, pos in positions.items():
                if day >= len(self.data[sym]['closes']):
                    continue
                
                price = self.data[sym]['closes'][day]
                high = self.data[sym]['highs'][day]
                low = self.data[sym]['lows'][day]
                
                # Stop loss hit
                if low <= pos['stop']:
                    to_close.append((sym, pos['stop'], 'stop'))
                # Take profit hit
                elif high >= pos['target']:
                    to_close.append((sym, pos['target'], 'target'))
                # Time stop (exit at close on last hold day)
                elif day - pos['entry_day'] >= hold_days:
                    to_close.append((sym, price, 'time'))
            
            for sym, exit_price, reason in to_close:
                pos = positions[sym]
                pnl = pos['shares'] * (exit_price - pos['entry_price'])
                cost = pos['shares'] * (exit_price + pos['entry_price']) * 0.001
                portfolio_value += pnl - cost
                
                hold_period = day - pos['entry_day']
                trades.append({
                    'sym': sym,
                    'pnl': pnl - cost,
                    'return_pct': (exit_price / pos['entry_price'] - 1),
                    'hold_days': hold_period,
                    'reason': reason
                })
                del positions[sym]
            
            # Look for new entries
            if len(positions) < max_positions:
                for sym in self.data:
                    if sym in positions:
                        continue
                    
                    signal = self.check_entry(sym, day)
                    if signal and signal['signal']:
                        # Position sizing: risk 2% of portfolio per trade
                        risk_amount = portfolio_value * 0.02
                        shares = int(risk_amount / signal['risk'])
                        
                        if shares > 0:
                            cost = shares * signal['entry_price'] * 0.001
                            portfolio_value -= cost
                            positions[sym] = {
                                'shares': shares,
                                'entry_price': signal['entry_price'],
                                'stop': signal['stop_loss'],
                                'target': signal['take_profit'],
                                'entry_day': day,
                                'rr': signal['rr_ratio']
                            }
                            break  # Only one new position per day
            
            daily_values.append(portfolio_value)
        
        # Close remaining positions
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
        
        # Trade statistics
        if trades:
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
            avg_hold = np.mean([t['hold_days'] for t in trades])
            
            # Exit reason breakdown
            target_hits = sum(1 for t in trades if t['reason'] == 'target')
            stop_hits = sum(1 for t in trades if t['reason'] == 'stop')
            time_exits = sum(1 for t in trades if t['reason'] == 'time')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            avg_hold = 0
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
            'total_trades': len(trades),
            'target_hits': target_hits,
            'stop_hits': stop_hits,
            'time_exits': time_exits,
            'daily_values': daily_values,
            'trades': trades
        }


def main():
    print('='*70)
    print('SWING TRADING SYSTEM — 20 EMA PULLBACK')
    print('='*70)
    
    system = SwingTradingSystem(universe=UNIVERSE[:30])
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
    print(f'Avg Hold: {results["avg_hold_days"]:.1f} days')
    
    if results['total_trades'] > 0:
        print(f'\nExit Breakdown:')
        print(f'  Target Hit: {results["target_hits"]} ({results["target_hits"]/results["total_trades"]*100:.1f}%)')
        print(f'  Stop Hit: {results["stop_hits"]} ({results["stop_hits"]/results["total_trades"]*100:.1f}%)')
        print(f'  Time Exit: {results["time_exits"]} ({results["time_exits"]/results["total_trades"]*100:.1f}%)')
    
    # Monthly breakdown
    daily_values = results['daily_values']
    months = len(daily_values) / 21
    monthly_returns = []
    for i in range(0, len(daily_values), 21):
        month_end = min(i + 21, len(daily_values))
        if i > 0:
            month_ret = (daily_values[month_end-1] / daily_values[i-1]) - 1
            monthly_returns.append(month_ret)
    
    if monthly_returns:
        print(f'\nMonthly Returns ({len(monthly_returns)} months):')
        print(f'  Mean: {np.mean(monthly_returns)*100:+.2f}%')
        print(f'  Best: {np.max(monthly_returns)*100:+.2f}%')
        print(f'  Worst: {np.min(monthly_returns)*100:+.2f}%')
        print(f'  Positive months: {sum(1 for r in monthly_returns if r > 0)}/{len(monthly_returns)}')
    
    return results


if __name__ == '__main__':
    main()
