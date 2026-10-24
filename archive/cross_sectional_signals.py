"""
Cross-Sectional Mean Reversion Signal Generator
Based on Ernest Chan's "Algorithmic Trading: Winning Strategies and Their Rationale"
and Teddy Koker's backtest (Sharpe 1.17, 7.98% annual, 6.58% max drawdown).

Key insight: Don't predict absolute direction. Rank stocks relative to each other.
Buy the biggest underperformers, short the biggest outperformers.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE


class CrossSectionalMeanReversion:
    """
    Cross-sectional mean reversion strategy.
    
    Each day, compute each stock's return relative to the universe mean.
    Weight = -(stock_return - mean_return) / sum(|deviations|)
    
    Negative weight = SHORT (outperformer expected to revert down)
    Positive weight = LONG (underperformer expected to revert up)
    """
    
    def __init__(self, lookback=5, universe=None, min_volume=1e6):
        self.lookback = lookback
        self.universe = universe or UNIVERSE[:30]  # Top 30 liquid stocks
        self.min_volume = min_volume
        self.data = {}
        self.returns = {}
        self.volumes = {}
        
    def fetch_all(self, days=252):
        """Fetch data for entire universe."""
        print(f'Fetching {len(self.universe)} symbols...')
        for sym in self.universe:
            d = fetch_data(sym, days=days)
            if d is not None and len(d[0]) > 50:
                closes = np.array(d[0])
                volumes = np.array(d[1])
                # Filter low-volume stocks
                if np.mean(volumes[-20:]) >= self.min_volume:
                    self.data[sym] = closes
                    self.volumes[sym] = volumes
                    # Compute daily returns
                    self.returns[sym] = np.diff(closes) / closes[:-1]
        print(f'Loaded {len(self.data)} symbols')
        return len(self.data)
    
    def compute_weights(self, day_idx):
        """
        Compute cross-sectional mean reversion weights for a given day.
        
        Returns dict: {symbol: weight}
        Positive weight = LONG, Negative weight = SHORT
        """
        if day_idx < self.lookback:
            return {}
        
        # Compute lookback returns for each stock
        sym_returns = {}
        for sym, rets in self.returns.items():
            if day_idx < len(rets):
                start_idx = max(0, day_idx - self.lookback)
                sym_returns[sym] = np.mean(rets[start_idx:day_idx])
        
        if len(sym_returns) < 5:
            return {}
        
        # Cross-sectional mean
        mean_ret = np.mean(list(sym_returns.values()))
        
        # Deviations from mean
        deviations = {sym: ret - mean_ret for sym, ret in sym_returns.items()}
        
        # Normalize weights
        abs_sum = sum(abs(d) for d in deviations.values())
        if abs_sum < 1e-10:
            return {}
        
        weights = {sym: -dev / abs_sum for sym, dev in deviations.items()}
        
        return weights
    
    def generate_signals(self, day_idx, top_n=5):
        """
        Generate LONG/SHORT signals for a given day.
        
        Returns: {
            'long': [(symbol, weight), ...],  # top N underperformers
            'short': [(symbol, weight), ...]  # top N outperformers
        }
        """
        weights = self.compute_weights(day_idx)
        if not weights:
            return {'long': [], 'short': []}
        
        # Sort by weight (most negative = most outperformed = SHORT)
        sorted_syms = sorted(weights.items(), key=lambda x: x[1])
        
        # SHORT the top N outperformers (most negative weight)
        shorts = sorted_syms[:top_n]
        
        # LONG the top N underperformers (most positive weight)
        longs = sorted_syms[-top_n:]
        longs = [(sym, w) for sym, w in longs if w > 0]
        
        return {
            'long': longs,
            'short': [(sym, w) for sym, w in shorts if w < 0]
        }
    
    def backtest(self, initial_capital=100_000, transaction_cost=0.001):
        """
        Walk-forward backtest with proper validation.
        """
        print('\n' + '='*70)
        print('CROSS-SECTIONAL MEAN REVERSION BACKTEST')
        print('='*70)
        
        # Find common date range
        min_len = min(len(closes) for closes in self.data.values())
        print(f'Common history: {min_len} days')
        
        # Walk-forward: rebalance daily, hold for lookback period
        portfolio_value = initial_capital
        daily_values = []
        trades = []
        
        # Track positions: {symbol: (shares, entry_price)}
        positions = {}
        
        for day in range(self.lookback + 20, min_len - 1):
            # Close positions that have been held for lookback period
            positions_to_close = []
            for sym, (shares, entry_day, entry_price) in positions.items():
                if day - entry_day >= self.lookback:
                    positions_to_close.append(sym)
            
            for sym in positions_to_close:
                if sym in self.data and day < len(self.data[sym]):
                    exit_price = self.data[sym][day]
                    shares, entry_day, entry_price = positions[sym]
                    pnl = shares * (exit_price - entry_price) * (1 if shares > 0 else -1)
                    cost = abs(shares) * (exit_price + entry_price) * transaction_cost
                    portfolio_value += pnl - cost
                    trades.append({
                        'sym': sym, 'entry_day': entry_day, 'exit_day': day,
                        'pnl': pnl - cost, 'type': 'long' if shares > 0 else 'short'
                    })
                    del positions[sym]
            
            # Generate new signals
            signals = self.generate_signals(day, top_n=3)
            
            # Execute LONG signals
            for sym, weight in signals['long']:
                if sym not in self.data or day >= len(self.data[sym]):
                    continue
                if sym in positions:
                    continue
                
                price = self.data[sym][day]
                # Allocate capital based on weight
                allocation = portfolio_value * min(abs(weight) * 2, 0.10)  # Max 10% per position
                shares = int(allocation / price)
                if shares > 0:
                    cost = shares * price * transaction_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price)
            
            # Execute SHORT signals (if margin allows)
            for sym, weight in signals['short']:
                if sym not in self.data or day >= len(self.data[sym]):
                    continue
                if sym in positions:
                    continue
                
                price = self.data[sym][day]
                allocation = portfolio_value * min(abs(weight) * 2, 0.05)  # Max 5% per short
                shares = -int(allocation / price)
                if shares < 0:
                    cost = abs(shares) * price * transaction_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price)
            
            daily_values.append(portfolio_value)
        
        # Close all remaining positions
        for sym, (shares, entry_day, entry_price) in positions.items():
            if sym in self.data and min_len - 1 < len(self.data[sym]):
                exit_price = self.data[sym][min_len - 1]
                pnl = shares * (exit_price - entry_price) * (1 if shares > 0 else -1)
                cost = abs(shares) * (exit_price + entry_price) * transaction_cost
                portfolio_value += pnl - cost
        
        daily_values.append(portfolio_value)
        
        # Compute metrics
        daily_values = np.array(daily_values)
        daily_rets = np.diff(daily_values) / daily_values[:-1]
        
        total_return = (portfolio_value - initial_capital) / initial_capital
        annual_return = (1 + total_return) ** (252 / len(daily_values)) - 1
        sharpe = np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252)
        max_dd = np.min(daily_values / np.maximum.accumulate(daily_values) - 1)
        
        print(f'\nResults:')
        print(f'  Total Return: {total_return*100:.2f}%')
        print(f'  Annual Return: {annual_return*100:.2f}%')
        print(f'  Sharpe Ratio: {sharpe:.3f}')
        print(f'  Max Drawdown: {max_dd*100:.2f}%')
        print(f'  Total Trades: {len(trades)}')
        
        if trades:
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
            avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in trades) else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
            print(f'  Win Rate: {win_rate*100:.1f}%')
            print(f'  Avg Win: ${avg_win:.2f}')
            print(f'  Avg Loss: ${avg_loss:.2f}')
            print(f'  Profit Factor: {abs(avg_win/avg_loss):.2f}' if avg_loss != 0 else '')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'trades': trades,
            'daily_values': daily_values
        }


if __name__ == '__main__':
    cs = CrossSectionalMeanReversion(lookback=5, universe=UNIVERSE[:30])
    cs.fetch_all(days=500)
    results = cs.backtest()
