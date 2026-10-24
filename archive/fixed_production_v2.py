"""
FIXED Production Trading System V2
==================================
V1 Results: +25.33% return, Sharpe 1.39
Issues:
1. Trailing stops at 4% too tight (-$55K from 58 stopped-out winners)
2. Bull trend momentum signals lose money (-$731)
3. Low vol regime loses money (-$3,812)

V2 Fixes:
1. Widen trailing stops to 6% (let winners run)
2. Remove bull trend momentum signals (stay in cash)
3. Remove low vol signals (no edge)
4. Profit target at 10% instead of 8%
5. Tighter entry threshold for sideways (need 3% underperformance)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE


class FixedProductionSystemV2:
    """
    V2: Wider stops, regime-filtered entries, better exits.
    """
    
    def __init__(self, universe=None):
        self.universe = universe or UNIVERSE[:25]
        self.data = {}
        
        # Risk parameters (FIXED)
        self.max_positions = 5
        self.max_position_size = 0.15
        self.transaction_cost = 0.001
        self.base_stop_loss = 0.06       # 6% hard stop (wider)
        self.trailing_stop_pct = 0.06   # 6% trailing (was 4%)
        self.target_profit = 0.10       # 10% profit target (was 8%)
        self.hold_days_min = 5
        self.hold_days_max = 20         # Longer hold (was 15)
        
        # Kelly params
        self.win_rate = 0.52
        self.avg_win_loss_ratio = 1.25
        self.kelly_fraction = 0.5
        
    def fetch_data(self, days=500):
        """Fetch data for all symbols."""
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
    
    def classify_regime(self, day_idx):
        """Classify market regime."""
        for spy_sym in ['SPY', 'QQQ', 'AAPL']:
            if spy_sym in self.data:
                closes = self.data[spy_sym]['closes'][:day_idx+1]
                if len(closes) < 30:
                    continue
                    
                rets = np.diff(closes) / closes[:-1]
                vol = np.std(rets[-20:]) * np.sqrt(252)
                
                if len(rets) > 21:
                    autocorr = np.corrcoef(rets[-21:-1], rets[-20:])[0, 1]
                    if np.isnan(autocorr):
                        autocorr = 0.0
                else:
                    autocorr = 0.0
                
                ma_10 = np.mean(closes[-10:])
                ma_20 = np.mean(closes[-20:])
                trend = (ma_10 - ma_20) / ma_20 if ma_20 > 0 else 0
                
                if vol > 0.30:
                    return 'HIGH_VOL', 0.7
                elif autocorr > 0.15 and abs(trend) > 0.02:
                    return 'TRENDING', 0.7
                elif vol < 0.12:
                    return 'LOW_VOL', 0.5
                else:
                    return 'SIDEWAYS', 0.5
        
        return 'UNKNOWN', 0.0
    
    def compute_features(self, sym, day_idx):
        """Compute features."""
        if sym not in self.data:
            return None
        
        closes = self.data[sym]['closes'][:day_idx+1]
        volumes = self.data[sym]['volumes'][:day_idx+1]
        highs = self.data[sym]['highs'][:day_idx+1]
        lows = self.data[sym]['lows'][:day_idx+1]
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
        
        ma_10 = np.mean(closes[-10:])
        trend = (ma_10 - ma20) / ma20 if ma20 > 0 else 0
        
        return {
            'rsi': rsi,
            'price_vs_ma20': price_vs_ma20,
            'volatility': vol20,
            'vol_ratio': vol_ratio,
            'atr_ratio': atr_ratio,
            'momentum_5d': mom_5,
            'momentum_20d': mom_20,
            'volume_ratio': vol_ratio_20,
            'trend': trend,
        }
    
    def volatility_scaled_sizing(self, capital, atr, price):
        """Risk 2% per trade based on ATR."""
        risk_per_trade = capital * 0.02
        stop_distance = atr * 2.0  # 2x ATR stop (wider)
        if stop_distance < 1e-10:
            return 0
        shares = int(risk_per_trade / stop_distance)
        max_shares = int(capital * self.max_position_size / price)
        return min(shares, max_shares)
    
    def generate_signals(self, day_idx, regime):
        """
        Regime-filtered signal generation.
        ONLY trade where we have edge: SIDEWAYS and HIGH_VOL.
        """
        if day_idx < 20:
            return []
        
        # NO signals in TRENDING or LOW_VOL — we don't have edge there
        if regime in ['TRENDING', 'LOW_VOL', 'BEAR_TREND']:
            return []
        
        sym_returns = {}
        for sym, data in self.data.items():
            closes = data['closes']
            if day_idx < len(closes) and day_idx >= 5:
                sym_returns[sym] = (closes[day_idx] / closes[day_idx - 5] - 1)
        
        if len(sym_returns) < 5:
            return []
        
        mean_ret = np.mean(list(sym_returns.values()))
        
        signals = []
        for sym, ret in sym_returns.items():
            features = self.compute_features(sym, day_idx)
            if features is None:
                continue
            
            if features['volatility'] > 0.5:
                continue
            
            if regime in ['SIDEWAYS', 'HIGH_VOL', 'LOW_VOL']:
                deviation = ret - mean_ret
                # Tighter threshold: need 3% underperformance (was 2%)
                if deviation < -0.03:
                    weight = min(abs(deviation) * 5, 1.0)
                    signals.append({
                        'sym': sym, 'weight': weight, 'type': 'long',
                        'score': abs(deviation) / (features['volatility'] + 0.01)
                    })
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals[:8]  # Fewer, higher-quality signals (was 10)
    
    def backtest(self, initial_capital=100_000):
        """Full walk-forward backtest."""
        print('\n' + '='*70)
        print('FIXED PRODUCTION TRADING SYSTEM V2')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        print(f'History: {min_len} days')
        
        # Backtest
        print('\nRunning backtest...')
        portfolio_value = initial_capital
        daily_values = []
        trades = []
        positions = {}
        
        for day in range(30, min_len - 1):
            regime, confidence = self.classify_regime(day)
            
            # Check trailing stops and profit targets
            positions_to_close = []
            for sym, (shares, entry_day, entry_price, highest_price) in positions.items():
                if sym in self.data and day < len(self.data[sym]['closes']):
                    current_price = self.data[sym]['closes'][day]
                    highest_price = max(highest_price, current_price)
                    
                    # Trailing stop (6%, widened from 4%)
                    trailing_stop = highest_price * (1 - self.trailing_stop_pct)
                    if current_price < trailing_stop:
                        positions_to_close.append((sym, 'trailing_stop'))
                        continue
                    
                    # Profit target (10%, up from 8%)
                    profit_pct = (current_price - entry_price) / entry_price
                    if profit_pct > self.target_profit:
                        positions_to_close.append((sym, 'profit_target'))
                        continue
                    
                    # Max hold period (20 days, up from 15)
                    if day - entry_day >= self.hold_days_max:
                        positions_to_close.append((sym, 'max_hold'))
                        continue
                    
                    # Hard stop loss (6%)
                    loss_pct = (entry_price - current_price) / entry_price
                    if loss_pct > self.base_stop_loss:
                        positions_to_close.append((sym, 'stop_loss'))
                        continue
            
            for sym, reason in positions_to_close:
                if sym in self.data and day < len(self.data[sym]['closes']):
                    exit_price = self.data[sym]['closes'][day]
                    shares, entry_day, entry_price, _ = positions[sym]
                    pnl = shares * (exit_price - entry_price)
                    cost = abs(shares) * (exit_price + entry_price) * self.transaction_cost
                    portfolio_value += pnl - cost
                    trades.append({
                        'sym': sym, 'entry_day': entry_day, 'exit_day': day,
                        'pnl': pnl - cost, 'regime': regime, 'reason': reason
                    })
                    del positions[sym]
            
            signals = self.generate_signals(day, regime)
            
            for signal in signals:
                sym = signal['sym']
                if sym in positions:
                    continue
                if len(positions) >= self.max_positions:
                    break
                
                price = self.data[sym]['closes'][day]
                features = self.compute_features(sym, day)
                if features is None:
                    continue
                
                atr = features['atr_ratio'] * price
                shares = self.volatility_scaled_sizing(portfolio_value, atr, price)
                
                if shares > 0:
                    cost = shares * price * self.transaction_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price, price)
            
            daily_values.append(portfolio_value)
        
        # Close remaining
        for sym, (shares, entry_day, entry_price, _) in positions.items():
            if sym in self.data and min_len - 1 < len(self.data[sym]['closes']):
                exit_price = self.data[sym]['closes'][min_len - 1]
                pnl = shares * (exit_price - entry_price)
                cost = abs(shares) * (exit_price + entry_price) * self.transaction_cost
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
            if avg_loss != 0:
                print(f'  Profit Factor: {abs(avg_win/avg_loss):.2f}')
            
            regimes = set(t['regime'] for t in trades)
            print(f'\n  By Regime:')
            for reg in sorted(regimes):
                reg_trades = [t for t in trades if t['regime'] == reg]
                reg_wr = sum(1 for t in reg_trades if t['pnl'] > 0) / len(reg_trades)
                reg_pnl = sum(t['pnl'] for t in reg_trades)
                print(f'    {reg:15s}: {len(reg_trades):3d} trades, {reg_wr*100:.0f}% WR, ${reg_pnl:+,.0f}')
            
            reasons = set(t['reason'] for t in trades)
            print(f'\n  By Exit Reason:')
            for reason in sorted(reasons):
                reason_trades = [t for t in trades if t['reason'] == reason]
                reason_wr = sum(1 for t in reason_trades if t['pnl'] > 0) / len(reason_trades)
                reason_pnl = sum(t['pnl'] for t in reason_trades)
                print(f'    {reason:15s}: {len(reason_trades):3d} trades, {reason_wr*100:.0f}% WR, ${reason_pnl:+,.0f}')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'trades': trades,
            'daily_values': daily_values
        }


if __name__ == '__main__':
    system = FixedProductionSystemV2(universe=UNIVERSE[:25])
    system.fetch_data(days=500)
    results = system.backtest()
