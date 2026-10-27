"""
V4 TUNED: Less conservative vol targeting, looser cluster limits
"""
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE


class FixedProductionSystemV4Tuned:
    def __init__(self, universe=None):
        self.universe = universe or UNIVERSE[:25]
        self.data = {}
        self.max_positions = 5
        self.max_position_size = 0.15
        self.transaction_cost = 0.001
        self.base_stop_loss = 0.08
        self.target_profit = 0.12
        self.hold_days_max = 25
        
        # V4 TUNED: More aggressive vol targeting
        self.target_vol = 0.18          # 18% target (higher)
        self.vol_half_life = 10
        self.no_trade_threshold = 0.03  # 3% no-tighter region
        
        # V4 TUNED: Looser cluster limits
        self.max_correlation = 0.75     # 0.75 threshold
        self.max_cluster_exposure = 0.40  # 40% max
        
        # Risk circuit breakers
        self.daily_loss_limit = 0.03
        self.max_drawdown_limit = 0.15
        
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
            'rsi': rsi, 'price_vs_ma20': price_vs_ma20, 'volatility': vol20,
            'atr_ratio': atr_ratio, 'momentum_5d': mom_5, 'momentum_20d': mom_20,
            'volume_ratio': vol_ratio_20, 'trend': trend, 'atr': atr,
            'returns': rets
        }
    
    def volatility_targeted_size(self, capital, price, features, current_portfolio_vol):
        """V4 TUNED: More aggressive sizing"""
        if features is None:
            return 0
        
        stock_vol = features['volatility']
        if stock_vol < 1e-10:
            return 0
        
        risk_per_trade = capital * 0.02
        atr = features['atr']
        stop_distance = atr * 2.0
        if stop_distance < 1e-10:
            return 0
        
        base_shares = int(risk_per_trade / stop_distance)
        
        # V4 TUNED: Less aggressive vol scaling
        vol_scale = min(self.target_vol / stock_vol, 2.0)  # Allow up to 2x
        
        # V4 TUNED: Only reduce if portfolio vol very high
        if current_portfolio_vol > self.target_vol * 2.0:
            vol_scale *= 0.8
        
        adjusted_shares = int(base_shares * vol_scale)
        max_shares = int(capital * self.max_position_size / price)
        return min(adjusted_shares, max_shares)
    
    def correlation_adjusted_size(self, sym, base_shares, price, positions, day_idx):
        """V4 TUNED: Less aggressive correlation reduction"""
        if not positions or base_shares <= 0:
            return base_shares
        
        correlations = []
        for held_sym in positions:
            if held_sym in self.data and sym in self.data:
                held_closes = self.data[held_sym]['closes'][:day_idx+1]
                sym_closes = self.data[sym]['closes'][:day_idx+1]
                
                if len(held_closes) >= 61 and len(sym_closes) >= 61:
                    held_rets = np.diff(held_closes[-61:]) / held_closes[-61:-1]
                    sym_rets = np.diff(sym_closes[-61:]) / sym_closes[-61:-1]
                    
                    min_len = min(len(held_rets), len(sym_rets))
                    corr = np.corrcoef(held_rets[:min_len], sym_rets[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if not correlations:
            return base_shares
        
        max_corr = max(correlations)
        
        if max_corr > self.max_correlation:
            reduction = 1 - (max_corr - self.max_correlation) / (1 - self.max_correlation)
            reduction = max(reduction, 0.5)  # At most 50% reduction (was 0.3)
            base_shares = int(base_shares * reduction)
        
        return max(base_shares, 0)
    
    def check_cluster_exposure(self, sym, price, shares, positions, day_idx):
        """V4 TUNED: Looser cluster exposure check"""
        if not positions:
            return True
        
        cluster_value = 0
        total_position_value = 0
        
        for held_sym, (held_shares, _, held_entry_price) in positions.items():
            if held_sym in self.data and sym in self.data:
                held_closes = self.data[held_sym]['closes'][:day_idx+1]
                sym_closes = self.data[sym]['closes'][:day_idx+1]
                
                if len(held_closes) >= 61 and len(sym_closes) >= 61:
                    held_rets = np.diff(held_closes[-61:]) / held_closes[-61:-1]
                    sym_rets = np.diff(sym_closes[-61:]) / sym_closes[-61:-1]
                    
                    min_len = min(len(held_rets), len(sym_rets))
                    corr = np.corrcoef(held_rets[:min_len], sym_rets[:min_len])[0, 1]
                    
                    if abs(corr) > self.max_correlation:
                        cluster_value += held_shares * held_entry_price
                
                total_position_value += held_shares * held_entry_price
        
        if total_position_value == 0:
            return True
        
        new_position_value = shares * price
        cluster_ratio = (cluster_value + new_position_value) / (total_position_value + new_position_value)
        
        return cluster_ratio <= self.max_cluster_exposure
    
    def generate_signals(self, day_idx, regime):
        if day_idx < 20:
            return []
        if regime in ['TRENDING', 'LOW_VOL']:
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
            
            deviation = ret - mean_ret
            if deviation < -0.03:
                weight = min(abs(deviation) * 5, 1.0)
                signals.append({
                    'sym': sym, 'weight': weight, 'type': 'long',
                    'score': abs(deviation) / (features['volatility'] + 0.01),
                    'features': features
                })
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals[:8]
    
    def estimate_portfolio_volatility(self, positions, day_idx):
        if not positions:
            return 0.0
        
        total_vol = 0
        total_value = 0
        
        for sym, (shares, _, entry_price) in positions.items():
            features = self.compute_features(sym, day_idx)
            if features:
                value = shares * entry_price
                total_vol += features['volatility'] * value
                total_value += value
        
        return total_vol / total_value if total_value > 0 else 0.0
    
    def backtest(self, initial_capital=100_000):
        print('\n' + '='*70)
        print('V4 TUNED: Less conservative vol targeting + looser cluster limits')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        print(f'History: {min_len} days')
        print(f'Parameters:')
        print(f'  Target Vol: {self.target_vol*100:.0f}%')
        print(f'  Max Correlation: {self.max_correlation}')
        print(f'  Max Cluster Exposure: {self.max_cluster_exposure*100:.0f}%')
        print()
        print('Running backtest...')
        
        portfolio_value = initial_capital
        daily_values = []
        trades = []
        positions = {}
        peak_value = initial_capital
        daily_pnl_start = portfolio_value
        
        for day in range(30, min_len - 1):
            peak_value = max(peak_value, portfolio_value)
            current_drawdown = (portfolio_value - peak_value) / peak_value
            
            # Circuit breaker: close everything if DD too large
            if current_drawdown < -self.max_drawdown_limit:
                for sym in list(positions.keys()):
                    if sym in self.data and day < len(self.data[sym]['closes']):
                        exit_price = self.data[sym]['closes'][day]
                        shares, entry_day, entry_price = positions[sym]
                        pnl = shares * (exit_price - entry_price)
                        cost = abs(shares) * (exit_price + entry_price) * self.transaction_cost
                        portfolio_value += pnl - cost
                        trades.append({
                            'sym': sym, 'entry_day': entry_day, 'exit_day': day,
                            'pnl': pnl - cost, 'regime': 'HALT', 'reason': 'circuit_breaker'
                        })
                positions.clear()
                daily_values.append(portfolio_value)
                continue
            
            regime, confidence = self.classify_regime(day)
            
            # Check exits
            positions_to_close = []
            for sym, (shares, entry_day, entry_price) in positions.items():
                if sym in self.data and day < len(self.data[sym]['closes']):
                    current_price = self.data[sym]['closes'][day]
                    profit_pct = (current_price - entry_price) / entry_price
                    loss_pct = (entry_price - current_price) / entry_price
                    
                    if profit_pct > self.target_profit:
                        positions_to_close.append((sym, 'profit_target'))
                    elif loss_pct > self.base_stop_loss:
                        positions_to_close.append((sym, 'stop_loss'))
                    elif day - entry_day >= self.hold_days_max:
                        positions_to_close.append((sym, 'max_hold'))
            
            for sym, reason in positions_to_close:
                if sym in self.data and day < len(self.data[sym]['closes']):
                    exit_price = self.data[sym]['closes'][day]
                    shares, entry_day, entry_price = positions[sym]
                    pnl = shares * (exit_price - entry_price)
                    cost = abs(shares) * (exit_price + entry_price) * self.transaction_cost
                    portfolio_value += pnl - cost
                    trades.append({
                        'sym': sym, 'entry_day': entry_day, 'exit_day': day,
                        'pnl': pnl - cost, 'regime': regime, 'reason': reason
                    })
                    del positions[sym]
            
            # Daily loss check
            daily_return = (portfolio_value - daily_pnl_start) / daily_pnl_start
            if daily_return > -self.daily_loss_limit:
                signals = self.generate_signals(day, regime)
                portfolio_vol = self.estimate_portfolio_volatility(positions, day)
                
                for signal in signals:
                    sym = signal['sym']
                    if sym in positions or len(positions) >= self.max_positions:
                        continue
                    
                    price = self.data[sym]['closes'][day]
                    features = signal['features']
                    
                    shares = self.volatility_targeted_size(portfolio_value, price, features, portfolio_vol)
                    shares = self.correlation_adjusted_size(sym, shares, price, positions, day)
                    
                    if not self.check_cluster_exposure(sym, price, shares, positions, day):
                        continue
                    
                    if shares > 0:
                        cost = shares * price * self.transaction_cost
                        portfolio_value -= cost
                        positions[sym] = (shares, day, price)
            
            daily_values.append(portfolio_value)
            daily_pnl_start = portfolio_value
        
        for sym, (shares, entry_day, entry_price) in positions.items():
            if sym in self.data and min_len - 1 < len(self.data[sym]['closes']):
                exit_price = self.data[sym]['closes'][min_len - 1]
                pnl = shares * (exit_price - entry_price)
                cost = abs(shares) * (exit_price + entry_price) * self.transaction_cost
                portfolio_value += pnl - cost
        
        daily_values.append(portfolio_value)
        
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
            
            reasons = set(t['reason'] for t in trades)
            print(f'\n  By Exit Reason:')
            for reason in sorted(reasons):
                reason_trades = [t for t in trades if t['reason'] == reason]
                reason_wr = sum(1 for t in reason_trades if t['pnl'] > 0) / len(reason_trades)
                reason_pnl = sum(t['pnl'] for t in reason_trades)
                print(f'    {reason:15s}: {len(reason_trades):3d} trades, {reason_wr*100:.0f}% WR, ${reason_pnl:+,.0f}')
        
        return {
            'total_return': total_return, 'annual_return': annual_return,
            'sharpe': sharpe, 'max_drawdown': max_dd, 'trades': trades
        }


if __name__ == '__main__':
    system = FixedProductionSystemV4Tuned(universe=UNIVERSE[:25])
    system.fetch_data(days=500)
    results = system.backtest()
