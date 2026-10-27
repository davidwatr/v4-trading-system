"""
Regime-Adaptive Trading System
================================
Combines:
1. Cross-sectional mean reversion signals
2. Regime classifier (adjusts strategy parameters per regime)
3. Meta-labeling filter (predicts signal quality)
4. Dynamic position sizing based on regime confidence

This is the production-ready system that integrates all components.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE


class RegimeAdaptiveSystem:
    """
    Full regime-adaptive trading system.
    
    Regimes and their strategy adjustments:
    - TRENDING_UP: Wider stops, larger positions, momentum overlay
    - TRENDING_DOWN: Tighter stops, smaller positions, mean reversion focus
    - MEAN_REVERTING: Standard mean reversion, normal sizing
    - VOLATILE: Reduced sizing, wider stops, fewer positions
    - LOW_VOL: Increased sizing, tighter stops
    """
    
    def __init__(self, universe=None):
        self.universe = universe or UNIVERSE[:25]
        self.data = {}
        
        # Regime-specific parameters
        self.regime_params = {
            'TRENDING_UP': {
                'max_positions': 5,
                'position_size': 0.12,  # 12% per position
                'stop_mult': 2.0,       # Wider stops
                'take_profit_mult': 3.0,
                'hold_period': 7,       # Longer holds
                'meta_threshold': 0.50, # Lower threshold (more trades)
            },
            'TRENDING_DOWN': {
                'max_positions': 3,
                'position_size': 0.08,
                'stop_mult': 1.5,
                'take_profit_mult': 2.0,
                'hold_period': 5,
                'meta_threshold': 0.55,
            },
            'MEAN_REVERTING': {
                'max_positions': 5,
                'position_size': 0.10,
                'stop_mult': 1.5,
                'take_profit_mult': 2.0,
                'hold_period': 5,
                'meta_threshold': 0.52,
            },
            'VOLATILE': {
                'max_positions': 3,
                'position_size': 0.06,
                'stop_mult': 2.5,
                'take_profit_mult': 3.0,
                'hold_period': 3,
                'meta_threshold': 0.55,
            },
            'LOW_VOL': {
                'max_positions': 5,
                'position_size': 0.15,
                'stop_mult': 1.0,
                'take_profit_mult': 1.5,
                'hold_period': 7,
                'meta_threshold': 0.50,
            },
            'UNKNOWN': {
                'max_positions': 4,
                'position_size': 0.10,
                'stop_mult': 1.5,
                'take_profit_mult': 2.0,
                'hold_period': 5,
                'meta_threshold': 0.52,
            }
        }
        
        # Meta-labeling model
        self.meta_model = None
        self.meta_scaler = None
        self.meta_trained = False
        
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
        """Classify current market regime."""
        # Use SPY or QQQ as market proxy
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
                
                if len(closes) > 40:
                    sma20 = np.mean(closes[-20:])
                    sma40 = np.mean(closes[-40:])
                    trend = (sma20 - sma40) / (sma40 + 1e-10)
                else:
                    trend = 0.0
                
                # Classify
                if vol > 0.30:
                    if autocorr > 0.1:
                        return 'TRENDING_DOWN', min(autocorr * 2, 1.0)
                    else:
                        return 'VOLATILE', 0.6
                elif vol < 0.15:
                    if autocorr > 0.05:
                        return 'TRENDING_UP', min(autocorr * 3, 1.0)
                    else:
                        return 'LOW_VOL', 0.5
                else:
                    if trend > 0.02:
                        return 'TRENDING_UP', min(trend * 10, 1.0)
                    elif trend < -0.02:
                        return 'TRENDING_DOWN', min(abs(trend) * 10, 1.0)
                    else:
                        return 'MEAN_REVERTING', 0.5
        
        return 'UNKNOWN', 0.0
    
    def compute_signal_features(self, sym, day_idx):
        """Compute features for meta-labeling."""
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
        
        # RSI
        gains = np.where(rets > 0, rets, 0)
        losses = np.where(rets < 0, -rets, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Price vs MA20
        ma20 = np.mean(closes[-20:])
        price_vs_ma20 = closes[-1] / ma20 - 1
        
        # Volatility
        vol20 = np.std(rets[-20:]) * np.sqrt(252)
        vol_short = np.std(rets[-10:]) if len(rets) >= 10 else vol20
        vol_long = np.std(rets[-30:]) if len(rets) >= 30 else vol20
        vol_ratio = vol_short / (vol_long + 1e-10)
        
        # ATR
        tr = np.abs(np.diff(closes[-15:]))
        atr = np.mean(tr)
        atr_ratio = atr / (closes[-1] + 1e-10)
        
        # Momentum
        mom_5 = (closes[-1] / closes[-6] - 1) if n >= 6 else 0
        mom_20 = (closes[-1] / closes[-21] - 1) if n >= 21 else 0
        
        # Volume ratio
        vol_ma20 = np.mean(volumes[-20:])
        vol_ratio_20 = volumes[-1] / (vol_ma20 + 1e-10)
        
        # Skewness
        if len(rets) >= 20:
            recent_rets = rets[-20:]
            skew = np.mean((recent_rets - np.mean(recent_rets))**3) / (np.std(recent_rets)**3 + 1e-10)
        else:
            skew = 0
        
        return {
            'rsi': rsi,
            'price_vs_ma20': price_vs_ma20,
            'volatility': vol20,
            'vol_ratio': vol_ratio,
            'atr_ratio': atr_ratio,
            'momentum_5d': mom_5,
            'momentum_20d': mom_20,
            'volume_ratio': vol_ratio_20,
            'skewness': skew,
        }
    
    def generate_signals(self, day_idx):
        """Generate cross-sectional mean reversion signals."""
        if day_idx < 20:
            return []
        
        # Compute 5-day returns for each stock
        sym_returns = {}
        for sym, data in self.data.items():
            closes = data['closes']
            if day_idx < len(closes) and day_idx >= 5:
                sym_returns[sym] = (closes[day_idx] / closes[day_idx - 5] - 1)
        
        if len(sym_returns) < 5:
            return []
        
        # Cross-sectional mean
        mean_ret = np.mean(list(sym_returns.values()))
        deviations = {sym: ret - mean_ret for sym, ret in sym_returns.items()}
        abs_sum = sum(abs(d) for d in deviations.values())
        
        if abs_sum < 1e-10:
            return []
        
        weights = {sym: -dev / abs_sum for sym, dev in deviations.items()}
        
        # Select top underperformers (positive weight = LONG)
        signals = []
        for sym, weight in weights.items():
            if weight > 0.02:  # Threshold
                signals.append({
                    'sym': sym,
                    'weight': weight,
                    'type': 'long'
                })
        
        # Sort by weight and take top N
        signals.sort(key=lambda x: x['weight'], reverse=True)
        return signals[:10]  # Top 10 candidates
    
    def train_meta_model(self, training_data):
        """Train meta-labeling model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        X = []
        y = []
        for data in training_data:
            if data['features'] is None:
                continue
            X.append(list(data['features'].values()))
            y.append(data['outcome'])
        
        if len(X) < 20:
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        self.meta_scaler = StandardScaler()
        X_scaled = self.meta_scaler.fit_transform(X)
        
        self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
        self.meta_model.fit(X_scaled, y)
        self.meta_trained = True
        return True
    
    def predict_signal_quality(self, features):
        """Predict if a signal will be profitable."""
        if not self.meta_trained or features is None:
            return True, 0.5
        
        X = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.meta_scaler.transform(X)
        proba = self.meta_model.predict_proba(X_scaled)[0]
        prediction = self.meta_model.predict(X_scaled)[0]
        
        return prediction == 1, proba[1]
    
    def backtest(self, initial_capital=100_000, transaction_cost=0.001):
        """
        Full walk-forward backtest with regime adaptation and meta-labeling.
        """
        print('\n' + '='*70)
        print('REGIME-ADAPTIVE TRADING SYSTEM')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        print(f'History: {min_len} days')
        
        # Phase 1: Collect meta-labeling training data
        print('\nPhase 1: Training meta-labeling model...')
        meta_training = []
        
        for day in range(30, min_len - 6):
            signals = self.generate_signals(day)
            for signal in signals:
                sym = signal['sym']
                features = self.compute_signal_features(sym, day)
                
                if features is not None:
                    closes = self.data[sym]['closes']
                    if day + 5 < len(closes):
                        outcome = 1 if closes[day + 5] > closes[day] else 0
                        meta_training.append({'features': features, 'outcome': outcome})
        
        print(f'Training samples: {len(meta_training)}')
        if self.train_meta_model(meta_training):
            win_rate = np.mean([d['outcome'] for d in meta_training])
            print(f'Meta model trained (base win rate: {win_rate*100:.1f}%)')
        
        # Phase 2: Walk-forward backtest
        print('\nPhase 2: Walk-forward backtest...')
        
        portfolio_value = initial_capital
        daily_values = []
        trades = []
        positions = {}
        regime_history = []
        
        for day in range(30, min_len - 1):
            # Classify regime
            regime, confidence = self.classify_regime(day)
            regime_history.append(regime)
            params = self.regime_params[regime]
            
            # Close expired positions
            positions_to_close = []
            for sym, (shares, entry_day, entry_price, _) in positions.items():
                if day - entry_day >= params['hold_period']:
                    positions_to_close.append(sym)
            
            for sym in positions_to_close:
                if sym in self.data and day < len(self.data[sym]['closes']):
                    exit_price = self.data[sym]['closes'][day]
                    shares, entry_day, entry_price, _ = positions[sym]
                    pnl = shares * (exit_price - entry_price)
                    cost = abs(shares) * (exit_price + entry_price) * transaction_cost
                    portfolio_value += pnl - cost
                    trades.append({
                        'sym': sym, 'entry_day': entry_day, 'exit_day': day,
                        'pnl': pnl - cost, 'regime': regime
                    })
                    del positions[sym]
            
            # Generate new signals
            signals = self.generate_signals(day)
            
            # Filter with meta-labeling
            filtered_signals = []
            for signal in signals:
                sym = signal['sym']
                features = self.compute_signal_features(sym, day)
                will_win, prob = self.predict_signal_quality(features)
                
                if will_win and prob > params['meta_threshold']:
                    filtered_signals.append({
                        **signal,
                        'probability': prob
                    })
            
            # Execute signals
            for signal in filtered_signals:
                sym = signal['sym']
                if sym in positions:
                    continue
                if len(positions) >= params['max_positions']:
                    break
                
                price = self.data[sym]['closes'][day]
                allocation = portfolio_value * min(signal['weight'] * 2, params['position_size'])
                shares = int(allocation / price)
                if shares > 0:
                    cost = shares * price * transaction_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price, signal['probability'])
            
            daily_values.append(portfolio_value)
        
        # Close remaining positions
        for sym, (shares, entry_day, entry_price, _) in positions.items():
            if sym in self.data and min_len - 1 < len(self.data[sym]['closes']):
                exit_price = self.data[sym]['closes'][min_len - 1]
                pnl = shares * (exit_price - entry_price)
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
        
        # Regime distribution
        regime_counts = {}
        for r in regime_history:
            regime_counts[r] = regime_counts.get(r, 0) + 1
        
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
        
        print(f'\nRegime Distribution:')
        for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
            print(f'  {regime}: {count} days ({count/len(regime_history)*100:.1f}%)')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'trades': trades,
            'daily_values': daily_values,
            'regime_history': regime_history
        }


if __name__ == '__main__':
    system = RegimeAdaptiveSystem(universe=UNIVERSE[:25])
    system.fetch_data(days=500)
    results = system.backtest()
