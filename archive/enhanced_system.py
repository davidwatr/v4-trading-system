"""
Regime Classifier + Meta-Labeling Filter
========================================
1. Regime Classifier: Detect if market is trending, mean-reverting, or volatile
   using volatility, ATR expansion, and return autocorrelation.
2. Meta-Labeling: Predict which heuristic signals will win using cross-sectional features.

Based on:
- NautilusTrader's AI training framework
- Ernest Chan's regime detection
- Lopez de Prado's meta-labeling (AFML Ch.3)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE


class RegimeClassifier:
    """
    Classify market regime using volatility and trend features.
    
    Regimes:
    - TRENDING_UP: Low volatility, positive autocorrelation
    - TRENDING_DOWN: High volatility, negative autocorrelation  
    - MEAN_REVERTING: High volatility, near-zero autocorrelation
    - TRANSITION: Mixed signals
    """
    
    def __init__(self, lookback=20):
        self.lookback = lookback
    
    def compute_features(self, closes, volumes=None):
        """Compute regime detection features."""
        rets = np.diff(closes) / closes[:-1]
        
        if len(rets) < self.lookback + 1:
            return None
        
        # Realized volatility
        vol = np.std(rets[-self.lookback:]) * np.sqrt(252)
        
        # Volatility of volatility (vol regime)
        if len(rets) > self.lookback * 2:
            vol_short = np.std(rets[-self.lookback:])
            vol_long = np.std(rets[-self.lookback*2:])
            vol_ratio = vol_short / (vol_long + 1e-10)
        else:
            vol_ratio = 1.0
        
        # Return autocorrelation (trend vs mean-reversion)
        if len(rets) > self.lookback + 1:
            autocorr = np.corrcoef(rets[-self.lookback-1:-1], rets[-self.lookback:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
        
        # Trend strength (SMA slope)
        if len(closes) > self.lookback:
            sma = np.mean(closes[-self.lookback:])
            sma_prev = np.mean(closes[-self.lookback*2:-self.lookback]) if len(closes) > self.lookback * 2 else sma
            trend = (sma - sma_prev) / (sma_prev + 1e-10)
        else:
            trend = 0.0
        
        # ATR expansion
        if len(closes) > self.lookback:
            tr = np.abs(np.diff(closes[-self.lookback:]))
            atr = np.mean(tr)
            atr_prev = np.mean(np.abs(np.diff(closes[-self.lookback*2:-self.lookback]))) if len(closes) > self.lookback * 2 else atr
            atr_ratio = atr / (atr_prev + 1e-10)
        else:
            atr_ratio = 1.0
        
        return {
            'volatility': vol,
            'vol_ratio': vol_ratio,
            'autocorrelation': autocorr,
            'trend': trend,
            'atr_ratio': atr_ratio
        }
    
    def classify(self, closes, volumes=None):
        """
        Classify regime based on features.
        
        Returns: {
            'regime': str,
            'confidence': float,
            'features': dict
        }
        """
        feats = self.compute_features(closes, volumes)
        if feats is None:
            return {'regime': 'UNKNOWN', 'confidence': 0.0, 'features': {}}
        
        vol = feats['volatility']
        autocorr = feats['autocorrelation']
        trend = feats['trend']
        vol_ratio = feats['vol_ratio']
        
        # Classification logic
        if vol > 0.30:  # High volatility
            if autocorr > 0.1:
                regime = 'TRENDING_DOWN'
                confidence = min(autocorr * 2, 1.0)
            elif autocorr < -0.1:
                regime = 'MEAN_REVERTING'
                confidence = min(abs(autocorr) * 2, 1.0)
            else:
                regime = 'VOLATILE'
                confidence = 0.5
        elif vol < 0.15:  # Low volatility
            if autocorr > 0.05:
                regime = 'TRENDING_UP'
                confidence = min(autocorr * 3, 1.0)
            else:
                regime = 'LOW_VOL'
                confidence = 0.5
        else:  # Medium volatility
            if trend > 0.02:
                regime = 'TRENDING_UP'
                confidence = min(trend * 10, 1.0)
            elif trend < -0.02:
                regime = 'TRENDING_DOWN'
                confidence = min(abs(trend) * 10, 1.0)
            else:
                regime = 'MEAN_REVERTING'
                confidence = 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'features': feats
        }


class MetaLabelingFilter:
    """
    Meta-labeling: predict which heuristic signals will win.
    
    Instead of generating signals, this filter predicts whether a given
    signal will be profitable or not.
    
    Based on AFML Ch.3: Use ML to size bets, not generate them.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
    
    def compute_signal_features(self, closes, volumes, highs, lows, signal_type='long'):
        """
        Compute features for a specific signal.
        All features are causal (past-only).
        """
        n = len(closes)
        if n < 25:
            return None
        
        rets = np.diff(closes) / closes[:-1]
        
        # RSI (14-day)
        gains = np.where(rets > 0, rets, 0)
        losses = np.where(rets < 0, -rets, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Price vs MA20
        ma20 = np.mean(closes[-20:])
        price_vs_ma20 = closes[-1] / ma20 - 1
        
        # Volatility (20-day)
        vol20 = np.std(rets[-20:]) * np.sqrt(252)
        
        # Volatility ratio (short/long)
        vol_short = np.std(rets[-10:]) if len(rets) >= 10 else vol20
        vol_long = np.std(rets[-30:]) if len(rets) >= 30 else vol20
        vol_ratio = vol_short / (vol_long + 1e-10)
        
        # ATR (14-day)
        tr = np.abs(np.diff(closes[-15:]))
        atr = np.mean(tr)
        atr_ratio = atr / (closes[-1] + 1e-10)
        
        # Momentum (5-day)
        mom_5 = (closes[-1] / closes[-6] - 1) if n >= 6 else 0
        
        # Momentum (20-day)
        mom_20 = (closes[-1] / closes[-21] - 1) if n >= 21 else 0
        
        # Volume ratio
        vol_ma20 = np.mean(volumes[-20:])
        vol_ratio_20 = volumes[-1] / (vol_ma20 + 1e-10)
        
        # Skewness (20-day returns)
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
            'signal_type': 1 if signal_type == 'long' else -1
        }
    
    def prepare_training_data(self, signals_data):
        """
        Prepare training data from historical signals.
        
        signals_data: list of dicts with keys:
            - features: dict from compute_signal_features
            - outcome: 1 if signal was profitable, 0 otherwise
        """
        X = []
        y = []
        
        for data in signals_data:
            if data['features'] is None:
                continue
            X.append(list(data['features'].values()))
            y.append(data['outcome'])
        
        if len(X) < 10:
            return None, None
        
        return np.array(X), np.array(y)
    
    def train(self, X, y):
        """Train the meta-labeling model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        if len(X) < 20:
            return False
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        return True
    
    def predict(self, features):
        """
        Predict whether a signal will be profitable.
        
        Returns: {
            'will_win': bool,
            'confidence': float,
            'probability': float
        }
        """
        if not self.is_trained or features is None:
            return {'will_win': True, 'confidence': 0.5, 'probability': 0.5}
        
        X = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]
        
        return {
            'will_win': prediction == 1,
            'confidence': max(proba),
            'probability': proba[1]
        }


class EnhancedTradingSystem:
    """
    Enhanced trading system combining:
    1. Cross-sectional mean reversion signals
    2. Regime classifier (adjusts strategy parameters)
    3. Meta-labeling filter (predicts signal quality)
    """
    
    def __init__(self, universe=None):
        self.universe = universe or UNIVERSE[:25]
        self.regime_clf = RegimeClassifier(lookback=20)
        self.meta_filter = MetaLabelingFilter()
        self.data = {}
        
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
    
    def generate_signals(self, day_idx):
        """
        Generate signals using cross-sectional mean reversion.
        """
        if day_idx < 20:
            return {'long': [], 'short': [], 'regime': 'UNKNOWN'}
        
        # Compute returns for each stock
        sym_returns = {}
        for sym, data in self.data.items():
            closes = data['closes']
            if day_idx < len(closes) and day_idx >= 5:
                sym_returns[sym] = (closes[day_idx] / closes[day_idx - 5] - 1)
        
        if len(sym_returns) < 5:
            return {'long': [], 'short': [], 'regime': 'UNKNOWN'}
        
        # Cross-sectional mean
        mean_ret = np.mean(list(sym_returns.values()))
        deviations = {sym: ret - mean_ret for sym, ret in sym_returns.items()}
        abs_sum = sum(abs(d) for d in deviations.values())
        
        if abs_sum < 1e-10:
            return {'long': [], 'short': [], 'regime': 'UNKNOWN'}
        
        weights = {sym: -dev / abs_sum for sym, dev in deviations.items()}
        
        # Classify regime using SPY or first available
        regime = 'UNKNOWN'
        for spy_sym in ['SPY', 'QQQ', 'AAPL']:
            if spy_sym in self.data:
                result = self.regime_clf.classify(self.data[spy_sym]['closes'][:day_idx+1])
                regime = result['regime']
                break
        
        # Select top signals
        sorted_syms = sorted(weights.items(), key=lambda x: x[1])
        
        signals = {
            'long': [(sym, w) for sym, w in sorted_syms[-5:] if w > 0.02],
            'short': [(sym, w) for sym, w in sorted_syms[:5:] if w < -0.02],
            'regime': regime
        }
        
        return signals
    
    def backtest(self, initial_capital=100_000, transaction_cost=0.001):
        """
        Full backtest with regime detection and meta-labeling.
        """
        print('\n' + '='*70)
        print('ENHANCED TRADING SYSTEM BACKTEST')
        print('='*70)
        
        min_len = min(len(d['closes']) for d in self.data.values())
        print(f'History: {min_len} days')
        
        # Phase 1: Collect training data for meta-labeling
        print('\nPhase 1: Collecting meta-labeling training data...')
        meta_training_data = []
        
        lookback = 5
        for day in range(30, min_len - lookback - 1):
            signals = self.generate_signals(day)
            
            for sym, weight in signals['long']:
                if sym not in self.data:
                    continue
                closes = self.data[sym]['closes']
                volumes = self.data[sym]['volumes']
                highs = self.data[sym]['highs']
                lows = self.data[sym]['lows']
                
                features = self.meta_filter.compute_signal_features(
                    closes[:day+1], volumes[:day+1], highs[:day+1], lows[:day+1], 'long'
                )
                
                # Outcome: did price go up in next 5 days?
                if day + lookback < len(closes):
                    outcome = 1 if closes[day + lookback] > closes[day] else 0
                    meta_training_data.append({'features': features, 'outcome': outcome})
        
        print(f'Collected {len(meta_training_data)} training samples')
        
        # Train meta-labeling filter
        X, y = self.meta_filter.prepare_training_data(meta_training_data)
        if X is not None:
            self.meta_filter.train(X, y)
            print(f'Meta-labeling model trained (win rate: {np.mean(y)*100:.1f}%)')
        
        # Phase 2: Walk-forward backtest
        print('\nPhase 2: Walk-forward backtest...')
        
        portfolio_value = initial_capital
        daily_values = []
        trades = []
        positions = {}
        
        for day in range(30, min_len - 1):
            # Close expired positions
            positions_to_close = []
            for sym, (shares, entry_day, entry_price, _) in positions.items():
                if day - entry_day >= lookback:
                    positions_to_close.append(sym)
            
            for sym in positions_to_close:
                if sym in self.data and day < len(self.data[sym]['closes']):
                    exit_price = self.data[sym]['closes'][day]
                    shares, entry_day, entry_price, _ = positions[sym]
                    pnl = shares * (exit_price - entry_price) * (1 if shares > 0 else -1)
                    cost = abs(shares) * (exit_price + entry_price) * transaction_cost
                    portfolio_value += pnl - cost
                    trades.append({
                        'sym': sym, 'entry_day': entry_day, 'exit_day': day,
                        'pnl': pnl - cost, 'type': 'long' if shares > 0 else 'short'
                    })
                    del positions[sym]
            
            # Generate new signals
            signals = self.generate_signals(day)
            
            # Apply meta-labeling filter
            filtered_longs = []
            for sym, weight in signals['long']:
                if sym not in self.data:
                    continue
                closes = self.data[sym]['closes']
                volumes = self.data[sym]['volumes']
                highs = self.data[sym]['highs']
                lows = self.data[sym]['lows']
                
                features = self.meta_filter.compute_signal_features(
                    closes[:day+1], volumes[:day+1], highs[:day+1], lows[:day+1], 'long'
                )
                
                prediction = self.meta_filter.predict(features)
                
                # Only take signals where meta-model predicts win
                if prediction['will_win'] and prediction['probability'] > 0.52:
                    filtered_longs.append((sym, weight, prediction['probability']))
            
            # Execute filtered signals
            for sym, weight, prob in filtered_longs:
                if sym in positions:
                    continue
                if len(positions) >= 5:  # Max 5 positions
                    break
                
                price = self.data[sym]['closes'][day]
                allocation = portfolio_value * min(abs(weight) * 2, 0.10)
                shares = int(allocation / price)
                if shares > 0:
                    cost = shares * price * transaction_cost
                    portfolio_value -= cost
                    positions[sym] = (shares, day, price, prob)
            
            daily_values.append(portfolio_value)
        
        # Close remaining
        for sym, (shares, entry_day, entry_price, _) in positions.items():
            if sym in self.data and min_len - 1 < len(self.data[sym]['closes']):
                exit_price = self.data[sym]['closes'][min_len - 1]
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
            if avg_loss != 0:
                print(f'  Profit Factor: {abs(avg_win/avg_loss):.2f}')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'trades': trades,
            'daily_values': daily_values
        }


if __name__ == '__main__':
    system = EnhancedTradingSystem(universe=UNIVERSE[:25])
    system.fetch_data(days=500)
    results = system.backtest()
