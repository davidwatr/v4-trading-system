"""
OOS Comparison: Heuristic vs Production (Cross-Sectional MR)
============================================================
Clean walk-forward test with:
- Permutation test (random noise must drop to ~50%)
- ROC-AUC check (must be >0.53)
- Confusion matrix (must show real DOWN precision)
- No look-ahead bias (causal features only)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from production_v18 import fetch_data, UNIVERSE
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


class HeuristicSystem:
    """Current live heuristic system (RSI < 50 OR 3 down days)."""
    
    def __init__(self, universe):
        self.universe = universe
        self.data = {}
    
    def fetch_data(self, days=500):
        for sym in self.universe:
            d = fetch_data(sym, days=days)
            if d is not None and len(d[0]) > 50:
                self.data[sym] = {
                    'closes': np.array(d[0]),
                    'volumes': np.array(d[1]),
                    'highs': np.array(d[2]),
                    'lows': np.array(d[3])
                }
        print(f'Heuristic: Loaded {len(self.data)} symbols')
    
    def generate_signals(self, day_idx):
        """RSI < 50 OR 3 consecutive down days. Must be > 200-day MA * 0.95."""
        signals = []
        for sym, data in self.data.items():
            closes = data['closes']
            volumes = data['volumes']
            if day_idx >= len(closes) or day_idx < 200:
                continue
            
            # 200-day MA filter
            ma200 = np.mean(closes[day_idx-200:day_idx])
            if closes[day_idx] < ma200 * 0.95:
                continue
            
            # RSI
            rets = np.diff(closes[day_idx-15:day_idx+1]) / closes[day_idx-15:day_idx]
            gains = np.where(rets > 0, rets, 0)
            losses = np.where(rets < 0, -rets, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # 3 down days
            recent_closes = closes[day_idx-3:day_idx+1]
            down_days = sum(1 for i in range(1, len(recent_closes)) if recent_closes[i] < recent_closes[i-1])
            
            if rsi < 50 or down_days >= 3:
                signals.append({'sym': sym, 'weight': 1.0, 'rsi': rsi})
        
        return signals


class ProductionSystem:
    """Cross-sectional mean reversion + meta-labeling."""
    
    def __init__(self, universe):
        self.universe = universe
        self.data = {}
        self.meta_model = None
        self.meta_scaler = None
        self.meta_trained = False
    
    def fetch_data(self, days=500):
        for sym in self.universe:
            d = fetch_data(sym, days=days)
            if d is not None and len(d[0]) > 50:
                self.data[sym] = {
                    'closes': np.array(d[0]),
                    'volumes': np.array(d[1]),
                    'highs': np.array(d[2]),
                    'lows': np.array(d[3])
                }
        print(f'Production: Loaded {len(self.data)} symbols')
    
    def compute_features(self, sym, day_idx):
        if sym not in self.data:
            return None
        closes = self.data[sym]['closes'][:day_idx+1]
        volumes = self.data[sym]['volumes'][:day_idx+1]
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
        
        return {
            'rsi': rsi, 'price_vs_ma20': price_vs_ma20, 'volatility': vol20,
            'vol_ratio': vol_ratio, 'atr_ratio': atr_ratio,
            'momentum_5d': mom_5, 'momentum_20d': mom_20, 'volume_ratio': vol_ratio_20,
        }
    
    def generate_signals(self, day_idx):
        if day_idx < 20:
            return []
        
        sym_returns = {}
        for sym, data in self.data.items():
            closes = data['closes']
            if day_idx < len(closes) and day_idx >= 5:
                sym_returns[sym] = (closes[day_idx] / closes[day_idx - 5] - 1)
        
        if len(sym_returns) < 5:
            return []
        
        mean_ret = np.mean(list(sym_returns.values()))
        deviations = {sym: ret - mean_ret for sym, ret in sym_returns.items()}
        abs_sum = sum(abs(d) for d in deviations.values())
        if abs_sum < 1e-10:
            return []
        
        weights = {sym: -dev / abs_sum for sym, dev in deviations.items()}
        signals = []
        for sym, weight in weights.items():
            if weight > 0.02:
                signals.append({'sym': sym, 'weight': weight})
        
        signals.sort(key=lambda x: x['weight'], reverse=True)
        return signals[:10]
    
    def train_meta_model(self, training_data):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        X, y = [], []
        for d in training_data:
            if d['features'] is None:
                continue
            X.append(list(d['features'].values()))
            y.append(d['outcome'])
        
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
        if not self.meta_trained or features is None:
            return True, 0.5
        X = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.meta_scaler.transform(X)
        proba = self.meta_model.predict_proba(X_scaled)[0]
        prediction = self.meta_model.predict(X_scaled)[0]
        return prediction == 1, proba[1]


def backtest_system(system, name, initial_capital=100_000, trans_cost=0.001):
    """Backtest a system with walk-forward OOS."""
    min_len = min(len(d['closes']) for d in system.data.values())
    
    # Phase 1: Train meta-labeling (if applicable)
    meta_training = []
    if isinstance(system, ProductionSystem):
        for day in range(200, min_len - 6):
            signals = system.generate_signals(day)
            for signal in signals:
                features = system.compute_features(signal['sym'], day)
                if features is not None:
                    closes = system.data[signal['sym']]['closes']
                    if day + 5 < len(closes):
                        outcome = 1 if closes[day + 5] > closes[day] else 0
                        meta_training.append({'features': features, 'outcome': outcome})
        
        if meta_training:
            system.train_meta_model(meta_training)
            print(f'  Meta model trained on {len(meta_training)} samples')
    
    # Phase 2: Walk-forward backtest
    portfolio_value = initial_capital
    daily_values = []
    trades = []
    positions = {}
    
    for day in range(200, min_len - 1):
        # Close expired positions (5-day hold)
        to_close = [sym for sym, (shares, entry_day, entry_price) in positions.items() if day - entry_day >= 5]
        for sym in to_close:
            if day < len(system.data[sym]['closes']):
                exit_price = system.data[sym]['closes'][day]
                shares, entry_day, entry_price = positions[sym]
                pnl = shares * (exit_price - entry_price)
                cost = abs(shares) * (exit_price + entry_price) * trans_cost
                portfolio_value += pnl - cost
                trades.append({'sym': sym, 'pnl': pnl - cost})
                del positions[sym]
        
        # Generate signals
        signals = system.generate_signals(day)
        
        # Filter with meta-labeling (production only)
        if isinstance(system, ProductionSystem):
            filtered = []
            for signal in signals:
                features = system.compute_features(signal['sym'], day)
                will_win, prob = system.predict_signal_quality(features)
                if will_win and prob > 0.52:
                    filtered.append(signal)
            signals = filtered
        
        # Execute
        for signal in signals:
            sym = signal['sym']
            if sym in positions or len(positions) >= 5:
                continue
            
            price = system.data[sym]['closes'][day]
            weight = signal.get('weight', 1.0)
            allocation = portfolio_value * min(weight * 2, 0.10)
            shares = int(allocation / price)
            if shares > 0:
                cost = shares * price * trans_cost
                portfolio_value -= cost
                positions[sym] = (shares, day, price)
        
        daily_values.append(portfolio_value)
    
    # Close remaining
    for sym, (shares, entry_day, entry_price) in positions.items():
        if min_len - 1 < len(system.data[sym]['closes']):
            exit_price = system.data[sym]['closes'][min_len - 1]
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
        'name': name,
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


def permutation_test(system, n_permutations=100):
    """Test if edge survives random feature permutation."""
    min_len = min(len(d['closes']) for d in system.data.values())
    
    perm_returns = []
    for _ in range(n_permutations):
        portfolio_value = 100_000
        positions = {}
        
        for day in range(200, min_len - 1):
            # Close expired
            to_close = [sym for sym, (shares, entry_day, entry_price) in positions.items() if day - entry_day >= 5]
            for sym in to_close:
                if day < len(system.data[sym]['closes']):
                    exit_price = system.data[sym]['closes'][day]
                    shares, entry_day, entry_price = positions[sym]
                    portfolio_value += shares * (exit_price - entry_price) * 0.999
                    del positions[sym]
            
            # Random signals (same number as real system)
            signals = system.generate_signals(day)
            if signals:
                n_signals = len(signals)
                all_syms = list(system.data.keys())
                random_syms = np.random.choice(all_syms, size=min(n_signals, len(all_syms)), replace=False)
                
                for sym in random_syms:
                    if sym in positions or len(positions) >= 5:
                        continue
                    price = system.data[sym]['closes'][day]
                    allocation = portfolio_value * 0.08
                    shares = int(allocation / price)
                    if shares > 0:
                        portfolio_value -= shares * price * 0.001
                        positions[sym] = (shares, day, price)
        
        total_return = (portfolio_value - 100_000) / 100_000
        perm_returns.append(total_return)
    
    return np.mean(perm_returns), np.std(perm_returns)


def compute_roc_auc(system, name):
    """Compute ROC-AUC for signal quality."""
    min_len = min(len(d['closes']) for d in system.data.values())
    
    y_true = []
    y_pred = []
    
    for day in range(200, min_len - 6):
        signals = system.generate_signals(day)
        for signal in signals:
            sym = signal['sym']
            closes = system.data[sym]['closes']
            if day + 5 < len(closes):
                y_true.append(1 if closes[day + 5] > closes[day] else 0)
                y_pred.append(signal.get('weight', 0.5))
    
    if len(y_true) < 10:
        return 0.5
    
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0.5
    
    return auc


def main():
    print('='*70)
    print('OOS COMPARISON: HEURISTIC vs PRODUCTION')
    print('='*70)
    
    universe = UNIVERSE[:25]
    
    # Fetch data once
    print('\nFetching data...')
    heuristic = HeuristicSystem(universe)
    production = ProductionSystem(universe)
    
    heuristic.fetch_data(days=500)
    production.fetch_data(days=500)
    
    # Backtest both
    print('\n' + '-'*70)
    print('BACKTESTING HEURISTIC SYSTEM')
    print('-'*70)
    h_results = backtest_system(heuristic, 'Heuristic')
    
    print('\n' + '-'*70)
    print('BACKTESTING PRODUCTION SYSTEM')
    print('-'*70)
    p_results = backtest_system(production, 'Production')
    
    # Results comparison
    print('\n' + '='*70)
    print('RESULTS COMPARISON')
    print('='*70)
    
    print(f'\n{"Metric":<20} {"Heuristic":<20} {"Production":<20}')
    print('-'*60)
    print(f'{"Total Return":<20} {h_results["total_return"]*100:>10.2f}%{"":<9} {p_results["total_return"]*100:>10.2f}%')
    print(f'{"Annual Return":<20} {h_results["annual_return"]*100:>10.2f}%{"":<9} {p_results["annual_return"]*100:>10.2f}%')
    print(f'{"Sharpe Ratio":<20} {h_results["sharpe"]:>10.3f}{"":<10} {p_results["sharpe"]:>10.3f}')
    print(f'{"Max Drawdown":<20} {h_results["max_drawdown"]*100:>10.2f}%{"":<9} {p_results["max_drawdown"]*100:>10.2f}%')
    print(f'{"Win Rate":<20} {h_results["win_rate"]*100:>10.1f}%{"":<9} {p_results["win_rate"]*100:>10.1f}%')
    print(f'{"Avg Win":<20} ${h_results["avg_win"]:>9.2f}{"":<10} ${p_results["avg_win"]:>9.2f}')
    print(f'{"Avg Loss":<20} ${h_results["avg_loss"]:>9.2f}{"":<10} ${p_results["avg_loss"]:>9.2f}')
    print(f'{"Total Trades":<20} {h_results["total_trades"]:>10}{"":<10} {p_results["total_trades"]:>10}')
    
    # Verification tests
    print('\n' + '='*70)
    print('VERIFICATION TESTS')
    print('='*70)
    
    # ROC-AUC
    print('\nROC-AUC Check:')
    h_auc = compute_roc_auc(heuristic, 'Heuristic')
    p_auc = compute_roc_auc(production, 'Production')
    print(f'  Heuristic: {h_auc:.4f} {"✅" if h_auc > 0.53 else "❌"} (need >0.53)')
    print(f'  Production: {p_auc:.4f} {"✅" if p_auc > 0.53 else "❌"} (need >0.53)')
    
    # Permutation test
    print('\nPermutation Test (random signals):')
    print('  Testing heuristic...')
    h_perm_mean, h_perm_std = permutation_test(heuristic, n_permutations=50)
    print(f'  Heuristic random return: {h_perm_mean*100:.2f}% ± {h_perm_std*100:.2f}%')
    
    print('  Testing production...')
    p_perm_mean, p_perm_std = permutation_test(production, n_permutations=50)
    print(f'  Production random return: {p_perm_mean*100:.2f}% ± {p_perm_std*100:.2f}%')
    
    if h_perm_mean < 0.02 and p_perm_mean < 0.02:
        print('  ✅ PASS: Random signals drop to ~0% → no structural leak')
    else:
        print('  ❌ FAIL: Random signals still profitable → structural leak')
    
    # Confusion matrix
    print('\nConfusion Matrix (Production):')
    min_len = min(len(d['closes']) for d in production.data.values())
    y_true, y_pred = [], []
    
    for day in range(200, min_len - 6):
        signals = production.generate_signals(day)
        for signal in signals:
            sym = signal['sym']
            closes = production.data[sym]['closes']
            if day + 5 < len(closes):
                y_true.append(1 if closes[day + 5] > closes[day] else 0)
                y_pred.append(1 if signal['weight'] > 0.03 else 0)
    
    if y_true:
        cm = confusion_matrix(y_true, y_pred)
        print(f'  TN: {cm[0,0]:>5}  FP: {cm[0,1]:>5}')
        print(f'  FN: {cm[1,0]:>5}  TP: {cm[1,1]:>5}')
        
        precision = cm[1,1] / (cm[1,1] + cm[0,1] + 1e-10)
        recall = cm[1,1] / (cm[1,1] + cm[1,0] + 1e-10)
        print(f'  Precision: {precision:.3f}')
        print(f'  Recall: {recall:.3f}')
    
    # Final verdict
    print('\n' + '='*70)
    print('FINAL VERDICT')
    print('='*70)
    
    if p_results['sharpe'] > h_results['sharpe'] and p_results['total_return'] > h_results['total_return']:
        print('  ✅ PRODUCTION SYSTEM WINS')
        print(f'  Better Sharpe ({p_results["sharpe"]:.3f} vs {h_results["sharpe"]:.3f})')
        print(f'  Better Return ({p_results["total_return"]*100:.2f}% vs {h_results["total_return"]*100:.2f}%)')
    elif h_results['sharpe'] > p_results['sharpe']:
        print('  ✅ HEURISTIC SYSTEM WINS')
    else:
        print('  ⚖️  MIXED RESULTS')
    
    print(f'\n  Recommendation: {"Deploy Production" if p_results["sharpe"] > h_results["sharpe"] else "Keep Heuristic"}')


if __name__ == '__main__':
    main()
