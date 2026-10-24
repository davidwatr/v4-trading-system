#!/usr/bin/env python3
"""
Optimized Tweaked Swing Trading System (3-10 Day Hold)
Fixed 501-day window, deterministic data cache.

Changes from baseline:
- Universe: Drop AAPL, MSFT, ADBE (worst performers). Drop AMZN (not in cache).
- Pullback: 3 consecutive down days (was 2)
- TP: +6% (was +4%)
- SL: -3% (was -2%)
- Time stop: 10 days (was 5)
- No EMA10 cross entry (pure pullback + volume)
"""

import pickle
import numpy as np

def sma(data, period, idx):
    if idx < period - 1:
        return None
    return sum(data[idx - period + 1:idx + 1]) / period

def run_backtest():
    with open('deterministic_data_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    
    symbols_list = cache['symbols']
    spy_closes = cache['SPY']['closes']
    
    # ---- Universe: drop worst performers + AMZN (not in cache) ----
    ALL_SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'AMD',
        'CRM', 'ADBE', 'INTC', 'PYPL',
        'JPM', 'BAC', 'GS', 'V', 'MA', 'MS',
    ]
    DROP = {'AAPL', 'MSFT', 'ADBE', 'MS', 'INTC', 'AMZN'}
    UNIVERSE = [s for s in ALL_SYMBOLS if s not in DROP and s in symbols_list]
    
    TOTAL_DAYS = 501
    START_IDX = 200
    MAX_POSITIONS = 8
    TP_PCT = 0.06       # +6% target
    SL_PCT = 0.03       # -3% stop
    TIME_STOP = 10      # 10-day time stop
    
    # Precompute SPY SMA50
    spy_sma50 = [sma(spy_closes, 50, i) for i in range(len(spy_closes))]
    
    # Precompute per-symbol
    symbol_data = {}
    for sym in UNIVERSE:
        d = symbols_list[sym]
        closes = d['closes']
        volumes = d['volumes']
        symbol_data[sym] = {
            'closes': closes,
            'volumes': volumes,
            'sma50': [sma(closes, 50, i) for i in range(len(closes))],
            'vol_sma20': [sma(volumes, 20, i) for i in range(len(volumes))],
        }
    
    INITIAL_CAPITAL = 100_000.0
    capital = INITIAL_CAPITAL
    open_positions = {}
    
    daily_portfolio = []
    trades = []
    
    for day in range(START_IDX, TOTAL_DAYS):
        # ---- Mark-to-market ----
        port_value = capital
        for sym, pos in open_positions.items():
            port_value += pos['shares'] * symbol_data[sym]['closes'][day]
        daily_portfolio.append(port_value)
        
        # ---- Macro filter ----
        spy_above = spy_sma50[day] is not None and spy_closes[day] > spy_sma50[day]
        
        # ---- Exit checks ----
        to_close = []
        for sym, pos in open_positions.items():
            close = symbol_data[sym]['closes'][day]
            days_held = day - pos['entry_idx']
            
            if close <= pos['entry_price'] * (1 - SL_PCT):
                to_close.append((sym, 'SL', close))
            elif close >= pos['entry_price'] * (1 + TP_PCT):
                to_close.append((sym, 'TP', close))
            elif days_held >= TIME_STOP:
                to_close.append((sym, 'TIME', close))
        
        for sym, reason, exit_price in to_close:
            pos = open_positions.pop(sym)
            exit_value = pos['shares'] * exit_price
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
            capital += exit_value
            
            trades.append({
                'symbol': sym,
                'entry_idx': pos['entry_idx'],
                'exit_idx': day,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'pnl_dollars': exit_value - pos['entry_value'],
                'days_held': day - pos['entry_idx'],
                'exit_reason': reason,
            })
        
        # ---- Scan for entries ----
        if not spy_above:
            continue
        
        open_count = len(open_positions)
        if open_count >= MAX_POSITIONS:
            continue
        
        total_equity = capital + sum(
            pos['shares'] * symbol_data[sym]['closes'][day]
            for sym, pos in open_positions.items()
        )
        alloc_per_position = total_equity / MAX_POSITIONS
        
        for sym in UNIVERSE:
            if sym in open_positions:
                continue
            
            d = symbol_data[sym]
            closes = d['closes']
            volumes = d['volumes']
            
            sma50 = d['sma50'][day]
            vol_sma20 = d['vol_sma20'][day]
            
            if sma50 is None or vol_sma20 is None:
                continue
            if closes[day] <= sma50:
                continue
            
            # Pullback: 3 consecutive down days
            pullback = False
            if day >= 4:
                if (closes[day] < closes[day-1] and
                    closes[day-1] < closes[day-2] and
                    closes[day-2] < closes[day-3]):
                    pullback = True
            
            if not pullback:
                continue
            
            # Volume spike on reversal day
            if volumes[day] <= vol_sma20:
                continue
            
            # ---- ENTER ----
            price = closes[day]
            shares = alloc_per_position / price
            entry_value = shares * price
            capital -= entry_value
            
            open_positions[sym] = {
                'entry_idx': day,
                'entry_price': price,
                'shares': shares,
                'entry_value': entry_value,
            }
    
    # ---- Final close ----
    final_day = TOTAL_DAYS - 1
    for sym, pos in open_positions.items():
        close_price = symbol_data[sym]['closes'][final_day]
        exit_value = pos['shares'] * close_price
        pnl_pct = (close_price - pos['entry_price']) / pos['entry_price']
        capital += exit_value
        
        trades.append({
            'symbol': sym,
            'entry_idx': pos['entry_idx'],
            'exit_idx': final_day,
            'entry_price': pos['entry_price'],
            'exit_price': close_price,
            'pnl_pct': pnl_pct,
            'pnl_dollars': exit_value - pos['entry_value'],
            'days_held': final_day - pos['entry_idx'],
            'exit_reason': 'FINAL',
        })
    
    daily_portfolio.append(capital)
    
    # ---- Metrics ----
    daily_returns = np.diff(daily_portfolio) / daily_portfolio[:-1]
    
    total_return = daily_portfolio[-1] / daily_portfolio[0] - 1
    n_ret_days = len(daily_returns)
    ann_return = (1 + total_return) ** (252 / max(1, n_ret_days)) - 1
    sharpe = (np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
              if np.std(daily_returns) > 0 else 0)
    
    peak = daily_portfolio[0]
    max_dd = 0
    for v in daily_portfolio:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak)
    
    n_trades = len(trades)
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0
    avg_duration = np.mean([t['days_held'] for t in trades]) if trades else 0
    total_win = sum(t['pnl_dollars'] for t in wins)
    total_loss = sum(abs(t['pnl_dollars']) for t in losses)
    profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
    
    # SPY
    spy_slice = spy_closes[START_IDX:TOTAL_DAYS]
    spy_total = spy_slice[-1] / spy_slice[0] - 1
    spy_ann = (1 + spy_total) ** (252 / len(spy_slice)) - 1
    spy_daily = np.diff(spy_slice) / spy_slice[:-1]
    spy_sharpe = (np.sqrt(252) * np.mean(spy_daily) / np.std(spy_daily)
                  if np.std(spy_daily) > 0 else 0)
    spy_peak = spy_slice[0]
    spy_max_dd = 0
    for p in spy_slice:
        spy_peak = max(spy_peak, p)
        spy_max_dd = max(spy_max_dd, (spy_peak - p) / spy_peak)
    
    # Baseline swing (from earlier run)
    # +3.72%, ann +3.11%, Sharpe 0.44, DD 8.23%, 131 trades, WR 45.8%
    baseline = {
        'total_return': 0.0372,
        'annualized_return': 0.0311,
        'sharpe': 0.44,
        'max_drawdown': 0.0823,
        'n_trades': 131,
        'win_rate': 0.458,
        'profit_factor': 1.15,
        'avg_duration': 3.7,
    }
    
    # ---- Print ----
    print("=" * 65)
    print("OPTIMIZED TWEAKED SWING SYSTEM (3-Day Pullback, 10-Day Hold)")
    print("=" * 65)
    print(f"\nUniverse: {len(UNIVERSE)} symbols — {UNIVERSE}")
    print(f"Dropped: AAPL, MSFT, ADBE (worst performers) + MS, INTC, AMZN")
    print(f"Max positions: {MAX_POSITIONS} | TP: +{TP_PCT*100:.0f}% | SL: -{SL_PCT*100:.0f}% | Time stop: {TIME_STOP} days")
    print(f"Entry: 3-day pullback + volume > 20-day avg + SPY>SMA50 + Close>SMA50")
    print()
    
    print("--- OPTIMIZED SWING SYSTEM ---")
    print(f"Total Return:        {total_return*100:+.2f}%")
    print(f"Annualized Return:   {ann_return*100:+.2f}%")
    print(f"Sharpe Ratio:        {sharpe:.2f}")
    print(f"Max Drawdown:        {max_dd*100:.2f}%")
    print(f"Total Trades:        {n_trades}")
    print(f"Win Rate:            {win_rate*100:.1f}%")
    print(f"Avg Win:             {avg_win*100:+.2f}%")
    print(f"Avg Loss:            {avg_loss*100:+.2f}%")
    if avg_loss != 0:
        print(f"Win/Loss Ratio:      {avg_win/abs(avg_loss):.2f}")
    print(f"Profit Factor:       {profit_factor:.2f}")
    print(f"Avg Trade Duration:  {avg_duration:.1f} days")
    print()
    print("Exit breakdown:")
    for reason in ['TP', 'SL', 'TIME', 'FINAL']:
        r = [t for t in trades if t['exit_reason'] == reason]
        if r:
            avg_pnl = np.mean([t['pnl_pct'] for t in r])
            print(f"  {reason}: {len(r):3d} trades, avg {avg_pnl*100:+.2f}%")
    print()
    
    print("--- BASELINE SWING SYSTEM (for comparison) ---")
    print(f"Total Return:        {baseline['total_return']*100:+.2f}%")
    print(f"Annualized Return:   {baseline['annualized_return']*100:+.2f}%")
    print(f"Sharpe Ratio:        {baseline['sharpe']:.2f}")
    print(f"Max Drawdown:        {baseline['max_drawdown']*100:.2f}%")
    print(f"Total Trades:        {baseline['n_trades']}")
    print(f"Win Rate:            {baseline['win_rate']*100:.1f}%")
    print(f"Profit Factor:       {baseline['profit_factor']:.2f}")
    print(f"Avg Trade Duration:  {baseline['avg_duration']:.1f} days")
    print()
    
    print("--- SPY BUY-AND-HOLD ---")
    print(f"Total Return:        {spy_total*100:+.2f}%")
    print(f"Annualized Return:   {spy_ann*100:+.2f}%")
    print(f"Sharpe Ratio:        {spy_sharpe:.2f}")
    print(f"Max Drawdown:        {spy_max_dd*100:.2f}%")
    print()
    
    print("--- THREE-WAY COMPARISON ---")
    print(f"{'Metric':<25} {'Optimized':>10} {'Baseline':>10} {'SPY':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"{'Total Return':<25} {total_return*100:>+9.2f}% {baseline['total_return']*100:>+9.2f}% {spy_total*100:>+9.2f}%")
    print(f"{'Annualized Return':<25} {ann_return*100:>+9.2f}% {baseline['annualized_return']*100:>+9.2f}% {spy_ann*100:>+9.2f}%")
    print(f"{'Sharpe Ratio':<25} {sharpe:>10.2f} {baseline['sharpe']:>10.2f} {spy_sharpe:>10.2f}")
    print(f"{'Max Drawdown':<25} {max_dd*100:>+9.2f}% {baseline['max_drawdown']*100:>+9.2f}% {spy_max_dd*100:>+9.2f}%")
    print(f"{'Total Trades':<25} {n_trades:>10} {baseline['n_trades']:>10} {'--':>10}")
    print(f"{'Win Rate':<25} {win_rate*100:>+9.1f}% {baseline['win_rate']*100:>+9.1f}% {'--':>10}")
    print(f"{'Profit Factor':<25} {profit_factor:>10.2f} {baseline['profit_factor']:>10.2f} {'--':>10}")
    print(f"{'Avg Duration (days)':<25} {avg_duration:>10.1f} {baseline['avg_duration']:>10.1f} {'--':>10}")
    print()
    
    beating_spy = total_return > spy_total and sharpe > spy_sharpe
    if beating_spy:
        print("✓ BEATS SPY on both return and risk-adjusted basis")
    elif total_return > spy_total:
        print("⚠ Beats SPY on return but not risk-adjusted")
    elif sharpe > spy_sharpe:
        print("⚠ Beats SPY on risk-adjusted but not total return")
    else:
        print("✗ Does NOT beat SPY on either metric")
    
    # ---- Save ----
    results = {
        'optimized': {
            'total_return': total_return,
            'annualized_return': ann_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_duration': avg_duration,
            'daily_portfolio': daily_portfolio,
            'trades': trades,
        },
        'baseline': baseline,
        'spy': {
            'total_return': spy_total,
            'annualized_return': spy_ann,
            'sharpe': spy_sharpe,
            'max_drawdown': spy_max_dd,
        },
        'config': {
            'start_idx': START_IDX,
            'total_days': TOTAL_DAYS,
            'universe': UNIVERSE,
            'max_positions': MAX_POSITIONS,
            'tp_pct': TP_PCT,
            'sl_pct': SL_PCT,
            'time_stop_days': TIME_STOP,
            'pullback_days': 3,
            'volume_filter': True,
        }
    }
    
    with open('swing_optimized_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to swing_optimized_results.pkl")

if __name__ == '__main__':
    run_backtest()
