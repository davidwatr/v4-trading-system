#!/usr/bin/env python3
"""
Short-Term Swing Trading System (2-5 Day Hold) — Fixed Capital Model
Fixed 501-day window, deterministic data cache.

Rules:
- Macro: SPY > SMA50, no entries when below
- Trend: Close > SMA50 of the asset
- Entry: 2+ day pullback OR touch 10-day EMA, with volume > 20-day avg volume
- Exit: +4% TP, -2% SL, 5-day time stop
- Universe: financials + tech (no consumer, no dead symbols)
"""

import pickle
import numpy as np

def sma(data, period, idx):
    """SMA ending at idx (inclusive). Returns None if not enough data."""
    if idx < period - 1:
        return None
    return sum(data[idx - period + 1:idx + 1]) / period

def ema(data, period, idx):
    """EMA ending at idx (inclusive)."""
    if idx < period - 1:
        return None
    k = 2.0 / (period + 1)
    ema_val = data[idx - period + 1]
    for i in range(idx - period + 2, idx + 1):
        ema_val = data[i] * k + ema_val * (1 - k)
    return ema_val

def run_backtest():
    with open('deterministic_data_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    
    symbols_list = cache['symbols']
    spy_closes = cache['SPY']['closes']
    
    # ---- Universe ----
    ALL_SYMBOLS = [
        'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'AMD',
        'CRM', 'ADBE', 'INTC', 'PYPL',
        'JPM', 'BAC', 'GS', 'V', 'MA', 'MS',
    ]
    # Drop dead symbols from prior analysis
    DEAD = {'MS', 'INTC'}
    UNIVERSE = [s for s in ALL_SYMBOLS if s not in DEAD and s in symbols_list]
    
    TOTAL_DAYS = 501
    START_IDX = 200
    MAX_POSITIONS = 8
    TP_PCT = 0.04
    SL_PCT = 0.02
    TIME_STOP = 5
    
    # ---- Precompute SPY SMA50 ----
    spy_sma50 = [sma(spy_closes, 50, i) for i in range(len(spy_closes))]
    
    # ---- Precompute per-symbol indicators ----
    symbol_data = {}
    for sym in UNIVERSE:
        d = symbols_list[sym]
        closes = d['closes']
        volumes = d['volumes']
        symbol_data[sym] = {
            'closes': closes,
            'volumes': volumes,
            'sma50': [sma(closes, 50, i) for i in range(len(closes))],
            'ema10': [ema(closes, 10, i) for i in range(len(closes))],
            'vol_sma20': [sma(volumes, 20, i) for i in range(len(volumes))],
        }
    
    # ---- Portfolio ----
    INITIAL_CAPITAL = 100_000.0
    capital = INITIAL_CAPITAL  # cash available
    
    # Positions: sym -> {entry_idx, entry_price, shares, entry_value}
    open_positions = {}
    
    daily_portfolio = []
    trades = []
    
    for day in range(START_IDX, TOTAL_DAYS):
        # ---- Mark-to-market: compute portfolio value ----
        port_value = capital
        for sym, pos in open_positions.items():
            close = symbol_data[sym]['closes'][day]
            port_value += pos['shares'] * close
        daily_portfolio.append(port_value)
        
        # ---- Macro filter ----
        spy_above_sma = spy_sma50[day] is not None and spy_closes[day] > spy_sma50[day]
        
        # ---- Check exits ----
        to_close = []
        for sym, pos in open_positions.items():
            close = symbol_data[sym]['closes'][day]
            days_held = day - pos['entry_idx']
            
            # SL: -2%
            if close <= pos['entry_price'] * (1 - SL_PCT):
                to_close.append((sym, 'SL', close))
            # TP: +4%
            elif close >= pos['entry_price'] * (1 + TP_PCT):
                to_close.append((sym, 'TP', close))
            # Time stop
            elif days_held >= TIME_STOP:
                to_close.append((sym, 'TIME', close))
        
        for sym, reason, exit_price in to_close:
            pos = open_positions.pop(sym)
            exit_value = pos['shares'] * exit_price
            pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price']
            capital += exit_value  # refund cash
            
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
        if not spy_above_sma:
            continue
        
        open_count = len(open_positions)
        if open_count >= MAX_POSITIONS:
            continue
        
        # Capital per position: equal weight on total portfolio value
        # Each position = 1/MAX_POSITIONS of total portfolio
        total_equity = capital + sum(pos['shares'] * symbol_data[sym]['closes'][day]
                                      for sym, pos in open_positions.items())
        alloc_per_position = total_equity / MAX_POSITIONS
        
        for sym in UNIVERSE:
            if sym in open_positions:
                continue
            
            d = symbol_data[sym]
            closes = d['closes']
            volumes = d['volumes']
            
            sma50 = d['sma50'][day]
            ema10 = d['ema10'][day]
            vol_sma20 = d['vol_sma20'][day]
            
            if sma50 is None or ema10 is None or vol_sma20 is None:
                continue
            if closes[day] <= sma50:
                continue
            
            # Pullback: 2 consecutive down days OR cross below EMA10
            pullback = False
            if day >= 3:
                if closes[day] < closes[day-1] and closes[day-1] < closes[day-2]:
                    pullback = True
            if not pullback and day >= 1:
                if closes[day] <= ema10 and closes[day-1] > ema10:
                    pullback = True
            
            if not pullback:
                continue
            
            # Volume spike
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
    
    total_return = (daily_portfolio[-1] / daily_portfolio[0]) - 1
    n_ret_days = len(daily_returns)
    ann_return = (1 + total_return) ** (252 / max(1, n_ret_days)) - 1
    sharpe = (np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
              if np.std(daily_returns) > 0 else 0)
    
    peak = daily_portfolio[0]
    max_dd = 0
    for val in daily_portfolio:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        max_dd = max(max_dd, dd)
    
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
    
    # SPY benchmark
    spy_slice = spy_closes[START_IDX:TOTAL_DAYS]
    spy_total = (spy_slice[-1] / spy_slice[0]) - 1
    spy_ann = (1 + spy_total) ** (252 / len(spy_slice)) - 1
    spy_daily = np.diff(spy_slice) / spy_slice[:-1]
    spy_sharpe = (np.sqrt(252) * np.mean(spy_daily) / np.std(spy_daily)
                  if np.std(spy_daily) > 0 else 0)
    spy_peak = spy_slice[0]
    spy_max_dd = 0
    for p in spy_slice:
        spy_peak = max(spy_peak, p)
        spy_max_dd = max(spy_max_dd, (spy_peak - p) / spy_peak)
    
    # ---- Output ----
    print("=" * 60)
    print("SHORT-TERM SWING TRADING SYSTEM (2-5 DAY HOLD)")
    print("=" * 60)
    print(f"\nWindow: day {START_IDX} to {TOTAL_DAYS-1} ({TOTAL_DAYS - START_IDX} trading days)")
    print(f"Universe: {len(UNIVERSE)} symbols — {UNIVERSE}")
    print(f"Max positions: {MAX_POSITIONS} | TP: +{TP_PCT*100:.0f}% | SL: -{SL_PCT*100:.0f}% | Time stop: {TIME_STOP} days")
    print(f"Entry: 2-day pullback or EMA10 cross + volume > 20-day avg")
    print()
    
    print("--- SWING SYSTEM ---")
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
    
    print("--- SPY BUY-AND-HOLD ---")
    print(f"Total Return:        {spy_total*100:+.2f}%")
    print(f"Annualized Return:   {spy_ann*100:+.2f}%")
    print(f"Sharpe Ratio:        {spy_sharpe:.2f}")
    print(f"Max Drawdown:        {spy_max_dd*100:.2f}%")
    print()
    
    print("--- COMPARISON (Swing System vs SPY) ---")
    print(f"Total Return diff:   {total_return*100 - spy_total*100:+.2f} pp")
    print(f"Sharpe diff:         {sharpe - spy_sharpe:+.2f}")
    print(f"Max DD diff:         {(max_dd - spy_max_dd)*100:+.2f} pp (negative = less drawdown)")
    
    beating_spy_return = total_return > spy_total
    beating_spy_sharpe = sharpe > spy_sharpe
    
    print()
    if beating_spy_return and beating_spy_sharpe:
        print("✓ BEATS SPY on both return and risk-adjusted basis")
    elif beating_spy_return:
        print("⚠ Beats SPY on return but not risk-adjusted (Sharpe)")
    elif beating_spy_sharpe:
        print("⚠ Beats SPY on risk-adjusted (Sharpe) but not total return")
    else:
        print("✗ Does NOT beat SPY on either metric")
    
    # ---- Save ----
    results = {
        'system': {
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
        }
    }
    
    with open('swing_backtest_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nResults saved to swing_backtest_results.pkl")

if __name__ == '__main__':
    run_backtest()
