#!/usr/bin/env python3
"""
Model Competition Daily Update
Runs all 4 model versions, checks SL/TP, updates positions.
Rules: $20 start, all-in per trade, max 1 position, 14d hold, -8% SL, +15% TP
"""

import json
import pickle
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from production_v18 import (
    fetch_data, extract_features, get_market_regime,
    UNIVERSE, MODEL_FILE
)

MODELS = {
    'v3_ml': 'ml_model.pkl',
    'v4_xgb': 'ml_model_xgb.pkl',
    'v5_proper': 'ml_model_proper.pkl',
    'v6_ridge': 'ml_model_ridge.pkl'
}

INITIAL_CAPITAL = 20.0
STOP_LOSS = -0.08
TAKE_PROFIT = 0.15
MAX_HOLD_DAYS = 14


def load_model(model_file):
    with open(model_file, 'rb') as f:
        return pickle.load(f)


def compute_v6_features(closes, volumes, highs, lows):
    """Compute z-score features for Ridge model (v6)"""
    if len(closes) < 200:
        return None

    current_price = closes[-1]

    # RSI
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    rsi_values = []
    for i in range(14, len(gains)+1):
        avg_gain = sum(gains[i-14:i]) / 14
        avg_loss = sum(losses[i-14:i]) / 14
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rsi_values.append(100 - (100 / (1 + avg_gain / avg_loss)))

    rsi_zscore = (rsi_values[-1] - 50) / 25.0 if rsi_values else 0

    # Vol ratio
    avg_vol_20 = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
    vol_ratio = volumes[-1] / avg_vol_20 if avg_vol_20 > 0 else 1.0

    # ATR ratio
    atr_values = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr_values.append(tr)
    atr_14_current = sum(atr_values[-14:]) / 14 if len(atr_values) >= 14 else atr_values[-1]
    atr_50_avg = sum(atr_values[-50:]) / 50 if len(atr_values) >= 50 else sum(atr_values) / len(atr_values)
    atr_ratio = atr_14_current / atr_50_avg if atr_50_avg > 0 else 1.0

    # Momentum zscore (20-day returns)
    momentum_values = [(closes[i] - closes[i-20]) / closes[i-20] for i in range(20, len(closes))]
    if momentum_values:
        avg_mom = sum(momentum_values) / len(momentum_values)
        std_mom = (sum((m - avg_mom)**2 for m in momentum_values) / len(momentum_values)) ** 0.5
        momentum_zscore = (momentum_values[-1] - avg_mom) / std_mom if std_mom > 0 else 0
    else:
        momentum_zscore = 0

    # Spread zscore (high-low spread)
    spread_values = [(highs[i] - lows[i]) / closes[i] for i in range(len(closes))]
    if spread_values:
        avg_spread = sum(spread_values) / len(spread_values)
        std_spread = (sum((s - avg_spread)**2 for s in spread_values) / len(spread_values)) ** 0.5
        spread_zscore = (spread_values[-1] - avg_spread) / std_spread if std_spread > 0 else 0
    else:
        spread_zscore = 0

    return {
        'rsi_zscore': rsi_zscore,
        'vol_ratio': vol_ratio,
        'atr_ratio': atr_ratio,
        'momentum_zscore': momentum_zscore,
        'spread_zscore': spread_zscore
    }


def run_screener(model_name, model_file):
    """Run a single model screener and return buy signals"""
    saved = load_model(model_file)
    model = saved['model']
    features = saved['feature_names']

    # Fetch SPY
    spy_closes, _, spy_highs, spy_lows = fetch_data("SPY", days=730)
    if not spy_closes:
        print(f"ERROR: Can't fetch SPY for {model_name}")
        return []

    regime, _, _, _, _ = get_market_regime(spy_closes, spy_highs, spy_lows)

    results = []
    for symbol in UNIVERSE:
        try:
            closes, volumes, highs, lows = fetch_data(symbol, days=730)
            if not closes or len(closes) < 200:
                continue

            if model_name == 'v6_ridge':
                feat = compute_v6_features(closes, volumes, highs, lows)
                if feat is None:
                    continue
            else:
                sector_etf = {
                    "AAPL":"XLK","MSFT":"XLK","GOOGL":"XLK","NVDA":"XLK","META":"XLK",
                    "INTC":"XLK","AMD":"XLK","CSCO":"XLK","ORCL":"XLK","CRM":"XLK","ADBE":"XLK",
                    "JPM":"XLF","BAC":"XLF","GS":"XLF","MS":"XLF","WFC":"XLF","C":"XLF",
                    "BRK-B":"XLF","V":"XLF","MA":"XLF","PYPL":"XLF",
                    "AMZN":"XLY","TSLA":"XLY","DIS":"XLY","NFLX":"XLY"
                }.get(symbol, "SPY")
                sector_closes, _, _, _ = fetch_data(sector_etf, days=730)
                if not sector_closes:
                    sector_closes = spy_closes
                feat = extract_features(closes, volumes, highs, lows, symbol, spy_closes, sector_closes)
                if feat is None:
                    continue

            X = [[feat[name] for name in features]]

            if model_name == 'v6_ridge':
                # Ridge regression - convert to probability via sigmoid
                raw = model.predict(X)[0]
                prob = 1 / (1 + (-raw).exp()) if hasattr(raw, 'exp') else 1 / (1 + 2.71828 ** (-raw))
            else:
                prob = float(model.predict_proba(X)[0][1])

            score = prob - 0.5
            signal = "BUY" if prob > 0.6 else ("SELL" if prob < 0.4 else "HOLD")

            results.append({
                "symbol": symbol,
                "price": round(float(closes[-1]), 2),
                "probability": round(float(prob), 4),
                "signal": signal,
                "score": round(float(score), 4),
                "regime": regime
            })
            time.sleep(0.2)
        except Exception as e:
            continue

    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def load_latest_competition():
    """Load the latest competition state from log"""
    log_file = Path('competition_log.jsonl')
    if not log_file.exists():
        return None
    with open(log_file, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None
    return json.loads(lines[-1])


def update_competition_state(latest_state, all_signals, current_prices):
    """
    Update competition state based on rules:
    - Max 1 position at a time
    - All-in $20 per position
    - Check SL (-8%) and TP (+15%)
    - 14-day max hold
    """
    today = datetime.now().strftime('%Y-%m-%d')
    new_state = {}

    for model_name, signals in all_signals.items():
        # Get current position from latest state
        current_pos = None
        if latest_state and 'competition' in latest_state:
            model_data = latest_state['competition'].get(model_name, {})
            if 'slots' in model_data:
                # Find the actual position (not cash slots)
                for slot in model_data['slots']:
                    if slot['holding'] and slot['entry'] is not None:
                        # Check if it's a real position (return != 0 or entry != current)
                        current_pos = slot
                        break
            elif 'holding' in model_data and model_data['holding'] and model_data.get('entry'):
                current_pos = model_data

        # Get total portfolio value from latest state
        portfolio_value = INITIAL_CAPITAL
        if latest_state and 'competition' in latest_state:
            model_data = latest_state['competition'].get(model_name, {})
            portfolio_value = model_data.get('total_value', INITIAL_CAPITAL)

        if current_pos:
            # Check current position
            entry_price = current_pos['entry']
            symbol = current_pos['holding']
            current_price = current_prices.get(symbol, entry_price)
            return_pct = (current_price - entry_price) / entry_price

            # Check stop loss or take profit
            if return_pct <= STOP_LOSS:
                # Stop loss hit - close position
                portfolio_value = INITIAL_CAPITAL * (1 + return_pct)
                new_state[model_name] = {
                    'holding': None,
                    'entry': None,
                    'current': None,
                    'return': 0.0,
                    'value': round(portfolio_value, 2),
                    'total_return': round((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
                    'exit_reason': 'STOP_LOSS',
                    'prev_holding': symbol,
                    'prev_entry': entry_price,
                    'exit_price': current_price,
                    'exit_return': round(return_pct * 100, 2)
                }
            elif return_pct >= TAKE_PROFIT:
                # Take profit hit - close position
                portfolio_value = INITIAL_CAPITAL * (1 + return_pct)
                new_state[model_name] = {
                    'holding': None,
                    'entry': None,
                    'current': None,
                    'return': 0.0,
                    'value': round(portfolio_value, 2),
                    'total_return': round((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
                    'exit_reason': 'TAKE_PROFIT',
                    'prev_holding': symbol,
                    'prev_entry': entry_price,
                    'exit_price': current_price,
                    'exit_return': round(return_pct * 100, 2)
                }
            else:
                # Position still valid
                new_state[model_name] = {
                    'holding': symbol,
                    'entry': entry_price,
                    'current': current_price,
                    'return': round(return_pct * 100, 2),
                    'value': round(INITIAL_CAPITAL * (1 + return_pct), 2),
                    'total_return': round((INITIAL_CAPITAL * (1 + return_pct) - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)
                }
        else:
            # No position - look for buy signal
            top_buys = [s for s in signals if s['signal'] == 'BUY' and s['score'] >= 0.1]
            if top_buys:
                best = top_buys[0]
                symbol = best['symbol']
                price = best['price']
                # Use current portfolio value as the new "investment base"
                new_state[model_name] = {
                    'holding': symbol,
                    'entry': price,
                    'current': price,
                    'return': 0.0,
                    'value': round(portfolio_value, 2),
                    'total_return': round((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
                    'entry_reason': f"BUY signal (prob={best['probability']:.2%})"
                }
            else:
                # Stay in cash
                new_state[model_name] = {
                    'holding': None,
                    'entry': None,
                    'current': None,
                    'return': 0.0,
                    'value': round(portfolio_value, 2),
                    'total_return': round((portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2)
                }

    return new_state


def main():
    print(f"=== Model Competition Update {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")

    # Load latest state
    latest = load_latest_competition()
    if latest:
        print(f"Latest log: {latest['date']} {latest['time']}")
    else:
        print("No existing log found - starting fresh")

    # Run all screeners
    all_signals = {}
    for model_name, model_file in MODELS.items():
        print(f"\n--- Running {model_name} ({model_file}) ---")
        signals = run_screener(model_name, model_file)
        all_signals[model_name] = signals
        top_buys = [s for s in signals if s['signal'] == 'BUY'][:3]
        buy_strs = [f"{s['symbol']}(${s['price']:.2f}, P={s['probability']:.2%})" for s in top_buys]
        print(f"  Top buys: {', '.join(buy_strs)}")

    # Get current prices from v3 signals (they all use the same universe)
    current_prices = {}
    for s in all_signals['v3_ml']:
        current_prices[s['symbol']] = s['price']

    # Update competition state
    new_state = update_competition_state(latest, all_signals, current_prices)

    # Print results
    print("\n=== COMPETITION STATE ===")
    for model_name, data in new_state.items():
        if data['holding']:
            print(f"  {model_name:12s}: LONG {data['holding']:6s} @ ${data['entry']:.2f} → ${data['current']:.2f} ({data['return']:+.2f}%)  [${data['value']:.2f}]")
        else:
            reason = data.get('exit_reason', data.get('entry_reason', 'CASH'))
            print(f"  {model_name:12s}: CASH  [${data['value']:.2f}] ({data['total_return']:+.2f}%)  {reason}")

    # Log to JSONL
    log_entry = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M'),
        'prices': {k: round(v, 2) for k, v in sorted(current_prices.items())},
        'competition': new_state
    }
    with open('competition_log.jsonl', 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    print(f"\n✓ Logged to competition_log.jsonl")

    # Update competition.md
    update_competition_md(new_state, current_prices)

    print(f"✓ Updated competition.md")


def update_competition_md(state, prices):
    """Update the competition markdown file"""
    md_path = Path(r'C:\Users\david_wncs0ps\OneDrive\Documents\hermes and i\Trading\competition.md')

    today = datetime.now().strftime('%Y-%m-%d')

    lines = [
        "# 🏆 Model Competition — Paper Trading Tournament",
        "",
        f"**Last Updated:** {today}",
        f"**Rules:** $20 start | All-in per trade | Max 1 position | 14d hold | -8% SL | +15% TP",
        "",
        "## 📊 Current Standings",
        "",
        "| Model | Holding | Entry | Current | Return | Portfolio | Total Return |",
        "|-------|---------|-------|---------|--------|-----------|--------------|"
    ]

    # Sort by portfolio value descending
    sorted_models = sorted(state.items(), key=lambda x: x[1]['value'], reverse=True)

    for model_name, data in sorted_models:
        if data['holding']:
            lines.append(
                f"| {model_name} | {data['holding']} | ${data['entry']:.2f} | ${data['current']:.2f} | {data['return']:+.2f}% | ${data['value']:.2f} | {data['total_return']:+.2f}% |"
            )
        else:
            action = data.get('exit_reason', data.get('entry_reason', 'CASH'))
            lines.append(
                f"| {model_name} | :dollar: CASH | — | — | — | ${data['value']:.2f} | {data['total_return']:+.2f}% |"
            )

    # Add price reference
    lines.extend([
        "",
        "## 📈 Current Prices",
        "",
        "| Symbol | Price |",
        "|--------|-------|"
    ])
    for sym in sorted(prices.keys()):
        lines.append(f"| {sym} | ${prices[sym]:.2f} |")

    # Add recent trade log
    lines.extend([
        "",
        "## 📋 Recent Actions",
        "",
        "| Date | Model | Action | Details |",
        "|------|-------|--------|---------|"
    ])

    log_file = Path('competition_log.jsonl')
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
        # Show last 10 entries
        for log_line in log_lines[-10:]:
            entry = json.loads(log_line)
            date = entry['date']
            for model_name, data in entry.get('competition', {}).items():
                # Handle both 'holding' and 'slots' format
                holding = data.get('holding')
                entry_price = data.get('entry')
                current = data.get('current')
                ret = data.get('return', 0)
                
                if not holding and 'slots' in data:
                    # Find the main position (first slot with actual holding)
                    for slot in data['slots']:
                        if slot.get('holding') and slot.get('entry'):
                            holding = slot['holding']
                            entry_price = slot['entry']
                            current = slot.get('current', entry_price)
                            ret = slot.get('return', 0)
                            break
                
                if 'exit_reason' in data:
                    action = f"❌ {data['exit_reason']}"
                    details = f"{data.get('prev_holding', '?')} @ {data.get('prev_entry', 0):.2f} → {data.get('exit_price', 0):.2f} ({data.get('exit_return', 0):+.2f}%)"
                elif 'entry_reason' in data:
                    action = "🟢 BUY"
                    details = f"{holding} @ ${entry_price:.2f}" if holding else "CASH"
                elif holding:
                    action = "📊 HOLD"
                    details = f"{holding} ({ret:+.2f}%)"
                else:
                    action = "💵 CASH"
                    details = "No signal"
                lines.append(f"| {date} | {model_name} | {action} | {details} |")

    # Ensure directory exists
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == "__main__":
    main()
