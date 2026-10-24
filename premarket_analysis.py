from production_v18 import fetch_data, get_market_regime, calc_atr
import json
from datetime import datetime

spy_closes, _, spy_highs, spy_lows = fetch_data('SPY', days=730)
if spy_closes is None:
    print("ERROR: Could not fetch SPY data")
    exit(1)

result = get_market_regime(spy_closes, spy_highs, spy_lows)
regime = result[0]
penalty = result[1]
max_pos = result[2]
kelly_mult = result[3]
atr_val = calc_atr(spy_highs, spy_lows, spy_closes)

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print('='*50)
print('PRE-MARKET ANALYSIS')
print('='*50)
print(f'Timestamp: {now}')
print(f'Market Regime: {regime}')
print(f'Max Positions: {max_pos}')
print(f'Kelly Multiplier: {kelly_mult}x')
print(f'SPY Price: ${spy_closes[-1]:.2f}')
print(f'ATR(14): {atr_val:.2f}')
print('='*50)

# Log to file
log_path = r"C:\Users\david_wncs0ps\OneDrive\Documents\hermes and i\Trading\premarket_log.md"
log_entry = f"""

## {now}
- **Market Regime**: {regime}
- **Max Positions**: {max_pos}
- **Kelly Multiplier**: {kelly_mult}x
- **SPY Price**: ${spy_closes[-1]:.2f}
- **ATR(14)**: {atr_val:.2f}
"""

try:
    with open(log_path, 'a') as f:
        f.write(log_entry)
    print(f"Logged to: {log_path}")
except Exception as e:
    print(f"Logging error: {e}")
