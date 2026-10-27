import json
import os
from datetime import datetime

with open('screener_results.json') as f:
    data = json.load(f)

buy_signals = data.get('buy_signals', [])
sell_signals = data.get('sell_signals', [])

today = datetime.now().strftime('%Y-%m-%d')
vault_dir = r'C:/Users/david_wncs0ps/OneDrive/Documents/hermes and i/Trading'
os.makedirs(vault_dir, exist_ok=True)
markdown_path = os.path.join(vault_dir, f'{today}.md')

lines = []
lines.append(f'# Trading Signals — {today}')
lines.append('')
lines.append('## Buy Signals')
lines.append('')
lines.append('| Symbol | Type | Price | Score | Signals |')
lines.append('|--------|------|-------|-------|---------|')
for s in buy_signals:
    symbol = s.get('symbol', 'N/A')
    sig_type = s.get('type', 'N/A')
    price = s.get('price', 'N/A')
    score = s.get('score', 'N/A')
    signals = s.get('signals', [])
    if isinstance(signals, list):
        signals_str = ', '.join(str(sig) for sig in signals)
    else:
        signals_str = str(signals)
    lines.append(f'| {symbol} | {sig_type} | {price:.2f} | {score} | {signals_str} |')

lines.append('')
lines.append('## Sell Signals')
lines.append('')
lines.append('| Symbol | Type | Price | Score | Signals |')
lines.append('|--------|------|-------|-------|---------|')
for s in sell_signals:
    symbol = s.get('symbol', 'N/A')
    sig_type = s.get('type', 'N/A')
    price = s.get('price', 'N/A')
    score = s.get('score', 'N/A')
    signals = s.get('signals', [])
    if isinstance(signals, list):
        signals_str = ', '.join(str(sig) for sig in signals)
    else:
        signals_str = str(signals)
    lines.append(f'| {symbol} | {sig_type} | {price:.2f} | {score} | {signals_str} |')

lines.append('')
lines.append('## Paper Trades Entered')
lines.append('')
lines.append('| Symbol | Entry Price | Exit Date |')
lines.append('|--------|-------------|-----------|')
lines.append('| | | |')

lines.append('')
lines.append('## Notes')
lines.append('')
lines.append(f'- Data source: screener_results.json (timestamp: {data.get("timestamp", "unknown")})')
lines.append('- production_v18.py has a feature mismatch bug (22 vs 24 features) — screener.py not found in directory')
lines.append('- Market regime: bull (from last successful run)')
lines.append('- Top buy: PYPL (score 7, oversold + support + volume spike)')
lines.append('- Top sell: TSLA (score -6, overbought + resistance)')

content = '\n'.join(lines) + '\n'

with open(markdown_path, 'w') as f:
    f.write(content)

print(f'Markdown file written to {markdown_path}')
print(f'Buy signals: {len(buy_signals)}')
print(f'Sell signals: {len(sell_signals)}')
