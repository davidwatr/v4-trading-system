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

content = f'# Trading Signals — {today}\\n\\n'
content += '## Buy Signals\\n\\n'
content += '| Symbol | Type | Price | Score | Signals |\\n'
content += '|--------|------|-------|-------|---------|\\n'
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
    content += f'| {symbol} | {sig_type} | {price} | {score} | {signals_str} |\\n'

content += '\\n## Sell Signals\\n\\n'
content += '| Symbol | Type | Price | Score | Signals |\\n'
content += '|--------|------|-------|-------|---------|\\n'
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
    content += f'| {symbol} | {sig_type} | {price} | {score} | {signals_str} |\\n'

content += '\\n## Paper Trades Entered\\n\\n'
content += '| Symbol | Entry Price | Exit Date |\\n'
content += '|--------|-------------|-----------|\\n'
content += '| | | |\\n'

content += '\\n## Notes\\n\\n'
content += 'Used existing screener_results.json (screener.py not found).\\n'

with open(markdown_path, 'w') as f:
    f.write(content)

print(f'Markdown file written to {markdown_path}')