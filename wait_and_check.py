import os
import time
time.sleep(15)

from alpaca.trading.client import TradingClient

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

positions = client.get_all_positions()
print('Positions:')
for p in positions:
    print(f'  {p.symbol}: {int(p.qty)} shares @ ${float(p.avg_entry_price):.2f}')
