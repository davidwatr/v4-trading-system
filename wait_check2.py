import os
import time
time.sleep(20)

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

positions = client.get_all_positions()
print('Positions:')
for p in positions:
    print(f'  {p.symbol}: {int(p.qty)} shares @ ${float(p.avg_entry_price):.2f}')

orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=10))
print('\nRecent orders:')
for o in orders:
    print(f'  {o.symbol} | {o.side} | {o.type} | {o.status} | {o.qty}')
