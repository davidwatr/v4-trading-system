import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))

print('TSLA orders after cancel:')
for o in orders:
    if o.symbol == 'TSLA':
        print(f'  {o.id} | {o.side} | {o.type} | {o.status} | {o.qty} @ {o.limit_price}')
