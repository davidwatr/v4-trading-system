import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100))

print('All TSLA orders:')
for o in orders:
    if o.symbol == 'TSLA':
        print(f'  ID: {o.id}, Side: {o.side}, Type: {o.type}, Status: {o.status}, Qty: {o.qty}, Limit: {o.limit_price}, Stop: {o.stop_price}')
