import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

# Cancel existing GOOGL sell orders
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))
for o in orders:
    if o.symbol == 'GOOGL' and o.side == OrderSide.SELL and str(o.status) == 'NEW':
        client.cancel_order_by_id(o.id)
        print(f'Canceled GOOGL sell: {o.id} @ ${o.limit_price}')

print('Done')
