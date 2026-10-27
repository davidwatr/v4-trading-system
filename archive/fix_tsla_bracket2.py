import os
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide
from alpaca.common.rest import APIError

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

# Cancel ALL existing TSLA sell orders
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderType

orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
for o in orders:
    if o.symbol == 'TSLA' and o.side == OrderSide.SELL:
        client.cancel_order_by_id(o.id)
        print(f'Canceled: {o.type} {o.id}')

# Submit bracket order (Alpaca API v2 with order_class=bracket)
import json
data = {
    'symbol': 'TSLA',
    'qty': 57,
    'side': 'sell',
    'type': 'limit',
    'limit_price': 370.00,
    'time_in_force': 'gtc',
    'order_class': 'bracket',
    'stop_loss': {'stop_price': 326.39},
    'take_profit': {'limit_price': 370.00}
}

try:
    response = client.post('/orders', data)
    print(f'Bracket submitted: {json.dumps(response, indent=2)}')
except APIError as e:
    print(f'Error: {e}')
    # Fallback: just set SL, TP will be handled by cron monitor
    from alpaca.trading.requests import StopOrderRequest
    from alpaca.trading.enums import TimeInForce
    sl = StopOrderRequest(symbol='TSLA', qty=57, side=OrderSide.SELL, type=OrderType.STOP, time_in_force=TimeInForce.GTC, stop_price=326.39)
    client.submit_order(sl)
    print('Fallback: SL set at $326.39, TP handled by cron')

print('Done')
