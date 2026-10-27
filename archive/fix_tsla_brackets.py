import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

# Cancel the NEW TSLA sell order
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))
for o in orders:
    if o.symbol == 'TSLA' and o.side == OrderSide.SELL and str(o.status) == 'NEW':
        client.cancel_order_by_id(o.id)
        print(f'Canceled TSLA sell: {o.id} @ ${o.limit_price}')

# Now set brackets
positions = client.get_all_positions()
tsla = [p for p in positions if p.symbol == 'TSLA'][0]
qty = int(tsla.qty)
current = float(tsla.current_price)
atr = 18.0
tp = round(current + (2.0 * atr), 2)
sl = round(current - (1.5 * atr), 2)

print(f'TSLA: {qty} shares @ ${current:.2f}')
print(f'TP: ${tp}, SL: ${sl}')

tp_order = LimitOrderRequest(symbol='TSLA', qty=qty, side=OrderSide.SELL, type=OrderType.LIMIT, time_in_force=TimeInForce.GTC, limit_price=tp)
client.submit_order(tp_order)
print(f'TP set: ${tp}')

sl_order = StopOrderRequest(symbol='TSLA', qty=qty, side=OrderSide.SELL, type=OrderType.STOP, time_in_force=TimeInForce.GTC, stop_price=sl)
client.submit_order(sl_order)
print(f'SL set: ${sl}')
