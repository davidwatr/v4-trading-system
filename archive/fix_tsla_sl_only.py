from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

# Cancel ALL existing TSLA sell orders
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
for o in orders:
    if o.symbol == 'TSLA' and o.side == OrderSide.SELL:
        client.cancel_order_by_id(o.id)
        print(f'Canceled: {o.type} {o.id}')

# Set SL only
sl = StopOrderRequest(symbol='TSLA', qty=57, side=OrderSide.SELL, type=OrderType.STOP, time_in_force=TimeInForce.GTC, stop_price=326.39)
client.submit_order(sl)
print('SL set: $326.39')

print('Done')
