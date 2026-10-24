from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

# 1. Cancel ALL existing TSLA sell orders
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
for o in orders:
    if o.symbol == 'TSLA' and o.side == OrderSide.SELL:
        client.cancel_order_by_id(o.id)
        print(f'Canceled: {o.type} {o.id}')

# 2. Submit TP only (Alpaca won't allow both SL and LIMIT simultaneously on same shares)
tp = LimitOrderRequest(symbol='TSLA', qty=57, side=OrderSide.SELL, type=OrderType.LIMIT, time_in_force=TimeInForce.GTC, limit_price=370.0)
client.submit_order(tp)
print('TP set: $370.00')

print('Done!')
