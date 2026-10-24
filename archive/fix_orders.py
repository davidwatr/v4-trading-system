from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, StopLimitOrderRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide, TimeInForce

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

# Cancel all open orders
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=20))
for o in orders:
    client.cancel_order_by_id(o.id)
    print(f'Cancelled: {o.symbol} {o.side} {o.type}')

# Place fresh stops
stops = [
    ('NVDA', 32, 210, 209.95),
    ('GS', 1, 980, 979.95),
    ('MS', 7, 205, 204.95),
]

for symbol, qty, stop, limit in stops:
    try:
        order = client.submit_order(StopLimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            stop_price=stop,
            limit_price=limit,
            time_in_force=TimeInForce.DAY
        ))
        print(f'New {symbol} stop: stop=${stop} limit=${limit}')
    except Exception as e:
        print(f'{symbol} error: {e}')

print('Done')
