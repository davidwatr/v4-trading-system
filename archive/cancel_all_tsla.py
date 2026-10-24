from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

# Cancel ALL TSLA sell orders
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))
canceled = []
for o in orders:
    if o.symbol == 'TSLA' and o.side == OrderSide.SELL and str(o.status) == 'NEW':
        client.cancel_order_by_id(o.id)
        canceled.append(o.id)
        print(f'Canceled: {o.id}')

print(f'Canceled {len(canceled)} orders')

# Verify
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))
print('\nRemaining TSLA orders:')
for o in orders:
    if o.symbol == 'TSLA':
        print(f'  {o.side} | {o.type} | {o.status} | {o.qty}')
