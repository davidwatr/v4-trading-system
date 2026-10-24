from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))

print('GOOGL orders:')
for o in orders:
    if o.symbol == 'GOOGL':
        print(f'  ID: {o.id}, Side: {o.side}, Type: {o.type}, Status: {o.status}, Qty: {o.qty}, Limit: {o.limit_price}')
