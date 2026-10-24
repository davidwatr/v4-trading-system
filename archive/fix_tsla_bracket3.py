from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest, LimitOrderRequest, GetOrdersRequest
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

# Set SL first (this holds the shares)
sl = StopOrderRequest(symbol='TSLA', qty=57, side=OrderSide.SELL, type=OrderType.STOP, time_in_force=TimeInForce.GTC, stop_price=326.39)
client.submit_order(sl)
print('SL set: $326.39')

# Now use OCO order for TP (One-Cancels-Other - will auto-cancel if SL hits)
# OCO requires qty and limit_price
oco_data = {
    'symbol': 'TSLA',
    'qty': 57,
    'side': 'sell',
    'type': 'limit',
    'limit_price': 370.00,
    'time_in_force': 'gtc',
    'order_class': 'oco',
    'stop_loss': {'stop_price': 326.39}
}

try:
    response = client.post('/orders', oco_data)
    print(f'OCO submitted: TP=$370, SL=$326.39')
except Exception as e:
    print(f'OCO failed: {e}')
    # Fallback: just TP with no SL hold
    tp = LimitOrderRequest(symbol='TSLA', qty=57, side=OrderSide.SELL, type=OrderType.LIMIT, time_in_force=TimeInForce.GTC, limit_price=370.0)
    client.submit_order(tp)
    print('Fallback: TP set at $370')

print('Done')
