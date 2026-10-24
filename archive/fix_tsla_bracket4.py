from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, StopOrderRequest
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

# Submit bracket order using raw API
import json

# Try bracket order with correct structure for SELL (closing long)
bracket_data = {
    'symbol': 'TSLA',
    'qty': '57',
    'side': 'sell',
    'type': 'limit',
    'limit_price': '370.00',
    'time_in_force': 'gtc',
    'order_class': 'bracket',
    'stop_loss': {'stop_price': '326.39'},
    'take_profit': {'limit_price': '370.00'}
}

try:
    response = client.post('/orders', bracket_data)
    print(f'Bracket submitted: {json.dumps(response, indent=2)}')
except Exception as e:
    print(f'Bracket failed: {e}')
    
    # Fallback: OCO order
    oco_data = {
        'symbol': 'TSLA',
        'qty': '57',
        'side': 'sell',
        'type': 'limit',
        'limit_price': '370.00',
        'time_in_force': 'gtc',
        'order_class': 'oco',
        'stop_loss': {'stop_price': '326.39'}
    }
    
    try:
        response = client.post('/orders', oco_data)
        print(f'OCO submitted: TP=$370, SL=$326.39')
    except Exception as e2:
        print(f'OCO failed: {e2}')
        
        # Last resort: just TP, monitor SL via cron
        tp = LimitOrderRequest(symbol='TSLA', qty=57, side=OrderSide.SELL, type=OrderType.LIMIT, time_in_force=TimeInForce.GTC, limit_price=370.0)
        client.submit_order(tp)
        print('Last resort: TP set at $370, SL handled by cron monitor')

print('Done')
