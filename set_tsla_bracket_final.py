from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

positions = client.get_all_positions()
tsla = [p for p in positions if p.symbol == 'TSLA'][0]
qty = int(tsla.qty)
current = float(tsla.current_price)
atr = 18.0
tp = round(current + (2.0 * atr), 2)
sl = round(current - (1.5 * atr), 2)

print(f'TSLA: {qty} shares @ ${current:.2f}')
print(f'TP: ${tp}, SL: ${sl}')

# Set SL first (doesn't hold shares until triggered)
sl_order = StopOrderRequest(symbol='TSLA', qty=qty, side=OrderSide.SELL, type=OrderType.STOP, time_in_force=TimeInForce.GTC, stop_price=sl)
client.submit_order(sl_order)
print(f'SL set: ${sl}')

# Set TP (limit order holds shares)
tp_order = LimitOrderRequest(symbol='TSLA', qty=qty, side=OrderSide.SELL, type=OrderType.LIMIT, time_in_force=TimeInForce.GTC, limit_price=tp)
client.submit_order(tp_order)
print(f'TP set: ${tp}')

print('Done!')
