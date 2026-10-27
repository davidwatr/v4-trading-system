import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
import time

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

# Get TSLA position
positions = client.get_all_positions()
tsla_pos = None
for p in positions:
    if p.symbol == 'TSLA':
        tsla_pos = p
        break

if not tsla_pos:
    print('No TSLA position found')
    exit()

qty = int(tsla_pos.qty)
entry = float(tsla_pos.avg_entry_price)
current = float(tsla_pos.current_price)

# Calculate TP/SL
atr = 18.0  # approximate ATR for TSLA
take_profit = round(current + (2.0 * atr), 2)
stop_loss = round(current - (1.5 * atr), 2)

print(f'TSLA: {qty} shares @ ${entry:.2f}')
print(f'Current: ${current:.2f}')
print(f'Take-profit: ${take_profit:.2f}')
print(f'Stop-loss: ${stop_loss:.2f}')

# Submit take-profit limit order
tp_order = LimitOrderRequest(
    symbol='TSLA',
    qty=qty,
    side=OrderSide.SELL,
    type=OrderType.LIMIT,
    time_in_force=TimeInForce.GTC,
    limit_price=take_profit
)
tp = client.submit_order(tp_order)
print(f'\nTP order submitted: {tp.id} @ ${take_profit:.2f}')

time.sleep(2)

# Submit stop-loss stop order
sl_order = StopOrderRequest(
    symbol='TSLA',
    qty=qty,
    side=OrderSide.SELL,
    type=OrderType.STOP,
    time_in_force=TimeInForce.GTC,
    stop_price=stop_loss
)
sl = client.submit_order(sl_order)
print(f'SL order submitted: {sl.id} @ ${stop_loss:.2f}')

print('\nBracket orders set on TSLA!')
