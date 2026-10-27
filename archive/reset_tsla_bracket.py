import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, StopOrderRequest, MarketOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

# Cancel all TSLA sell orders first
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))
for o in orders:
    if o.symbol == 'TSLA' and o.side == OrderSide.SELL and str(o.status) == 'NEW':
        client.cancel_order_by_id(o.id)
        print(f'Canceled: {o.id}')

# Get TSLA position
positions = client.get_all_positions()
tsla = [p for p in positions if p.symbol == 'TSLA'][0]
qty = int(tsla.qty)
current = float(tsla.current_price)

# Sell all shares first
sell = MarketOrderRequest(symbol='TSLA', qty=qty, side=OrderSide.SELL, type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
client.submit_order(sell)
print(f'Sold {qty} TSLA shares')

# Wait for fill
import time
time.sleep(5)

# Re-buy with bracket order
atr = 18.0
tp = round(current + (2.0 * atr), 2)
sl = round(current - (1.5 * atr), 2)

# Calculate position size
account = client.get_account()
portfolio = float(account.portfolio_value)
cash = float(account.cash)
position_value = portfolio * 0.20
buy_qty = int(position_value / current)

buy = MarketOrderRequest(symbol='TSLA', qty=buy_qty, side=OrderSide.BUY, type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
client.submit_order(buy)
print(f'Bought {buy_qty} TSLA shares')

time.sleep(5)

# Now set TP/SL on the new position
positions = client.get_all_positions()
tsla = [p for p in positions if p.symbol == 'TSLA'][0]
new_qty = int(tsla.qty)

tp_order = LimitOrderRequest(symbol='TSLA', qty=new_qty, side=OrderSide.SELL, type=OrderType.LIMIT, time_in_force=TimeInForce.GTC, limit_price=tp)
client.submit_order(tp_order)
print(f'TP set: ${tp}')

sl_order = StopOrderRequest(symbol='TSLA', qty=new_qty, side=OrderSide.SELL, type=OrderType.STOP, time_in_force=TimeInForce.GTC, stop_price=sl)
client.submit_order(sl_order)
print(f'SL set: ${sl}')

print('Done!')
