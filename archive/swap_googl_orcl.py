import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

# Sell GOOGL
positions = client.get_all_positions()
googl = [p for p in positions if p.symbol == 'GOOGL'][0]
qty = int(googl.qty)
sell = MarketOrderRequest(symbol='GOOGL', qty=qty, side=OrderSide.SELL, type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
client.submit_order(sell)
print(f'Sold GOOGL: {qty} shares')

import time
time.sleep(5)

# Buy ORCL
from production_v18 import fetch_data
closes, _, _, _ = fetch_data('ORCL', days=10)
orcl_price = closes[-1]

account = client.get_account()
portfolio = float(account.portfolio_value)
cash = float(account.cash)
position_value = portfolio * 0.20
orcl_qty = int(position_value / orcl_price)

buy = MarketOrderRequest(symbol='ORCL', qty=orcl_qty, side=OrderSide.BUY, type=OrderType.MARKET, time_in_force=TimeInForce.DAY)
client.submit_order(buy)
print(f'Bought ORCL: {orcl_qty} shares @ ${orcl_price:.2f}')

time.sleep(5)

# Set brackets on ORCL
positions = client.get_all_positions()
orcl = [p for p in positions if p.symbol == 'ORCL'][0]
new_qty = int(orcl.qty)
current = float(orcl.current_price)
atr = 5.0
tp = round(current + (2.0 * atr), 2)
sl = round(current - (1.5 * atr), 2)

sl_order = StopOrderRequest(symbol='ORCL', qty=new_qty, side=OrderSide.SELL, type=OrderType.STOP, time_in_force=TimeInForce.GTC, stop_price=sl)
client.submit_order(sl_order)
print(f'SL set: ${sl}')

tp_order = LimitOrderRequest(symbol='ORCL', qty=new_qty, side=OrderSide.SELL, type=OrderType.LIMIT, time_in_force=TimeInForce.GTC, limit_price=tp)
client.submit_order(tp_order)
print(f'TP set: ${tp}')

print('Done!')
