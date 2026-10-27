import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
import time

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

# Check positions
account = client.get_account()
positions = client.get_all_positions()

print('Current positions:')
for p in positions:
    print(f'  {p.symbol}: {int(p.qty)} shares @ ${float(p.avg_entry_price):.2f}')

# Get TSLA price from Alpaca
try:
    snap = client.get_snapshot('TSLA')
    tsla_price = snap.latest_trade.price
    print(f'\nTSLA price: ${tsla_price:.2f}')
except:
    # Fallback to known price
    tsla_price = 375.0
    print(f'\nTSLA price (fallback): ${tsla_price:.2f}')

# Calculate position size
portfolio = float(account.portfolio_value)
cash = float(account.cash)
print(f'Portfolio: ${portfolio:,.2f}')
print(f'Cash: ${cash:,.2f}')

# Use 20% of portfolio for TSLA
position_value = portfolio * 0.20
tsla_qty = int(position_value / tsla_price)
print(f'Buying TSLA: {tsla_qty} shares (${position_value:,.2f})')

if tsla_qty > 0:
    buy_order = MarketOrderRequest(
        symbol='TSLA',
        qty=tsla_qty,
        side=OrderSide.BUY,
        type=OrderType.MARKET,
        time_in_force=TimeInForce.DAY
    )
    buy = client.submit_order(buy_order)
    print(f'Buy order submitted: {buy.id}')
    
    time.sleep(8)
    buy_check = client.get_order_by_id(buy.id)
    print(f'Buy status: {buy_check.status}')
