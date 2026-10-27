import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

client = TradingClient(
    api_key=os.environ.get('APCA_API_KEY_ID'),
    secret_key=os.environ.get('APCA_API_SECRET_KEY'),
    paper=True
)

# Account
acc = client.get_account()
print('=== ACCOUNT ===')
print(f'Equity: ${float(acc.equity):,.2f}')
print(f'Cash: ${float(acc.cash):,.2f}')
print(f'Buying Power: ${float(acc.buying_power):,.2f}')
print(f'Portfolio Value: ${float(acc.portfolio_value):,.2f}')

# Positions
print()
print('=== POSITIONS ===')
positions = client.get_all_positions()
for p in positions:
    qty = float(p.qty)
    entry = float(p.avg_entry_price) if p.avg_entry_price else 0
    price = float(p.current_price) if p.current_price else 0
    pl = float(p.unrealized_pl) if p.unrealized_pl else 0
    plpc = float(p.unrealized_plpc) * 100 if p.unrealized_plpc else 0
    print(f'{p.symbol:6s} | {qty:4.0f} sh | Avg ${entry:7.2f} | Cur ${price:7.2f} | P&L ${pl:+8.2f} ({plpc:+.1f}%)')

# Orders
print()
print('=== ORDERS ===')
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
for o in orders:
    qty_str = str(o.qty)
    price = float(o.limit_price or o.stop_price or 0)
    print(f'{o.side.value:4s} {qty_str:>4s} {o.symbol:6s} @ ${price:7.2f} [{o.type.value}] ({o.status.value})')
