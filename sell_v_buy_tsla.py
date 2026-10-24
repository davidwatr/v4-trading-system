from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
import time

client = TradingClient(
    api_key='PKMJGOE2QUBWHZGTWNT7RZZLNO',
    secret_key='DFKvaZNXLVMNP74k3QRiFwGXw4uJQpLQKYThRpeLsfrn',
    paper=True
)

# Get positions
account = client.get_account()
positions = client.get_all_positions()

v_pos = None
for p in positions:
    if p.symbol == 'V':
        v_pos = p
        break

if not v_pos:
    print('No V position found')
    exit()

qty = int(v_pos.qty)
print(f'Selling V: {qty} shares...')

sell_order = MarketOrderRequest(
    symbol='V',
    qty=qty,
    side=OrderSide.SELL,
    type=OrderType.MARKET,
    time_in_force=TimeInForce.DAY
)
sell = client.submit_order(sell_order)
print(f'Sell order submitted: {sell.id}')

# Wait for fill
time.sleep(8)
sell_check = client.get_order_by_id(sell.id)
print(f'Sell status: {sell_check.status}')

if str(sell_check.status) == 'FILLED':
    # Get TSLA price
    from production_v18 import fetch_data
    closes, _, _, _ = fetch_data('TSLA', days=10)
    tsla_price = closes[-1]
    print(f'TSLA price: ${tsla_price:.2f}')
    
    # Calculate position size (use ~20% of portfolio)
    account = client.get_account()
    portfolio = float(account.portfolio_value)
    cash_after_sell = float(account.cash)
    position_value = portfolio * 0.20
    tsla_qty = int(position_value / tsla_price)
    
    if tsla_qty > 0:
        print(f'Buying TSLA: {tsla_qty} shares @ ${tsla_price:.2f}...')
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
    else:
        print('Insufficient cash to buy TSLA')
else:
    print('Sell did not fill immediately')
    print(f'Sell status: {sell_check.status}')
