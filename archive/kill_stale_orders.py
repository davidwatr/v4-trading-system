#!/usr/bin/env python3
"""
Emergency Kill Script — Cancel all stale orders and reset bracket logic.
Run this first to unblock the executor.
"""
import os, sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderStatus

def load_env():
    env_file = __import__('pathlib').Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

client = TradingClient(
    api_key=os.environ.get("APCA_API_KEY_ID", ""),
    secret_key=os.environ.get("APCA_API_SECRET_KEY", ""),
    paper=True
)

print("=" * 60)
print("EMERGENCY KILL — Cancelling all stale orders")
print("=" * 60)

# Get all open orders
request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50)
orders = client.get_orders(request)

print(f"\nFound {len(orders)} open orders:")
for o in orders:
    print(f"  {o.symbol:6s} | {str(o.side):4s} | {str(o.type):6s} | Qty: {str(o.qty):4s} | Status: {o.status}")

# Cancel all
cancelled = 0
for o in orders:
    try:
        client.cancel_order_by_id(o.id)
        cancelled += 1
        print(f"  ✓ Cancelled {o.symbol} {o.type} order")
    except Exception as e:
        print(f"  ✗ Failed to cancel {o.symbol}: {e}")
    time.sleep(0.2)

print(f"\nCancelled {cancelled}/{len(orders)} orders")

# Show positions
positions = client.get_all_positions()
print(f"\nCurrent positions ({len(positions)}):")
for p in positions:
    print(f"  {p.symbol:6s} | {int(p.qty):4d} shares | Entry: ${float(p.avg_entry_price):,.2f} | Current: ${float(p.current_price):,.2f} | PnL: ${float(p.unrealized_pl):+,.2f}")

# Show account
account = client.get_account()
print(f"\nAccount Summary:")
print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
print(f"  Cash:            ${float(account.cash):,.2f}")
print(f"  Buying Power:    ${float(account.buying_power):,.2f}")

print("\n" + "=" * 60)
print("DONE — Executor should now be able to submit new brackets")
print("=" * 60)
