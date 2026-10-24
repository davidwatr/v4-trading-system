#!/usr/bin/env python3
"""
Mid-Day Portfolio Monitor — checks open positions, PnL, and alerts if something's wrong.
US market hours: 9:30 AM - 4:00 PM Eastern = 16:30 - 23:00 Israel.
Runs 2x during trading day to catch issues early.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Load Alpaca keys
env_file = Path(__file__).parent / '.env'
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY", "")
APCA_PAPER = True

def main():
    print("=" * 60)
    print("  MID-DAY PORTFOLIO MONITOR")
    print("  Time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 60)
    
    client = TradingClient(api_key=APCA_API_KEY_ID, secret_key=APCA_API_SECRET_KEY, paper=APCA_PAPER)
    account = client.get_account()
    
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    
    print(f"\nPortfolio: ${portfolio_value:,.2f}")
    print(f"Cash: ${cash:,.2f}")
    
    # Get positions
    positions = client.get_all_positions()
    print(f"\nOpen Positions: {len(positions)}")
    print("-" * 60)
    
    alerts = []
    
    for pos in positions:
        symbol = pos.symbol
        qty = int(pos.qty)
        entry = float(pos.avg_entry_price)
        current = float(pos.current_price)
        unrealized = float(pos.unrealized_pl)
        pnl_pct = (unrealized / (entry * qty)) * 100 if qty > 0 else 0
        
        # Check bracket orders (alpaca-py v1 uses params dict)
        from alpaca.trading.requests import GetOrdersRequest
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
        symbol_orders = [o for o in orders if o.symbol == symbol]
        
        status = "✅"
        if pnl_pct < -3:
            status = "🔴 LOSS >3%"
            alerts.append(f"{symbol} down {pnl_pct:.1f}% — consider review")
        elif pnl_pct > 5:
            status = "🟢 WIN >5%"
        
        print(f"{symbol:6s} | {qty:3d} shares | Entry: ${entry:.2f} | Current: ${current:.2f} | PnL: ${unrealized:+,.2f} ({pnl_pct:+.1f}%) | {len(symbol_orders)} open orders | {status}")
    
    # Check stale orders (alpaca-py v1 uses params dict)
    from alpaca.trading.requests import GetOrdersRequest
    open_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=100))
    stale = []
    for order in open_orders:
        if order.submitted_at:
            age_hours = (datetime.now(order.submitted_at.tzinfo) - order.submitted_at).total_seconds() / 3600
            if age_hours > 6:
                stale.append(f"{order.symbol} ({age_hours:.1f}h old)")
    
    if stale:
        print(f"\n⚠️  STALE ORDERS: {', '.join(stale)}")
        alerts.append(f"Stale orders: {', '.join(stale)}")
    
    # Summary
    print("\n" + "=" * 60)
    if alerts:
        print("⚠️  ALERTS:")
        for a in alerts:
            print(f"  - {a}")
    else:
        print("✅ All positions healthy. No alerts.")
    print("=" * 60)

if __name__ == "__main__":
    main()
