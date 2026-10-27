#!/usr/bin/env python3
"""
Small Cap Alpaca Executor — bridges the small cap scanner with real Alpaca paper trading.
Reads small_cap_signals.json, submits real bracket orders via Alpaca.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
    TakeProfitRequest, StopLossRequest, GetOrdersRequest, OrderClass
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus

# ============================================================
# CONFIGURATION
# ============================================================
APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY", "")

SIGNAL_FILE = Path("small_cap_signals.json")
PORTFOLIO_FILE = Path("small_cap_alpaca_portfolio.json")
MAX_POSITIONS = 3
RISK_PER_TRADE = 0.05  # 5% risk per trade (more aggressive for small caps)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('small_cap_alpaca.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# ALPACA CLIENT
# ============================================================
class SmallCapAlpacaExecutor:
    def __init__(self):
        if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
            raise ValueError("Alpaca API keys not configured")
        
        self.client = TradingClient(
            api_key=APCA_API_KEY_ID,
            secret_key=APCA_API_SECRET_KEY,
            paper=True
        )
        
        try:
            account = self.client.get_account()
            logger.info(f"Connected to Alpaca Paper Trading")
            logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            raise
    
    def get_account(self):
        return self.client.get_account()
    
    def get_positions(self):
        return self.client.get_all_positions()
    
    def get_position_count(self):
        return len(self.get_positions())
    
    def get_orders(self, status=QueryOrderStatus.OPEN, limit=50):
        try:
            request = GetOrdersRequest(status=status, limit=limit)
            return self.client.get_orders(request)
        except:
            return []
    
    def cancel_stale_orders(self, max_age_hours=24):
        """Cancel stale orders."""
        try:
            orders = self.get_orders(status=QueryOrderStatus.OPEN)
            cancelled = 0
            for order in orders:
                if order.submitted_at:
                    age_hours = (datetime.now(order.submitted_at.tzinfo) - order.submitted_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        try:
                            self.client.cancel_order_by_id(order.id)
                            cancelled += 1
                        except:
                            pass
            return cancelled
        except:
            return 0
    
    def submit_bracket_order(self, symbol, entry_price, stop_loss, take_profit, shares):
        """Submit a bracket order."""
        try:
            # Check if we already have this position
            positions = self.get_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    logger.warning(f"Already have position in {symbol}. Skipping.")
                    return None
            
            # Check position count
            if self.get_position_count() >= MAX_POSITIONS:
                logger.warning(f"Max positions ({MAX_POSITIONS}) reached. Skipping {symbol}.")
                return None
            
            shares = int(shares)
            if shares <= 0:
                logger.warning(f"Invalid share count for {symbol}: {shares}")
                return None
            
            entry_price = round(entry_price, 2)
            stop_loss = round(stop_loss, 2)
            take_profit = round(take_profit, 2)
            
            # Safety check
            if take_profit <= entry_price:
                take_profit = round(entry_price * 1.05, 2)
            if stop_loss >= entry_price:
                stop_loss = round(entry_price * 0.95, 2)
            
            bracket_order = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit),
                stop_loss=StopLossRequest(stop_price=stop_loss, limit_price=stop_loss)
            )
            
            order = self.client.submit_order(bracket_order)
            logger.info(f"BRACKET ORDER: {symbol} | Qty: {shares} | TP: ${take_profit} | SL: ${stop_loss}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to submit bracket order for {symbol}: {e}")
            return None
    
    def close_position(self, symbol):
        try:
            order = self.client.close_position(symbol)
            logger.info(f"Closed position in {symbol}")
            return order
        except Exception as e:
            logger.error(f"Failed to close position in {symbol}: {e}")
            return None

# ============================================================
# PORTFOLIO TRACKER
# ============================================================
class PortfolioTracker:
    def __init__(self):
        self.portfolio = self.load_portfolio()
    
    def load_portfolio(self):
        if PORTFOLIO_FILE.exists():
            with open(PORTFOLIO_FILE) as f:
                return json.load(f)
        return {
            "start_date": datetime.now().isoformat(),
            "initial_balance": 0,
            "current_balance": 0,
            "positions": [],
            "closed_trades": []
        }
    
    def save_portfolio(self):
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(self.portfolio, f, indent=2)

# ============================================================
# MAIN EXECUTOR
# ============================================================
def main():
    print("=" * 60)
    print("  SMALL CAP ALPACA EXECUTOR")
    print("=" * 60)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        executor = SmallCapAlpacaExecutor()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return
    
    account = executor.get_account()
    portfolio_value = float(account.portfolio_value)
    buying_power = float(account.buying_power)
    
    print(f"\n  Portfolio Value: ${portfolio_value:,.2f}")
    print(f"  Buying Power: ${buying_power:,.2f}")
    
    # Cancel stale orders
    cancelled = executor.cancel_stale_orders(max_age_hours=4)
    print(f"  Cancelled {cancelled} stale orders")
    
    # Read signals from scanner
    if not SIGNAL_FILE.exists():
        print("  No signals file found. Run small_cap_scanner.py first.")
        return
    
    with open(SIGNAL_FILE) as f:
        signals = json.load(f)
    
    print(f"  Found {len(signals)} signals from scanner")
    
    # Get current positions
    positions = executor.get_positions()
    existing_symbols = {p.symbol for p in positions}
    
    print(f"  Current positions: {len(positions)} ({', '.join(existing_symbols) if existing_symbols else 'none'})")
    
    # Execute trades
    executed = 0
    skipped = 0
    
    for signal in signals:
        if executed >= MAX_POSITIONS - len(positions):
            break
        
        symbol = signal.get("symbol")
        if not symbol or symbol in existing_symbols:
            continue
        
        entry = signal.get("entry")
        stop = signal.get("stop")
        target = signal.get("target_1")
        
        if not all([entry, stop, target]):
            continue
        
        # Calculate position size (5% of portfolio)
        risk_amount = portfolio_value * RISK_PER_TRADE
        risk_per_share = entry - stop
        if risk_per_share <= 0:
            continue
        
        shares = int(risk_amount / risk_per_share)
        if shares <= 0:
            continue
        
        # Check buying power
        cost = shares * entry
        if cost > buying_power:
            # Try smaller size
            shares = int(buying_power / entry * 0.95)
            if shares <= 0:
                skipped += 1
                continue
        
        order = executor.submit_bracket_order(symbol, entry, stop, target, shares)
        if order:
            executed += 1
            print(f"  ✓ {symbol}: {shares} shares | Entry: ${entry} | SL: ${stop} | TP: ${target}")
        else:
            skipped += 1
    
    print(f"\n  Executed: {executed} | Skipped: {skipped}")
    
    # Show updated positions
    positions = executor.get_positions()
    if positions:
        print(f"\n  CURRENT POSITIONS:")
        for pos in positions:
            print(f"    {pos.symbol:6s} | {int(pos.qty):4d} shares | Entry: ${float(pos.avg_entry_price):,.2f} | PnL: ${float(pos.unrealized_pl):+,.2f}")
    
    print(f"\n{'=' * 60}")
    print(f"  EXECUTION COMPLETE")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
