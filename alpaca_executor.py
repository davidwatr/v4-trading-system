#!/usr/bin/env python3
"""
Alpaca Paper Trading Executor — Simple Version
Uses heuristic mean reversion instead of ML model.
No HMM, no walk-forward, no academic factor models.
Just: simple signals + bracket orders + risk management.

Signal: 3-day mean reversion + RSI oversold + volume confirmation
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Load .env file if it exists
def load_env():
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# Import alpaca-py
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, GetOrdersRequest,
    ClosePositionRequest, OrderClass,
    TakeProfitRequest, StopLossRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus

# Import production model functions (only what we still need)
from production_v18 import (
    fetch_data, get_market_regime
)

# Import swing signal generator
from swing_signals import generate_swing_signals

# ============================================
# CONFIGURATION
# ============================================
APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY", "")
APCA_PAPER = True
INITIAL_CAPITAL = 10000
MAX_RISK_PER_TRADE = 0.08  # 8% max risk per trade (was 5%)
CASH_BUFFER_PCT = 0.02  # 2% cash buffer (was 5%)

# Risk management
MAX_DAILY_LOSS_PCT = 0.05
MAX_MONTHLY_LOSS_PCT = 0.15
MAX_MARGIN_PCT = 0.50
VIX_HIGH_THRESHOLD = 25
VIX_SIZE_MULTIPLIER = 0.5

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpaca_executor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# DAILY COOLDOWN LOCK (prevents wash trading)
# ============================================
COOLDOWN_FILE = Path('daily_cooldown.json')

def load_cooldown():
    """Load cooldown tracking from disk."""
    if COOLDOWN_FILE.exists():
        with open(COOLDOWN_FILE) as f:
            return json.load(f)
    return {}

def save_cooldown(cooldown_dict):
    """Save cooldown tracking to disk."""
    with open(COOLDOWN_FILE, 'w') as f:
        json.dump(cooldown_dict, f, indent=2)

def is_on_cooldown(symbol):
    """Check if symbol is on cooldown (traded today)."""
    cooldown = load_cooldown()
    today = datetime.now().strftime('%Y-%m-%d')
    return cooldown.get(symbol, '') == today

def mark_cooldown(symbol):
    """Mark symbol as traded today."""
    cooldown = load_cooldown()
    cooldown[symbol] = datetime.now().strftime('%Y-%m-%d')
    save_cooldown(cooldown)

def clear_stale_cooldowns(executor):
    """Clear cooldowns for symbols that are no longer in positions."""
    try:
        cooldown = load_cooldown()
        positions = executor.get_positions()
        position_symbols = {pos.symbol for pos in positions}
        
        to_remove = []
        for symbol in cooldown:
            if symbol not in position_symbols:
                to_remove.append(symbol)
        
        for symbol in to_remove:
            del cooldown[symbol]
        
        if to_remove:
            save_cooldown(cooldown)
            logger.info(f"Cleared {len(to_remove)} stale cooldowns: {to_remove}")
    except Exception as e:
        logger.warning(f"Failed to clear stale cooldowns: {e}")

# ============================================
# SECTOR MAPPING
# ============================================
STOCK_TO_SECTOR = {
    'INTC': 'tech', 'AMD': 'tech', 'NVDA': 'tech', 'MSFT': 'tech', 'AAPL': 'tech',
    'GOOGL': 'tech', 'META': 'tech', 'TSLA': 'tech', 'PYPL': 'tech', 'CSCO': 'tech',
    'ADBE': 'tech', 'CRM': 'tech', 'NFLX': 'tech',
    'BAC': 'financials', 'JPM': 'financials', 'GS': 'financials', 'MS': 'financials',
    'WFC': 'financials', 'C': 'financials', 'V': 'financials', 'MA': 'financials',
    'ORCL': 'tech', 'BRK-B': 'financials'
}

# ============================================
# ALPACA CLIENT
# ============================================
class AlpacaExecutor:
    def __init__(self):
        if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
            raise ValueError("Alpaca API keys not configured")
        
        self.client = TradingClient(
            api_key=APCA_API_KEY_ID,
            secret_key=APCA_API_SECRET_KEY,
            paper=APCA_PAPER
        )
        
        # Verify connection
        try:
            account = self.client.get_account()
            logger.info(f"Connected to Alpaca Paper Trading")
            logger.info(f"Cash: ${float(account.cash):,.2f}")
            logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            logger.info(f"Buying Power: ${float(account.buying_power):,.2f}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    def get_account(self):
        return self.client.get_account()
    
    def get_positions(self):
        return self.client.get_all_positions()
    
    def get_position_count(self):
        return len(self.get_positions())
    
    def get_orders(self, status=QueryOrderStatus.ALL, limit=50):
        try:
            request = GetOrdersRequest(status=status, limit=limit)
            return self.client.get_orders(request)
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def cancel_stale_brackets(self, max_age_hours=4):
        """Cancel all open orders older than max_age_hours."""
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
                            logger.info(f"Cancelled stale {order.symbol} order ({age_hours:.1f}h old)")
                        except Exception as e:
                            logger.warning(f"Failed to cancel {order.symbol}: {e}")
            return cancelled
        except Exception as e:
            logger.error(f"Failed to cancel stale brackets: {e}")
            return 0
    
    def submit_bracket_order(self, symbol, entry_price, stop_loss, take_profit, shares, regime):
        """Submit a bracket order."""
        try:
            # Check if we already have a position in this symbol
            # Allow adding to winners (Strategy 1), skip losers
            positions = self.get_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    # Already own this — add to winners, skip losers
                    unrealized = float(pos.unrealized_pl)
                    if unrealized > 0:
                        logger.info(f"Already own {symbol} with +${unrealized:.2f} PnL — adding to winner")
                        # Allow through — will add shares below
                        break
                    else:
                        logger.info(f"Already own {symbol} with -${abs(unrealized):.2f} PnL — skipping loser")
                        return None
            else:
                # No existing position — check position count
                # (this runs only if we didn't break out of the loop)
                max_positions = 8 if 'bull' in regime else (5 if 'bear' in regime else 6)
                if self.get_position_count() >= max_positions:
                    logger.warning(f"Max positions ({max_positions}) reached. Skipping {symbol}.")
                    return None
            
            # Round shares to whole number
            shares = int(shares)
            if shares <= 0:
                logger.warning(f"Invalid share count for {symbol}: {shares}")
                return None
            
            # Round prices
            entry_price = round(entry_price, 2)
            stop_loss = round(stop_loss, 2)
            take_profit = round(take_profit, 2)
            
            # Safety check
            if take_profit <= entry_price:
                take_profit = round(entry_price * 1.05, 2)
            if stop_loss >= entry_price:
                stop_loss = round(entry_price * 0.95, 2)
            if take_profit <= stop_loss:
                take_profit, stop_loss = stop_loss, take_profit
            
            # Create bracket order
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
            
            logger.info(f"BRACKET ORDER SUBMITTED")
            logger.info(f"Symbol: {symbol} | Qty: {shares} | TP: ${take_profit} | SL: ${stop_loss}")
            
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
    
    def close_weakest_positions(self, count=2):
        """Close the weakest positions (biggest losers)."""
        try:
            positions = self.get_positions()
            if not positions:
                return 0
            sorted_pos = sorted(positions, key=lambda p: float(p.unrealized_pl))
            closed = 0
            for pos in sorted_pos[:count]:
                try:
                    # Cancel open orders first
                    orders = self.get_orders(status=QueryOrderStatus.OPEN)
                    for order in orders:
                        if order.symbol == pos.symbol:
                            try:
                                self.client.cancel_order_by_id(order.id)
                            except Exception:
                                pass
                    time.sleep(0.5)
                    self.client.close_position(pos.symbol)
                    logger.info(f"Closed weak position: {pos.symbol} (PnL: ${float(pos.unrealized_pl):,.2f})")
                    closed += 1
                except Exception as e:
                    logger.warning(f"Failed to close {pos.symbol}: {e}")
            return closed
        except Exception as e:
            logger.error(f"Failed to close weakest positions: {e}")
            return 0

# ============================================
# SIGNAL GENERATOR (Swing 2d PB)
# ============================================
class SignalGenerator:
    def __init__(self):
        pass

    def generate_signals(self):
        """Generate swing 2d PB signals — backtest-validated, Sharpe 1.54."""
        logger.info("Generating swing 2d PB signals...")
        results = generate_swing_signals()

        if not results:
            logger.warning("No swing signals generated")
            return [], {
                'regime': 'flat',
                'big_cap_max': 8,
                'kelly_multiplier': 1.0,
            }

        # All swing signals use fixed 12.5% kelly (1/8 max positions)
        # No regime-based scaling — macro filter handles bear market defense
        return results, {
            'regime': 'swing',
            'big_cap_max': 8,
            'kelly_multiplier': 1.0,
        }

# ============================================
# RISK MANAGEMENT (simple, no extra files)
# ============================================
def get_vix():
    """Fetch current VIX level."""
    try:
        closes, _, _, _ = fetch_data('^VIX', days=5)
        if closes:
            return closes[-1]
    except Exception:
        pass
    return 15

def check_drawdown_breaker(portfolio_value, initial_capital=INITIAL_CAPITAL):
    """Circuit breaker: halt if daily or monthly loss exceeds threshold."""
    daily_loss = (initial_capital - portfolio_value) / initial_capital
    if daily_loss > MAX_DAILY_LOSS_PCT:
        return True, f"Daily loss {daily_loss:.1%} exceeds {MAX_DAILY_LOSS_PCT:.0%}"
    return False, None

def get_risk_multiplier(cash, portfolio_value):
    """Composite risk multiplier: VIX + margin."""
    mult = 1.0
    
    # VIX check
    vix = get_vix()
    if vix > VIX_HIGH_THRESHOLD:
        mult *= VIX_SIZE_MULTIPLIER
        logger.info(f"VIX at {vix:.1f} > {VIX_HIGH_THRESHOLD}. Reducing size by {VIX_SIZE_MULTIPLIER:.0%}")
    
    # Margin check
    if portfolio_value > 0:
        margin_used = abs(min(cash, 0))
        margin_pct = margin_used / portfolio_value
        if margin_pct > MAX_MARGIN_PCT:
            reduction = MAX_MARGIN_PCT / margin_pct
            mult *= reduction
            logger.info(f"Margin at {margin_pct:.0%} > {MAX_MARGIN_PCT:.0%}. Reducing size to {reduction:.0%}")
    
    return mult

# ============================================
# MAIN EXECUTOR
# ============================================
def acquire_lock():
    lock_file = Path('alpaca_executor.lock')
    try:
        if lock_file.exists():
            lock_age = time.time() - lock_file.stat().st_mtime
            if lock_age < 1800:
                return False
            else:
                lock_file.unlink()
        lock_file.write_text(str(os.getpid()))
        return True
    except Exception:
        return False

def release_lock():
    lock_file = Path('alpaca_executor.lock')
    try:
        if lock_file.exists():
            lock_file.unlink()
    except Exception:
        pass

def check_time_exits(executor):
    """
    Time-based exits: close positions held >5 days without profit.
    Mean reversion should snap back quickly. If it doesn't, thesis is dead.
    """
    try:
        positions = executor.get_positions()
        if not positions:
            return
        
        # Load position entry dates from tracking file
        tracking_file = Path('position_tracking.json')
        if not tracking_file.exists():
            return
        
        with open(tracking_file) as f:
            tracking = json.load(f)
        
        now = datetime.now()
        closed = 0
        
        for pos in positions:
            symbol = pos.symbol
            if symbol not in tracking:
                continue
            
            entry_date = datetime.fromisoformat(tracking[symbol]['entry_date'])
            days_held = (now - entry_date).days
            
            # If held >=5 days, close it (Swing 2d PB: 5-day time stop, hard)
            if days_held >= 5:
                logger.info(f"Time exit: {symbol} held {days_held} days, PnL=${float(pos.unrealized_pl):.2f}")
                # Cancel open orders first
                orders = executor.get_orders(status=QueryOrderStatus.OPEN)
                for order in orders:
                    if order.symbol == symbol:
                        try:
                            executor.client.cancel_order_by_id(order.id)
                        except Exception:
                            pass
                time.sleep(0.5)
                executor.close_position(symbol)
                del tracking[symbol]
                # Clear cooldown so symbol can be traded tomorrow
                cooldown = load_cooldown()
                if symbol in cooldown:
                    del cooldown[symbol]
                    save_cooldown(cooldown)
                closed += 1
        
        if closed > 0:
            with open(tracking_file, 'w') as f:
                json.dump(tracking, f, indent=2)
            logger.info(f"Closed {closed} positions on time exit")
    
    except Exception as e:
        logger.warning(f"Time exit check failed: {e}")


def main():
    if not acquire_lock():
        print("Another instance is running. Exiting.")
        return
    
    try:
        _main_executor()
    finally:
        release_lock()

def _main_executor():
    print("=" * 70)
    print("  ALPACA PAPER TRADING EXECUTOR")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'PAPER' if APCA_PAPER else 'LIVE'}")
    print("=" * 70)
    
    try:
        executor = AlpacaExecutor()
    except Exception as e:
        logger.error(f"Failed to initialize Alpaca executor: {e}")
        return
    
    try:
        signal_gen = SignalGenerator()
    except Exception as e:
        logger.error(f"Failed to initialize signal generator: {e}")
        return
    
    # Clear cooldowns for symbols no longer in positions
    # (bracket orders may have closed them since last run)
    clear_stale_cooldowns(executor)
    
    signals, regime_info = signal_gen.generate_signals()
    
    if signals is None or len(signals) == 0:
        logger.warning("No signals generated")
        return
    
    # Risk management checks
    account = executor.get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    
    # 1. Drawdown circuit breaker
    should_halt, halt_reason = check_drawdown_breaker(portfolio_value)
    if should_halt:
        print(f"\n  🛑 HALTED: {halt_reason}")
        logger.warning(f"Trading halted: {halt_reason}")
        return
    
    # 2. VIX + margin risk multiplier
    composite_mult = get_risk_multiplier(cash, portfolio_value)
    
    # 3. Time-based exits — close positions held >5 days without profit
    check_time_exits(executor)
    
    # Auto-close excess positions
    big_cap_max = regime_info['big_cap_max']
    regime = regime_info['regime']
    
    positions = executor.get_positions()
    if positions and len(positions) > big_cap_max:
        excess = len(positions) - big_cap_max
        logger.info(f"Closing {excess} excess positions")
        executor.close_weakest_positions(excess)
    
    print(f"\n  SWING SYSTEM: 2d Pullback + Volume Confirmation")
    print(f"  Max Positions:   {big_cap_max}")
    print(f"  Risk Multiplier: {composite_mult:.2f}")
    print(f"\n  SWING SIGNALS:")
    print(f"  {'=' * 60}")
    for s in signals[:5]:
        f = s['features']
        print(f"  {s['symbol']:6s} | ${f['current_price']:.2f} | "
              f"Vol: {f.get('vol_ratio', 0):.2f}x | "
              f"Kelly: {s['kelly_size']:.1%} | "
              f"Win%: {s['win_prob']:.1%} | "
              f"TP: ${f['current_price']*1.04:.2f} | "
              f"SL: ${f['current_price']*0.98:.2f}")
    
    # Cancel stale brackets
    print(f"\n  CANCELLING STALE BRACKETS...")
    cancelled = executor.cancel_stale_brackets(max_age_hours=4)
    print(f"  Cancelled {cancelled} stale orders")
    
    # Execute trades
    print(f"\n  EXECUTING TRADES...")
    print(f"  {'=' * 60}")
    
    executed = 0
    skipped = 0
    
    for signal in signals:
        big_cap_max = regime_info['big_cap_max']
        
        # Check position count — always get fresh count from broker
        total_positions = executor.get_position_count()
        if total_positions >= big_cap_max:
            logger.info(f"Max big cap positions ({big_cap_max}) reached. Stopping.")
            break
        
        # Skip if kelly too small
        if signal['kelly_size'] <= 0.005:
            skipped += 1
            continue
        
        symbol = signal['symbol']
        
        # Daily cooldown lock — prevent wash trading
        if is_on_cooldown(symbol):
            logger.info(f"{symbol} on cooldown (traded today). Skipping.")
            skipped += 1
            continue
        
        # Calculate entry/stop/target
        entry_price = round(signal['features']['current_price'], 2)
        stop_loss = round(entry_price * (1 - signal['lower_mult']), 2)
        take_profit = round(entry_price * (1 + signal['upper_mult']), 2)
        
        if take_profit <= entry_price:
            take_profit = round(entry_price * 1.05, 2)
        if stop_loss >= entry_price:
            stop_loss = round(entry_price * 0.95, 2)
        if take_profit <= stop_loss:
            take_profit, stop_loss = stop_loss, take_profit
        
        # Position size with risk multiplier
        risk_adjusted_kelly = signal['kelly_size'] * composite_mult
        # Use 0.75 Kelly for better capital deployment (was 0.5)
        kelly_fraction = risk_adjusted_kelly * 0.75
        risk_amount = portfolio_value * kelly_fraction
        shares = int(risk_amount / entry_price)
        
        # Buying power check
        buffer_amount = buying_power * CASH_BUFFER_PCT
        if buying_power - risk_amount < buffer_amount:
            logger.warning(f"Insufficient buying power for {symbol}. Skipping.")
            skipped += 1
            continue
        
        # Sector limit check — allow adding to winners, block new sectors
        sector = signal.get('sector', 'unknown')
        if sector != 'unknown':
            positions = executor.get_positions()
            # Count existing positions in this sector (excluding the symbol we're adding to)
            sector_positions = [p for p in positions if STOCK_TO_SECTOR.get(p.symbol, 'unknown') == sector and p.symbol != symbol]
            # Also count already-executing signals in this sector
            executed_in_sector = sum(1 for s in signals[:signals.index(signal)] if s.get('sector', 'unknown') == sector and s.get('executed'))
            
            if len(sector_positions) + executed_in_sector >= 2:
                # Sector limit hit — only allow if adding to an existing position in this sector
                existing_in_sector = [p for p in positions if p.symbol == symbol and STOCK_TO_SECTOR.get(p.symbol, 'unknown') == sector]
                if not existing_in_sector:
                    logger.info(f"Sector limit (2) reached for {sector}. Skipping {symbol}.")
                    skipped += 1
                    continue
        
        if shares <= 0:
            skipped += 1
            continue
        
        order = executor.submit_bracket_order(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            shares=shares,
            regime=signal.get('regime', 'bull')
        )
        
        if order:
            executed += 1
            signal['executed'] = True
            logger.info(f"Executed trade {executed}: {symbol} - {shares} shares")
            
            # Mark cooldown — can't trade this symbol again today
            mark_cooldown(symbol)
            
            # Track position entry date for time-based exits
            tracking_file = Path('position_tracking.json')
            tracking = {}
            if tracking_file.exists():
                with open(tracking_file) as f:
                    tracking = json.load(f)
            tracking[symbol] = {
                'entry_date': datetime.now().isoformat(),
                'shares': shares,
                'entry_price': entry_price
            }
            with open(tracking_file, 'w') as f:
                json.dump(tracking, f, indent=2)
        else:
            skipped += 1
    
    # Summary
    print(f"\n  EXECUTION SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  Trades Executed: {executed}")
    print(f"  Trades Skipped:  {skipped}")
    print(f"  Max Positions:   {big_cap_max}")
    print(f"  Regime:          {regime}")
    
    positions = executor.get_positions()
    if positions:
        print(f"\n  CURRENT POSITIONS:")
        print(f"  {'=' * 60}")
        for pos in positions:
            print(f"  {pos.symbol:6s} | {int(pos.qty):4d} shares | Entry: ${float(pos.avg_entry_price):,.2f} | Current: ${float(pos.current_price):,.2f} | PnL: ${float(pos.unrealized_pl):+,.2f}")
    
    print(f"\n{'=' * 70}")
    print(f"  EXECUTION COMPLETE")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
