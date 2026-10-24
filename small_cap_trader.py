#!/usr/bin/env python3
"""
Small Cap Paper Trader — Explosive moves, not boring blue chips.
Reads small_cap_signals.json, enters on high-score signals, holds until SL/TP.
"""

import json
import sys
import pickle
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from production_v18 import (
    fetch_data, extract_features, get_market_regime,
    calc_atr, calc_rsi, MODEL_FILE
)

PORTFOLIO_FILE = Path('small_cap_portfolio.json')
INITIAL_CASH = 1300  # ILS
MAX_POSITIONS = 3
RISK_PER_TRADE = 0.025  # 2.5% risk per trade (more aggressive for small caps)


def load_portfolio():
    """Load portfolio from file or create new."""
    if PORTFOLIO_FILE.exists():
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        "cash_ils": INITIAL_CASH,
        "initial_cash_ils": INITIAL_CASH,
        "positions": [],
        "closed_trades": [],
        "start_date": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat()
    }


def save_portfolio(portfolio):
    """Save portfolio to file."""
    portfolio["last_update"] = datetime.now().isoformat()
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)


def get_current_price(symbol):
    """Get current price from Yahoo."""
    closes, _, _, _ = fetch_data(symbol, days=5)
    if closes:
        return closes[-1]
    return None


def check_stop_loss(position, current_price):
    """Check if stop loss was hit."""
    return current_price <= position["stop"]


def check_take_profit(position, current_price):
    """Check if take profit was hit."""
    return current_price >= position["target_1"]


def simulate_entry(portfolio, signal, ils_usd_rate=3.6):
    """Simulate entering a position."""
    symbol = signal["symbol"]
    entry_price = signal["entry"]
    stop = signal["stop"]
    target_1 = signal["target_1"]
    target_2 = signal.get("target_2", target_1)
    
    # Check if we already have this symbol
    for pos in portfolio["positions"]:
        if pos["symbol"] == symbol:
            return False, f"Already in {symbol}"
    
    # Check max positions
    if len(portfolio["positions"]) >= MAX_POSITIONS:
        return False, "Max positions reached"
    
    # Calculate position size
    cash_usd = portfolio["cash_ils"] / ils_usd_rate
    
    # For small accounts: equal-weight across positions
    if portfolio["cash_ils"] < 5000:
        target_value_usd = cash_usd / (MAX_POSITIONS - len(portfolio["positions"]))
        shares = target_value_usd / entry_price
    else:
        # Larger accounts: 2.5% risk per trade
        risk_usd = portfolio["cash_ils"] * RISK_PER_TRADE / ils_usd_rate
        risk_per_share_usd = entry_price - stop
        if risk_per_share_usd <= 0:
            return False, "Invalid risk per share"
        shares = risk_usd / risk_per_share_usd
    
    if shares <= 0:
        return False, "Can't afford any shares"
    
    # Round to 2 decimal places for fractional shares
    shares = round(shares, 2)
    
    cost_usd = shares * entry_price
    cost_ils = cost_usd * ils_usd_rate
    
    if cost_ils > portfolio["cash_ils"]:
        # Try minimum position
        shares = 0.01
        cost_usd = shares * entry_price
        cost_ils = cost_usd * ils_usd_rate
        if cost_ils > portfolio["cash_ils"]:
            return False, "Insufficient cash"
    
    position = {
        "symbol": symbol,
        "shares": shares,
        "entry_price": entry_price,
        "stop": stop,
        "target_1": target_1,
        "target_2": target_2,
        "current_price": entry_price,
        "cost_ils": round(cost_ils, 2),
        "value_ils": round(cost_ils, 2),
        "pnl_ils": 0,
        "pnl_pct": 0,
        "entry_date": datetime.now().isoformat(),
        "signal_type": signal["signal_type"],
        "score": signal["score"],
        "confidence": signal["confidence"],
        "beta": signal["beta"],
        "rsi": signal["rsi"]
    }
    
    portfolio["positions"].append(position)
    portfolio["cash_ils"] -= cost_ils
    
    return True, f"BUY {shares:.2f} {symbol} @ ${entry_price:.2f} = {cost_ils:.0f} ILS"


def simulate_exit(portfolio, position, exit_price, reason):
    """Simulate exiting a position."""
    symbol = position["symbol"]
    shares = position["shares"]
    entry_price = position["entry_price"]
    
    proceeds_usd = shares * exit_price
    cost_usd = shares * entry_price
    pnl_usd = proceeds_usd - cost_usd
    
    ils_usd_rate = 3.6
    pnl_ils = pnl_usd * ils_usd_rate
    proceeds_ils = proceeds_usd * ils_usd_rate
    
    trade = {
        "symbol": symbol,
        "shares": shares,
        "entry_price": entry_price,
        "exit_price": round(exit_price, 2),
        "cost_ils": position["cost_ils"],
        "proceeds_ils": round(proceeds_ils, 2),
        "pnl_ils": round(pnl_ils, 2),
        "pnl_pct": round((exit_price - entry_price) / entry_price * 100, 2),
        "entry_date": position["entry_date"],
        "exit_date": datetime.now().isoformat(),
        "reason": reason,
        "signal_type": position["signal_type"],
        "score": position["score"],
        "confidence": position["confidence"]
    }
    
    portfolio["closed_trades"].append(trade)
    portfolio["cash_ils"] += proceeds_ils
    
    # Remove from open positions
    portfolio["positions"] = [p for p in portfolio["positions"] if p["symbol"] != symbol]
    
    return trade


def update_positions(portfolio):
    """Update all positions with current prices."""
    closed = []
    
    for position in portfolio["positions"][:]:
        symbol = position["symbol"]
        current_price = get_current_price(symbol)
        
        if current_price is None:
            continue
        
        position["current_price"] = current_price
        
        # Calculate P&L
        cost_usd = position["shares"] * position["entry_price"]
        value_usd = position["shares"] * current_price
        pnl_usd = value_usd - cost_usd
        
        ils_usd_rate = 3.6
        position["value_ils"] = round(value_usd * ils_usd_rate, 2)
        position["pnl_ils"] = round(pnl_usd * ils_usd_rate, 2)
        position["pnl_pct"] = round((current_price - position["entry_price"]) / position["entry_price"] * 100, 2)
        
        # Check stop loss
        if check_stop_loss(position, current_price):
            trade = simulate_exit(portfolio, position, current_price, "STOP_LOSS")
            closed.append(trade)
        
        # Check take profit
        elif check_take_profit(position, current_price):
            trade = simulate_exit(portfolio, position, current_price, "TAKE_PROFIT")
            closed.append(trade)
    
    return closed


def print_portfolio(portfolio):
    """Print portfolio status."""
    print(f"\n{'=' * 60}")
    print(f"  SMALL CAP PORTFOLIO — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 60}")
    print(f"  Cash: {portfolio['cash_ils']:.0f} ILS")
    
    total_value = portfolio["cash_ils"]
    
    if portfolio["positions"]:
        print(f"\n  OPEN POSITIONS:")
        for pos in portfolio["positions"]:
            pnl_emoji = "🟢" if pos["pnl_ils"] >= 0 else "🔴"
            print(f"    {pnl_emoji} {pos['symbol']:6s} {pos['shares']:6.2f} shares | "
                  f"Entry: ${pos['entry_price']:.2f} | Now: ${pos['current_price']:.2f} | "
                  f"P&L: {pos['pnl_ils']:+.0f} ILS ({pos['pnl_pct']:+.1f}%) | "
                  f"Score {pos['score']} Conf {pos['confidence']:.0%}")
            total_value += pos["value_ils"]
    
    total_pnl = total_value - portfolio["initial_cash_ils"]
    
    print(f"\n  TOTAL VALUE: {total_value:.0f} ILS")
    print(f"  TOTAL P&L: {total_pnl:+.0f} ILS ({total_pnl / portfolio['initial_cash_ils'] * 100:+.1f}%)")
    
    if portfolio["closed_trades"]:
        wins = sum(1 for t in portfolio["closed_trades"] if t["pnl_ils"] > 0)
        losses = sum(1 for t in portfolio["closed_trades"] if t["pnl_ils"] <= 0)
        print(f"\n  CLOSED TRADES: {len(portfolio['closed_trades'])} (W: {wins} / L: {losses})")
        
        for trade in portfolio["closed_trades"][-5:]:
            pnl_emoji = "🟢" if trade["pnl_ils"] >= 0 else "🔴"
            print(f"    {pnl_emoji} {trade['symbol']:6s} | Entry: ${trade['entry_price']:.2f} | "
                  f"Exit: ${trade['exit_price']:.2f} | P&L: {trade['pnl_ils']:+.0f} ILS ({trade['pnl_pct']:+.1f}%) | "
                  f"{trade['reason']}")
    
    print(f"{'=' * 60}")


def main():
    print("=" * 60)
    print(f"  SMALL CAP PAPER TRADER — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Load portfolio
    portfolio = load_portfolio()
    
    # Step 1: Update existing positions
    print("\n[1] Updating positions...")
    closed = update_positions(portfolio)
    for trade in closed:
        emoji = "🟢" if trade["pnl_ils"] >= 0 else "🔴"
        print(f"    {emoji} {trade['symbol']} closed @ ${trade['exit_price']:.2f} | "
              f"P&L: {trade['pnl_ils']:+.0f} ILS ({trade['pnl_pct']:+.1f}%) | {trade['reason']}")
    
    # Step 2: Check for new entries from scanner
    print("\n[2] Checking for new entries...")
    scanner_file = Path('small_cap_signals.json')
    if scanner_file.exists():
        with open(scanner_file) as f:
            signals = json.load(f)
        
        # Take top signals that we don't already have
        existing_symbols = {p["symbol"] for p in portfolio["positions"]}
        
        for signal in signals[:10]:
            if signal["symbol"] in existing_symbols:
                continue
            if len(portfolio["positions"]) >= MAX_POSITIONS:
                break
            
            # Skip signals with missing data
            if not all(k in signal for k in ["entry", "stop", "target_1"]):
                continue
            
            # Only enter high-confidence signals
            if signal.get("confidence", 0) < 0.7:
                continue
            
            success, msg = simulate_entry(portfolio, signal)
            print(f"    {msg}")
            if success:
                existing_symbols.add(signal["symbol"])
    else:
        print("    No scanner results - run 'python small_cap_scanner.py' first")
    
    # Step 3: Save and print
    save_portfolio(portfolio)
    print_portfolio(portfolio)
    
    return portfolio


if __name__ == "__main__":
    main()
