#!/usr/bin/env python3
"""
Earnings & Event Detector — Find explosive catalysts
Uses yfinance to get upcoming earnings dates.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

def get_earnings_info(symbol):
    """
    Get earnings info for a stock.
    Returns dict with earnings date, estimate, etc.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        earnings_date = None
        eps_estimate = None
        revenue_estimate = None
        
        # Method 1: earningsTimestamp
        ts = info.get('earningsTimestamp')
        if ts:
            try:
                earnings_date = datetime.fromtimestamp(ts)
            except:
                pass
        
        # Method 2: calendar
        if earnings_date is None:
            try:
                calendar = ticker.calendar
                if calendar and 'Earnings Date' in calendar:
                    dates = calendar['Earnings Date']
                    if dates:
                        if isinstance(dates[0], datetime):
                            earnings_date = dates[0]
                        elif hasattr(dates[0], 'date'):
                            earnings_date = datetime.combine(dates[0].date(), datetime.min.time())
            except:
                pass
        
        # Get estimates from calendar
        try:
            calendar = ticker.calendar
            if calendar:
                eps_estimate = calendar.get('Earnings Average')
                revenue_estimate = calendar.get('Revenue Average')
        except:
            pass
        
        if earnings_date:
            now = datetime.now()
            days_until = (earnings_date - now).days
            
            return {
                'has_earnings': True,
                'date': earnings_date.strftime('%Y-%m-%d'),
                'days_until': days_until,
                'eps_estimate': eps_estimate,
                'revenue_estimate': revenue_estimate,
                'is_upcoming': 0 <= days_until <= 14
            }
        
        return {'has_earnings': False}
        
    except Exception as e:
        return {'has_earnings': False, 'error': str(e)}


def has_upcoming_catalyst(symbol, days_ahead=14):
    """
    Check if a stock has an upcoming catalyst within the next N days.
    Returns (bool, dict)
    """
    info = get_earnings_info(symbol)
    
    if info.get('has_earnings') and info.get('is_upcoming'):
        return True, {
            'has_catalyst': True,
            'catalyst_type': 'earnings',
            'date': info['date'],
            'days_until': info['days_until'],
            'eps_estimate': info['eps_estimate'],
            'revenue_estimate': info['revenue_estimate']
        }
    
    return False, {}


def batch_check_catalysts(symbols, days_ahead=14):
    """
    Check multiple stocks for upcoming catalysts.
    Returns dict of symbol -> catalyst_info
    """
    results = {}
    
    for symbol in symbols:
        has_cat, info = has_upcoming_catalyst(symbol, days_ahead)
        if has_cat:
            results[symbol] = info
    
    return results


def main():
    """Test the earnings detector."""
    print("=" * 60)
    print("  EARNINGS & EVENT DETECTOR")
    print("=" * 60)
    
    test_symbols = ['SEDG', 'ARM', 'LRCX', 'AMD', 'NVDA', 'BE', 'MRNA', 'BNTX', 'QCOM', 'KLAC']
    
    print(f"\nChecking {len(test_symbols)} stocks for upcoming catalysts...\n")
    
    for symbol in test_symbols:
        has_cat, info = has_upcoming_catalyst(symbol, days_ahead=14)
        
        if has_cat:
            days = info.get('days_until', '?')
            date = info.get('date', '?')
            eps = info.get('eps_estimate')
            eps_str = f"${eps:.2f}" if eps else "N/A"
            print(f"  {symbol}: ✅ Catalyst in {days} days ({date}) | EPS est: {eps_str}")
        else:
            # Check if we got any earnings data at all
            earn_info = get_earnings_info(symbol)
            if earn_info.get('has_earnings'):
                days = earn_info.get('days_until', '?')
                date = earn_info.get('date', '?')
                print(f"  {symbol}: ⏳ Earnings in {days} days ({date}) — outside window")
            else:
                print(f"  {symbol}: ❌ No catalyst found")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
