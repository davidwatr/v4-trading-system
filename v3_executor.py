#!/usr/bin/env python3
"""
V3 ALPACA EXECUTOR
==================
Integrates the V3 backtest-validated strategy into live paper trading.

Strategy (from V3 backtest):
- Regime filter: ONLY trade SIDEWAYS and HIGH_VOL
- Signal: 3% underperformance vs universe mean (5-day lookback)
- Sizing: 2% risk per trade, 2x ATR stop distance
- Exit: 12% profit target OR 8% hard stop OR 25-day max hold
- NO trailing stops (they destroy value)

Backtest: +28.5% total, 1.6 Sharpe, -10.4% max DD
OOS: +1.05% avg (conservative), 65% MC positive
"""

import os
import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Load .env
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

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, GetOrdersRequest,
    ClosePositionRequest, OrderClass,
    TakeProfitRequest, StopLossRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, QueryOrderStatus
from production_v18 import fetch_data, UNIVERSE

# ============================================
# CONFIG
# ============================================
APCA_API_KEY_ID = os.environ.get("APCA_API_KEY_ID", "")
APCA_API_SECRET_KEY = os.environ.get("APCA_API_SECRET_KEY", "")
APCA_PAPER = True

# V3 Parameters
MAX_POSITIONS = 5
TRANSACTION_COST = 0.001
BASE_STOP_LOSS = 0.08       # 8% hard stop
TARGET_PROFIT = 0.12        # 12% profit target
HOLD_DAYS_MAX = 25          # 25 days max hold
ENTRY_THRESHOLD = 0.03      # 3% underperformance
RISK_PER_TRADE = 0.02       # 2% risk per trade
ATR_STOP_MULT = 2.0         # 2x ATR stop distance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('v3_executor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# ALPACA CLIENT
# ============================================
class V3Executor:
    def __init__(self):
        if not APCA_API_KEY_ID or not APCA_API_SECRET_KEY:
            raise ValueError("Alpaca API keys not configured")
        
        self.client = TradingClient(
            api_key=APCA_API_KEY_ID,
            secret_key=APCA_API_SECRET_KEY,
            paper=APCA_PAPER
        )
        
        try:
            account = self.client.get_account()
            logger.info(f"Connected to Alpaca Paper Trading")
            logger.info(f"Cash: ${float(account.cash):,.2f}")
            logger.info(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
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
    
    def cancel_all_orders(self):
        try:
            orders = self.get_orders(status=QueryOrderStatus.OPEN)
            for order in orders:
                self.client.cancel_order_by_id(order.id)
                logger.info(f"Cancelled: {order.symbol}")
            return len(orders)
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return 0
    
    def close_position(self, symbol):
        try:
            self.client.close_position(symbol)
            logger.info(f"Closed position in {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            return False
    
    def submit_bracket_order(self, symbol, shares, entry_price, stop_loss, take_profit):
        """Submit bracket order with TP/SL."""
        try:
            shares = int(shares)
            if shares <= 0:
                return None
            
            entry_price = round(entry_price, 2)
            stop_loss = round(stop_loss, 2)
            take_profit = round(take_profit, 2)
            
            # Safety
            if take_profit <= entry_price:
                take_profit = round(entry_price * 1.12, 2)
            if stop_loss >= entry_price:
                stop_loss = round(entry_price * 0.92, 2)
            
            bracket = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=take_profit),
                stop_loss=StopLossRequest(stop_price=stop_loss, limit_price=stop_loss)
            )
            
            order = self.client.submit_order(bracket)
            logger.info(f"BRACKET: {symbol} {shares}sh | Entry ${entry_price} | TP ${take_profit} | SL ${stop_loss}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to submit {symbol}: {e}")
            return None


# ============================================
# V3 SIGNAL GENERATOR
# ============================================
class V3SignalGenerator:
    def __init__(self, universe=None):
        self.universe = universe or UNIVERSE[:25]
        self.data = {}
    
    def fetch_data(self, days=500):
        """Fetch data for all symbols."""
        logger.info(f"Fetching {len(self.universe)} symbols...")
        for i, sym in enumerate(self.universe):
            if i > 0:
                time.sleep(0.6)  # Rate limit: Yahoo allows ~1 req/sec
            # Retry up to 3 times
            for attempt in range(3):
                d = fetch_data(sym, days=days)
                if d is not None and d[0] is not None and len(d[0]) > 50:
                    self.data[sym] = {
                        'closes': np.array(d[0]),
                        'volumes': np.array(d[1]),
                        'highs': np.array(d[2]),
                        'lows': np.array(d[3])
                    }
                    break
                else:
                    time.sleep(5)
        logger.info(f"Loaded {len(self.data)} symbols")
    
    def classify_regime(self):
        """Classify current market regime."""
        for spy_sym in ['SPY', 'QQQ', 'AAPL']:
            if spy_sym in self.data:
                closes = self.data[spy_sym]['closes']
                if len(closes) < 30:
                    continue
                
                rets = np.diff(closes) / closes[:-1]
                vol = np.std(rets[-20:]) * np.sqrt(252)
                
                if len(rets) > 21:
                    autocorr = np.corrcoef(rets[-21:-1], rets[-20:])[0, 1]
                    if np.isnan(autocorr):
                        autocorr = 0.0
                else:
                    autocorr = 0.0
                
                ma_10 = np.mean(closes[-10:])
                ma_20 = np.mean(closes[-20:])
                trend = (ma_10 - ma_20) / ma_20 if ma_20 > 0 else 0
                
                if vol > 0.30:
                    return 'HIGH_VOL', 0.7
                elif autocorr > 0.15 and abs(trend) > 0.02:
                    return 'TRENDING', 0.7
                elif vol < 0.12:
                    return 'LOW_VOL', 0.5
                else:
                    return 'SIDEWAYS', 0.5
        
        return 'UNKNOWN', 0.0
    
    def compute_features(self, sym):
        """Compute features for a symbol."""
        if sym not in self.data:
            return None
        
        closes = self.data[sym]['closes']
        volumes = self.data[sym]['volumes']
        highs = self.data[sym]['highs']
        lows = self.data[sym]['lows']
        n = len(closes)
        
        if n < 25:
            return None
        
        rets = np.diff(closes) / closes[:-1]
        
        # ATR
        tr = np.abs(np.diff(closes[-15:]))
        atr = np.mean(tr)
        atr_ratio = atr / (closes[-1] + 1e-10)
        
        # Momentum
        mom_5 = (closes[-1] / closes[-6] - 1) if n >= 6 else 0
        mom_20 = (closes[-1] / closes[-21] - 1) if n >= 21 else 0
        
        # Volatility
        vol20 = np.std(rets[-20:]) * np.sqrt(252)
        
        # Volume ratio
        vol_ma20 = np.mean(volumes[-20:])
        vol_ratio_20 = volumes[-1] / (vol_ma20 + 1e-10)
        
        return {
            'atr': atr,
            'atr_ratio': atr_ratio,
            'momentum_5d': mom_5,
            'momentum_20d': mom_20,
            'volatility': vol20,
            'volume_ratio': vol_ratio_20,
        }
    
    def generate_signals(self, regime):
        """
        Generate V3 signals.
        ONLY trade SIDEWAYS and HIGH_VOL.
        """
        if regime in ['TRENDING', 'LOW_VOL']:
            return []
        
        # Compute 5-day returns for each stock
        sym_returns = {}
        for sym, data in self.data.items():
            closes = data['closes']
            if len(closes) >= 6:
                sym_returns[sym] = (closes[-1] / closes[-6] - 1)
        
        if len(sym_returns) < 5:
            return []
        
        mean_ret = np.mean(list(sym_returns.values()))
        
        signals = []
        for sym, ret in sym_returns.items():
            features = self.compute_features(sym)
            if features is None:
                continue
            
            # Skip if too volatile
            if features['volatility'] > 0.5:
                continue
            
            deviation = ret - mean_ret
            if deviation < -ENTRY_THRESHOLD:
                weight = min(abs(deviation) * 5, 1.0)
                signals.append({
                    'sym': sym,
                    'weight': weight,
                    'score': abs(deviation) / (features['volatility'] + 0.01),
                    'atr': features['atr'],
                    'price': self.data[sym]['closes'][-1]
                })
        
        signals.sort(key=lambda x: x['score'], reverse=True)
        return signals[:8]


# ============================================
# MAIN EXECUTOR
# ============================================
def main():
    print("=" * 70)
    print("  V3 ALPACA PAPER TRADING EXECUTOR")
    print("=" * 70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'PAPER' if APCA_PAPER else 'LIVE'}")
    print("=" * 70)
    
    try:
        executor = V3Executor()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return
    
    # Fetch data and generate signals
    sig_gen = V3SignalGenerator()
    sig_gen.fetch_data(days=500)
    
    # Classify regime
    regime, confidence = sig_gen.classify_regime()
    logger.info(f"Regime: {regime} (confidence: {confidence:.0%})")
    
    # Check existing positions
    positions = executor.get_positions()
    logger.info(f"Open positions: {len(positions)}")
    
    # Time-based exits (V3: close positions held >25 days)
    tracking_file = Path('v3_position_tracking.json')
    if tracking_file.exists():
        with open(tracking_file) as f:
            tracking = json.load(f)
    else:
        tracking = {}
    
    now = datetime.now()
    closed_count = 0
    for pos in positions:
        sym = pos.symbol
        if sym in tracking:
            entry_date = datetime.fromisoformat(tracking[sym]['entry_date'])
            days_held = (now - entry_date).days
            if days_held >= HOLD_DAYS_MAX:
                logger.info(f"Time exit: {sym} held {days_held} days")
                executor.close_position(sym)
                del tracking[sym]
                closed_count += 1
    
    if closed_count > 0:
        with open(tracking_file, 'w') as f:
            json.dump(tracking, f, indent=2)
        logger.info(f"Closed {closed_count} positions on time exit")
    
    # Generate new signals
    signals = sig_gen.generate_signals(regime)
    logger.info(f"Signals generated: {len(signals)}")
    
    if not signals:
        logger.info("No signals — staying in cash")
        return
    
    # Get account info
    account = executor.get_account()
    portfolio_value = float(account.portfolio_value)
    cash = float(account.cash)
    
    # Execute signals
    executed = 0
    for signal in signals:
        sym = signal['sym']
        price = signal['price']
        atr = signal['atr']
        
        # Skip if already in position
        already_in = False
        for pos in positions:
            if pos.symbol == sym:
                already_in = True
                break
        if already_in:
            continue
        
        # Check position count
        if executor.get_position_count() >= MAX_POSITIONS:
            logger.info(f"Max positions ({MAX_POSITIONS}) reached")
            break
        
        # Check cash
        if cash < price * 10:  # Need at least $10 worth
            continue
        
        # Volatility-scaled sizing (2% risk per trade)
        risk_per_trade = portfolio_value * RISK_PER_TRADE
        stop_distance = atr * ATR_STOP_MULT
        if stop_distance < 1e-10:
            continue
        shares = int(risk_per_trade / stop_distance)
        max_shares = int(portfolio_value * 0.15 / price)
        shares = min(shares, max_shares)
        
        if shares <= 0:
            continue
        
        # Calculate TP/SL
        stop_loss = round(price * (1 - BASE_STOP_LOSS), 2)
        take_profit = round(price * (1 + TARGET_PROFIT), 2)
        
        # Submit bracket order
        order = executor.submit_bracket_order(sym, shares, price, stop_loss, take_profit)
        if order:
            tracking[sym] = {
                'entry_date': datetime.now().isoformat(),
                'entry_price': price,
                'shares': shares,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            cash -= shares * price
            executed += 1
    
    with open(tracking_file, 'w') as f:
        json.dump(tracking, f, indent=2)
    
    logger.info(f"Executed {executed} signals")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
