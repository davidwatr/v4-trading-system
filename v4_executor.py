"""
V4.1 ALPACA EXECUTOR — LIVE
============================
Uses tuple-based data format (production_v18 compatible).
Full portfolio transition: close old positions, build V4 portfolio.
"""
import os
import sys
import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import time

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOrdersRequest, MarketOrderRequest, LimitOrderRequest,
    ClosePositionRequest
)
from alpaca.trading.enums import (
    QueryOrderStatus, OrderSide, TimeInForce, OrderType
)

# Local data fetcher
sys.path.insert(0, str(Path(__file__).parent))
from production_v18 import fetch_data, UNIVERSE

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
class Config:
    TARGET_VOL = 0.18
    VOL_HALF_LIFE = 10
    CORR_THRESHOLD = 0.75
    MAX_CLUSTER_EXPOSURE = 0.30
    MAX_POSITIONS = 5
    RISK_PER_TRADE = 0.02
    MAX_DAILY_LOSS = 0.03
    MAX_DRAWDOWN = 0.15
    STOP_LOSS = 0.08
    PROFIT_TARGET = 0.12
    MAX_HOLD_DAYS = 25
    ENTRY_THRESHOLD = 0.03
    MAX_POSITION_SIZE = 0.15
    TRANSACTION_COST = 0.001

cfg = Config()

# ─────────────────────────────────────────────
# ALPACA HELPERS
# ─────────────────────────────────────────────
class AlpacaHelper:
    def __init__(self):
        self._load_env()
        self.client = TradingClient(
            api_key=os.environ.get('APCA_API_KEY_ID', ''),
            secret_key=os.environ.get('APCA_API_SECRET_KEY', ''),
            paper=True
        )

    def _load_env(self):
        # First check if env vars are already set (GitHub Actions, Streamlit Cloud)
        if os.environ.get('APCA_API_KEY_ID') and os.environ.get('APCA_API_SECRET_KEY'):
            return
        # Otherwise load from .env file (local development)
        env_file = Path(__file__).parent / '.env'
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

    def get_account(self):
        return self.client.get_account()

    def get_positions(self):
        return self.client.get_all_positions()

    def cancel_all_orders(self):
        orders = self.client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
        cancelled = 0
        for order in orders:
            try:
                self.client.cancel_order_by_id(order.id)
                cancelled += 1
            except Exception:
                pass
        return cancelled

    def close_position(self, symbol):
        try:
            result = self.client.close_position(symbol)
            return True, result
        except Exception as e:
            return False, str(e)

    def submit_market_order(self, symbol, qty, side):
        try:
            order = self.client.submit_order(MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY
            ))
            return True, order
        except Exception as e:
            return False, str(e)

    def submit_limit_order(self, symbol, qty, side, limit_price):
        try:
            order = self.client.submit_order(LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price
            ))
            return True, order
        except Exception as e:
            return False, str(e)


# ─────────────────────────────────────────────
# MARKET DATA (tuple-based)
# ─────────────────────────────────────────────
class MarketData:
    def __init__(self):
        self.cache = {}

    def fetch(self, symbol, days=500):
        """Returns (closes, volumes, highs, lows) tuple or None"""
        if symbol in self.cache:
            return self.cache[symbol]
        d = fetch_data(symbol, days=days)
        if d is not None and len(d[0]) > 50:
            self.cache[symbol] = d
            return d
        return None

    def get_closes(self, symbol, days=500):
        data = self.fetch(symbol, days)
        return data[0] if data else None


# ─────────────────────────────────────────────
# REGIME DETECTION
# ─────────────────────────────────────────────
class RegimeDetector:
    def __init__(self, data: MarketData):
        self.data = data

    def get_regime(self):
        closes = self.data.get_closes('SPY', 250)
        if closes is None or len(closes) < 30:
            return 'SIDEWAYS', 0.5

        closes = np.array(closes)
        rets = np.diff(closes) / closes[:-1]
        vol = float(np.std(rets[-20:])) * np.sqrt(252)

        if len(rets) > 21:
            autocorr = np.corrcoef(rets[-21:-1], rets[-20:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0

        ma20 = float(np.mean(closes[-20:]))
        ma50 = float(np.mean(closes[-50:]))
        current = float(closes[-1])

        trend = 0
        if current > ma20 > ma50:
            trend = 1
        elif current < ma20 < ma50:
            trend = -1

        if trend == 1 and vol > 0.20:
            regime = 'TRENDING_UP'
        elif trend == 1:
            regime = 'LOW_VOL'
        elif trend == -1 and vol > 0.20:
            regime = 'TRENDING_DOWN'
        elif trend == -1:
            regime = 'CRASH'
        elif vol > 0.20:
            regime = 'HIGH_VOL'
        else:
            regime = 'SIDEWAYS'

        confidence = min(1.0, abs(trend) * 0.5 + (vol / 0.3) * 0.5)
        return regime, confidence


# ─────────────────────────────────────────────
# V4 STRATEGY ENGINE
# ─────────────────────────────────────────────
class V4Strategy:
    def __init__(self, data: MarketData):
        self.data = data

    def calc_atr(self, symbol, period=14):
        data = self.data.fetch(symbol, 60)
        if data is None or len(data[0]) < period + 1:
            return None
        highs = np.array(data[2])
        lows = np.array(data[1])
        closes = np.array(data[0])
        trs = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(1, len(closes))]
        return float(np.mean(trs[-period:]))

    def volatility_target_size(self, symbol, capital, held_symbols):
        """Returns (qty, atr) with vol targeting + correlation sizing"""
        risk_amount = capital * cfg.RISK_PER_TRADE
        atr = self.calc_atr(symbol)
        if atr is None or atr == 0:
            return 0, None

        raw_qty = int(risk_amount / (atr * 1.5))
        if raw_qty <= 0:
            return 0, None

        # Vol targeting
        closes = self.data.get_closes(symbol, 60)
        if closes is not None and len(closes) >= 20:
            closes = np.array(closes)
            rets = np.diff(closes) / closes[:-1]
            current_vol = float(pd.Series(rets).ewm(halflife=cfg.VOL_HALF_LIFE).std().iloc[-1])
            if not np.isnan(current_vol) and current_vol > 0:
                target_vol = cfg.TARGET_VOL / np.sqrt(252)
                vol_ratio = target_vol / max(current_vol, 0.005)
                vol_ratio = np.clip(vol_ratio, 0.5, 1.5)
                raw_qty = int(raw_qty * vol_ratio)

        # Correlation sizing
        if held_symbols:
            target_closes = self.data.get_closes(symbol, 60)
            if target_closes is not None and len(target_closes) >= 30:
                target_closes = np.array(target_closes)
                target_rets = np.diff(target_closes) / target_closes[:-1]
                high_corr_count = 0
                total_corr_count = 0
                for hsym in held_symbols:
                    held_closes = self.data.get_closes(hsym, 60)
                    if held_closes is not None and len(held_closes) >= 30:
                        held_closes = np.array(held_closes)
                        held_rets = np.diff(held_closes) / held_closes[:-1]
                        min_len = min(len(target_rets), len(held_rets))
                        if min_len > 10:
                            corr = np.corrcoef(target_rets[:min_len], held_rets[:min_len])[0, 1]
                            if not np.isnan(corr):
                                total_corr_count += 1
                                if corr > cfg.CORR_THRESHOLD:
                                    high_corr_count += 1
                if high_corr_count > 0 and total_corr_count > 0:
                    cluster_ratio = high_corr_count / total_corr_count
                    if cluster_ratio > cfg.MAX_CLUSTER_EXPOSURE:
                        raw_qty = int(raw_qty * 0.5)
                    else:
                        raw_qty = int(raw_qty * (1 - cluster_ratio * 0.3))

        return raw_qty, atr

    def generate_signals(self, regime, held_symbols):
        if regime not in ('SIDEWAYS', 'HIGH_VOL'):
            return []

        signals = []
        for symbol in UNIVERSE:
            if symbol in held_symbols:
                continue

            closes = self.data.get_closes(symbol, 250)
            if closes is None or len(closes) < 50:
                continue

            closes = np.array(closes)
            ma20 = float(np.mean(closes[-20:]))
            current_price = float(closes[-1])

            if np.isnan(ma20) or ma20 == 0:
                continue

            deviation = (current_price - ma20) / ma20

            if abs(deviation) >= cfg.ENTRY_THRESHOLD:
                atr = self.calc_atr(symbol)
                if atr is None or atr == 0:
                    continue

                signals.append({
                    'symbol': symbol,
                    'price': current_price,
                    'ma20': ma20,
                    'deviation': float(deviation),
                    'atr': atr,
                })

        signals.sort(key=lambda x: abs(x['deviation']), reverse=True)
        return signals[:cfg.MAX_POSITIONS]


# ─────────────────────────────────────────────
# RISK MANAGEMENT
# ─────────────────────────────────────────────
class RiskManager:
    def __init__(self, helper: AlpacaHelper):
        self.helper = helper

    def check_circuit_breakers(self, portfolio_value, peak_value):
        current_dd = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0
        if current_dd <= -cfg.MAX_DRAWDOWN:
            return False, f"MAX DRAWDOWN BREACH: {current_dd:.1%}"
        return True, "OK"

    def risk_check(self, signal, account, positions):
        if float(account.buying_power) < float(account.portfolio_value) * 0.01:
            return False, "Insufficient buying power"

        held_symbols = [p.symbol for p in positions]
        if signal['symbol'] in held_symbols:
            return False, "Already in portfolio"

        return True, "ALLOW"


# ─────────────────────────────────────────────
# PORTFOLIO TRANSITION MANAGER
# ─────────────────────────────────────────────
class PortfolioTransition:
    def __init__(self, helper: AlpacaHelper, strategy: V4Strategy, risk: RiskManager):
        self.helper = helper
        self.strategy = strategy
        self.risk = risk

    def cancel_all_and_close_unwanted(self, wanted_symbols):
        cancelled = self.helper.cancel_all_orders()
        if cancelled > 0:
            logger.info(f"Cancelled {cancelled} stale orders")

        positions = self.helper.get_positions()
        if not positions:
            return []

        closed = []
        for pos in positions:
            if pos.symbol not in wanted_symbols:
                ok, result = self.helper.close_position(pos.symbol)
                if ok:
                    closed.append(pos.symbol)
                    logger.info(f"Closed unwanted position: {pos.symbol}")
                else:
                    logger.warning(f"Failed to close {pos.symbol}: {result}")

        if closed:
            logger.info(f"Closed {len(closed)} unwanted positions: {closed}")
            time.sleep(3)

        return closed

    def build_new_portfolio(self, regime):
        logger.info("=" * 60)
        logger.info("PORTFOLIO TRANSITION — V4")
        logger.info("=" * 60)

        account = self.helper.get_account()
        pv = float(account.portfolio_value)
        positions = self.helper.get_positions()
        held_symbols = [p.symbol for p in positions]

        logger.info(f"Portfolio Value: ${pv:,.2f}")
        logger.info(f"Cash: ${float(account.cash):,.2f}")
        logger.info(f"Current Positions: {len(positions)}")
        logger.info(f"Regime: {regime}")

        # Generate V4 signals
        signals = self.strategy.generate_signals(regime, [])
        wanted_symbols = set(s['symbol'] for s in signals[:cfg.MAX_POSITIONS])

        logger.info(f"V4 wants: {sorted(wanted_symbols)}")
        logger.info(f"Currently hold: {sorted(held_symbols)}")

        # Close unwanted positions
        to_close = set(held_symbols) - wanted_symbols
        if to_close:
            self.cancel_all_and_close_unwanted(wanted_symbols)
            time.sleep(2)

        # Recalculate after closing
        positions = self.helper.get_positions()
        held_symbols = [p.symbol for p in positions]
        remaining_positions = len(positions)

        # Execute new signals
        executed = []
        for signal in signals:
            if signal['symbol'] in held_symbols:
                continue
            if remaining_positions >= cfg.MAX_POSITIONS:
                break

            qty, atr = self.strategy.volatility_target_size(
                signal['symbol'], pv, held_symbols
            )
            if qty <= 0:
                continue

            # Check max position position
            max_shares = int(pv * cfg.MAX_POSITION_SIZE / signal['price'])
            qty = min(qty, max_shares)

            ok, result = self.helper.submit_market_order(
                signal['symbol'], qty, OrderSide.BUY
            )
            if ok:
                executed.append(signal['symbol'])
                held_symbols.append(signal['symbol'])
                remaining_positions += 1
                logger.info(f"  EXECUTED BUY: {signal['symbol']} qty={qty} @ market")
            else:
                logger.warning(f"  FAILED BUY: {signal['symbol']}: {result}")

        logger.info(f"Executed {len(executed)} new positions: {executed}")
        return executed


# ─────────────────────────────────────────────
# MAIN EXECUTOR
# ─────────────────────────────────────────────
def run_executor():
    logger.info("=" * 60)
    logger.info("V4.1 ALPACA PAPER TRADING EXECUTOR")
    logger.info("=" * 60)
    logger.info(f"Time: {datetime.now()}")
    logger.info("Mode: PAPER")
    logger.info("=" * 60)

    helper = AlpacaHelper()
    data = MarketData()
    strategy = V4Strategy(data)
    risk = RiskManager(helper)
    transition = PortfolioTransition(helper, strategy, risk)
    regime_detector = RegimeDetector(data)

    account = helper.get_account()
    pv = float(account.portfolio_value)
    cash = float(account.cash)

    logger.info(f"Portfolio Value: ${pv:,.2f}")
    logger.info(f"Cash: ${cash:,.2f}")

    # Circuit breakers
    peak = pv
    can_trade, reason = risk.check_circuit_breakers(pv, peak)
    if not can_trade:
        logger.warning(f"CIRCUIT BREAKER: {reason}")
        return

    # Get regime
    regime, confidence = regime_detector.get_regime()
    logger.info(f"Regime: {regime} (confidence: {confidence:.0%})")

    # If not tradable regime, stay in cash
    if regime not in ('SIDEWAYS', 'HIGH_VOL'):
        logger.info(f"Regime '{regime}' not suitable — staying in cash")
        positions = helper.get_positions()
        for pos in positions:
            helper.close_position(pos.symbol)
        return

    # Do full portfolio transition
    executed = transition.build_new_portfolio(regime)

    # Summary
    account = helper.get_account()
    positions = helper.get_positions()
    new_pv = float(account.portfolio_value)

    logger.info("=" * 60)
    logger.info("PORTFOLIO SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Portfolio Value: ${new_pv:,.2f}")
    logger.info(f"Cash: ${float(account.cash):,.2f}")
    logger.info(f"Positions: {len(positions)}")

    for pos in positions:
        qty = float(pos.qty)
        entry = float(pos.avg_entry_price)
        current = float(pos.current_price)
        val = qty * current
        pl_pct = (current - entry) / entry * 100
        logger.info(f"  {pos.symbol:6s} | {qty:4.0f} shares | ${entry:8.2f} → ${current:8.2f} | ${val:9,.2f} | {pl_pct:+.1f}%")

    logger.info("=" * 60)


if __name__ == '__main__':
    run_executor()
