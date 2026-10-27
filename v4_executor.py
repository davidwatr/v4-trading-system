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

    def get_order_by_id(self, order_id):
        if not order_id:
            return None
        try:
            return self.client.get_order_by_id(order_id)
        except Exception:
            return None

    def get_open_orders(self):
        try:
            return self.client.get_orders(
                filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except Exception:
            return []


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
        if d is not None and d[0] is not None and len(d[0]) > 40:
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
        lows = np.array(data[3])
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
        self.max_hold_days = cfg.MAX_HOLD_DAYS  # Max days to hold a position
        self.position_log_file = Path("v4_position_tracking.json")

    def _close_stale_positions(self, positions):
        """Close positions held longer than MAX_HOLD_DAYS."""
        now = datetime.now()
        closed = []
        if not positions:
            return closed
        for pos in positions:
            symbol = pos.symbol
            # Prefer explicit entry_date; alpaca-py positions expose opened_at
            entry_date = getattr(pos, 'entry_date', None) or getattr(pos, 'opened_at', None)
            if entry_date:
                hold_days = (now - entry_date).days
            else:
                # Fallback: assume positions held since V4 system start (2026-09-16)
                entry_date = datetime(2026, 9, 16)
                hold_days = (now - entry_date).days
            if hold_days >= self.max_hold_days:
                ok, result = self.helper.close_position(symbol)
                if ok:
                    closed.append(symbol)
                    logger.info(f"CLOSED STALE position: {symbol} (held {hold_days} days >= {self.max_hold_days} days)")
                    self._log_position_close(symbol, hold_days)
                else:
                    logger.warning(f"FAILED to close stale {symbol}: {result}")
        return closed

    def _load_tracking(self):
        """Load tracking file; supports the legacy list format."""
        if not self.position_log_file.exists():
            return {"pending": [], "history": [], "closed": []}
        try:
            with open(self.position_log_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"pending": [], "history": [], "closed": []}
        if isinstance(data, list):  # legacy: closed-position log only
            return {"pending": [], "history": [], "closed": data}
        data.setdefault("pending", [])
        data.setdefault("history", [])
        data.setdefault("closed", [])
        return data

    def _save_tracking(self, data):
        try:
            with open(self.position_log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save tracking file: {e}")

    def _track_submitted_order(self, symbol, qty, order_id):
        """Record a submitted order as pending until its fill is confirmed."""
        data = self._load_tracking()
        data["pending"].append({
            "symbol": symbol,
            "qty": float(qty),
            "order_id": str(order_id) if order_id else None,
            "side": "buy",
            "submitted_at": datetime.now().isoformat(),
            "status": "accepted",
        })
        data["pending"] = data["pending"][-100:]
        self._save_tracking(data)

    def _move_pending(self, order_id, status, extra=None):
        """Move a pending order to history with a terminal status."""
        data = self._load_tracking()
        for o in data["pending"]:
            if o.get("order_id") == str(order_id):
                o["status"] = status
                o["ended_at"] = datetime.now().isoformat()
                if extra:
                    o.update(extra)
                data["history"].append(o)
                data["pending"] = [x for x in data["pending"] if x.get("order_id") != str(order_id)]
                break
        data["history"] = data["history"][-200:]
        self._save_tracking(data)

    def _await_fill(self, order_id, symbol, qty, timeout=25):
        """Poll briefly for a fill; in-session market fills are near-instant."""
        if not order_id:
            logger.info(f"  no order id returned for {symbol} — fill unverified")
            return False
        deadline = time.time() + timeout
        while time.time() < deadline:
            order = self.helper.get_order_by_id(order_id)
            status = getattr(order, 'status', None) if order else None
            if status == 'filled':
                logger.info(f"  CONFIRMED FILL: {symbol} qty={qty}")
                self._move_pending(order_id, "filled")
                return True
            if status in ('rejected', 'expired', 'canceled'):
                logger.warning(f"  ORDER {status.upper()}: {symbol} qty={qty}")
                self._move_pending(order_id, status)
                return False
            time.sleep(3)
        logger.info(f"  PENDING (unfilled yet): {symbol} qty={qty} id={order_id}")
        return False

    def reconcile_pending_orders(self):
        """Check orders left pending by previous runs; update their status."""
        data = self._load_tracking()
        if not data["pending"]:
            return
        still_pending = []
        for o in data["pending"]:
            if not o.get("order_id"):
                still_pending.append(o)
                continue
            order = self.helper.get_order_by_id(o["order_id"])
            status = getattr(order, 'status', None) if order else None
            if status == 'filled':
                logger.info(f"PREVIOUS ORDER FILLED: {o['symbol']} qty={o['qty']}")
                self._move_pending(o["order_id"], "filled")
            elif status in ('rejected', 'expired', 'canceled'):
                logger.info(f"PREVIOUS ORDER {status.upper()}: {o['symbol']} qty={o['qty']}")
                self._move_pending(o["order_id"], status)
            else:
                still_pending.append(o)
        data = self._load_tracking()
        data["pending"] = still_pending
        self._save_tracking(data)

    def _log_position_close(self, symbol: str, hold_days: int):
        """Log closed position to tracking file."""
        try:
            data = self._load_tracking()
            data["closed"].append({
                "symbol": symbol,
                "hold_days": hold_days,
                "closed_at": datetime.now().isoformat(),
                "action": "stale_close"
            })
            data["closed"] = data["closed"][-100:]
            self._save_tracking(data)
        except Exception as e:
            logger.warning(f"Failed to log position close: {e}")

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
        # Close stale positions (held too long) - CRITICAL: fix position staleness
        stale_closed = self._close_stale_positions(positions)
        if stale_closed:
            logger.info(f"Closed {len(stale_closed)} stale positions held >{cfg.MAX_HOLD_DAYS} days")
        time.sleep(2)

        # Recalculate after closing
        positions = self.helper.get_positions()
        held_symbols = [p.symbol for p in positions]
        # Symbols with open BUY orders count as held — server-side state survives
        # ephemeral GH Actions runners (local tracking file does not).
        open_orders = self.helper.get_open_orders()
        pending_buy_symbols = {
            o.symbol for o in open_orders
            if str(getattr(o, 'side', '')).upper() == 'BUY'
        }
        if pending_buy_symbols:
            logger.info(f"Pending open buy orders (counting as held): {sorted(pending_buy_symbols)}")
        held_symbols = sorted(set(held_symbols) | pending_buy_symbols)
        remaining_positions = len(held_symbols)

        # Execute new signals (submit + verify fill; track pending orders)
        submitted = []
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
                order_id = getattr(result, 'id', None)
                self._track_submitted_order(signal['symbol'], qty, order_id)
                submitted.append(signal['symbol'])
                held_symbols.append(signal['symbol'])
                remaining_positions += 1
                logger.info(f"  ORDER SUBMITTED: {signal['symbol']} qty={qty} id={order_id}")
                self._await_fill(order_id, signal['symbol'], qty)
            else:
                logger.warning(f"  FAILED BUY: {signal['symbol']}: {result}")

        logger.info(f"Submitted {len(submitted)} buy orders: {submitted} — market orders fill NEXT session (in-session fill check not applicable); pending orders tracked in v4_position_tracking.json; verify fills via daily_review.py")
        return submitted


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

    # Reconcile orders submitted by previous runs (fills/expiries overnight)
    transition.reconcile_pending_orders()

    account = helper.get_account()
    pv = float(account.portfolio_value)
    cash = float(account.cash)

    logger.info(f"Portfolio Value: ${pv:,.2f}")
    logger.info(f"Cash: ${cash:,.2f}")

    # Circuit breakers — rolling peak persisted in v4_peak_portfolio.json
    # (ratchets up only; drawdown measured vs all-time peak, survives restarts)
    peak_file = Path(__file__).parent / "v4_peak_portfolio.json"
    try:
        peak = float(json.loads(peak_file.read_text())["peak"])
    except Exception:
        peak = pv
    if pv > peak:
        peak = pv
        peak_file.write_text(json.dumps({"peak": peak, "updated": datetime.utcnow().isoformat() + "Z"}))
    can_trade, reason = risk.check_circuit_breakers(pv, peak)
    logger.info(f"Circuit breakers: pv=${pv:,.2f} peak=${peak:,.2f} dd={(pv - peak) / peak * 100:.2f}% can_trade={can_trade}")
    if not can_trade:
        logger.warning(f"CIRCUIT BREAKER: {reason}")
        return

    # Weekend guard: Sat/Sun — skip regime check (stale data would trigger
    # liquidations) and skip new buys (stale signals). Pendings reconciled above.
    if datetime.utcnow().weekday() >= 5:
        logger.warning("WEEKEND GUARD: Sat/Sun — holding portfolio as-is; no regime check, no liquidations, no new buys")
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
