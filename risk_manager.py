#!/usr/bin/env python3
"""
Risk Management Module — Post Quant Review Fixes.
Implements every actionable fix from the brutal review.
"""

import os
import json
import pickle
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================
MAX_DAILY_LOSS_PCT = 0.05      # Halt trading if daily loss > 5%
MAX_MONTHLY_LOSS_PCT = 0.15    # Halt trading if monthly loss > 15%
MAX_MARGIN_PCT = 0.50          # Max 50% margin utilization
VIX_HIGH_THRESHOLD = 25        # Reduce size when VIX > 25
VIX_SIZE_MULTIPLIER = 0.5      # Half size when VIX high
HALF_KELLY = True              # Use half-Kelly (aggressive ceiling for pros)
SLIPPAGE_BPS_LARGE = 10        # 0.1% slippage on large caps (in bps)
SLIPPAGE_BPS_SMALL = 50        # 0.5% slippage on small caps (in bps)
SHARPE_KILL_THRESHOLD = 0.5    # Kill strategy if 30-day Sharpe < 0.5
SHARPE_LOOKBACK_DAYS = 30
MAX_POSITIONS_BULL = 5
MAX_POSITIONS_BEAR = 2
MAX_POSITIONS_SIDEWAYS = 3

class RiskManager:
    """Centralized risk management — fixes the D+ grade."""
    
    def __init__(self, state_file='risk_state.json'):
        self.state_file = Path(state_file)
        self.state = self.load_state()
    
    def load_state(self):
        """Load or initialize risk state."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {
            'daily_pnl': {},
            'monthly_pnl': {},
            'trade_history': [],
            'sharpe_window': deque(maxlen=SHARPE_LOOKBACK_DAYS),
            'halted': False,
            'halt_reason': None,
            'peak_portfolio_value': 0,
            'max_drawdown': 0,
            'current_vix': 15,
            'last_updated': datetime.now().isoformat()
        }
    
    def save_state(self):
        """Persist risk state."""
        # Convert deque to list for JSON
        state = self.state.copy()
        state['sharpe_window'] = list(state['sharpe_window'])
        state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def get_vix(self):
        """Fetch current VIX level."""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from production_v18 import fetch_data
            closes, _, _, _ = fetch_data('^VIX', days=5)
            if closes:
                return closes[-1]
        except Exception as e:
            logger.warning(f"Failed to fetch VIX: {e}")
        return 15  # Default if unavailable
    
    def check_drawdown_breaker(self, portfolio_value):
        """
        Circuit breaker: halt if daily or monthly loss exceeds threshold.
        Returns: (should_halt, reason)
        """
        today = datetime.now().strftime('%Y-%m-%d')
        month = datetime.now().strftime('%Y-%m')
        
        # Update peak
        if portfolio_value > self.state['peak_portfolio_value']:
            self.state['peak_portfolio_value'] = portfolio_value
        
        # Calculate drawdown
        if self.state['peak_portfolio_value'] > 0:
            drawdown = (self.state['peak_portfolio_value'] - portfolio_value) / self.state['peak_portfolio_value']
            self.state['max_drawdown'] = max(self.state['max_drawdown'], drawdown)
        
        # Daily check
        daily_pnl = self.state['daily_pnl'].get(today, 0)
        if daily_pnl < -portfolio_value * MAX_DAILY_LOSS_PCT:
            return True, f"Daily loss {daily_pnl:.2f} exceeds {MAX_DAILY_LOSS_PCT:.0%} limit"
        
        # Monthly check
        monthly_pnl = self.state['monthly_pnl'].get(month, 0)
        if monthly_pnl < -portfolio_value * MAX_MONTHLY_LOSS_PCT:
            return True, f"Monthly loss {monthly_pnl:.2f} exceeds {MAX_MONTHLY_LOSS_PCT:.0%} limit"
        
        return False, None
    
    def check_margin_cap(self, cash, portfolio_value):
        """
        Reduce position if margin usage exceeds 50%.
        Returns: max_position_size_multiplier (0.0 to 1.0)
        """
        if portfolio_value <= 0:
            return 0
        
        margin_used = abs(min(cash, 0))  # Negative cash = margin
        margin_pct = margin_used / portfolio_value
        
        if margin_pct > MAX_MARGIN_PCT:
            # Reduce new position sizes proportionally
            reduction = MAX_MARGIN_PCT / margin_pct
            logger.warning(f"Margin at {margin_pct:.0%} > {MAX_MARGIN_PCT:.0%}. Reducing size to {reduction:.0%}")
            return reduction
        
        return 1.0
    
    def get_volatility_multiplier(self):
        """
        Volatility targeting: reduce size when VIX > 25.
        Returns: size multiplier (0.0 to 1.0)
        """
        vix = self.get_vix()
        self.state['current_vix'] = vix
        
        if vix > VIX_HIGH_THRESHOLD:
            logger.info(f"VIX at {vix:.1f} > {VIX_HIGH_THRESHOLD}. Reducing size by {VIX_SIZE_MULTIPLIER:.0%}")
            return VIX_SIZE_MULTIPLIER
        
        return 1.0
    
    def calculate_kelly(self, win_prob, payout_ratio, max_risk=0.05):
        """
        Half-Kelly position sizing (aggressive ceiling for pros).
        Kelly = (win_prob * payout_ratio - (1 - win_prob)) / payout_ratio
        Half-Kelly = Kelly / 2
        """
        if payout_ratio <= 0 or win_prob <= 0:
            return 0
        
        # Full Kelly
        kelly = (win_prob * payout_ratio - (1 - win_prob)) / payout_ratio
        
        # Half-Kelly (conservative)
        if HALF_KELLY:
            kelly = kelly / 2
        
        # Cap at max_risk
        kelly = min(kelly, max_risk)
        
        # No negative Kelly
        return max(kelly, 0)
    
    def apply_slippage(self, expected_return, is_small_cap=False):
        """
        Slippage model: reduce expected returns by estimated slippage.
        Large caps: 0.1% (10 bps)
        Small caps: 0.5% (50 bps)
        """
        slippage = SLIPPAGE_BPS_SMALL / 10000 if is_small_cap else SLIPPAGE_BPS_LARGE / 10000
        return expected_return - slippage
    
    def track_trade(self, symbol, entry_price, exit_price, shares, is_small_cap=False):
        """Track closed trade for Sharpe calculation."""
        pnl = (exit_price - entry_price) * shares
        return_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        
        # Apply slippage cost
        slippage_cost = (SLIPPAGE_BPS_SMALL if is_small_cap else SLIPPAGE_BPS_LARGE) / 10000 * entry_price * shares
        net_pnl = pnl - slippage_cost
        
        self.state['trade_history'].append({
            'symbol': symbol,
            'entry': entry_price,
            'exit': exit_price,
            'shares': shares,
            'pnl': pnl,
            'slippage_cost': slippage_cost,
            'net_pnl': net_pnl,
            'return': return_pct - (SLIPPAGE_BPS_SMALL / 10000 if is_small_cap else SLIPPAGE_BPS_LARGE / 10000),
            'is_small_cap': is_small_cap,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update daily/monthly PnL
        today = datetime.now().strftime('%Y-%m-%d')
        month = datetime.now().strftime('%Y-%m')
        self.state['daily_pnl'][today] = self.state['daily_pnl'].get(today, 0) + net_pnl
        self.state['monthly_pnl'][month] = self.state['monthly_pnl'].get(month, 0) + net_pnl
        
        # Update Sharpe window
        self.state['sharpe_window'].append(return_pct)
        
        self.save_state()
        return net_pnl
    
    def check_sharpe_kill_switch(self, portfolio_value):
        """
        Kill switch: if 30-day Sharpe ratio < 0.5, halt trading.
        Returns: (should_kill, sharpe_ratio)
        """
        if len(self.state['sharpe_window']) < 10:  # Need at least 10 trades
            return False, None
        
        returns = list(self.state['sharpe_window'])
        if len(returns) < 2:
            return False, None
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return False, None
        
        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        
        if sharpe < SHARPE_KILL_THRESHOLD:
            logger.warning(f"30-day Sharpe {sharpe:.2f} < {SHARPE_KILL_THRESHOLD}. KILL SWITCH.")
            return True, sharpe
        
        return False, sharpe
    
    def get_portfolio_correlation(self, positions, new_symbol_sector, stock_to_sector):
        """
        Correlation-adjusted sizing: limit sector exposure.
        Returns: correlation penalty (0.0 to 1.0)
        """
        if not positions:
            return 1.0
        
        # Count positions in same sector
        sector_count = 0
        for pos in positions:
            pos_sector = stock_to_sector.get(pos.symbol, 'unknown')
            if pos_sector == new_symbol_sector:
                sector_count += 1
        
        # Max 2 per sector
        if sector_count >= 2:
            return 0  # Block
        
        # Reduce size if 1 already in sector
        if sector_count == 1:
            return 0.5
        
        return 1.0
    
    def get_max_positions(self, regime):
        """Get max positions based on regime."""
        if 'bull' in regime:
            return MAX_POSITIONS_BULL
        elif 'bear' in regime:
            return MAX_POSITIONS_BEAR
        else:
            return MAX_POSITIONS_SIDEWAYS
    
    def get_composite_risk_multiplier(self, portfolio_value, cash, positions, new_symbol_sector, stock_to_sector):
        """
        Calculate composite risk multiplier combining all checks.
        Returns: (multiplier, reasons_list)
        """
        reasons = []
        multiplier = 1.0
        
        # 1. Drawdown breaker
        should_halt, reason = self.check_drawdown_breaker(portfolio_value)
        if should_halt:
            self.state['halted'] = True
            self.state['halt_reason'] = reason
            self.save_state()
            return 0, [f"HALT: {reason}"]
        
        # 2. Margin cap
        margin_mult = self.check_margin_cap(cash, portfolio_value)
        if margin_mult < 1.0:
            reasons.append(f"Margin cap: {margin_mult:.0%}")
            multiplier *= margin_mult
        
        # 3. Volatility targeting
        vol_mult = self.get_volatility_multiplier()
        if vol_mult < 1.0:
            reasons.append(f"VIX high: {vol_mult:.0%}")
            multiplier *= vol_mult
        
        # 4. Correlation/sector limit
        corr_mult = self.get_portfolio_correlation(positions, new_symbol_sector, stock_to_sector)
        if corr_mult == 0:
            reasons.append("Sector limit reached")
            return 0, reasons
        elif corr_mult < 1.0:
            reasons.append(f"Sector risk: {corr_mult:.0%}")
            multiplier *= corr_mult
        
        # 5. Sharpe kill switch
        should_kill, sharpe = self.check_sharpe_kill_switch(portfolio_value)
        if should_kill:
            self.state['halted'] = True
            self.state['halt_reason'] = f"Sharpe kill switch: {sharpe:.2f}"
            self.save_state()
            return 0, [f"HALT: Sharpe {sharpe:.2f} < {SHARPE_KILL_THRESHOLD}"]
        
        self.save_state()
        return multiplier, reasons
    
    def get_risk_report(self):
        """Generate risk report for cron job."""
        should_halt, halt_reason = self.check_drawdown_breaker(
            self.state['peak_portfolio_value']
        )
        
        should_kill, sharpe = self.check_sharpe_kill_switch(
            self.state['peak_portfolio_value']
        )
        
        return {
            'max_drawdown': self.state['max_drawdown'],
            'current_vix': self.state['current_vix'],
            'halted': self.state['halted'],
            'halt_reason': self.state['halt_reason'],
            'sharpe_30d': sharpe,
            'trade_count': len(self.state['trade_history']),
            'peak_value': self.state['peak_portfolio_value']
        }

# Global instance
risk_manager = RiskManager()
