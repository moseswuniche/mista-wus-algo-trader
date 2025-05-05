"""Handles risk management logic, including SL/TP/TSL and max loss checks."""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

# Import dependent components
from .position_manager import PositionManager
from .trade_logger import (
    CsvTradeLogger,
)  # Needed for cumulative profit access (temporary)

logger = logging.getLogger(__name__)


class RiskManager:
    """Monitors and enforces overall trading risk limits."""

    def __init__(
        self,
        position_manager: PositionManager,
        max_cumulative_loss: Optional[float] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        trailing_stop_loss_pct: Optional[float] = None,
        # Add other potential risk limits here (e.g., max drawdown, max open positions)
    ):
        self.position_manager = position_manager
        self.max_cumulative_loss = (
            abs(max_cumulative_loss) if max_cumulative_loss is not None else None
        )
        self._risk_limit_breached: bool = False
        self.stop_loss_pct: Optional[float] = stop_loss_pct
        self.take_profit_pct: Optional[float] = take_profit_pct
        self.trailing_stop_loss_pct: Optional[float] = trailing_stop_loss_pct
        self._max_loss_stop_triggered: bool = (
            False  # Internal flag if max loss caused stop
        )

        log_msg = "RiskManager initialized."
        if self.max_cumulative_loss is not None:
            log_msg += (
                f" Max Cumulative Loss Threshold: -{self.max_cumulative_loss:.2f}"
            )
        else:
            log_msg += " No Max Cumulative Loss threshold set."
        logger.info(log_msg)

    @property
    def risk_limit_breached(self) -> bool:
        """Returns True if a risk limit has been breached and trading should stop."""
        return self._risk_limit_breached

    def check_risk_limits(self) -> bool:
        """Checks if any defined risk limits have been breached.
        Updates the internal state and returns True if a limit was breached *now*,
        False otherwise (or if already breached).
        """
        if self._risk_limit_breached:
            return False  # Already breached, no new breach

        breached_now = False

        # 1. Check Max Cumulative Loss
        if self.max_cumulative_loss is not None:
            current_cum_profit = self.position_manager.cumulative_profit
            if current_cum_profit <= -self.max_cumulative_loss:
                logger.critical(
                    f"RISK LIMIT BREACHED: Maximum cumulative loss exceeded! "
                    f"(Current Profit: {current_cum_profit:.4f}, Limit: -{self.max_cumulative_loss:.4f})"
                )
                self._risk_limit_breached = True
                breached_now = True

        # 2. Add checks for other limits (e.g., daily loss, drawdown) here
        # Example:
        # if self.max_daily_loss is not None:
        #     daily_pnl = self._calculate_daily_pnl() # Requires tracking trades/pnl per day
        #     if daily_pnl <= -self.max_daily_loss:
        #         logger.critical("RISK LIMIT BREACHED: Maximum daily loss exceeded!")
        #         self._risk_limit_breached = True
        #         breached_now = True

        if breached_now:
            logger.warning("RiskManager flagged breach. Trading should halt.")

        return breached_now

    def reset_breached_flag(self) -> None:
        """Manually resets the breached flag (use with caution, e.g., after manual intervention)."""
        logger.warning("Risk limit breached flag manually reset.")
        self._risk_limit_breached = False

    # --- State Loading/Saving Helpers (for Orchestrator) ---
    # RiskManager state might be simpler, often just the breached flag if config is external

    def get_state(self) -> Dict[str, Any]:
        """Returns the current managed state as a dictionary."""
        return {
            "risk_limit_breached": self._risk_limit_breached
            # Only save dynamic state, config params are loaded at init
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Loads state from a dictionary."""
        logger.info("Loading risk manager state...")
        self._risk_limit_breached = state.get("risk_limit_breached", False)
        logger.info(f"Loaded state: RiskLimitBreached={self._risk_limit_breached}")

    def update_risk_parameters(
        self,
        stop_loss_pct: Optional[float] = np.nan,
        take_profit_pct: Optional[float] = np.nan,
        trailing_stop_loss_pct: Optional[float] = np.nan,
        max_cumulative_loss: Optional[float] = None,
    ):
        """Updates risk parameters, e.g., from runtime config."""
        changed = False
        if not pd.isna(stop_loss_pct) and stop_loss_pct != self.stop_loss_pct:
            self.stop_loss_pct = stop_loss_pct
            logger.info(f"RiskManager: Updated stop_loss_pct to {self.stop_loss_pct}")
            changed = True
        if not pd.isna(take_profit_pct) and take_profit_pct != self.take_profit_pct:
            self.take_profit_pct = take_profit_pct
            logger.info(
                f"RiskManager: Updated take_profit_pct to {self.take_profit_pct}"
            )
            changed = True
        if (
            not pd.isna(trailing_stop_loss_pct)
            and trailing_stop_loss_pct != self.trailing_stop_loss_pct
        ):
            self.trailing_stop_loss_pct = trailing_stop_loss_pct
            logger.info(
                f"RiskManager: Updated trailing_stop_loss_pct to {self.trailing_stop_loss_pct}"
            )
            changed = True
        new_max_loss = (
            abs(max_cumulative_loss) if max_cumulative_loss is not None else None
        )
        if new_max_loss != self.max_cumulative_loss:
            self.max_cumulative_loss = new_max_loss
            logger.info(
                f"RiskManager: Updated max_cumulative_loss to {self.max_cumulative_loss}"
            )
            changed = True

    def reset_sl_tp_state(self):
        """Resets SL/TP/TSL tracking variables, called when position is closed."""
        logger.debug("RiskManager SL/TP/TSL levels reset.")

    def calculate_set_initial_sl_tp(self):
        """Calculates and stores initial SL/TP levels based on entry price from PositionManager."""
        position = self.position_manager.position
        entry_price, _, _ = self.position_manager.get_entry_details()

        if position == 0 or entry_price is None:
            return

        # Logic moved to PositionManager
        # if self.stop_loss_pct:
        #     if position == 1: self.current_stop_loss = entry_price * (1 - self.stop_loss_pct)
        #     elif position == -1: self.current_stop_loss = entry_price * (1 + self.stop_loss_pct)
        #     logger.info(f"Initial Stop Loss set at: {self.current_stop_loss:.5f}")
        # else: self.current_stop_loss = None

        # Logic moved to PositionManager
        # if self.take_profit_pct:
        #     if position == 1: self.current_take_profit = entry_price * (1 + self.take_profit_pct)
        #     elif position == -1: self.current_take_profit = entry_price * (1 - self.take_profit_pct)
        #     logger.info(f"Take Profit set at: {self.current_take_profit:.5f}")
        # else: self.current_take_profit = None

        # Initialize TSL Peak Price
        # Logic moved to PositionManager
        # if self.trailing_stop_loss_pct:
        #     self.tsl_peak_price = entry_price
        #     logger.info(f"Trailing Stop Loss activated. Initial Peak: {self.tsl_peak_price:.5f}")
        # else: self.tsl_peak_price = None

        # REMOVED recalculation logic - PositionManager handles SL/TP on entry/update
        # if changed and self.position_manager.position != 0:
        #     logger.warning("Risk parameters updated while in position. SL/TP will use new values on *next* entry or TSL update.")

    # REMOVING second check_risk_limits definition (lines 168-227) ---
    # The first definition correctly handles max cumulative loss check.
    # SL/TP/TSL logic is now primarily in PositionManager.check_sl_tp_and_update_tsl

    # Logic moved to PositionManager
    # if self.trailing_stop_loss_pct and self.tsl_peak_price is not None:
    #     initial_sl = self.current_stop_loss
    #     potential_tsl_stop = None
    #     if position == 1:
    #         self.tsl_peak_price = max(self.tsl_peak_price, current_high)
    #         potential_tsl_stop = self.tsl_peak_price * (1 - self.trailing_stop_loss_pct)
    #         if potential_tsl_stop > (initial_sl or -np.inf): self.current_stop_loss = potential_tsl_stop
    #     elif position == -1:
    #         self.tsl_peak_price = min(self.tsl_peak_price, current_low)
    #         potential_tsl_stop = self.tsl_peak_price * (1 + self.trailing_stop_loss_pct)
    #         if potential_tsl_stop < (initial_sl or np.inf): self.current_stop_loss = potential_tsl_stop
    #     if self.current_stop_loss != initial_sl: logger.info(f"Trailing Stop Loss updated to: {self.current_stop_loss:.5f} (Peak: {self.tsl_peak_price:.5f})")

    # Logic moved to PositionManager
    # exit_reason = None
    # exit_price = None
    # if self.current_stop_loss is not None:
    #     if position == 1 and current_low <= self.current_stop_loss: exit_reason, exit_price = "Stop Loss / TSL", self.current_stop_loss
    #     elif position == -1 and current_high >= self.current_stop_loss: exit_reason, exit_price = "Stop Loss / TSL", self.current_stop_loss
    # if exit_reason is None and self.current_take_profit is not None:
    #     if position == 1 and current_high >= self.current_take_profit: exit_reason, exit_price = "Take Profit", self.current_take_profit
    #     elif position == -1 and current_low <= self.current_take_profit: exit_reason, exit_price = "Take Profit", self.current_take_profit

    # Logic moved to PositionManager
    # if exit_reason:
    #     logger.info(f"RiskManager triggering exit: {exit_reason} at price ~{exit_price:.5f}")
    #     # PositionManager handles the actual closing and state update
    #     closed_successfully = self.position_manager.close_position(context_message=f"Exit due to {exit_reason}")
    #     # If close was successful, PositionManager resets its state,
    #     # RiskManager needs its SL/TP state reset too.
    #     if closed_successfully:
    #          self.reset_sl_tp_state()
    #          # The caller (main Trader loop) should check _max_loss_stop_triggered if needed
    #     return True # Indicate an exit was triggered

    # return False # No exit triggered by risk limits
