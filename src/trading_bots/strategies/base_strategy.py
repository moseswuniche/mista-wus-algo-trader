import pandas as pd
from typing import Dict, Any, Tuple, ClassVar
import backtrader as bt
import datetime
import logging

# Define a type alias for parameter definitions
ParamDefinition = Tuple[str, Any]


# --- Backtrader Compatible Base Strategy --- #
class BaseStrategy(bt.Strategy):
    """
    Base strategy for backtrader integration, providing common methods
    for order management (SL/TP), logging, and parameter handling.
    Assumes strategy parameters include 'stop_loss_pct' and 'take_profit_pct' if SL/TP is desired.
    """

    # Default base parameters (can be overridden by subclasses)
    params: ClassVar[Tuple[ParamDefinition, ...]] = (
        ("stop_loss_pct", None),  # Percentage (e.g., 0.02 for 2%)
        ("take_profit_pct", None),  # Percentage (e.g., 0.04 for 4%)
        ("time_window", None),  # String like "HH:MM-HH:MM" or list of tuples
        ("liquidity_threshold", None),  # In quote currency
        ("apply_atr_filter", False),
        ("atr_period", 14),
        ("atr_threshold_multiplier", 1.5),
        ("atr_threshold", None),
        # Add other common params if needed (e.g., logging level)
    )

    def __init__(self):
        """Base initializer. Can be extended by subclasses."""
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.trade_count = 0
        self.log(f"Strategy Initialized: {self.__class__.__name__}")
        self.log(f"Parameters: {self.p._getkwargs()}")

    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f"{dt.isoformat()} - {self.__class__.__name__} - {txt}")

    def notify_order(self, order):
        """Handles order notifications and SL/TP bracket management."""
        if order.status in [order.Submitted, order.Accepted]:
            # An active order exists - nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"BUY EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
                self.entry_price = order.executed.price
                self.entry_bar = len(self)
            elif order.issell():
                self.log(
                    f"SELL EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}"
                )
                self.entry_price = order.executed.price
                self.entry_bar = len(self)

            # Check if the completed order was the main entry/exit order
            if (
                order.ref == self.order.ref
            ):  # Check if it's the primary order we submitted
                self.order = None  # Main order completed, allow new orders

        elif order.status in [
            order.Canceled,
            order.Margin,
            order.Rejected,
            order.Expired,
        ]:
            self.log(
                f"Order Canceled/Margin/Rejected/Expired - Status: {order.getstatusname()}"
            )
            # If the main order failed, reset self.order
            if self.order and order.ref == self.order.ref:
                self.order = None

        # Note: If using bracket orders, SL/TP orders might complete or be cancelled here.
        # The core logic doesn't explicitly track bracket orders after submission,
        # relying on backtrader to manage them.
        # If a SL/TP hits, the position will be closed, and self.order should be None allowing new entries.

    def _is_within_time_window(self) -> bool:
        """Check if the current time is within the allowed trading window."""
        if not self.p.time_window:
            return True  # No window defined, always allowed

        try:
            current_time = self.datas[0].datetime.time(0)
            start_str, end_str = self.p.time_window.split("-")
            start_time = datetime.datetime.strptime(start_str, "%H:%M").time()
            end_time = datetime.datetime.strptime(end_str, "%H:%M").time()

            if start_time <= end_time:
                is_allowed = start_time <= current_time < end_time
            else:  # Handles overnight window (e.g., 22:00-06:00)
                is_allowed = current_time >= start_time or current_time < end_time
            return bool(is_allowed)
        except Exception as e:
            self.log(f"ERROR parsing time window {self.p.time_window}: {e}")
            return True  # Default to True on error

    def _is_liquid_enough(self) -> bool:
        """Check if the current candle meets the minimum liquidity threshold."""
        if self.p.liquidity_threshold is None:
            return True  # No threshold set

        try:
            if len(self.data.volume) == 0 or len(self.data.close) == 0:
                self.log(
                    "Warning: Volume or Close data not available for liquidity check."
                )
                return True

            current_volume = self.data.volume[0]
            current_close = self.data.close[0]
            current_volume_quote = current_volume * current_close
            is_liquid = current_volume_quote >= self.p.liquidity_threshold
            return bool(is_liquid)
        except Exception as e:
            self.log(f"ERROR checking liquidity: {e}")
            return True

    def _check_atr_volatility(self) -> bool:
        """Check if ATR is below the configured threshold (if filter enabled)."""
        if not self.p.apply_atr_filter:
            return True  # Filter not enabled

        try:
            current_atr = self.data.atr[
                0
            ]  # Assumes ATR is calculated and available on self.data
            threshold = self.p.atr_threshold

            if threshold is None and self.p.atr_threshold_multiplier is not None:
                threshold = current_atr * self.p.atr_threshold_multiplier
                if threshold is None:
                    self.log(
                        "Warning: Could not calculate ATR threshold from multiplier."
                    )
                    return True
            elif threshold is None:
                self.log(
                    "Warning: ATR filter enabled but no threshold or multiplier set."
                )
                return True

            if pd.isna(current_atr) or pd.isna(threshold):
                self.log("Warning: ATR or Threshold is NaN, filter inactive.")
                return True

            is_below_threshold = current_atr < threshold
            if not is_below_threshold:
                # self.log(f"ATR volatility too high: {current_atr:.4f} >= {threshold:.4f}")
                pass
            return bool(is_below_threshold)
        except IndexError:
            self.log("Warning: Not enough data for ATR calculation.")
            return True
        except AttributeError:
            self.log(
                "Warning: self.data.atr not available. Ensure ATR indicator is added."
            )
            return True
        except Exception as e:
            self.log(f"ERROR checking ATR volatility: {e}")
            return True

    # --- Override buy/sell/close for common checks and SL/TP ---

    def enter_long(self, **kwargs):
        """Enters a long position with optional SL/TP."""
        if not self._is_within_time_window() or not self._is_liquid_enough():
            return None  # Don't place order if outside window or not liquid

        sl_price = None
        tp_price = None
        current_price = self.data.close[0]

        if self.p.stop_loss_pct:
            sl_price = current_price * (1 - self.p.stop_loss_pct)
        if self.p.take_profit_pct:
            tp_price = current_price * (1 + self.p.take_profit_pct)

        if sl_price is not None and tp_price is not None:
            self.log(
                f"Submitting BUY Bracket: Entry ~{current_price:.5f}, TP: {tp_price:.5f}, SL: {sl_price:.5f}"
            )
            self.order = self.buy_bracket(
                price=current_price, stopprice=sl_price, limitprice=tp_price, **kwargs
            )
        elif sl_price is not None:
            self.log(
                f"Submitting BUY + SL: Entry ~{current_price:.5f}, SL: {sl_price:.5f}"
            )
            self.order = self.buy(exectype=bt.Order.Market, **kwargs)
            # Note: backtrader doesn't have a simple buy + separate stop order submission in one go easily
            # self.sell(exectype=bt.Order.Stop, price=sl_price, parent=self.order) # Requires careful management
            # Using buy_bracket is generally preferred if platform supports it.
            # For simplicity here, we rely on manual SL check in `next` if only SL is set.
        elif tp_price is not None:
            self.log(
                f"Submitting BUY + TP: Entry ~{current_price:.5f}, TP: {tp_price:.5f}"
            )
            self.order = self.buy(exectype=bt.Order.Market, **kwargs)
            # self.sell(exectype=bt.Order.Limit, price=tp_price, parent=self.order) # Requires careful management
            # For simplicity, rely on manual TP check.
        else:
            self.log(f"Submitting BUY Market @ ~{current_price:.5f}")
            self.order = self.buy(**kwargs)

        return self.order

    def enter_short(self, **kwargs):
        """Enters a short position with optional SL/TP."""
        if not self._is_within_time_window() or not self._is_liquid_enough():
            return None

        sl_price = None
        tp_price = None
        current_price = self.data.close[0]

        if self.p.stop_loss_pct:
            sl_price = current_price * (1 + self.p.stop_loss_pct)
        if self.p.take_profit_pct:
            tp_price = current_price * (1 - self.p.take_profit_pct)

        if sl_price is not None and tp_price is not None:
            self.log(
                f"Submitting SELL Bracket: Entry ~{current_price:.5f}, TP: {tp_price:.5f}, SL: {sl_price:.5f}"
            )
            self.order = self.sell_bracket(
                price=current_price, stopprice=sl_price, limitprice=tp_price, **kwargs
            )
        elif sl_price is not None:
            self.log(
                f"Submitting SELL + SL: Entry ~{current_price:.5f}, SL: {sl_price:.5f}"
            )
            self.order = self.sell(exectype=bt.Order.Market, **kwargs)
            # self.buy(exectype=bt.Order.Stop, price=sl_price, parent=self.order)
        elif tp_price is not None:
            self.log(
                f"Submitting SELL + TP: Entry ~{current_price:.5f}, TP: {tp_price:.5f}"
            )
            self.order = self.sell(exectype=bt.Order.Market, **kwargs)
            # self.buy(exectype=bt.Order.Limit, price=tp_price, parent=self.order)
        else:
            self.log(f"Submitting SELL Market @ ~{current_price:.5f}")
            self.order = self.sell(**kwargs)

        return self.order

    def close_position(self, **kwargs):
        """Closes the current position."""
        self.log(f"Submitting CLOSE @ {self.data.close[0]:.5f}")
        self.order = self.close(**kwargs)  # Use backtrader's close
        return self.order

    # --- Default next method - subclasses should override --- #
    def next(self):
        # Check if an order is pending
        if self.order:
            return

        # Check if basic filters pass before evaluating strategy logic
        if not self._is_within_time_window() or not self._is_liquid_enough():
            # If we are in a position and time/liquidity becomes invalid, consider closing?
            if self.position:
                self.log("Condition (Time/Liquidity) invalid, closing position.")
                self.close_position()
            return  # Skip signal generation

        # Subclasses should implement their signal generation and call
        # self.enter_long(), self.enter_short(), or self.close_position()
        pass

    def get_required_lookback(self) -> int:
        """Calculates the minimum number of bars required based on parameters."""
        required = 0
        potential_period_params = [
            "period",
            "fast_period",
            "slow_period",
            "ema_fast",
            "ema_slow",
            "rsi_period",
            "bb_period",
            "atr_period",
            "zscore_period",
            "short_window",
            "long_window",  # Add other common period names
        ]
        # Use _getkwargs().items() to iterate through parameters
        for param_name, value in self.p._getkwargs().items():
            # Check if param name suggests it's a period/lookback
            if any(pp in param_name.lower() for pp in potential_period_params):
                if isinstance(value, int) and value > required:
                    required = value
        # Add a small buffer just in case (e.g., for SMA calculation within ATR)
        return required + 5 if required > 0 else 50  # Default to 50 if no periods found
