"""Manages the current trading position state (long/short/flat, entry details)."""

import logging
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from binance.exceptions import BinanceAPIException

# Import dependent components
from .order_executor import OrderExecutor
from .trade_logger import CsvTradeLogger
from .client_manager import ClientManager  # Corrected import

logger = logging.getLogger(__name__)

# --- Type Aliases --- (Consider moving to a shared types module)
Position = int  # -1, 0, 1
Symbol = str
OrderId = Optional[str]
Timestamp = Optional[pd.Timestamp]
# --- End Type Aliases ---


class PositionManager:
    """Manages the trading position state, including entry price, SL/TP levels, and PnL."""

    def __init__(
        self,
        symbol: Symbol,
        units: float,  # Added units back
        order_executor: OrderExecutor,
        trade_logger: CsvTradeLogger,
        client_manager: ClientManager,  # Inject ClientManager
        initial_position: Position = 0,
        # SL/TP configuration
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        trailing_stop_loss_pct: Optional[float] = None,
        # Commission for PnL
        commission_bps: float = 0.0,  # e.g., 7.5 for 0.075%
        initial_entry_price: Optional[float] = None,
        initial_entry_time: Optional[pd.Timestamp] = None,
        initial_entry_order_id: Optional[str] = None,
    ):
        self.symbol = symbol
        self.units = units
        self.executor = order_executor
        self.logger = trade_logger
        self.client_manager = client_manager

        # Core position state
        self._position: Position = initial_position
        self.entry_price: Optional[float] = initial_entry_price
        self.entry_time: Timestamp = initial_entry_time
        self.entry_order_id: OrderId = initial_entry_order_id

        # SL/TP configuration and state
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_loss_pct = trailing_stop_loss_pct
        self.current_stop_loss: Optional[float] = None
        self.current_take_profit: Optional[float] = None
        self.tsl_peak_price: Optional[float] = None

        # PnL Tracking
        self.commission_rate = commission_bps / 10000.0
        self.cumulative_profit: float = 0.0

        logger.info(
            f"PositionManager initialized for {symbol}. Initial Pos: {self._position}"
        )

    # --- Position State Access --- #

    @property
    def position(self) -> Position:
        return self._position

    def is_flat(self) -> bool:
        return self._position == 0

    def is_long(self) -> bool:
        return self._position == 1

    def is_short(self) -> bool:
        return self._position == -1

    def get_entry_details(
        self,
    ) -> tuple[Optional[float], Optional[pd.Timestamp], Optional[str]]:
        """Returns the current entry price, time, and order ID."""
        return self.entry_price, self.entry_time, self.entry_order_id

    def set_state(
        self,
        position: Position,
        entry_price: Optional[float],
        entry_time: Optional[pd.Timestamp],
        entry_order_id: Optional[str],
    ):
        """Allows setting the state, e.g., when loading from StateManager."""
        self._position = position
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.entry_order_id = entry_order_id
        logger.info(
            f"PositionManager state set: Pos={self._position}, EntryPrice={self.entry_price}"
        )

    def _get_base_asset(self) -> str:
        """Extracts the base asset from the symbol (e.g., XRP from XRPUSDT)."""
        # Basic implementation, assumes USDT, BUSD, BTC, ETH etc. as quote
        quote_assets = ["USDT", "BUSD", "BTC", "ETH", "EUR", "GBP", "AUD"]
        for quote in quote_assets:
            if self.symbol.endswith(quote):
                return self.symbol[: -len(quote)]
        logger.warning(f"Could not determine base asset from symbol: {self.symbol}")
        return self.symbol[:3]

    def _get_current_position_size(self) -> float:
        """Queries the exchange for the free balance of the base asset.
        Returns 0.0 on error or if client not available.
        """
        base_asset = self._get_base_asset()
        balance_info = self.client_manager.get_asset_balance(asset=base_asset)
        if balance_info and "free" in balance_info:
            try:
                size = float(balance_info["free"])
                logger.debug(f"Current free balance query for {base_asset}: {size}")
                return size
            except (ValueError, TypeError):
                logger.error(
                    f"Could not convert balance 'free' value to float: {balance_info['free']}"
                )
                return 0.0
        else:
            logger.warning(
                f"Could not get balance info or 'free' field for {base_asset}. Response: {balance_info}"
            )
            return 0.0

    # --- Position Update Methods --- #

    def record_entry(self, order: Dict[str, Any]) -> None:
        """Records a new position entry based on a successful order."""
        side = order.get("side")
        if side not in ["BUY", "SELL"]:
            logger.error(f"Cannot record entry: Invalid side '{side}' in order.")
            return

        new_position = 1 if side == "BUY" else -1
        if self._position != 0:
            logger.warning(
                f"Recording entry for {side} while already in position {self._position}. Overwriting previous entry state."
            )

        self._position = new_position
        self.entry_order_id = order.get("orderId")

        # Set entry time
        entry_time_ms = order.get("transactTime")
        self.entry_time = (
            pd.Timestamp(entry_time_ms, unit="ms")
            if entry_time_ms
            else pd.Timestamp.now(tz="UTC")
        )

        # Calculate entry price and set initial SL/TP
        self._calculate_set_entry_price_sl_tp(order)

        logger.info(
            f"Position entry recorded: Position={self._position}, EntryPrice={self.entry_price}, EntryTime={self.entry_time}, OrderID={self.entry_order_id}"
        )
        # Note: State persistence should be handled by the orchestrator

    def record_exit(
        self, order: Dict[str, Any], exit_reason: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Records a position exit, calculates PnL, resets state. Returns (Net PnL, Cumulative PnL)."""
        if self.is_flat():
            logger.warning("Attempted to record exit, but already flat.")
            return None, self.cumulative_profit

        executed_qty = float(order.get("executedQty", 0))
        if executed_qty <= 0:
            logger.warning(
                f"Exit order {order.get('orderId')} has zero executed quantity. Cannot calculate PnL."
            )
            # Still reset position state as the intent was to exit
            self._reset_position_state()
            return None, self.cumulative_profit

        # Calculate exit price
        exit_price = self._calculate_exit_price(order, executed_qty)
        if exit_price is None:
            logger.error(
                f"Could not determine exit price for order {order.get('orderId')}. Cannot calculate PnL."
            )
            # Still reset position state
            self._reset_position_state()
            return None, self.cumulative_profit

        # Get exit time
        exit_time_ms = order.get("transactTime")
        exit_time = (
            pd.Timestamp(exit_time_ms, unit="ms")
            if exit_time_ms
            else pd.Timestamp.now(tz="UTC")
        )

        # Calculate PnL
        net_pnl = self._calculate_pnl(exit_price, executed_qty)
        if net_pnl is not None:
            self.cumulative_profit += net_pnl
            logger.info(
                f"Position exit recorded: Reason='{exit_reason}', ExitPrice={exit_price:.5f}, ExitTime={exit_time}, NetPnL={net_pnl:.4f}, CumPnL={self.cumulative_profit:.4f}"
            )
        else:
            logger.warning(
                "Net PnL calculation failed (likely missing entry price). Cum PnL unchanged."
            )

        # Reset position state AFTER calculation and logging
        self._reset_position_state()

        return net_pnl, self.cumulative_profit

    def _reset_position_state(self) -> None:
        """Resets state variables associated with an open position."""
        logger.debug("Resetting position state (entry price, time, SL/TP, etc.).")
        self._position = 0
        self.entry_price = None
        self.entry_time = None
        self.entry_order_id = None
        self.current_stop_loss = None
        self.current_take_profit = None
        self.tsl_peak_price = None
        # Note: cumulative_profit is NOT reset here
        # Note: State persistence should be handled by the orchestrator

    # --- SL/TP Calculation and Checking --- #

    def _calculate_set_entry_price_sl_tp(self, entry_order: Dict[str, Any]) -> None:
        """Calculates entry price and sets initial SL/TP levels."""
        try:
            # Calculate average fill price
            fills = entry_order.get("fills", [])
            if fills:
                fill_prices = [float(fill["price"]) for fill in fills]
                fill_qtys = [float(fill["qty"]) for fill in fills]
                total_qty = sum(fill_qtys)
                if total_qty > 0:
                    self.entry_price = (
                        sum(p * q for p, q in zip(fill_prices, fill_qtys)) / total_qty
                    )
                else:
                    logger.warning(
                        "Entry order fills have zero total quantity. Cannot calculate weighted average price."
                    )
                    self.entry_price = None
            else:  # Fallback if no fills info (shouldn't happen for FILLED market order)
                logger.warning(
                    "No fill data in entry order. Using 'cummulativeQuoteQty' if available."
                )
                cum_quote_qty = float(entry_order.get("cummulativeQuoteQty", 0))
                executed_qty = float(entry_order.get("executedQty", 0))
                if executed_qty > 0:
                    self.entry_price = cum_quote_qty / executed_qty
                else:
                    self.entry_price = None

            if self.entry_price is None:
                logger.error(
                    "Could not determine entry price from order. SL/TP cannot be set."
                )
                self._reset_position_state()  # Ensure state consistency
                return

            logger.info(f"Calculated Entry Price: {self.entry_price:.5f}")

            # Calculate Fixed Stop Loss
            if self.stop_loss_pct:
                if self.is_long():
                    self.current_stop_loss = self.entry_price * (1 - self.stop_loss_pct)
                elif self.is_short():
                    self.current_stop_loss = self.entry_price * (1 + self.stop_loss_pct)
                logger.info(f"Initial Stop Loss set at: {self.current_stop_loss:.5f}")
            else:
                self.current_stop_loss = None

            # Calculate Fixed Take Profit
            if self.take_profit_pct:
                if self.is_long():
                    self.current_take_profit = self.entry_price * (
                        1 + self.take_profit_pct
                    )
                elif self.is_short():
                    self.current_take_profit = self.entry_price * (
                        1 - self.take_profit_pct
                    )
                logger.info(f"Take Profit set at: {self.current_take_profit:.5f}")
            else:
                self.current_take_profit = None

            # Initialize TSL Peak Price
            if self.trailing_stop_loss_pct:
                self.tsl_peak_price = self.entry_price
                logger.info(
                    f"Trailing Stop Loss activated. Initial Peak: {self.tsl_peak_price:.5f}"
                )
            else:
                self.tsl_peak_price = None

        except Exception as e:
            logger.error(
                f"Error calculating/setting entry price or SL/TP: {e}", exc_info=True
            )
            self._reset_position_state()  # Reset state on error

    def check_sl_tp_and_update_tsl(
        self, current_high: float, current_low: float
    ) -> Optional[Tuple[str, float]]:
        """Checks for SL/TP hits, updates TSL, and closes position if triggered.

        Args:
            current_high: The high price of the current (or most recent) bar.
            current_low: The low price of the current (or most recent) bar.

        Returns:
            A tuple (exit_reason, exit_price) if SL/TP triggered an exit, otherwise None.
            Note: PositionManager calls self.close_position internally if hit occurs.
        """
        if self.is_flat():
            return None

        exit_reason: Optional[str] = None
        exit_price: Optional[float] = None

        # --- Check Fixed SL/TP First ---
        if self.is_long():
            if (
                self.current_stop_loss is not None
                and current_low <= self.current_stop_loss
            ):
                exit_reason = "Stop Loss"
                exit_price = self.current_stop_loss  # Exit at SL price
            elif (
                self.current_take_profit is not None
                and current_high >= self.current_take_profit
            ):
                exit_reason = "Take Profit"
                exit_price = self.current_take_profit  # Exit at TP price
        elif self.is_short():
            if (
                self.current_stop_loss is not None
                and current_high >= self.current_stop_loss
            ):
                exit_reason = "Stop Loss"
                exit_price = self.current_stop_loss
            elif (
                self.current_take_profit is not None
                and current_low <= self.current_take_profit
            ):
                exit_reason = "Take Profit"
                exit_price = self.current_take_profit

        # --- Update Trailing Stop Loss (if no fixed SL/TP hit yet) ---
        if (
            exit_reason is None
            and self.trailing_stop_loss_pct is not None
            and self.entry_price is not None
        ):
            new_stop_loss = None
            if self.is_long():
                # Initialize peak if needed
                if (
                    self.tsl_peak_price is None
                    or self.tsl_peak_price < self.entry_price
                ):
                    self.tsl_peak_price = self.entry_price
                # Update peak
                self.tsl_peak_price = max(self.tsl_peak_price, current_high)
                # Calculate new TSL stop
                new_stop_loss = self.tsl_peak_price * (1 - self.trailing_stop_loss_pct)
                # Update current_stop_loss if TSL is higher
                if (
                    self.current_stop_loss is None
                    or new_stop_loss > self.current_stop_loss
                ):
                    if new_stop_loss != self.current_stop_loss:
                        logger.info(
                            f"Trailing Stop Loss updated for LONG: Peak={self.tsl_peak_price:.5f}, New SL={new_stop_loss:.5f}"
                        )
                        self.current_stop_loss = new_stop_loss
                    # No state saving here - done by Trader after this call returns

            elif self.is_short():
                # Initialize peak if needed
                if (
                    self.tsl_peak_price is None
                    or self.tsl_peak_price > self.entry_price
                ):
                    self.tsl_peak_price = self.entry_price
                # Update peak
                self.tsl_peak_price = min(self.tsl_peak_price, current_low)
                # Calculate new TSL stop
                new_stop_loss = self.tsl_peak_price * (1 + self.trailing_stop_loss_pct)
                # Update current_stop_loss if TSL is lower
                if (
                    self.current_stop_loss is None
                    or new_stop_loss < self.current_stop_loss
                ):
                    if new_stop_loss != self.current_stop_loss:
                        logger.info(
                            f"Trailing Stop Loss updated for SHORT: Peak={self.tsl_peak_price:.5f}, New SL={new_stop_loss:.5f}"
                        )
                        self.current_stop_loss = new_stop_loss
                    # No state saving here

            # --- Re-check if TSL hit after update ---
            if self.current_stop_loss is not None:
                if self.is_long() and current_low <= self.current_stop_loss:
                    exit_reason = "Trailing Stop Loss"
                    exit_price = self.current_stop_loss
                elif self.is_short() and current_high >= self.current_stop_loss:
                    exit_reason = "Trailing Stop Loss"
                    exit_price = self.current_stop_loss

        # --- Execute Exit if Triggered --- #
        if exit_reason and exit_price is not None:
            logger.info(
                f"{exit_reason} hit at estimated price {exit_price:.5f}. Closing position."
            )
            self.close_position(exit_reason=exit_reason)  # Pass reason internally
            return exit_reason, exit_price  # Return info indicating an exit occurred
        else:
            return None  # No exit triggered

    # --- PnL Calculation --- #

    def _calculate_exit_price(
        self, exit_order: Dict[str, Any], executed_qty: float
    ) -> Optional[float]:
        """Calculates the average exit price from the order details."""
        # Preferentially use cummulativeQuoteQty for accuracy
        cum_quote_qty = float(exit_order.get("cummulativeQuoteQty", 0))
        if cum_quote_qty > 0 and executed_qty > 0:
            return cum_quote_qty / executed_qty

        # Fallback to fills if cummulativeQuoteQty is missing/zero
        fills = exit_order.get("fills", [])
        if fills:
            fill_prices = [float(f["price"]) for f in fills]
            fill_qtys = [float(f["qty"]) for f in fills]
            total_qty_fills = sum(fill_qtys)
            # Use fills only if the total quantity matches the order's executedQty reasonably
            if (
                total_qty_fills > 0 and abs(total_qty_fills - executed_qty) < 1e-9
            ):  # Tolerance for float comparison
                avg_fill_price = (
                    sum(p * q for p, q in zip(fill_prices, fill_qtys)) / total_qty_fills
                )
                logger.debug(f"Calculated exit price from fills: {avg_fill_price:.5f}")
                return avg_fill_price
            else:
                logger.warning(
                    f"Fill quantities ({total_qty_fills}) don't match executedQty ({executed_qty}). Cannot reliably use fills for exit price."
                )

        logger.warning(
            f"Could not determine exit price for order {exit_order.get('orderId')} from available data."
        )
        return None

    def _calculate_pnl(self, exit_price: float, executed_qty: float) -> Optional[float]:
        """Calculates Gross and Net PnL for a closed trade."""
        if self.entry_price is None:
            logger.error("Cannot calculate PnL: Entry price is missing.")
            return None

        # Gross PnL
        pnl_per_unit = exit_price - self.entry_price
        gross_pnl = (
            pnl_per_unit * executed_qty * self._position
        )  # position is still +/- 1 here

        # Commission Calculation
        entry_value = self.entry_price * executed_qty
        exit_value = exit_price * executed_qty
        commission_paid = (abs(entry_value) + abs(exit_value)) * self.commission_rate

        # Net PnL
        net_pnl = gross_pnl - commission_paid

        logger.debug(
            f"PnL Calculation: Entry={self.entry_price:.5f}, Exit={exit_price:.5f}, Qty={executed_qty}, Pos={self._position}, GrossPnL={gross_pnl:.4f}, Comm={commission_paid:.4f}, NetPnL={net_pnl:.4f}"
        )
        return net_pnl

    # --- Exchange Interaction --- #

    def get_exchange_position_size(self) -> Optional[float]:
        """Queries the exchange for the free balance of the base asset.
        Returns the size as float on success, None on error.
        """
        client = self.client_manager.get_client()
        if not client:
            logger.error("Cannot get position size: Client not available.")
            return None

        base_asset = self._get_base_asset()
        if not base_asset:
            logger.error(
                "Cannot get position size: Base asset could not be determined."
            )
            return None

        try:
            balance_info = self.client_manager.retry_api_call(
                client.get_asset_balance, asset=base_asset
            )
            if balance_info and "free" in balance_info:
                size = float(balance_info["free"])
                logger.debug(
                    f"Query successful: Current free balance for {base_asset}: {size}"
                )
                return size
            else:
                logger.warning(
                    f"Could not get balance info or 'free' field for {base_asset}. Response: {balance_info}"
                )
                return None
        except BinanceAPIException as bae:
            logger.error(f"API Error getting balance for {base_asset}: {bae}")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error getting balance for {base_asset}: {e}", exc_info=True
            )
            return None

    # --- State Loading/Saving Helpers (for Orchestrator) --- #

    def get_state(self) -> Dict[str, Any]:
        """Returns the current managed state as a dictionary."""
        return {
            "position": self._position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_order_id": self.entry_order_id,
            "current_stop_loss": self.current_stop_loss,
            "current_take_profit": self.current_take_profit,
            "tsl_peak_price": self.tsl_peak_price,
            "cumulative_profit": self.cumulative_profit,
            # Configuration parameters are not part of the dynamic state here
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Loads state from a dictionary."""
        logger.info("Loading position manager state...")
        self._position = state.get("position", 0)
        self.entry_price = state.get("entry_price")
        entry_time_iso = state.get("entry_time")
        self.entry_time = pd.Timestamp(entry_time_iso) if entry_time_iso else None
        self.entry_order_id = state.get("entry_order_id")
        self.current_stop_loss = state.get("current_stop_loss")
        self.current_take_profit = state.get("current_take_profit")
        self.tsl_peak_price = state.get("tsl_peak_price")
        self.cumulative_profit = state.get("cumulative_profit", 0.0)
        logger.info(
            f"Loaded state: Position={self._position}, EntryPrice={self.entry_price}, CumProfit={self.cumulative_profit}"
        )
        # Add validation? E.g., if position is non-zero, entry_price should exist?
        if self._position != 0 and self.entry_price is None:
            logger.warning(
                "Loaded non-zero position but entry_price is missing. State might be inconsistent."
            )

    def open_position(self, side: str, quantity: float, context_message: str) -> bool:
        """Attempts to open a new position. Returns True if successful, False otherwise."""
        if self._position != 0:
            logger.error(
                f"Attempted to open new position, but already in position {self._position}. Aborting."
            )
            return False

        logger.info(
            f"Executing order to open new position ({side} {quantity} {self.symbol}). Reason: {context_message}"
        )
        order = self.executor.execute_market_order(
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            context_message=f"Opening Position - {context_message}",
        )

        if order and order.get("status") in ["FILLED", "PARTIALLY_FILLED"]:
            executed_qty = float(order.get("executedQty", 0))
            if executed_qty > 0:
                logger.info(
                    f"New position opened successfully (Order ID: {order.get('orderId')}). Executed Qty: {executed_qty}"
                )
                self._position = 1 if side == "BUY" else -1
                self.entry_order_id = order.get("orderId")
                # --- Set Entry Price/Time --- #
                if "fills" in order and order["fills"]:
                    fill_prices = [float(fill["price"]) for fill in order["fills"]]
                    fill_qtys = [float(fill["qty"]) for fill in order["fills"]]
                    if sum(fill_qtys) > 0:
                        self.entry_price = sum(
                            p * q for p, q in zip(fill_prices, fill_qtys)
                        ) / sum(fill_qtys)
                    else:
                        self.entry_price = float(order.get("price") or 0)
                elif order.get("price") and float(order["price"]) > 0:
                    self.entry_price = float(order["price"])
                else:
                    self.entry_price = None  # Could not determine entry price
                    logger.warning(
                        "Could not determine entry price from order. Risk management might be affected."
                    )

                entry_time_ms = order.get("transactTime")
                self.entry_time = (
                    pd.Timestamp(entry_time_ms, unit="ms")
                    if entry_time_ms
                    else pd.Timestamp.now()
                )
                # --- End Set Entry Price/Time --- #

                # Log the entry event
                self.logger.log_trade(
                    entry_time=self.entry_time,
                    entry_price=self.entry_price,
                    exit_time=None,
                    exit_price=None,
                    position_type=("Long" if self._position == 1 else "Short"),
                    executed_qty=executed_qty,
                    gross_pnl=None,
                    commission_paid=None,
                    net_pnl=None,
                    exit_reason="Trade Open",
                    entry_order_id=self.entry_order_id,
                    exit_order_id=None,
                    cumulative_profit=self.cumulative_profit,  # Log current profit at open
                )
                return True
            else:
                logger.warning(
                    "Open order status FILLED/PARTIALLY_FILLED but executed quantity 0. Position not opened."
                )
                self._position = 0
                return False
        else:
            logger.error(
                f"Failed to open new position. Order status: {order.get('status', 'N/A') if order else 'No order object'}. Position remains flat."
            )
            self._position = 0
            return False

    def close_position(self, exit_reason: str = "Closed by signal") -> bool:
        """Attempts to close the current open position. Returns True if successful, False otherwise."""
        if self._position == 0:
            logger.warning("Attempted to close position, but already flat.")
            return True

        side = "SELL" if self._position == 1 else "BUY"
        logger.info(
            f"Attempting to close position {self._position} ({side} {self.symbol}). Reason: {exit_reason}"
        )

        actual_size = self._get_current_position_size()
        if actual_size <= 0:
            logger.warning(
                f"Attempting to close {self._position} position, but current size query <= 0 ({actual_size}). Assuming flat."
            )
            self._reset_position_state()
            return True
        logger.info(f"Actual position size to close: {actual_size}")

        order = self.executor.execute_market_order(
            symbol=self.symbol,
            side=side,
            quantity=actual_size,
            context_message=f"Closing Position - {exit_reason}",
        )

        if order and order.get("status") in ["FILLED", "PARTIALLY_FILLED"]:
            executed_qty = float(order.get("executedQty", 0))
            tolerance = 1e-8

            if executed_qty > 0:
                # --- Calculate PnL and Log Trade --- #
                position_type = "Long" if self._position == 1 else "Short"
                exit_price = 0.0
                if order.get("cummulativeQuoteQty"):  # Use quote qty if available
                    try:
                        exit_price = float(order["cummulativeQuoteQty"]) / executed_qty
                    except (ValueError, ZeroDivisionError):
                        pass

                if exit_price == 0.0 and order.get(
                    "fills"
                ):  # Fallback to avg fill price
                    try:
                        fill_prices = [float(f["price"]) for f in order["fills"]]
                        fill_qtys = [float(f["qty"]) for f in order["fills"]]
                        if sum(fill_qtys) > 0:
                            exit_price = sum(
                                p * q for p, q in zip(fill_prices, fill_qtys)
                            ) / sum(fill_qtys)
                    except (ValueError, ZeroDivisionError):
                        pass

                if exit_price == 0.0:
                    exit_price = self.entry_price or 0.0  # Last resort

                gross_pnl = None
                commission_paid = None
                net_pnl = None
                if self.entry_price is not None:
                    gross_pnl = (
                        (exit_price - self.entry_price) * executed_qty * self._position
                    )
                    commission_rate = self.commission_rate
                    commission_paid = (
                        abs(self.entry_price * executed_qty)
                        + abs(exit_price * executed_qty)
                    ) * commission_rate
                    net_pnl = gross_pnl - commission_paid
                    self.cumulative_profit += net_pnl  # Update cumulative profit
                else:
                    logger.warning("Cannot calculate PnL: Entry price missing.")

                exit_time_ms = order.get("transactTime")
                exit_time_ts = (
                    pd.Timestamp(exit_time_ms, unit="ms")
                    if exit_time_ms
                    else pd.Timestamp.now()
                )

                self.logger.log_trade(
                    entry_time=self.entry_time,
                    entry_price=self.entry_price,
                    exit_time=exit_time_ts,
                    exit_price=exit_price,
                    position_type=position_type,
                    executed_qty=executed_qty,
                    gross_pnl=gross_pnl,
                    commission_paid=commission_paid,
                    net_pnl=net_pnl,
                    exit_reason=exit_reason,
                    entry_order_id=self.entry_order_id,
                    exit_order_id=order.get("orderId"),
                    cumulative_profit=self.cumulative_profit,
                )
                # --- End PnL Calc and Log --- #

            if abs(executed_qty - actual_size) < tolerance:
                logger.info(
                    f"Position closed successfully (Order ID: {order.get('orderId')}). Executed: {executed_qty}"
                )
                self._reset_position_state()
                return True
            elif executed_qty > 0:
                logger.warning(
                    f"Close order partially filled ({executed_qty}/{actual_size}). State NOT fully reset. Manual check advised."
                )
                # Don't reset state fully on partial close
                return False
            else:
                logger.error(
                    f"Close order status {order.get('status')} but executed quantity 0. Failed to close."
                )
                return False
        else:
            logger.error(
                f"Failed to close position. Order status: {order.get('status', 'N/A') if order else 'No order object'}."
            )
            return False

    def _reset_state(self):
        """Resets the internal position state to flat."""
        self._position = 0
        self.entry_price = None
        self.entry_time = None
        self.entry_order_id = None
        logger.debug("PositionManager state reset to flat.")

    def enter_long(self) -> bool:
        """Opens a long position using the configured units."""
        if not self.is_flat():
            logger.warning(
                f"Attempted to enter long while already in position {self.position}. Ignoring."
            )
            return False
        return self.open_position(
            side="BUY", quantity=self.units, context_message="Enter Long Signal"
        )

    def enter_short(self) -> bool:
        """Opens a short position using the configured units."""
        if not self.is_flat():
            logger.warning(
                f"Attempted to enter short while already in position {self.position}. Ignoring."
            )
            return False
        return self.open_position(
            side="SELL", quantity=self.units, context_message="Enter Short Signal"
        )
