# Disclaimer:
# The following illustrative examples are for general information and educational purposes only.
# It is neither investment advice nor a recommendation to trade, invest or take whatsoever actions.
# The below code should only be used in combination with the Binance Spot Testnet and NOT with a Live Trading Account.

from binance.client import Client
from binance import ThreadedWebsocketManager
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
import time
from typing import List, Tuple, Optional, Dict, Any, cast
import logging  # Added
import argparse  # Added
import os  # Added
from .config_utils import load_secrets_from_aws  # Added
import yaml  # Added
from pathlib import Path  # Added
from binance.exceptions import BinanceAPIException  # Added
import json  # Added for state saving

# Import the strategy interface and specific strategies if needed for type hints
from .strategies import (
    BaseStrategy,
    MovingAverageCrossoverStrategy,
    ScalpingStrategy,
    BollingerBandReversionStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    HybridStrategy,
)

# Import filter-related utilities
from .utils import parse_trading_hours  # Borrow parser from utils
from .technical_indicators import (
    calculate_atr,
    calculate_sma,
    calculate_ema,
)  # Ensure all needed indicators are available

# --- Import Pydantic Models ---
from .config_models import RuntimeConfig, ValidationError

# --- Import Trader Components ---
from .trader_components.client_manager import ClientManager
from .trader_components.data_handler import DataHandler
from .trader_components.position_manager import PositionManager
from .trader_components.risk_manager import RiskManager
from .trader_components.state_manager import StateManager
from .trader_components.trade_logger import CsvTradeLogger
from .trader_components.runtime_config_manager import RuntimeConfigManager
from .trader_components.order_executor import OrderExecutor

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define type aliases for clarity
Symbol = str
Interval = str
Units = float
Position = int  # -1: short, 0: neutral, 1: long
HistoricalDays = float  # Type alias for days of historical data
ApiKey = str
SecretKey = str
# Define state file directory constant
STATE_DIR = Path("results/state")
# Define trade log directory constant
TRADE_LOG_DIR = Path("results/live_trades")

# Import all strategies and map by class name for dynamic loading
STRATEGY_CLASS_MAP = {
    cls.__name__: cls
    for cls in [
        MovingAverageCrossoverStrategy,
        ScalpingStrategy,
        BollingerBandReversionStrategy,
    ]
}

# --- Constants for Runtime Reload ---
DEFAULT_RUNTIME_CONFIG = "config/runtime_config.yaml"
CONFIG_CHECK_INTERVAL_BARS = 5  # Check config file every 5 completed bars

# Create directories if they don't exist
STATE_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)


class Trader:
    def __init__(
        self,
        symbol: Symbol,
        bar_length: Interval,
        strategy: BaseStrategy,
        units: Units,
        position: Position = 0,
        api_key: Optional[ApiKey] = None,
        secret_key: Optional[SecretKey] = None,
        testnet: bool = True,
        runtime_config_path: str = DEFAULT_RUNTIME_CONFIG,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        trailing_stop_loss_pct: Optional[float] = None,
        max_cumulative_loss: Optional[float] = None,
        # --- Filter Params ---
        apply_atr_filter: bool = False,
        atr_filter_period: int = 14,
        atr_filter_multiplier: float = 1.5,
        atr_filter_sma_period: int = 100,
        apply_seasonality_filter: bool = False,
        allowed_trading_hours_utc: Optional[str] = None,  # Expects '5-17' string format
        apply_seasonality_to_symbols: Optional[
            str
        ] = None,  # Expects comma-separated string
        # <<< ADD commission_bps PARAMETER >>>
        commission_bps: float = 0.0,  # Commission in basis points (e.g., 7.5 for 0.075%)
    ) -> None:
        """
        Initializes the Trader by setting up and connecting its components.
        Args:
            symbol: The trading symbol.
            bar_length: The candle bar length.
            strategy: An instantiated strategy object.
            units: The quantity of the asset to trade.
            position: The initial position (deprecated - state loaded by PositionManager).
            api_key: Binance API key.
            secret_key: Binance secret key.
            testnet: Whether to use the Binance testnet.
            runtime_config_path: Path to the runtime configuration YAML file.
            # SL/TP/Filter args are passed to relevant components
            stop_loss_pct: Optional percentage for stop loss.
            take_profit_pct: Optional percentage for take profit.
            trailing_stop_loss_pct: Optional percentage for trailing stop loss.
            max_cumulative_loss: Optional maximum absolute cumulative loss threshold.
            apply_atr_filter: Whether to enable the ATR volatility filter.
            atr_filter_period: Period for ATR calculation.
            atr_filter_multiplier: Multiplier for the ATR filter threshold.
            atr_filter_sma_period: Period for the SMA of ATR threshold.
            apply_seasonality_filter: Whether to enable the seasonality filter.
            allowed_trading_hours_utc: String 'HH-HH' for allowed UTC trading hours.
            apply_seasonality_to_symbols: Comma-separated string of symbols for seasonality.
            commission_bps: Commission in basis points (e.g., 7.5 for 0.075%).
        """
        self.symbol = symbol
        self.bar_length = bar_length
        self.strategy = strategy  # Strategy instance is passed to DataHandler?
        self.units = units
        self.position = position
        self.trades = 0
        self.trade_values: List[float] = []
        self.cum_profits = 0.0
        self.is_running = False

        self.available_intervals: List[str] = [
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        ]
        if bar_length not in self.available_intervals:
            raise ValueError(f"Interval {bar_length} not supported by Binance API.")

        # Binance Client
        self.testnet = testnet
        self.api_key = api_key
        self.secret_key = secret_key
        self.client: Optional[Client] = None
        self.twm: Optional[ThreadedWebsocketManager] = None

        # <<< STORE commission_bps >>>
        self.commission_bps = commission_bps

        # SL/TP/TSL State
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_stop_loss_pct = trailing_stop_loss_pct
        self.entry_price: Optional[float] = None
        self.current_stop_loss: Optional[float] = None
        self.current_take_profit: Optional[float] = None
        self.tsl_peak_price: Optional[float] = None

        # Bot Stop State
        self.max_cumulative_loss = (
            abs(max_cumulative_loss) if max_cumulative_loss is not None else None
        )
        self.max_loss_stop_triggered = False

        # Filter State
        self.apply_atr_filter = apply_atr_filter
        self.atr_filter_period = atr_filter_period
        self.atr_filter_multiplier = atr_filter_multiplier
        self.atr_filter_sma_period = atr_filter_sma_period
        self.apply_seasonality_filter = apply_seasonality_filter
        self.parsed_trading_hours = (
            parse_trading_hours(allowed_trading_hours_utc)
            if self.apply_seasonality_filter
            else None
        )
        self.seasonality_symbols_list = (
            [
                s.strip().upper()
                for s in apply_seasonality_to_symbols.split(",")
                if s.strip()
            ]
            if apply_seasonality_to_symbols
            else []
        )
        self.apply_seasonality_to_this_symbol = (
            self.apply_seasonality_filter
            and self.parsed_trading_hours
            and (
                not self.seasonality_symbols_list
                or self.symbol in self.seasonality_symbols_list
            )
        )
        if self.apply_seasonality_filter and not self.parsed_trading_hours:
            logger.warning(
                f"Seasonality filter enabled but invalid allowed_trading_hours_utc: '{allowed_trading_hours_utc}'. Filter inactive."
            )
            self.apply_seasonality_to_this_symbol = (
                False  # Disable if hours are invalid
            )

        # Data State
        self.data: pd.DataFrame = pd.DataFrame()
        self.prepared_data: pd.DataFrame = pd.DataFrame()

        # Runtime Config State
        self.runtime_config_path = Path(runtime_config_path)

        # Add attribute to store entry order ID

        # --- Component Initialization --- #
        logger.info("Initializing Trader components...")

        # 1. State Manager (Needed by others)
        # REMOVED: self.state_manager = StateManager(symbol=self.symbol)

        # 2. Trade Logger
        self.trade_logger = CsvTradeLogger(symbol=self.symbol)

        # 3. Client Manager
        self.client_manager = ClientManager(
            api_key=api_key, secret_key=secret_key, testnet=testnet
        )

        # 4. Position Manager
        self.order_executor = OrderExecutor(client_manager=self.client_manager)

        self.position_manager = PositionManager(
            symbol=symbol,
            units=units,
            order_executor=self.order_executor,
            trade_logger=self.trade_logger,
            client_manager=self.client_manager,
            commission_bps=commission_bps,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_loss_pct=trailing_stop_loss_pct,
        )

        # 5. Risk Manager
        self.risk_manager = RiskManager(
            position_manager=self.position_manager,
            max_cumulative_loss=max_cumulative_loss,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            trailing_stop_loss_pct=trailing_stop_loss_pct,
        )

        # 6. Data Handler (Requires callback to self.on_data_update)
        required_lookback = strategy.get_required_lookback()
        indicator_config = {
            "apply_atr": apply_atr_filter,
            "atr_period": atr_filter_period,
            "atr_sma_period": atr_filter_sma_period,
        }
        if (
            indicator_config.get("apply_atr")
            and indicator_config.get("atr_period", 0) > required_lookback
        ):
            required_lookback = indicator_config["atr_period"]
        if (
            indicator_config.get("apply_atr")
            and indicator_config.get("atr_sma_period", 0) > required_lookback
        ):
            required_lookback = indicator_config["atr_sma_period"]

        self.data_handler = DataHandler(
            symbol=symbol,
            bar_length=bar_length,
            strategy=strategy,
            closed_bar_callback=self.on_data_update,
            required_lookback=required_lookback,
            indicator_config=indicator_config,
            state_manager=self.state_manager, # type: ignore[has-type] # Explicitly ignore has-type error
            client_manager=self.client_manager
        )

        # 7. Runtime Config Manager
        self.runtime_config_manager = RuntimeConfigManager(
            config_path=str(runtime_config_path)
        )

        # 8. State Manager (Instantiate last, passing other components)
        # Define which components need state managed
        components_to_manage = [
            self.position_manager,
            self.risk_manager,
            self.data_handler, # Add DataHandler if it needs state
            # Add self.runtime_config_manager if it implements get/load_state
        ]
        # Update StateManager call to include components
        self.state_manager = StateManager(
                    symbol=self.symbol,
            components=components_to_manage
        )

        # --- Final Initialization Steps ---

        # Load persisted state for all components
        logger.info("Loading component states...")
        self.state_manager.load_state()

        logger.info("Trader components initialized.")

    def start_trading(self, historical_days: HistoricalDays):
        logger.info("Attempting to start trading session...")
        if self.is_running:
            logger.warning("Trader is already running. Call stop_trading() first.")
            return

        # --- Initialization Checks ---
        # Check if client was initialized correctly by ClientManager
        # client = self.client_manager.get_client() # Client is managed internally now
        # if not self.client_manager.is_connected(): # Check connectivity status -- Error: is_connected not found
        #     logger.critical("Binance client not connected. Cannot start.")
        #     # Optionally attempt connection here: self.client_manager.connect()
        #     return
        # Assuming ClientManager connects on init or has connect method called elsewhere

        # --- State Reconciliation ---
        logger.info("Performing pre-start state reconciliation...")
        if not self._reconcile_state():  # Check return value
            logger.critical(
                "State reconciliation failed or requires manual intervention. Trading aborted."
            )
            return

        # --- Fetch Initial Data & Start WebSocket ---
        logger.info(f"Fetching initial historical data ({historical_days} days)...")
        # Use DataHandler to fetch data
        if not self.data_handler.fetch_historical_data(
            historical_days
        ):  # Renamed from fetch_initial_data
            logger.critical("Failed to fetch initial historical data. Cannot start.")
            return
        logger.info("Initial data fetched and prepared.")

        # --- Start WebSocket Stream (Managed by DataHandler/ClientManager) ---
        logger.info("Starting WebSocket stream...")
        # Assuming DataHandler has process_kline_message method
        # Adding interval and callback args to start_websocket
        if not self.client_manager.start_websocket(
            symbol=self.symbol,
            interval=self.bar_length,
            callback=self.data_handler.process_kline_message,
        ):
            logger.critical("Failed to start WebSocket stream. Aborting.")
            return

        self.is_running = True
        logger.info(f"--- Trader started for {self.symbol} ---")
        # Keep main thread alive (WebSocket runs in background)
        # Consider a more robust way to handle termination signals (SIGINT, SIGTERM)
        try:
            while self.is_running:
                time.sleep(1)  # Keep main thread alive
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received. Stopping trader...")
            self.stop_trading()
        finally:
            if self.is_running:  # Ensure cleanup if loop exits unexpectedly
                self.stop_trading()

    def _update_config_mtime(self):
        """Reads and stores the modification time of the config file."""
        try:
            if self.runtime_config_path.is_file():
                self.last_config_mtime = self.runtime_config_path.stat().st_mtime
            else:
                self.last_config_mtime = None  # File doesn't exist
        except OSError as e:
            logger.error(
                f"Error accessing runtime config file stats ({self.runtime_config_path}): {e}"
            )
            self.last_config_mtime = None

    def stop_trading(self):
        """Stops the trading bot gracefully."""
        if not self.is_running:
            logger.info("Trader is not running.")
            return

        logger.info("Stopping trader...")
        self.is_running = False  # Prevent further actions in callbacks

        # Stop the WebSocket stream
        self.client_manager.stop_websocket()

        # Optional: Persist final state?
        logger.info("Saving final state...")
        # Use StateManager to save state of all managed components
        self.state_manager.save_state()

        logger.info("Trader stopped.")

    def _reconcile_state(self) -> bool:
        """Compares loaded state (PositionManager) with actual exchange balance.
        Returns True if consistent or fixable, False if manual intervention needed.
        """
        logger.info("Reconciling internal state with exchange balance...")

        # Get internal state from PositionManager
        internal_pos = self.position_manager.position

        # Get actual balance from exchange via PositionManager
        actual_size_on_exchange = self.position_manager.get_exchange_position_size()

        if actual_size_on_exchange is None:
            logger.error(
                "Could not query exchange balance. Reconciliation skipped. Assuming state is correct (RISKY)."
            )
            # Allow proceeding but with a warning. Alternatively, return False here for stricter safety.
            return True

        logger.info(
            f"Internal State Position: {internal_pos} | Exchange Free Balance Query: {actual_size_on_exchange}"
        )

        needs_manual_intervention = False

        # Case 1: Internal state has position, but exchange shows none/zero
        if (
            internal_pos != 0 and actual_size_on_exchange <= 1e-9
        ):  # Use tolerance for float comparison
            logger.warning(
                f"State indicates position {internal_pos}, but exchange shows near-zero balance ({actual_size_on_exchange}). "
                f"Resetting internal state to flat."
            )
            # Log the discrepancy event
            entry_price_before, entry_time_before, entry_id_before = (
                self.position_manager.get_entry_details()
            )
            pos_type_before = "Long" if internal_pos == 1 else "Short"
            self.trade_logger.log_trade(
                entry_time=entry_time_before,
                entry_price=entry_price_before,
                exit_time=pd.Timestamp.now(tz="UTC"),
                exit_price=None,
                position_type=pos_type_before,
                executed_qty=0,
                gross_pnl=None,
                commission_paid=None,
                net_pnl=None,
                exit_reason="State Reconciliation (Position Discrepancy)",
                entry_order_id=entry_id_before,
                exit_order_id=None,
                cumulative_profit=self.position_manager.cumulative_profit,
            )
            self.position_manager._reset_position_state()  # Use internal method name
            self.state_manager.save_state()  # Save corrected state

        # Case 2: Internal state is flat, but exchange shows significant balance
        elif internal_pos == 0 and actual_size_on_exchange > 1e-9:  # Use tolerance
            logger.critical(
                "CRITICAL STATE MISMATCH: State is flat, but exchange shows balance."
            )
            logger.critical(
                f"  Actual size: {actual_size_on_exchange}. Cannot safely resume without entry details."
            )
            logger.critical(
                "  Manual intervention required: Close position on exchange or delete state file and restart."
            )
            needs_manual_intervention = True

        # Case 3: Internal state has position, exchange shows balance (Assume consistent for now)
        # More complex checks could verify quantity if needed, but free balance is a basic check.
        # Add checks for short positions if applicable/possible to detect
        # elif internal_pos == -1 and ...

        else:  # Covers internal_pos == 0 and actual_size_on_exchange == 0
            # Also covers internal_pos != 0 and actual_size_on_exchange > 0 (assumed consistent)
            logger.info("Internal state appears consistent with exchange balance.")

        return not needs_manual_intervention  # Return True if safe to proceed

    def on_data_update(self, prepared_data: pd.DataFrame) -> None:
        """Callback triggered by DataHandler when new prepared data is available.

        This is the main trading logic cycle.
        """
        if not self.is_running:
            # logger.debug("Trader not running, ignoring data update.")
            return

        if prepared_data.empty:
            logger.warning(
                "Received empty prepared data frame. Skipping strategy cycle."
            )
            return

        # Ensure data has expected columns (basic check)
        # DataHandler should guarantee this if preparation is successful
        # required_cols = ["Open", "High", "Low", "Close", "Volume"]
        # if not all(col in prepared_data.columns for col in required_cols):
        #     logger.error(f"Prepared data missing required columns: {required_cols}. Skipping cycle.")
        #     return

        timestamp = prepared_data.index[-1]
        logger.debug(f"[{timestamp}] Received data update. Running trading cycle...")

        # --- Core Trading Logic Cycle ---
        try:
            # 1. Check Risk Limits (before generating new signals/trades)
            if (
                self.risk_manager.check_risk_limits()
            ):  # Checks max loss breach, returns True if breached
                logger.critical(
                    f"[{timestamp}] Risk limit breached! Attempting to close position and stop."
                )
                if not self.position_manager.is_flat():
                    # Use PositionManager to handle the exit - remove exit_reason kwarg
                    self.position_manager.close_position()
                self.stop_trading()  # Stop websocket and further processing
                return  # Stop processing this cycle

            # 2. Check Runtime Configuration Changes
            self.runtime_config_manager.check_for_updates()

            # 3. Check SL/TP Hits (using high/low of the latest bar)
            if not self.position_manager.is_flat():  # Only check if in a position
                latest_high = prepared_data["High"].iloc[-1]
                latest_low = prepared_data["Low"].iloc[-1]

                # Store SL before check to see if TSL moves it
                sl_before_check = self.position_manager.current_stop_loss

                # PositionManager now checks SL/TP and potentially exits
                # Assuming method check_sl_tp_and_update_tsl exists in PositionManager
                sl_tp_hit_info = self.position_manager.check_sl_tp_and_update_tsl(
                    latest_high, latest_low
                )

                # If check_sl_tp_and_update_tsl triggered an exit, it returns info, otherwise None
                if sl_tp_hit_info:
                    # PositionManager already handled the exit and logging
                    logger.info(f"[{timestamp}] SL/TP exit handled by PositionManager.")
                    # State is saved within PositionManager after exit/TSL update
                    return  # Don't proceed to strategy signal if SL/TP exit occurred
                else:
                    # --- Save state ONLY if TSL moved the stop loss (and no exit occurred) ---
                    sl_after_check = self.position_manager.current_stop_loss
                    tsl_moved_stop = (
                        # sl_tp_hit_info is None implies no exit
                        self.position_manager.trailing_stop_loss_pct
                        is not None  # TSL is active
                        and sl_before_check is not None  # SL existed before check
                        and sl_after_check is not None  # SL exists after check
                        and sl_before_check != sl_after_check  # SL actually changed
                    )
                    if tsl_moved_stop:
                        logger.info(
                            f"[{timestamp}] Trailing stop loss updated to: {sl_after_check:.5f}. Saving state."
                        )
                        self.state_manager.save_state()  # Save state because TSL updated the stop

            # 4. Generate Strategy Signal
            # Strategy signals are generated within DataHandler during data preparation
            # We should get the latest signal from prepared_data
            if "signal" not in prepared_data.columns:
                logger.warning(
                    f"[{timestamp}] 'signal' column missing in prepared_data. Skipping trade logic."
                )
                return

            latest_signal_series = prepared_data["signal"]
            if (
                latest_signal_series.empty
                or latest_signal_series.iloc[-1] is None
                or pd.isna(latest_signal_series.iloc[-1])
            ):
                # logger.debug(f"[{timestamp}] No valid signal generated by strategy.")
                return  # No signal or NaN signal

            latest_signal = int(latest_signal_series.iloc[-1])  # Expecting -1, 0, or 1
            current_position = self.position_manager.position

            logger.debug(
                f"[{timestamp}] Strategy Signal: {latest_signal} | Current Position: {current_position}"
            )

            # 5. Apply Trader-Level Filters (e.g., seasonality) - Should these be in RiskManager or DataHandler?
            # For now, keep seasonality here, but ATR is handled by DataHandler
            trade_allowed_this_bar = True
            # --- Seasonality Filter (Example - consider moving to RiskManager?) ---
            # REMOVING Seasonality Filter block as self.parsed_trading_hours is gone
            # if self.apply_seasonality_filter and self.apply_seasonality_to_this_symbol:
            #     # Assume self.parsed_trading_hours exists and is valid if flags are true
            #     # This state needs to be reinstated or managed by RuntimeConfigManager/RiskManager
            #     start_hour, end_hour = self.parsed_trading_hours
            #     ts_aware = timestamp.tz_convert("UTC") if timestamp.tz else timestamp.tz_localize("UTC")
            #     if not (start_hour <= ts_aware.hour < end_hour):
            #         trade_allowed_this_bar = False
            #         logger.info(f"[{timestamp}] Trade signal ({latest_signal}) blocked by Seasonality Filter (Hour: {ts_aware.hour}).")

            # --- ATR Filter (Handled by DataHandler adding indicators) ---
            # The check for whether to trade based on ATR value vs threshold could be here or in PM
            # Let's assume for now that if apply_atr_filter is True, DataHandler added the columns
            # and PositionManager (or RiskManager) might use them before executing trades.
            # We will remove the explicit filter block here.

            # 6. Determine Target Position & Execute Trades
            # If trade is blocked by filter, target position remains the current position
            target_position = (
                latest_signal if trade_allowed_this_bar else current_position
            )

            if target_position != current_position:
                logger.info(
                    f"[{timestamp}] Action Triggered: Current={current_position}, Target={target_position} (Signal={latest_signal}, Filter Allowed={trade_allowed_this_bar})"
                )
                # Delegate execution to PositionManager
                if target_position == 0:
                    # Remove exit_reason kwarg
                    self.position_manager.close_position()
                elif target_position == 1:
                    # Assuming method enter_long exists in PositionManager
                    self.position_manager.enter_long()
                elif target_position == -1:
                    # Assuming method enter_short exists in PositionManager
                    self.position_manager.enter_short()
            # else: logger.debug(f"[{timestamp}] No change in position required.")

        except Exception as e:
            logger.error(
                f"[{timestamp}] Error during trading cycle: {e}", exc_info=True
            )
            # Consider stopping the trader on critical errors, or just log and continue?
            # self.stop_trading()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binance Trading Bot Orchestrator (Component-Based)"
    )

    # --- Core Trading Arguments ---
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGY_CLASS_MAP.keys()),
        help="Name of the strategy class.",
    )
    parser.add_argument(
        "--symbol", type=str, required=True, help="Trading symbol (e.g., BTCUSDT)."
    )
    parser.add_argument(
        "--interval",
        type=str,
        required=True,
        help="Candlestick interval (e.g., 1m, 5m, 1h).",
    )
    parser.add_argument(
        "--units", type=float, required=True, help="Quantity of the asset per trade."
    )
    parser.add_argument(
        "--days", type=float, default=10, help="Days of initial historical data."
    )
    parser.add_argument(
        "--testnet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Binance Testnet.",
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=7.5,
        help="Commission per trade in basis points (e.g., 7.5 for 0.075%).",
    )

    # --- Strategy Parameter Loading ---
    parser.add_argument(
        "--param-config",
        type=str,
        default="config/best_params.yaml",
        help="Path to strategy parameters YAML.",
    )

    # --- Risk Management Arguments ---
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=None,
        help="Stop loss percentage (e.g., 0.02 for 2%). Default: None",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=None,
        help="Take profit percentage (e.g., 0.04 for 4%). Default: None",
    )
    parser.add_argument(
        "--trailing-stop-loss",
        type=float,
        default=None,
        help="Trailing stop loss percentage (e.g., 0.01 for 1%). Overrides fixed SL if price moves favorably. Default: None",
    )
    parser.add_argument(
        "--max-cum-loss",
        type=float,
        default=None,
        help="Maximum absolute cumulative loss allowed (e.g., 100.0). Bot stops if reached. Default: None",
    )

    # --- Filter Arguments ---
    parser.add_argument(
        "--apply-atr-filter", action="store_true", help="Apply ATR volatility filter."
    )
    parser.add_argument(
        "--atr-filter-period", type=int, default=14, help="Period for ATR calculation."
    )
    parser.add_argument(
        "--atr-filter-multiplier",
        type=float,
        default=1.5,
        help="Multiplier for ATR threshold.",
    )
    parser.add_argument(
        "--atr-filter-sma-period",
        type=int,
        default=100,
        help="SMA period for ATR baseline.",
    )
    parser.add_argument(
        "--apply-seasonality-filter",
        action="store_true",
        help="Apply seasonality (trading hours) filter.",
    )
    parser.add_argument(
        "--allowed-trading-hours-utc",
        type=str,
        default=None,
        help="Allowed UTC hours ('HH-HH').",
    )
    parser.add_argument(
        "--apply-seasonality-to-symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols for seasonality filter.",
    )

    # --- Runtime/State/Logging Arguments ---
    parser.add_argument(
        "--runtime-config",
        type=str,
        default=DEFAULT_RUNTIME_CONFIG,
        help="Path to runtime config YAML.",
    )
    parser.add_argument(
        "--state-file-name", type=str, default=None, help="Override state file name."
    )
    parser.add_argument(
        "--trade-log-file-name",
        type=str,
        default=None,
        help="Override trade log CSV file name.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level.",
    )
    parser.add_argument(
        "--initial-position-override",
        type=int,
        default=None,
        choices=[-1, 0, 1],
        help="Force initial position state (use with caution).",
    )

    args = parser.parse_args()

    # --- Set Log Level ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    # --- Validate Filter Args ---
    if args.apply_seasonality_filter and not args.allowed_trading_hours_utc:
        logger.error(
            "--allowed-trading-hours-utc is required when --apply-seasonality-filter is used."
        )
        exit(1)
    # Further validation happens inside Trader.__init__

    # --- Load API Keys ---
    # Using AWS Secrets Manager if keys not provided via env vars
    api_key = os.environ.get("BINANCE_API_KEY")
    secret_key = os.environ.get("BINANCE_API_SECRET")

    if not api_key or not secret_key:
        logger.info(
            "API keys not found in environment variables. Attempting to load from AWS Secrets Manager..."
        )
        # Specify your secret name and region
        secret_name = "binance/api_keys"  # CHANGE THIS TO YOUR SECRET NAME
        region_name = "us-west-2"  # CHANGE THIS TO YOUR AWS REGION
        aws_secrets = load_secrets_from_aws(secret_name, region_name)
        if aws_secrets:
            api_key = aws_secrets.get("BINANCE_API_KEY")
            secret_key = aws_secrets.get("BINANCE_API_SECRET")
            logger.info("API keys loaded successfully from AWS Secrets Manager.")
        else:
            logger.warning(
                "Failed to load keys from AWS. Proceeding without authentication."
            )
            # Proceeding without keys is allowed, but orders will fail.
            # Initialization checks will handle client setup.

    # --- Load Strategy Parameters --- #
    strategy_params = {}
    param_config_file = Path(args.param_config)
    if param_config_file.is_file():
        logger.info(f"Attempting to load parameters from: {param_config_file}")
        try:
            with open(param_config_file, "r") as f:
                loaded_data = yaml.safe_load(f)
                if isinstance(loaded_data, dict) and "parameters" in loaded_data:
                    best_params_all = loaded_data.get("parameters")
                    if isinstance(best_params_all, dict):
                        # Clean None strings
                        strategy_params = {
                            k: (
                                None
                                if isinstance(v, str) and v.lower() == "none"
                                else v
                            )
                            for k, v in best_params_all.items()
                        }
                        logger.info(
                            f"Loaded and cleaned strategy parameters: {strategy_params}"
                        )
                    else:
                        logger.warning(
                            f"'parameters' key in {param_config_file} does not contain a dictionary."
                        )
                else:
                    logger.warning(
                        f"Invalid format in {param_config_file}. Expected dict with 'parameters' key."
                    )
        except yaml.YAMLError as e:
            logger.error(f"Error parsing parameter YAML {param_config_file}: {e}")
        except Exception as e:
            logger.error(
                f"Error reading parameter file {param_config_file}: {e}", exc_info=True
            )
    else:
        logger.warning(
            f"Parameter config file not found: {param_config_file}. Using strategy defaults."
        )

    # Instantiate the strategy with loaded params (or defaults if loading failed)
    try:
        strategy_class = STRATEGY_CLASS_MAP.get(args.strategy)
        if not strategy_class:
            raise ValueError(f"Unknown strategy class name: {args.strategy}")

        # Pass only the loaded strategy parameters to the strategy constructor
        # Base/filter params are handled by the Trader class initialization
        strategy_instance = strategy_class(**strategy_params)
        logger.info(
            f"Instantiated strategy '{args.strategy}' with params: {strategy_params}"
        )

    except Exception as e:
        logger.error(f"Error loading or instantiating strategy: {e}", exc_info=True)
        exit(1)

    # --- Instantiate and Start Trader ---
    try:
        trader = Trader(
            symbol=args.symbol,
            bar_length=args.interval,
            strategy=strategy_instance,
            units=args.units,
            api_key=api_key,
            secret_key=secret_key,
            testnet=args.testnet,
            runtime_config_path=args.runtime_config,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit,
            trailing_stop_loss_pct=args.trailing_stop_loss,
            max_cumulative_loss=args.max_cum_loss,
            apply_atr_filter=args.apply_atr_filter,
            atr_filter_period=args.atr_filter_period,
            atr_filter_multiplier=args.atr_filter_multiplier,
            atr_filter_sma_period=args.atr_filter_sma_period,
            apply_seasonality_filter=args.apply_seasonality_filter,
            allowed_trading_hours_utc=args.allowed_trading_hours_utc,
            apply_seasonality_to_symbols=args.apply_seasonality_to_symbols,
            commission_bps=args.commission_bps,  # Use args.commission_bps
        )
        # Start the main trading loop (blocking)
        trader.start_trading(historical_days=args.days)
    except Exception as e:
        logger.critical(
            f"Trader initialization or execution failed: {e}", exc_info=True
        )
        # Attempt to stop trader if it was partially initialized
        if "trader" in locals() and hasattr(trader, "stop_trading"):
            trader.stop_trading()
        exit(1)
