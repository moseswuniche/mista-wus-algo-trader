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
    Strategy,
    LongShortStrategy,
    MovingAverageCrossoverStrategy,
    RsiMeanReversionStrategy,
    BollingerBandReversionStrategy,
)

# Import filter-related utilities
from .backtest import parse_trading_hours  # Borrow parser from backtest
from .technical_indicators import (
    calculate_atr,
    calculate_sma,
    calculate_ema,
)  # Ensure all needed indicators are available

# --- Import Pydantic Models ---
from .config_models import RuntimeConfig, ValidationError

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
        LongShortStrategy,
        MovingAverageCrossoverStrategy,
        RsiMeanReversionStrategy,
        BollingerBandReversionStrategy,
    ]
}

# --- Constants for Runtime Reload ---
DEFAULT_RUNTIME_CONFIG = "config/runtime_config.yaml"
CONFIG_CHECK_INTERVAL_BARS = 5  # Check config file every 5 completed bars


class Trader:
    def __init__(
        self,
        symbol: Symbol,
        bar_length: Interval,
        strategy: Strategy,
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
        Initializes the Trader.
        Args:
            symbol: The trading symbol.
            bar_length: The candle bar length.
            strategy: An instantiated strategy object.
            units: The quantity of the asset to trade.
            position: The initial position.
            api_key: Binance API key.
            secret_key: Binance secret key.
            testnet: Whether to use the Binance testnet.
            runtime_config_path: Path to the runtime configuration YAML file.
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

        self.strategy = strategy
        self.units = units
        self.position = position
        self.trades = 0
        self.trade_values: List[float] = []
        self.cum_profits = 0.0

        # Binance Client
        self.testnet = testnet
        self.api_key = api_key
        self.secret_key = secret_key
        self.client: Optional[Client] = None
        self.twm: Optional[ThreadedWebsocketManager] = None
        self._initialize_client()

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
        self.required_historical_bars = self._determine_required_bars()

        # Runtime Config State
        self.runtime_config_path = Path(runtime_config_path)
        self.last_config_check_time = time.time()
        self.last_config_mtime: Optional[float] = None
        self.config_check_counter = 0
        self._update_config_mtime()

        # Add state file path
        self.state_file_path = STATE_DIR / f"trader_state_{self.symbol}.json"

        # Attempt to load state on initialization
        self._load_state()

        # Define trades CSV path and ensure directory exists
        TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.trades_csv_path = (
            TRADE_LOG_DIR
            / f"live_trades_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        logger.info(f"Live trades will be logged to: {self.trades_csv_path}")

        # Add attribute to store entry order ID
        self.entry_order_id: Optional[str] = None

    def _determine_required_bars(self) -> int:
        """Determine max lookback needed based on strategy and filters."""
        max_lookback = 0
        # Strategy params (assuming params dict stores periods)
        for period in self.strategy.params.values():
            if isinstance(period, int) and period > 0:
                max_lookback = max(max_lookback, period)
        # Filter params
        if self.apply_atr_filter:
            max_lookback = max(max_lookback, self.atr_filter_period)
            if self.atr_filter_sma_period > 0:
                # Need ATR period + SMA period for the SMA of ATR
                max_lookback = max(
                    max_lookback, self.atr_filter_period + self.atr_filter_sma_period
                )

        # Add buffer (e.g., 50 bars) for stability
        return max_lookback + 50

    def _initialize_client(self) -> None:
        """Initializes the Binance client using stored credentials."""
        if not self.api_key or not self.secret_key:
            logger.error("Cannot initialize client: API key or secret key is missing.")
            return
        if self.client is None:
            try:
                self.client = Client(
                    api_key=self.api_key,
                    api_secret=self.secret_key,
                    tld="com",
                    testnet=self.testnet,
                )
                # Perform a test connection / get account info
                self.client.ping()
                account_info = self.client.get_account()
                logger.info(
                    f"Binance Client Initialized (Testnet: {self.testnet}). Account status: {account_info.get('accountType', 'N/A')}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize or test Binance client: {e}", exc_info=True
                )
                self.client = None  # Ensure client is None if init fails
        else:
            logger.debug("Binance Client already initialized.")

    def _initialize_twm(self) -> bool:
        """Initializes or re-initializes the ThreadedWebsocketManager."""
        if not self.api_key or not self.secret_key:
            logger.error("API key/secret missing. Cannot start WebSocket Manager.")
            return False

        # Stop existing TWM if it's running
        if self.twm and self.twm.is_alive():
            logger.info("Stopping existing TWM before re-initializing...")
            self._stop_websocket()

        try:
            self.twm = ThreadedWebsocketManager(
                api_key=self.api_key, api_secret=self.secret_key
            )
            # Use start() which handles testnet URL internally now
            self.twm.start(testnet=self.testnet)
            logger.info(
                f"Threaded WebSocket Manager initialized and started (Testnet: {self.testnet})."
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to initialize or start ThreadedWebSocketManager: {e}",
                exc_info=True,
            )
            self.twm = None  # Ensure twm is None on failure
            return False

    def _get_base_asset(self) -> str:
        """Extracts the base asset from the symbol (e.g., XRP from XRPUSDT)."""
        # Basic implementation, assumes USDT, BUSD, BTC, ETH etc. as quote
        # Needs refinement for more complex pairs
        quote_assets = ["USDT", "BUSD", "BTC", "ETH", "EUR", "GBP", "AUD"]
        for quote in quote_assets:
            if self.symbol.endswith(quote):
                return self.symbol[: -len(quote)]
        # Fallback or error if quote asset not standard
        logger.warning(f"Could not determine base asset from symbol: {self.symbol}")
        # Returning the first part as a guess, might be incorrect
        return self.symbol[:3]

    def _get_position_size(self) -> float:
        """Queries the exchange for the free balance of the base asset.
        Returns 0.0 on error or if client not available.
        """
        if not self.client:
            logger.error("Cannot get position size: Client not initialized.")
            return 0.0

        base_asset = self._get_base_asset()
        try:
            # Use retry helper for the API call
            balance_info = self._retry_api_call(
                self.client.get_asset_balance, asset=base_asset
            )
            if balance_info and "free" in balance_info:
                size = float(balance_info["free"])
                logger.debug(f"Current free balance for {base_asset}: {size}")
                return size
            else:
                logger.warning(
                    f"Could not get balance info or 'free' field for {base_asset}. Response: {balance_info}"
                )
                return 0.0
        except BinanceAPIException as bae:
            logger.error(f"API Error getting balance for {base_asset}: {bae}")
            return 0.0
        except Exception as e:
            logger.error(
                f"Unexpected error getting balance for {base_asset}: {e}", exc_info=True
            )
            return 0.0

    def start_trading(self, historical_days: HistoricalDays) -> None:
        """
        Starts the trading session.
        Now requires API keys to be set during Trader initialization.
        Args:
            historical_days: How many days of historical data to fetch.
        """
        if not self.client:
            logger.error(
                "Binance client not initialized. Cannot start trading session."
            )
            return
        if not self.api_key or not self.secret_key:
            logger.error(
                "API key/secret missing. Cannot start WebSocket Manager for trading."
            )
            return

        try:
            self.twm = ThreadedWebsocketManager(
                api_key=self.api_key, api_secret=self.secret_key
            )
            if self.testnet:
                # Assuming python-binance >= 1.0.17 where start supports testnet directly
                # For older versions, you might need to set api_url/ws_url manually
                self.twm.start(testnet=True)
            else:
                self.twm.start()
            logger.info("Threaded WebSocket Manager started.")
        except Exception as e:
            logger.error(
                f"Failed to start ThreadedWebSocketManager: {e}", exc_info=True
            )
            return  # Exit if TWM fails to start

        # --- State Reconciliation ---
        logger.info("Verifying position state with exchange...")
        actual_size = self._get_position_size()
        expected_position = self.position  # From loaded state or default

        logger.info(
            f"Internal state position: {expected_position}, Exchange free balance query: {actual_size}"
        )

        needs_manual_intervention = False
        if (
            expected_position != 0 and actual_size <= 0
        ):  # Expected position but none exists
            logger.warning(
                f"State file indicates position {expected_position}, but exchange reports size {actual_size}. Resetting internal state to flat."
            )
            self.position = 0
            self._reset_sl_tp()
            self._save_state()  # Save the corrected state
        elif (
            expected_position == 0 and actual_size > 0
        ):  # Expected flat but position exists
            logger.critical(
                "CRITICAL STATE MISMATCH: State file indicates flat position, but exchange reports existing balance."
            )
            logger.critical(
                f"  Actual size: {actual_size}. Cannot safely resume trading without knowing entry price/SL/TP."
            )
            logger.critical(
                "  Manual intervention required: Please close the position on the exchange or delete the state file ({self.state_file_path}) to start fresh."
            )
            needs_manual_intervention = True
        elif (
            expected_position == 1 and actual_size < 0
        ):  # Expected long, found short (unlikely but possible)
            logger.critical(
                "CRITICAL STATE MISMATCH: State file indicates LONG position, but exchange reports SHORT balance/position."
            )
            logger.critical("  Manual intervention required.")
            needs_manual_intervention = True
        elif expected_position == -1 and actual_size > 0:  # Expected short, found long
            logger.critical(
                "CRITICAL STATE MISMATCH: State file indicates SHORT position, but exchange reports LONG balance/position."
            )
            logger.critical("  Manual intervention required.")
            needs_manual_intervention = True
        else:  # States seem consistent (0 and 0, or 1 and >0, or -1 and <0 - assuming _get_position_size handles short representation)
            logger.info(
                "Internal state position appears consistent with exchange balance."
            )

        if needs_manual_intervention:
            logger.error(
                "Trading will not start due to critical state mismatch requiring manual intervention."
            )
            return  # Prevent trading from starting
        # --- End State Reconciliation ---

        if self.bar_length in self.available_intervals:
            self.get_most_recent(historical_days)
            # Start socket - Indentation corrected here
            try:
                stream_name = self.twm.start_kline_socket(
                    callback=self.stream_candles,
                    symbol=self.symbol,
                    interval=self.bar_length,
                )
                if stream_name:  # Check if stream started successfully
                    logger.info(
                        f"Started Kline socket for {self.symbol} with interval {self.bar_length} (Stream: {stream_name})"
                    )
                    self.twm.join()  # Wait for TWM to finish
                    logger.info("Threaded WebSocket Manager stopped gracefully.")
                else:  # Handle failure to start stream properly
                    logger.error(
                        f"Failed to start Kline socket for {self.symbol}. Stream name was {stream_name}"
                    )
                    if self.twm:
                        self.twm.stop()  # Stop TWM if socket failed
            except Exception as e:
                logger.error(
                    f"Error during Kline socket operation or joining: {e}",
                    exc_info=True,
                )
                # Ensure TWM is stopped on exception during socket lifetime
                if self.twm:
                    self.twm.stop()
        else:
            logger.error(f"Interval {self.bar_length} not supported.")
            if self.twm:
                self.twm.stop()  # Stop TWM if interval is bad

        # Start the TWM
        if not self._initialize_twm():
            logger.critical("Failed to initialize TWM. Trading cannot start.")
            return

        # Fetch initial historical data
        if self.bar_length in self.available_intervals:
            self.get_most_recent(historical_days)
            # Subscribe to kline stream initially
            if self.twm:  # Ensure TWM was initialized
                try:
                    stream_name = self.twm.start_kline_socket(
                        callback=self.stream_candles,
                        symbol=self.symbol,
                        interval=self.bar_length,
                    )
                    if not stream_name:
                        logger.error(
                            f"Failed to start initial Kline socket for {self.symbol}."
                        )
                        self._stop_websocket()
                        return  # Exit if initial socket fails
                    else:
                        logger.info(
                            f"Started initial Kline socket for {self.symbol} (Stream: {stream_name})"
                        )
                except Exception as e:
                    logger.error(
                        f"Exception starting initial Kline socket: {e}", exc_info=True
                    )
                    self._stop_websocket()
                    return  # Exit if initial socket fails
            else:
                logger.error("TWM not initialized, cannot start socket.")
                return
        else:
            logger.error(f"Interval {self.bar_length} not supported.")
            self._stop_websocket()
            return

        # --- Main Monitoring Loop ---
        logger.info("Entering main monitoring loop...")
        check_interval = 60  # Seconds between health checks
        try:
            while True:
                # Check WebSocket status and attempt restart if needed
                self._check_and_handle_websocket(historical_days)

                # Check if TWM was stopped by the restart logic or other critical failure
                if not self.twm or not self.twm.is_alive():
                    logger.critical("TWM is not running. Exiting main loop.")
                    break

                # Optional: Add other periodic checks here if needed

                time.sleep(check_interval)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Stopping trader...")
        except Exception as e:
            logger.critical(
                f"Unhandled exception in main monitoring loop: {e}", exc_info=True
            )
        finally:
            logger.info("Exiting main loop. Stopping WebSocket...")
            self._stop_websocket()
            logger.info("Trader stopped.")
        # --- End Main Monitoring Loop ---

    def get_most_recent(self, days: HistoricalDays) -> None:
        """Fetches historical klines to initialize the DataFrame."""
        if not self.client:
            logger.error("Client not initialized. Cannot fetch historical data.")
            return

        now = datetime.utcnow()
        past = now - timedelta(days=days)
        start_str = past.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Fetching historical data for {self.symbol} from {start_str}...")

        try:
            bars = self.client.get_historical_klines(
                symbol=self.symbol,
                interval=self.bar_length,
                start_str=start_str,
                end_str=None,  # Fetch up to the latest
                limit=1000,  # Max limit per request
            )
        except Exception as e:
            logger.error(f"Error fetching historical klines: {e}", exc_info=True)
            return

        df = pd.DataFrame(
            bars,
            columns=[
                "Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close_time",
                "Quote_av",
                "Trades",
                "Tb_base_av",
                "Tb_quote_av",
                "Ignore",
            ],
        )
        df["Time"] = pd.to_datetime(df["Time"], unit="ms")
        df.set_index("Time", inplace=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col])

        # Keep only necessary columns
        self.data = df[["Open", "High", "Low", "Close", "Volume"]]
        logger.info(f"Initialized data with {len(self.data)} bars.")
        # Pre-prepare data immediately after fetching
        self._prepare_data()

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

    def _load_runtime_config(self) -> Optional[Dict[str, Any]]:
        """Loads and validates the runtime configuration from the YAML file."""
        if not self.runtime_config_path.is_file():
            logger.debug(f"Runtime config file not found: {self.runtime_config_path}")
            return None

        try:
            with open(self.runtime_config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if not isinstance(config_data, dict):
                    logger.error(
                        f"Invalid runtime config format in {self.runtime_config_path}. Expected dictionary."
                    )
                    return None

                # --- Validate using Pydantic ---
                try:
                    validated_config = RuntimeConfig.model_validate(config_data)
                    logger.debug(
                        f"Runtime config {self.runtime_config_path} validated successfully."
                    )
                except ValidationError as e:
                    logger.error(
                        f"Runtime config validation failed for {self.runtime_config_path}:\n{e}"
                    )
                    return None
                # --- End Validation ---

                # Basic validation (already partially covered by Pydantic, but keep for explicit checks)
                if validated_config.symbol != self.symbol:
                    logger.warning(
                        f"Symbol in runtime config ({validated_config.symbol}) does not match trader symbol ({self.symbol}). Ignoring config."
                    )
                    return None
                # Pydantic ensures strategy_name and strategy_params exist and params is dict

                return validated_config.model_dump()  # Return validated data as dict

        except yaml.YAMLError as e:
            logger.error(
                f"Error parsing runtime config YAML {self.runtime_config_path}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error reading runtime config file {self.runtime_config_path}: {e}",
                exc_info=True,
            )
            return None

    def _apply_runtime_config(self, config: Dict[str, Any]):
        """Applies parameters from the loaded runtime config if they exist and are valid."""
        logger.debug("Applying runtime configuration...")
        changes_applied = False

        # --- Update Trader Level Params --- #
        # Define parameters that can be updated at runtime
        updatable_params = {
            "units": float,
            "stop_loss_pct": float,
            "take_profit_pct": float,
            "trailing_stop_loss_pct": float,
            "max_cumulative_loss": float,
            "apply_atr_filter": bool,
            "apply_seasonality_filter": bool,
            # Potentially add ATR/Seasonality *value* params if needed,
            # but changing periods/hours might require strategy recalculation/restart.
        }

        for key, expected_type in updatable_params.items():
            if key in config:
                new_value = config[key]
                current_value = getattr(self, key, None)

                # Handle 'None' string for optional float params
                if (
                    expected_type is float
                    and isinstance(new_value, str)
                    and new_value.lower() == "none"
                ):
                    new_value = None

                try:
                    # Type checking/conversion
                    if new_value is not None and not isinstance(
                        new_value, expected_type
                    ):
                        logger.warning(
                            f"Runtime config: Invalid type for '{key}'. Expected {expected_type}, got {type(new_value)}. Skipping."
                        )
                        continue

                    # Apply absolute value for max_cumulative_loss
                    if key == "max_cumulative_loss" and new_value is not None:
                        new_value = abs(new_value)

                    if new_value != current_value:
                        setattr(self, key, new_value)
                        logger.info(
                            f"Runtime config: Updated '{key}' from {current_value} to {new_value}"
                        )
                        changes_applied = True
                except Exception as e:
                    logger.error(
                        f"Runtime config: Error applying value for '{key}': {e}",
                        exc_info=True,
                    )

        if not changes_applied:
            pass  # logger.debug("Runtime config: No changes detected or applied.")

        # --- IMPORTANT: Do NOT update strategy internal parameters here --- #
        # self.strategy.some_param = config.get("strategy_param", self.strategy.some_param)
        # This would likely cause inconsistencies as the strategy was initialized with specific params.
        # Strategy parameter changes should typically involve restarting the bot with new config.

    def _check_runtime_config(self):
        """Checks the runtime config file for updates and applies them."""
        self.config_check_counter += 1
        # Check less frequently than every bar
        if self.config_check_counter < CONFIG_CHECK_INTERVAL_BARS:
            return

        self.config_check_counter = 0  # Reset counter

        if not self.runtime_config_path.is_file():
            # logger.debug(f"Runtime config file not found at {self.runtime_config_path}. Skipping check.")
            return

        try:
            current_mtime = self.runtime_config_path.stat().st_mtime
            if self.last_config_mtime is None or current_mtime > self.last_config_mtime:
                logger.info(
                    f"Runtime configuration file change detected: {self.runtime_config_path}"
                )
                config = self._load_runtime_config()
                if config:
                    applied = self._apply_runtime_config(config)
                    if applied:
                        self._update_config_mtime()  # Update mtime only if change was successful
                    else:
                        logger.info(
                            "Config change loaded but not applied (e.g., position not flat)."
                        )
                        # Don't update mtime, try again next check
                else:
                    # Config invalid or mismatch, update mtime so we don't keep trying bad config
                    self._update_config_mtime()

        except OSError as e:
            logger.error(
                f"Error checking runtime config file stats ({self.runtime_config_path}): {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during runtime config check: {e}", exc_info=True
            )

    def stream_candles(self, msg: Dict[str, Any]) -> None:
        """Handles incoming candlestick data from the WebSocket."""
        # Extract candle data
        event_time = pd.to_datetime(msg["E"], unit="ms")
        candle = msg["k"]
        is_closed = candle["x"]
        close_price = float(candle["c"])
        high_price = float(candle["h"])
        low_price = float(candle["l"])
        # Start time of the current candle
        start_time = pd.to_datetime(candle["t"], unit="ms")

        # Log heartbeat or closed candle info
        # logger.debug(f"Stream: Time={event_time}, Symbol={candle['s']}, Close={close_price}, Closed={is_closed}")

        # --- Check SL/TP based on latest price update (High/Low within the candle) ---
        if self.position != 0:
            if self._check_sl_tp(high_price, low_price):  # Pass High and Low
                return  # Exit was triggered, wait for next bar

        # --- Process ONLY closed candles ---
        if is_closed:
            self.config_check_counter += 1
            # logger.info(f"Bar closed: {start_time} | Close: {close_price}")

            # Format new candle data
            new_data = pd.DataFrame(
                {
                    "Open": float(candle["o"]),
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": float(candle["v"]),
                },
                index=[start_time],
            )

            # Append new candle to historical data
            # Use pd.concat instead of append
            self.data = pd.concat([self.data, new_data])
            # Ensure data doesn't grow indefinitely (optional, keep e.g., last 5000 bars)
            # self.data = self.data.iloc[-5000:]

            # Only keep enough data for strategy calculations + buffer
            self.data = self.data.iloc[-self.required_historical_bars :]

            # Re-prepare data with the new bar
            self._prepare_data()

            # Define strategy based on FULL prepared data
            if not self.prepared_data.empty:
                self.define_strategy()
            else:
                logger.warning("Prepared data is empty, cannot define strategy.")

            # Check runtime config periodically
            if self.config_check_counter >= CONFIG_CHECK_INTERVAL_BARS:
                self._check_runtime_config()
                self.config_check_counter = 0  # Reset counter

    def _prepare_data(self) -> None:
        """Prepares data for strategy signal generation, including necessary indicators for filters."""
        if self.data.empty:
            self.prepared_data = pd.DataFrame()
            return

        df = self.data.copy()
        # Add base strategy indicators (handled by strategy.generate_signals)

        # Add indicators needed for filters
        if self.apply_atr_filter:
            df["atr"] = calculate_atr(df, period=self.atr_filter_period)
            if self.atr_filter_sma_period > 0:
                df["atr_sma"] = calculate_sma(
                    df["atr"], period=self.atr_filter_sma_period
                )

        self.prepared_data = df.copy()
        # logger.debug(f"Prepared data updated. Shape: {self.prepared_data.shape}")

    def define_strategy(self) -> None:
        """Generates signals and executes trades based on the latest prepared data."""
        if self.max_loss_stop_triggered:
            logger.warning("Max cumulative loss reached. Stopping trading.")
            self._stop_websocket()  # Corrected call
            return

        if self.prepared_data.empty:
            logger.warning("Prepared data is empty. Skipping strategy definition.")
            return

        # Generate signals using the strategy object
        signals = self.strategy.generate_signals(self.prepared_data)

        if signals.empty:
            logger.warning("Strategy generated empty signals series.")
            return

        # Get the latest signal (for the most recently closed bar)
        latest_signal = signals.iloc[-1]
        current_price = self.prepared_data["Close"].iloc[-1]
        timestamp = self.prepared_data.index[-1]

        # --- Apply Filters to Entry Signals ---
        trade_allowed_this_bar = True

        # Apply Seasonality Filter
        if self.apply_seasonality_to_this_symbol:
            start_hour, end_hour = self.parsed_trading_hours  # type: ignore
            ts_aware = (
                timestamp.tz_convert("UTC")
                if timestamp.tz
                else timestamp.tz_localize("UTC")
            )
            if not (start_hour <= ts_aware.hour < end_hour):
                trade_allowed_this_bar = False
                logger.info(
                    f"[{timestamp}] Trade signal ({latest_signal}) blocked by Seasonality Filter."
                )

        # Apply ATR Filter (if not already blocked)
        if self.apply_atr_filter and trade_allowed_this_bar:
            current_atr = self.prepared_data["atr"].iloc[-1]
            threshold = 0
            if self.atr_filter_sma_period > 0 and "atr_sma" in self.prepared_data:
                threshold = (
                    self.prepared_data["atr_sma"].iloc[-1] * self.atr_filter_multiplier
                )
            else:  # Fallback if SMA period is 0 or column missing
                threshold = current_atr * self.atr_filter_multiplier

            if pd.isna(current_atr) or pd.isna(threshold):
                logger.warning(
                    f"[{timestamp}] ATR ({current_atr}) or Threshold ({threshold}) is NaN. Filter inactive for this bar."
                )
            elif current_atr < threshold:
                trade_allowed_this_bar = False
                logger.info(
                    f"[{timestamp}] Trade signal ({latest_signal}) blocked by ATR Filter (ATR={current_atr:.4f} < Threshold={threshold:.4f})."
                )

        # --- Execute Trades based on Filtered Signal ---
        target_position = (
            latest_signal if trade_allowed_this_bar else self.position
        )  # Hold if filtered

        if self.position == target_position:
            # logger.debug(f"Holding position {self.position} at {timestamp}")
            return  # No change needed

        logger.info(
            f"Signal Update at {timestamp}: Current Pos={self.position}, Target Pos={target_position}, Price={current_price}"
        )
        self.execute_trades(target_position)

    def execute_trades(self, target_position: int) -> None:
        """Executes trades to reach the target position."""
        # Simplified: Assumes target_position is either 1, -1, or 0
        # Closes existing position first if reversing or going flat

        position_before_action = self.position  # Store initial position

        # --- Exit current position if needed --- #
        if self.position != 0 and target_position != self.position:
            logger.info(
                f"Attempting to close position {self.position} before targeting {target_position}"
            )
            closed_successfully = self._close_open_position(
                f"Signal change to {target_position}"
            )
            if not closed_successfully:
                logger.error(
                    "Failed to close existing position. Aborting further actions for this signal."
                )
                # Position remains unchanged from position_before_action
                return
            # If close was successful, self.position is now 0
            if self.max_loss_stop_triggered:
                logger.warning(
                    "Max loss triggered during position close. Stopping trading."
                )
                return  # Stop if max loss hit during close

        # --- Open new position if needed --- #
        # Now check if we need to enter a NEW position (only if currently flat)
        if target_position != 0 and self.position == 0:
            logger.info(f"Attempting to open new position {target_position}")
            side = "BUY" if target_position == 1 else "SELL"
            context = f"Opening {'Long' if target_position == 1 else 'Short'} Position"
            opened_successfully = self._open_new_position(side, self.units, context)
            if not opened_successfully:
                logger.error("Failed to open new position. Position remains flat (0).")
                # self.position should already be 0 here
                return
            # If open was successful, self.position is now target_position

        logger.debug(
            f"execute_trades finished. Position before: {position_before_action}, Position after: {self.position}, Target: {target_position}"
        )

    def _close_open_position(self, context_message: str) -> bool:
        """Attempts to close the current open position. Returns True if successful, False otherwise."""
        if self.position == 0:
            logger.warning("Attempted to close position, but already flat.")
            return True  # Considered successful as the goal is to be flat

        side = "SELL" if self.position == 1 else "BUY"
        logger.info(
            f"Executing order to close position ({side} {self.units} {self.symbol}). Reason: {context_message}"
        )

        # --- Get actual position size before closing ---
        actual_size = self._get_position_size()
        if actual_size <= 0:
            logger.warning(
                f"Attempting to close {self.position} position, but current size query returned {actual_size}. Assuming flat."
            )
            self.position = 0  # Correct state if needed
            self._reset_sl_tp()
            self._save_state()
            return True
        logger.info(f"Actual position size to close: {actual_size}")
        # --- End Get Size ---

        order = self._execute_order(
            side=side,
            quantity=actual_size,  # Use actual size for closing order
            context_message=f"Closing Position - {context_message}",
        )

        # Check if the order resulted in the position being closed
        if order and order.get("status") in ["FILLED", "PARTIALLY_FILLED"]:
            executed_qty = float(order.get("executedQty", 0))
            tolerance = 1e-8  # Adjust tolerance based on asset precision
            # Use the actual executed quantity for PnL calculation
            if executed_qty > 0:
                # --- Calculate PnL and Log Trade ---
                position_type = "Long" if self.position == 1 else "Short"
                exit_price = (
                    float(order.get("cummulativeQuoteQty", 0)) / executed_qty
                    if executed_qty
                    else 0
                )
                if exit_price == 0 and order.get("fills"):  # Fallback to avg fill price
                    fill_prices = [float(f["price"]) for f in order["fills"]]
                    fill_qtys = [float(f["qty"]) for f in order["fills"]]
                    if sum(fill_qtys) > 0:
                        exit_price = sum(
                            p * q for p, q in zip(fill_prices, fill_qtys)
                        ) / sum(fill_qtys)
                if exit_price == 0:
                    logger.warning(
                        "Could not determine exit price for PnL calculation. Using entry price as fallback."
                    )
                    exit_price = self.entry_price or 0  # Fallback

                if self.entry_price is not None:
                    gross_pnl = (
                        (exit_price - self.entry_price) * executed_qty * self.position
                    )
                    # Estimate commission (entry + exit)
                    commission_rate = self.commission_bps / 10000.0
                    commission_paid = (
                        abs(self.entry_price * executed_qty)
                        + abs(exit_price * executed_qty)
                    ) * commission_rate
                    net_pnl = gross_pnl - commission_paid
                    self.cum_profits += net_pnl  # Update cumulative profit

                    # Get exit timestamp (use event time from order if possible, else now)
                    exit_time_ms = order.get("transactTime")
                    exit_time_ts = (
                        pd.Timestamp(exit_time_ms, unit="ms")
                        if exit_time_ms
                        else pd.Timestamp.now()
                    )

                    self._log_trade_to_csv(
                        entry_time=self.entry_time,
                        entry_price=self.entry_price,
                        exit_time=exit_time_ts,
                        exit_price=exit_price,
                        position_type=position_type,
                        executed_qty=executed_qty,
                        gross_pnl=gross_pnl,
                        commission_paid=commission_paid,
                        net_pnl=net_pnl,
                        exit_reason=context_message,  # Use the reason passed in
                        entry_order_id=self.entry_order_id,
                        exit_order_id=order.get("orderId"),
                        cumulative_profit=self.cum_profits,
                    )
                else:
                    logger.warning(
                        "Cannot calculate PnL or log trade: Entry price is missing."
                    )
                # --- End PnL Calc and Log ---

            # Check if it was a full close
            if abs(executed_qty - actual_size) < tolerance:
                logger.info(
                    f"Position closed successfully (Order ID: {order.get('orderId')}). Executed: {executed_qty}, Expected: {actual_size}"
                )
                # <<< Call reset AFTER logging >>>
                self._reset_sl_tp()  # Reset SL/TP state
                self.position = 0  # Update position state
                self.entry_order_id = None  # Clear entry order ID
                self._save_state()
                return True
            elif executed_qty > 0:
                # Partial close - DON'T reset position, SL/TP, or save state fully reset
                # Just log what happened, state saving might need refinement for partials
                logger.warning(
                    f"Close order partially filled ({executed_qty}/{actual_size}). Position NOT fully closed. State partially updated (cum_profit). Manual check advised."
                )
                # We logged the partial PnL, but don't reset the main state
                self._save_state()  # Save updated cum_profits
                return False  # Indicate not fully closed
            else:  # executed_qty is 0 despite FILLED/PARTIALLY_FILLED status
                logger.error(
                    f"Close order status {order.get('status')} but executed quantity is 0. Failed to close. State unchanged."
                )
                return False
        else:
            logger.error(
                f"Failed to close position. Order status: {order.get('status', 'N/A') if order else 'No order object'}. Position state unchanged."
            )
            return False

    def _open_new_position(
        self, side: str, quantity: float, context_message: str
    ) -> bool:
        """Attempts to open a new position. Returns True if successful, False otherwise."""
        if self.position != 0:
            logger.error(
                f"Attempted to open new position, but already in position {self.position}. Aborting."
            )
            return False  # Don't open if already in position

        logger.info(
            f"Executing order to open new position ({side} {quantity} {self.symbol}). Reason: {context_message}"
        )
        order = self._execute_order(
            side=side,
            quantity=quantity,
            context_message=f"Opening Position - {context_message}",
        )

        if order and order.get("status") in ["FILLED", "PARTIALLY_FILLED"]:
            # For simplicity, assume FILLED or PARTIALLY_FILLED means entry occurred.
            # A robust system would check filled quantity.
            executed_qty = float(order.get("executedQty", 0))
            if executed_qty > 0:
                logger.info(
                    f"New position opened successfully (Order ID: {order.get('orderId')}). Executed Qty: {executed_qty}"
                )
                self.position = 1 if side == "BUY" else -1  # Update position state
                self.entry_order_id = order.get("orderId")
                # <<< Set Entry Time Here >>>
                entry_time_ms = order.get("transactTime")
                self.entry_time = (
                    pd.Timestamp(entry_time_ms, unit="ms")
                    if entry_time_ms
                    else pd.Timestamp.now()
                )
                # <<< End Set Entry Time >>>
                self._calculate_set_sl_tp(order)  # Set SL/TP based on entry
                self._save_state()
                # <<< LOG TRADE OPEN EVENT >>>
                # Need entry time - should be stored when calculating SL/TP or here
                # Let's assume self.entry_time is set correctly before this call
                if self.entry_time:
                    self._log_trade_to_csv(
                        entry_time=self.entry_time,
                        entry_price=self.entry_price,
                        exit_time=None,
                        exit_price=None,
                        position_type=(
                            "Long" if self.position == 1 else "Short"
                        ),  # Indicate position type
                        executed_qty=executed_qty,
                        gross_pnl=None,
                        commission_paid=None,
                        net_pnl=None,
                        exit_reason="Trade Open",  # Specific reason
                        entry_order_id=self.entry_order_id,
                        exit_order_id=None,
                        cumulative_profit=self.cum_profits,  # Log current cum_profit at open
                    )
                else:
                    logger.warning(
                        "Cannot log trade open event: self.entry_time not set."
                    )
                # <<< END LOG TRADE OPEN >>>
                return True
            else:
                logger.warning(
                    "Open order status is FILLED/PARTIALLY_FILLED but executed quantity is 0. Position not opened."
                )
                self.position = 0  # Ensure position is marked flat
                return False
        else:
            logger.error(
                f"Failed to open new position. Order status: {order.get('status', 'N/A') if order else 'No order object'}. Position remains flat."
            )
            self.position = 0  # Ensure position is marked flat
            return False

    def _execute_order(
        self, side: str, quantity: float, context_message: str
    ) -> Optional[Dict[str, Any]]:
        """Executes a market order via the Binance API.
        Returns the order dictionary on success (even partial fill), None on failure.
        """
        if not self.client:
            logger.error("Binance client not initialized.")
            return None

        position_before_order = self.position  # Store position before attempting order
        log_prefix = f"Order Attempt ({context_message}) - Side: {side}, Qty: {quantity}, PosBefore: {position_before_order}"
        logger.info(log_prefix)

        # Add safety check for quantity (optional, but good practice)
        if quantity <= 0:
            logger.error(
                f"{log_prefix} - Invalid quantity: {quantity}. Aborting order."
            )
            return None

        try:
            # Use testnet endpoint setting from client
            # self.client.API_URL = self.client.API_URL.replace('api.', 'testnet.') if self.testnet else self.client.API_URL.replace('testnet.', 'api.')
            logger.debug(f"Executing {side} order for {quantity} {self.symbol}")
            # Wrap the API call with the retry helper
            order = self._retry_api_call(
                self.client.create_order,
                symbol=self.symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
            )
            logger.info(f"Order submitted via API: {order}")

            # --- Process Order Result --- #
            if order and isinstance(order, dict):
                order_status = order.get("status")
                order_id = order.get("orderId")
                filled_qty = float(order.get("executedQty", 0))

                # Log trade details regardless of status for transparency
                # self.report_trade(order, context_message + f" (Status: {order_status})")

                if order_status in ["FILLED", "PARTIALLY_FILLED"]:
                    if filled_qty > 0:
                        logger.info(
                            f"Order {order_id} considered successful (Status: {order_status}, Filled Qty: {filled_qty})."
                        )
                        # State updates (position, SL/TP) should happen in the calling function (_close_open_position or _open_new_position)
                        # based on the success/failure return of this function.
                        return cast(
                            Dict[str, Any], order
                        )  # Return the successful order details
                    else:
                        logger.warning(
                            f"Order {order_id} has status {order_status} but filled quantity is 0. Treating as failed."
                        )
                        return None  # Treat as failure if no quantity filled
                elif order_status in ["REJECTED", "EXPIRED", "CANCELED", "NEW"]:
                    # NEW status might occur momentarily, but shouldn't persist for MARKET orders usually.
                    # Treat these as failures for state update purposes.
                    logger.error(
                        f"Order {order_id} failed or did not execute. Status: {order_status}. No position change."
                    )
                    return None  # Explicitly return None for failed states
                else:
                    logger.warning(
                        f"Order {order_id} has unexpected status: {order_status}. Treating as potentially failed."
                    )
                    # Consider returning order here if further checks are needed by caller, but safer to treat as None
                    return None
            else:
                logger.error(
                    f"API call for {side} {quantity} {self.symbol} returned invalid/unexpected data: {order}"
                )
                return None

        except BinanceAPIException as bae:
            # Handle specific API errors
            logger.error(
                f"Binance API Exception during order execution ({context_message}): Code={bae.code}, Message={bae.message}"
            )
            # Log details relevant to state
            logger.error(f"  Position before failed order: {position_before_order}")
            logger.error(
                f"  Current self.position: {self.position} (Should NOT have changed)"
            )
            # Depending on the error code, you might implement specific handling or retries
            # For now, simply return None to indicate failure
            return None
        except Exception as e:
            logger.error(
                f"Unexpected Exception during order execution ({context_message}): {e}",
                exc_info=True,
            )
            logger.error(f"  Position before failed order: {position_before_order}")
            logger.error(
                f"  Current self.position: {self.position} (Should NOT have changed)"
            )
            return None

    # --- SL/TP Management --- #
    def _calculate_set_sl_tp(self, entry_order: Dict[str, Any]):
        """Calculates and stores SL/TP levels based on entry price."""
        try:
            # Use average fill price if available, otherwise order price
            if "fills" in entry_order and entry_order["fills"]:
                fill_prices = [float(fill["price"]) for fill in entry_order["fills"]]
                fill_qtys = [float(fill["qty"]) for fill in entry_order["fills"]]
                if sum(fill_qtys) > 0:
                    self.entry_price = sum(
                        p * q for p, q in zip(fill_prices, fill_qtys)
                    ) / sum(fill_qtys)
                else:  # Fallback if fills somehow have 0 total qty
                    self.entry_price = float(entry_order.get("price") or 0)
                    if self.entry_price == 0:
                        logger.warning(
                            "Could not determine valid entry price from order fills or order price. Using last close."
                        )
                        self.entry_price = self.prepared_data["Close"].iloc[
                            -1
                        ]  # Last resort
            elif entry_order.get("price") and float(entry_order["price"]) > 0:
                self.entry_price = float(entry_order["price"])  # Use order price if > 0
            else:
                self.entry_price = self.prepared_data["Close"].iloc[
                    -1
                ]  # Fallback to last close
                logger.warning(
                    "Order price was zero or missing, using last close price for SL/TP calc."
                )

            if not self.entry_price:
                logger.error("Entry price is zero or None, cannot set SL/TP.")
                return

            logger.info(f"Setting SL/TP based on Entry Price: {self.entry_price:.5f}")

            # Calculate Fixed Stop Loss
            if self.stop_loss_pct:
                if self.position == 1:
                    self.current_stop_loss = self.entry_price * (1 - self.stop_loss_pct)
                elif self.position == -1:
                    self.current_stop_loss = self.entry_price * (1 + self.stop_loss_pct)
                logger.info(f"Initial Stop Loss set at: {self.current_stop_loss:.5f}")
            else:
                self.current_stop_loss = None

            # Calculate Fixed Take Profit
            if self.take_profit_pct:
                if self.position == 1:
                    self.current_take_profit = self.entry_price * (
                        1 + self.take_profit_pct
                    )
                elif self.position == -1:
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
            logger.error(f"Error calculating/setting SL/TP: {e}", exc_info=True)
            self._reset_sl_tp()  # Reset levels on error

    def _reset_sl_tp(self):
        """Resets SL/TP/TSL tracking variables."""
        self.entry_price = None
        self.current_stop_loss = None
        self.current_take_profit = None
        self.tsl_peak_price = None
        # logger.debug("SL/TP/TSL levels reset.")

    def _check_sl_tp(self, current_high: float, current_low: float) -> bool:
        """Checks if SL or TP levels were hit by the current bar's high/low price."""
        if self.position == 0:
            return False  # No position to check

        exit_reason = None
        exit_price = None

        # --- Update TSL first (if applicable) ---
        if self.trailing_stop_loss_pct and self.tsl_peak_price is not None:
            initial_sl = self.current_stop_loss
            potential_tsl_stop = None

            if self.position == 1:
                self.tsl_peak_price = max(self.tsl_peak_price, current_high)
                potential_tsl_stop = self.tsl_peak_price * (
                    1 - self.trailing_stop_loss_pct
                )
                # TSL only moves stop loss up
                if potential_tsl_stop > (initial_sl or -np.inf):
                    self.current_stop_loss = potential_tsl_stop
            elif self.position == -1:
                self.tsl_peak_price = min(self.tsl_peak_price, current_low)
                potential_tsl_stop = self.tsl_peak_price * (
                    1 + self.trailing_stop_loss_pct
                )
                # TSL only moves stop loss down
                if potential_tsl_stop < (initial_sl or np.inf):
                    self.current_stop_loss = potential_tsl_stop

            if self.current_stop_loss != initial_sl:
                logger.info(
                    f"Trailing Stop Loss updated to: {self.current_stop_loss:.5f} (Peak: {self.tsl_peak_price:.5f})"
                )

        # --- Check Exit Conditions ---
        # Check Stop Loss first
        if self.current_stop_loss is not None:
            if self.position == 1 and current_low <= self.current_stop_loss:
                exit_reason = "Stop Loss / TSL"
                exit_price = self.current_stop_loss  # Assume execution at SL level
            elif self.position == -1 and current_high >= self.current_stop_loss:
                exit_reason = "Stop Loss / TSL"
                exit_price = self.current_stop_loss

        # Check Take Profit only if SL wasn't hit
        if exit_reason is None and self.current_take_profit is not None:
            if self.position == 1 and current_high >= self.current_take_profit:
                exit_reason = "Take Profit"
                exit_price = self.current_take_profit  # Assume execution at TP level
            elif self.position == -1 and current_low <= self.current_take_profit:
                exit_reason = "Take Profit"
                exit_price = self.current_take_profit

        # --- Execute Exit Trade if Triggered ---
        if exit_reason:
            logger.info(f"Exit triggered: {exit_reason} at price ~{exit_price:.5f}")
            closed_successfully = self._close_open_position(
                context_message=f"Exit due to {exit_reason}"
            )
            if closed_successfully:
                self._save_state()
            return closed_successfully  # Indicate exit occurred (True if close was successful)

        return False  # No exit triggered

    def _stop_websocket(self):
        """Stops the WebSocket connection."""
        if self.twm:
            self.twm.stop()

    # --- API Call Retry Helper ---
    def _retry_api_call(
        self,
        api_func,
        *args,
        max_retries=3,
        initial_delay=1,
        backoff_factor=2,
        **kwargs,
    ):
        """Attempts to call a Binance API function with retries on specific errors."""
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                # Attempt the API call
                return api_func(*args, **kwargs)
            except BinanceAPIException as bae:
                logger.warning(
                    f"API Error on attempt {retries + 1}/{max_retries}: Code={bae.code}, Msg={bae.message}. Retrying..."
                )
                retries += 1
                # Specific handling for rate limits (longer wait)
                if bae.code == -1003:
                    wait_time = 60  # Wait 60 seconds for rate limit
                    logger.warning(f"Rate limit hit. Waiting {wait_time}s...")
                else:
                    wait_time = delay
                    delay *= backoff_factor  # Exponential backoff for other errors

                if retries >= max_retries:
                    logger.error(
                        f"Max retries ({max_retries}) reached for {api_func.__name__}. API Error: {bae}"
                    )
                    raise bae  # Reraise the last exception after max retries
                time.sleep(wait_time)

            except Exception as e:
                # Catch other potential exceptions (network errors, etc.)
                logger.warning(
                    f"Unexpected error on attempt {retries + 1}/{max_retries}: {e}. Retrying..."
                )
                retries += 1
                wait_time = delay
                delay *= backoff_factor

                if retries >= max_retries:
                    logger.error(
                        f"Max retries ({max_retries}) reached for {api_func.__name__}. Last Error: {e}"
                    )
                    raise e  # Reraise the last exception
                time.sleep(wait_time)

        # Should not be reached if max_retries > 0, but included for safety/type checking
        logger.error(f"Exited retry loop unexpectedly for {api_func.__name__}.")
        return None  # Or raise an exception indicating retry failure

    # --- End API Call Retry Helper ---

    # --- Added Websocket Health Check and Restart ---
    def _check_and_handle_websocket(self, historical_days: HistoricalDays) -> None:
        """Checks TWM status and attempts restart if not alive."""
        if self.twm and self.twm.is_alive():
            # All good, just log debug periodically if needed
            # logger.debug("TWM is alive.")
            return

        logger.warning(
            f"ThreadedWebsocketManager for {self.symbol} is not alive or not initialized. Attempting restart..."
        )

        # Attempt to stop cleanly first (might be redundant if already dead)
        self._stop_websocket()
        time.sleep(5)  # Brief pause before restarting

        # Re-initialize client and TWM
        logger.info("Re-initializing Binance client...")
        self._initialize_client()  # Re-check API connection
        if not self.client:
            logger.error("Failed to re-initialize client. Cannot restart TWM.")
            # Consider a longer sleep or different error handling here
            return

        logger.info("Re-initializing TWM...")
        if not self._initialize_twm():
            logger.error("Failed to re-initialize TWM.")
            # Consider a longer sleep or different error handling here
            return

        # Re-fetch recent data
        logger.info("Re-fetching recent historical data after TWM restart...")
        self.get_most_recent(historical_days)
        if self.data.empty:
            logger.error(
                "Failed to fetch recent data after TWM restart. Cannot resume strategy."
            )
            # Stop the newly started TWM?
            self._stop_websocket()
            return

        # Restart the kline socket subscription
        if self.twm:
            try:
                stream_name = self.twm.start_kline_socket(
                    callback=self.stream_candles,
                    symbol=self.symbol,
                    interval=self.bar_length,
                )
                if stream_name:
                    logger.info(
                        f"Successfully restarted Kline socket for {self.symbol} (Stream: {stream_name})"
                    )
                else:
                    logger.error(
                        f"Failed to restart Kline socket for {self.symbol} after TWM restart."
                    )
                    self._stop_websocket()  # Stop TWM again if socket fails
            except Exception as e:
                logger.error(f"Exception restarting Kline socket: {e}", exc_info=True)
                self._stop_websocket()
        else:
            logger.error(
                "TWM is None after restart attempt. Cannot restart Kline socket."
            )

    # --- End Websocket Health Check ---

    # --- State Management Methods ---
    def _get_state(self) -> Dict[str, Any]:
        """Gathers the current trader state into a dictionary."""
        return {
            "position": self.position,
            "entry_price": self.entry_price,
            "current_stop_loss": self.current_stop_loss,
            "current_take_profit": self.current_take_profit,
            "tsl_peak_price": self.tsl_peak_price,
            "trades": self.trades,  # Persist trade count
            "cum_profits": self.cum_profits,  # Persist cumulative profits
            # Add other simple state variables if needed
        }

    def _save_state(self) -> None:
        """Saves the current trader state to the JSON file."""
        state = self._get_state()
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.state_file_path, "w") as f:
                json.dump(state, f, indent=4)
            logger.info(f"Trader state saved to {self.state_file_path}")
        except Exception as e:
            logger.error(
                f"Failed to save trader state to {self.state_file_path}: {e}",
                exc_info=True,
            )

    def _load_state(self) -> None:
        """Loads trader state from the JSON file if it exists."""
        if not self.state_file_path.is_file():
            logger.info("No existing state file found. Starting with default state.")
            return

        try:
            with open(self.state_file_path, "r") as f:
                state = json.load(f)
            logger.info(f"Loading state from {self.state_file_path}")
            # Apply loaded state (with defaults for safety)
            self.position = state.get("position", 0)
            self.entry_price = state.get("entry_price")  # Keep None if not present
            self.current_stop_loss = state.get("current_stop_loss")
            self.current_take_profit = state.get("current_take_profit")
            self.tsl_peak_price = state.get("tsl_peak_price")
            self.trades = state.get("trades", 0)
            self.cum_profits = state.get("cum_profits", 0.0)
            logger.info(
                f"Loaded state: Position={self.position}, Entry={self.entry_price}, Trades={self.trades}, CumProfit={self.cum_profits}"
            )
            # If position is loaded as non-zero, SL/TP should ideally be present, but handle if not
            if self.position != 0 and self.entry_price is None:
                logger.warning(
                    "Loaded non-zero position but entry_price is missing in state file. SL/TP might be invalid."
                )
                # Consider resetting position to 0 here for safety?
                # self.position = 0
                # self._reset_sl_tp()

        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding state file {self.state_file_path}: {e}. Starting with default state."
            )
        except Exception as e:
            logger.error(
                f"Failed to load state from {self.state_file_path}: {e}. Starting with default state.",
                exc_info=True,
            )

    # --- End State Management Methods ---

    # --- Trade Logging Method ---
    def _log_trade_to_csv(
        self,
        entry_time: Optional[pd.Timestamp],
        entry_price: Optional[float],
        exit_time: Optional[pd.Timestamp],
        exit_price: Optional[float],
        position_type: str,  # "Long", "Short", or perhaps "ENTRY"?
        executed_qty: float,
        gross_pnl: Optional[float],
        commission_paid: Optional[float],
        net_pnl: Optional[float],
        exit_reason: Optional[str],
        entry_order_id: Optional[str],
        exit_order_id: Optional[str],
        cumulative_profit: Optional[float],
    ) -> None:
        """Appends a trade event (entry or exit) record to the CSV file."""
        trade_data = {
            "Symbol": self.symbol,
            "EntryTime": entry_time.isoformat() if entry_time else None,
            "EntryPrice": entry_price,
            "ExitTime": exit_time.isoformat() if exit_time else None,
            "ExitPrice": exit_price,
            "PositionType": position_type,
            "Quantity": executed_qty,
            "GrossPnL": gross_pnl,
            "Commission": commission_paid,
            "NetPnL": net_pnl,
            "ExitReason": exit_reason,
            "EntryOrderID": entry_order_id,
            "ExitOrderID": exit_order_id,
            "CumulativeProfit": cumulative_profit,  # Log profit *after* this trade
        }
        try:
            file_exists = self.trades_csv_path.is_file()
            df_trade = pd.DataFrame([trade_data])
            df_trade.to_csv(
                self.trades_csv_path, mode="a", header=not file_exists, index=False
            )
        except Exception as e:
            logger.error(
                f"Failed to log trade to CSV {self.trades_csv_path}: {e}", exc_info=True
            )

    # --- End Trade Logging Method ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance Trading Bot")

    # --- Core Trading Arguments ---
    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGY_CLASS_MAP.keys()),
        help="Short name of the strategy class to use.",
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
        "--units", type=float, required=True, help="Quantity of the asset to trade."
    )
    parser.add_argument(
        "--days",
        type=float,
        default=10,
        help="Days of historical data to load initially.",
    )
    parser.add_argument(
        "--testnet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Binance Testnet (default) or Live environment.",
    )

    # --- Strategy Parameter Loading ---
    parser.add_argument(
        "--param-config",
        type=str,
        default="config/best_params.yaml",  # Assume best params are used for live/testnet
        help="Path to YAML file containing strategy parameters (e.g., best_params.yaml).",
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
        "--apply-atr-filter",
        action="store_true",
        help="Apply ATR volatility filter.",
    )
    parser.add_argument(
        "--atr-filter-period",
        type=int,
        default=14,
        help="Period for ATR calculation.",
    )
    parser.add_argument(
        "--atr-filter-multiplier",
        type=float,
        default=1.5,
        help="Multiplier for ATR volatility threshold.",
    )
    parser.add_argument(
        "--atr-filter-sma-period",
        type=int,
        default=100,
        help="SMA period for ATR threshold baseline.",
    )
    parser.add_argument(
        "--apply-seasonality-filter",
        action="store_true",
        help="Apply seasonality filter (trading hours).",
    )
    parser.add_argument(
        "--allowed-trading-hours-utc",
        type=str,
        default=None,
        help="Allowed trading hours in UTC (e.g., '5-17'). Required if seasonality filter enabled.",
    )
    parser.add_argument(
        "--apply-seasonality-to-symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols to apply seasonality filter to (if empty, applies to the main symbol).",
    )

    # --- Runtime Configuration ---
    parser.add_argument(
        "--runtime-config",
        type=str,
        default=DEFAULT_RUNTIME_CONFIG,
        help="Path to the runtime configuration YAML file.",
    )

    # --- Logging ---
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    # --- Set Log Level ---
    log_level = getattr(logging, args.log.upper(), logging.INFO)
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

    # --- Load Strategy Parameters ---
    strategy_params = {}
    try:
        # Dynamically get the class based on the name
        strategy_class = STRATEGY_CLASS_MAP.get(args.strategy)
        if not strategy_class:
            raise ValueError(f"Unknown strategy class name: {args.strategy}")

        # Load params from the specified config file
        from .forward_test import load_best_params_from_config  # Borrow loader

        loaded_params = load_best_params_from_config(
            config_path=args.param_config,
            symbol=args.symbol,
            strategy_class_name=args.strategy,
        )
        if loaded_params:
            strategy_params = loaded_params
            logger.info(
                f"Loaded strategy parameters for {args.strategy} from {args.param_config}"
            )
        else:
            logger.warning(
                f"Could not load parameters from {args.param_config} for {args.strategy}/{args.symbol}. Using strategy defaults."
            )
            # Initialize with empty dict to use defaults in strategy __init__
            strategy_params = {}

        # Instantiate the strategy
        # Ensure strategy params are cleaned (e.g., remove SL/TP if present)
        core_strategy_params = {
            k: v
            for k, v in strategy_params.items()
            if k not in ["stop_loss_pct", "take_profit_pct", "trailing_stop_loss_pct"]
        }
        strategy_instance = strategy_class(**core_strategy_params)

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
            commission_bps=args.commission,  # Use args.commission
        )
        trader.start_trading(historical_days=args.days)
    except Exception as e:
        logger.critical(
            f"Trader initialization or execution failed: {e}", exc_info=True
        )
        # Attempt to stop trader if it was partially initialized
        if "trader" in locals() and hasattr(trader, "stop_trading"):
            trader.stop_trading()
        exit(1)
