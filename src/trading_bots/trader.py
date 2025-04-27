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
import logging # Added
import argparse # Added
import os # Added
from .config_utils import load_secrets_from_aws # Added
import yaml # Added
from pathlib import Path # Added
from binance.exceptions import BinanceAPIException # Added

# Import the strategy interface and specific strategies if needed for type hints
from .strategies import (
    Strategy,
    LongShortStrategy,
    MovingAverageCrossoverStrategy,
    RsiMeanReversionStrategy,
    BollingerBandReversionStrategy
)

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define type aliases for clarity
Symbol = str
Interval = str
Units = float
Position = int # -1: short, 0: neutral, 1: long
HistoricalDays = float
ApiKey = str
SecretKey = str

# Import all strategies and map by class name for dynamic loading
STRATEGY_CLASS_MAP = {
    cls.__name__: cls for cls in [
        LongShortStrategy, 
        MovingAverageCrossoverStrategy,
        RsiMeanReversionStrategy,
        BollingerBandReversionStrategy
    ]
}

# --- Constants for Runtime Reload ---
DEFAULT_RUNTIME_CONFIG = "config/runtime_config.yaml"
CONFIG_CHECK_INTERVAL_BARS = 5 # Check config file every 5 completed bars

class Trader():
    
    def __init__(self, symbol: Symbol, bar_length: Interval, strategy: Strategy, units: Units, position: Position = 0, 
                 api_key: Optional[ApiKey] = None, secret_key: Optional[SecretKey] = None,
                 testnet: bool = True, 
                 runtime_config_path: str = DEFAULT_RUNTIME_CONFIG) -> None: # Added runtime_config_path
        """
        Initializes the Trader.

        Args:
            symbol: The trading symbol (e.g., "BTCUSDT").
            bar_length: The candle bar length (e.g., "1m", "5m").
            strategy: An instance of a class implementing the Strategy interface.
            units: The quantity of the asset to trade.
            position: The initial position (default is 0).
            api_key: Binance API key (optional, needed for live trading/orders).
            secret_key: Binance secret key (optional).
            testnet: Whether to use the Binance testnet (default True).
            runtime_config_path: Path to the runtime configuration YAML file.
        """
        self.symbol: Symbol = symbol
        self.bar_length: Interval = bar_length
        self.strategy: Strategy = strategy # Store the strategy object
        self.units: Units = units
        self.position: Position = position
        self.testnet: bool = testnet # Store testnet status
        self.api_key: Optional[ApiKey] = api_key
        self.secret_key: Optional[SecretKey] = secret_key
        self.trades: int = 0
        self.trade_values: List[float] = []
        self.cum_profits: float = 0.0

        # Data related attributes
        self.data: pd.DataFrame = pd.DataFrame()
        self.prepared_data: pd.DataFrame = pd.DataFrame()

        # Binance specific attributes
        self.twm: Optional[ThreadedWebsocketManager] = None
        self.client: Optional[Client] = None # Added client attribute

        self.available_intervals: List[Interval] = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        # Removed strategy-specific attributes like self.return_thresh, self.volume_thresh

        # --- Runtime Config Attributes --- 
        self.runtime_config_path = Path(runtime_config_path)
        self.last_config_check_time = time.time()
        self.last_config_mtime: Optional[float] = None
        self.config_check_counter: int = 0
        self._update_config_mtime() # Initialize mtime

        # Initialize client immediately if keys are provided
        if self.api_key and self.secret_key:
            self._initialize_client()
        else:
            logger.warning("API key/secret not provided. Cannot execute orders or fetch private data.")
            # Initialize client without auth for public data access like historical klines?
            # self.client = Client() # Uncomment if needed for get_historical_klines without starting TWM
    
    def _initialize_client(self) -> None:
        """Initializes the Binance client using stored credentials."""
        if not self.api_key or not self.secret_key:
            logger.error("Cannot initialize client: API key or secret key is missing.")
            return
        if self.client is None:
            try:
                self.client = Client(api_key=self.api_key, api_secret=self.secret_key, tld="com", testnet=self.testnet)
                # Perform a test connection / get account info
                self.client.ping()
                account_info = self.client.get_account()
                logger.info(f"Binance Client Initialized (Testnet: {self.testnet}). Account status: {account_info.get('accountType', 'N/A')}")
            except Exception as e:
                 logger.error(f"Failed to initialize or test Binance client: {e}", exc_info=True)
                 self.client = None # Ensure client is None if init fails
        else:
            logger.debug("Binance Client already initialized.")

    def start_trading(self, historical_days: HistoricalDays) -> None:
        """
        Starts the trading session.
        Now requires API keys to be set during Trader initialization.
        Args:
            historical_days: How many days of historical data to fetch.
        """
        # Ensure client is ready for TWM
        if not self.client:
            logger.error("Binance client not initialized. Cannot start trading session.")
            return
        if not self.api_key or not self.secret_key:
             logger.error("API key/secret missing. Cannot start WebSocket Manager for trading.")
             return
             
        # Initialize TWM using stored credentials
        try:
            self.twm = ThreadedWebsocketManager(api_key=self.api_key, api_secret=self.secret_key)
            # Note: TWM defaults to production URLs. Add testnet=True if library supports it directly
            # or manage base URLs if needed. Check python-binance documentation for TWM testnet usage.
            self.twm.start()
            logger.info("Threaded WebSocket Manager started.")
        except Exception as e:
             logger.error(f"Failed to start ThreadedWebSocketManager: {e}", exc_info=True)
             return

        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol=self.symbol, interval=self.bar_length,
                                 days=historical_days)
            # Start socket
            try:
                self.twm.start_kline_socket(callback=self.stream_candles,
                                            symbol=self.symbol, interval=self.bar_length)
                logger.info(f"Started Klines socket for {self.symbol} with interval {self.bar_length}")
                self.twm.join() # Wait for TWM to finish (e.g., by stop() call or error)
                logger.info("Threaded WebSocket Manager stopped gracefully.")
            except Exception as e:
                 logger.error(f"Error starting or joining Kline socket: {e}", exc_info=True)
                 if self.twm: self.twm.stop()
        else:
            logger.error(f"Interval {self.bar_length} not supported.")
            if self.twm: self.twm.stop()

    def get_most_recent(self, symbol: Symbol, interval: Interval, days: HistoricalDays) -> None:
        """Fetches most recent historical klines."""
        # Use an unauthenticated client if the main one isn't set up?
        # Or require the main client to be initialized first?
        # Let's assume we need *a* client, preferably the authenticated one.
        temp_client = self.client if self.client else Client() # Use main client or temp unauth client
        if not temp_client:
             logger.error("Client unavailable. Cannot fetch historical data.")
             return

        now = datetime.now(dt.timezone.utc)
        past = str(now - timedelta(days=days))
        logger.info(f"Fetching {days if days > 1 else days * 24} days of recent historical data for {symbol}...")

        try:
            bars = temp_client.get_historical_klines(symbol=symbol, interval=interval,
                                                     start_str=past, end_str=None, limit=1000)
        except Exception as e:
            logger.error(f"Error fetching historical klines: {e}", exc_info=True)
            return

        df = pd.DataFrame(bars)
        if df.empty:
            logger.warning(f"Fetched historical data is empty for {symbol}.")
            self.data = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Complete"]).set_index("Date")
            return

        df["Date"] = pd.to_datetime(df.iloc[:,0], unit="ms")
        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Close Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        for column in df.columns:
            # Use apply with pd.to_numeric for potentially better performance
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Make sure 'Complete' column is boolean
        df["Complete"] = True
        if len(df) > 0:
            df.iloc[-1, df.columns.get_loc("Complete")] = False # Set last row to False

        self.data = df
        logger.info(f"Fetched and processed {len(self.data)} historical bars for {self.symbol}.")

    def _update_config_mtime(self):
        """Reads and stores the modification time of the config file."""
        try:
            if self.runtime_config_path.is_file():
                self.last_config_mtime = self.runtime_config_path.stat().st_mtime
            else:
                 self.last_config_mtime = None # File doesn't exist
        except OSError as e:
            logger.error(f"Error accessing runtime config file stats ({self.runtime_config_path}): {e}")
            self.last_config_mtime = None
            
    def _load_runtime_config(self) -> Optional[Dict[str, Any]]:
        """Loads and validates the runtime configuration from the YAML file."""
        if not self.runtime_config_path.is_file():
            logger.debug(f"Runtime config file not found: {self.runtime_config_path}")
            return None
        
        try:
            with open(self.runtime_config_path, 'r') as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict):
                    logger.error(f"Invalid runtime config format in {self.runtime_config_path}. Expected dictionary.")
                    return None
                # Basic validation
                if config.get('symbol') != self.symbol:
                     logger.warning(f"Symbol in runtime config ({config.get('symbol')}) does not match trader symbol ({self.symbol}). Ignoring config.")
                     return None
                if 'strategy_name' not in config or 'strategy_params' not in config:
                     logger.error(f"Runtime config missing 'strategy_name' or 'strategy_params'.")
                     return None
                if not isinstance(config['strategy_params'], dict):
                    logger.error(f"'strategy_params' in runtime config must be a dictionary.")
                    return None
                return config
        except yaml.YAMLError as e:
             logger.error(f"Error parsing runtime config YAML {self.runtime_config_path}: {e}")
             return None
        except Exception as e:
             logger.error(f"Error reading runtime config file {self.runtime_config_path}: {e}", exc_info=True)
             return None
             
    def _apply_runtime_config(self, config: Dict[str, Any]):
        """Applies the loaded runtime configuration to the trader."""
        new_strategy_name = config.get('strategy_name')
        new_params = config.get('strategy_params')

        if not new_strategy_name or not isinstance(new_params, dict):
             logger.error("Invalid config passed to _apply_runtime_config.")
             return

        current_strategy_name = self.strategy.__class__.__name__
        config_changed = False

        # Check if strategy needs changing
        if new_strategy_name != current_strategy_name:
            if new_strategy_name in STRATEGY_CLASS_MAP:
                 if self.position == 0: # Only switch strategy if flat
                    try:
                        logger.info(f"Runtime config change detected: Switching strategy from {current_strategy_name} to {new_strategy_name}")
                        # Validate parameters for the new strategy? (Basic check done in load)
                        self.strategy = STRATEGY_CLASS_MAP[new_strategy_name](**new_params)
                        logger.info(f"Successfully switched strategy to {new_strategy_name} with params: {new_params}")
                        config_changed = True
                    except Exception as e:
                         logger.error(f"Error instantiating new strategy {new_strategy_name} with params {new_params}: {e}", exc_info=True)
                         # Revert? Keep old strategy? Log error and continue with old.
                         logger.warning(f"Keeping old strategy {current_strategy_name} due to error.")
                 else:
                      logger.warning(f"Runtime config wants to switch strategy to {new_strategy_name}, but position is not flat ({self.position}). Deferring change.")
            else:
                logger.error(f"Unknown strategy_name '{new_strategy_name}' in runtime config. Cannot switch.")
        else:
            # Strategy is the same, check if parameters changed
            # Simple dict comparison (won't catch nested changes if params were complex objects)
            if self.strategy.params != new_params: # Assumes Strategy base class stores params
                if self.position == 0:
                    try:
                        logger.info(f"Runtime config change detected: Updating parameters for {current_strategy_name}")
                        # Option 1: Re-instantiate (safer if __init__ does setup)
                        self.strategy = STRATEGY_CLASS_MAP[current_strategy_name](**new_params)
                        # Option 2: Try to update params directly (if strategy supports it)
                        # self.strategy.params = new_params # Requires params to be mutable
                        # self.strategy.update_params(**new_params) # If strategy has an update method
                        logger.info(f"Successfully updated parameters for {current_strategy_name} to: {new_params}")
                        config_changed = True
                    except Exception as e:
                         logger.error(f"Error updating strategy parameters for {current_strategy_name} to {new_params}: {e}", exc_info=True)
                         logger.warning(f"Keeping old strategy parameters due to error.")
                else:
                     logger.warning(f"Runtime config wants to update parameters for {current_strategy_name}, but position is not flat ({self.position}). Deferring change.")

        return config_changed # Return True if a change was successfully applied

    def _check_runtime_config(self):
        """Checks if the runtime config file has changed and applies updates."""
        if not self.runtime_config_path.is_file():
            return # No config file to check

        try:
            current_mtime = self.runtime_config_path.stat().st_mtime
            if self.last_config_mtime is None or current_mtime > self.last_config_mtime:
                logger.info(f"Runtime configuration file change detected: {self.runtime_config_path}")
                config = self._load_runtime_config()
                if config:
                     applied = self._apply_runtime_config(config)
                     if applied:
                          self._update_config_mtime() # Update mtime only if change was successful
                     else:
                          logger.info("Config change loaded but not applied (e.g., position not flat).")
                          # Don't update mtime, try again next check
                else:
                     # Config invalid or mismatch, update mtime so we don't keep trying bad config
                     self._update_config_mtime()

        except OSError as e:
            logger.error(f"Error checking runtime config file stats ({self.runtime_config_path}): {e}")
        except Exception as e:
             logger.error(f"Unexpected error during runtime config check: {e}", exc_info=True)

    def stream_candles(self, msg: Dict[str, Any]) -> None:
        """Callback function for WebSocket Kline stream."""
        try:
            if 'e' in msg and msg['e'] == 'kline' and 'k' in msg:
                kline = msg['k']
                event_time = pd.to_datetime(msg["E"], unit="ms")
                start_time = pd.to_datetime(kline["t"], unit="ms")
                first   = float(kline["o"])
                high    = float(kline["h"])
                low     = float(kline["l"])
                close   = float(kline["c"])
                volume  = float(kline["v"])
                complete= bool(kline["x"])

                # Log tick data received, even if bar is not complete
                logger.debug(f"Tick: {self.symbol} | Start: {start_time} | Close: {close} | Vol: {volume} | Complete: {complete}")

                # --- Stop condition ---
                # Check stop condition BEFORE updating data, avoid acting on partial final bar if stopped
                if self.trades >= 5:
                    if self.twm:
                        try:
                            logger.info(f"Max trades reached ({self.trades}). Stopping stream at {event_time}")
                            self.twm.stop() 
                            self._close_open_position("GOING NEUTRAL AND STOP (Max Trades)")
                        except Exception as e:
                            logger.error(f"Error stopping stream or closing position after max trades: {e}", exc_info=True)
                    return

                # --- Update DataFrame ---
                self.data.loc[start_time] = [first, high, low, close, volume, complete]

                # --- Strategy Execution & Config Check ---
                if complete:
                    self.config_check_counter += 1
                    logger.info(f"Bar complete at {start_time}. Defining strategy and executing trades. (Counter: {self.config_check_counter})")
                    self.define_strategy()
                    self.execute_trades()
                    
                    # Check for runtime config changes periodically
                    if self.config_check_counter % CONFIG_CHECK_INTERVAL_BARS == 0:
                        logger.debug(f"Checking runtime configuration (Interval: {CONFIG_CHECK_INTERVAL_BARS} bars)")
                        self._check_runtime_config()
            elif 'e' in msg and msg['e'] == 'error':
                 logger.error(f"WebSocket Error received: {msg.get('m', 'Unknown error')}")
                 # Consider stopping TWM on critical errors?
                 # if self.twm: self.twm.stop()
            else:
                 logger.warning(f"Received unexpected WebSocket message format: {msg}")
        except Exception as e:
            logger.error(f"Error processing stream message: {e}", exc_info=True)
            # Decide if error is critical and requires stopping
            # if self.twm: self.twm.stop()

    def define_strategy(self) -> None:
        """Generates strategy signals using the assigned strategy object."""
        if self.data.empty:
            logger.warning("Data is empty, cannot define strategy.")
            return
        try:
            signals = self.strategy.generate_signals(self.data)
            self.prepared_data = self.data.copy()
            self.prepared_data['position'] = signals.reindex(self.data.index).fillna(0).astype(int)
            logger.info("Strategy signals defined.")
        except Exception as e:
            logger.error(f"Error defining strategy signals: {e}", exc_info=True)

    def execute_trades(self) -> None:
        """Executes trades based on the latest signal."""
        if self.prepared_data.empty:
            logger.warning("Prepared data is empty, cannot execute trades.")
            return
        try:
            latest_signal = self.prepared_data["position"].iloc[-1]

            if latest_signal == 1: # Signal: Long
                if self.position == 0:
                    self._execute_order("BUY", "GOING LONG")
                    self.position = 1
                elif self.position == -1:
                    self._execute_order("BUY", "GOING NEUTRAL FROM SHORT") # Close short
                    time.sleep(0.1) # Allow time for order processing
                    self._execute_order("BUY", "GOING LONG") # Open long
                    self.position = 1
                # If self.position is already 1, do nothing (already long)
            elif latest_signal == 0: # Signal: Neutral
                if self.position == 1:
                    self._execute_order("SELL", "GOING NEUTRAL FROM LONG")
                    self.position = 0
                elif self.position == -1:
                    self._execute_order("BUY", "GOING NEUTRAL FROM SHORT")
                    self.position = 0
                # If self.position is already 0, do nothing (already neutral)
            elif latest_signal == -1: # Signal: Short
                if self.position == 0:
                    self._execute_order("SELL", "GOING SHORT")
                    self.position = -1
                elif self.position == 1:
                    self._execute_order("SELL", "GOING NEUTRAL FROM LONG") # Close long
                    time.sleep(0.1) # Allow time for order processing
                    self._execute_order("SELL", "GOING SHORT") # Open short
                    self.position = -1
                # If self.position is already -1, do nothing (already short)
        except IndexError:
            logger.warning("Could not get latest signal from prepared data.")
        except Exception as e:
            logger.error(f"Error during trade execution logic: {e}", exc_info=True)

    def _execute_order(self, side: str, context_message: str) -> Optional[Dict[str, Any]]:
        """Executes a market order and reports the trade."""
        if not self.client:
            logger.error(f"Cannot execute {side} order. Binance client not initialized.")
            return None
        logger.info(f"Attempting to execute order: {side} {self.units} {self.symbol} ({context_message}) | Current Position: {self.position}")
        try:
            order = cast(Dict[str, Any], self.client.create_order(symbol=self.symbol, side=side, type="MARKET", quantity=self.units))
            self.report_trade(order, context_message)
            return order
        except BinanceAPIException as bae:
             logger.error(f"Binance API Error executing {side} order for {self.symbol}: Code={bae.code}, Message={bae.message}")
             return None
        except Exception as e:
            logger.error(f"Error executing {side} order for {self.symbol}: {e}", exc_info=True)
            return None

    def _close_open_position(self, context_message: str) -> None:
        """Closes any open long or short position."""
        logger.info(f"Attempting to close position if open ({context_message}). Current position: {self.position}")
        if self.position == 1:
            logger.info("Closing LONG position.")
            self._execute_order("SELL", context_message)
        elif self.position == -1:
            logger.info("Closing SHORT position.")
            self._execute_order("BUY", context_message)
        else:
            logger.info("No open position to close.")
        self.position = 0 # Ensure position is marked as neutral

    def report_trade(self, order: Dict[str, Any], going: str) -> None:
        """Formats and prints trade details and updates profit calculations."""
        try:
            side = order["side"]
            # Use 'updateTime' if 'transactTime' is 0 or missing, prefer 'transactTime'
            transact_time = order.get("transactTime", order.get("updateTime"))
            time_dt = pd.to_datetime(transact_time, unit="ms") if transact_time else datetime.now(dt.timezone.utc)

            base_units = float(order["executedQty"])
            quote_units = float(order["cummulativeQuoteQty"])

            # Avoid division by zero if executed quantity is zero
            price = round(quote_units / base_units, 5) if base_units != 0 else 0.0

            self.trades += 1
            trade_value = quote_units if side == "SELL" else -quote_units
            self.trade_values.append(trade_value)

            real_profit = 0.0
            # Calculate profit only on closing trades (even number of trades)
            if self.trades % 2 == 0 and len(self.trade_values) >= 2:
                real_profit = round(sum(self.trade_values[-2:]), 3)

            # Cumulative profit includes all trades
            self.cum_profits = round(sum(self.trade_values), 3)

            # Use INFO level for trade reports
            report_lines = []
            report_lines.append(100 * "-")
            report_lines.append(f"{time_dt} | {going}")
            report_lines.append(f"{time_dt} | Base_Units = {base_units} | Quote_Units = {quote_units} | Price = {price}")
            report_lines.append(f"{time_dt} | Trade Profit = {real_profit} | Cumulative Profits = {self.cum_profits}")
            report_lines.append(100 * "-")
            logger.info("\n" + "\n".join(report_lines)) # Log as multi-line info message

        except KeyError as e:
            logger.error(f"Error processing order report: Missing key {e} in order object: {order}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during trade reporting: {e}", exc_info=True)

if __name__ == "__main__":

    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Run Binance Trading Bot")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol.")
    parser.add_argument("--interval", type=str, default="1m", help="Kline interval.")
    parser.add_argument("--units", type=float, default=0.001, help="Trading units.")
    parser.add_argument("--strategy", type=str, default="LongShort", 
                        choices=["LongShort", "MACross", "RSIReversion", "BBReversion"], 
                        help="Strategy to use.")
    parser.add_argument("--hist_days", type=float, default=1/24, help="Days of historical data to fetch.")
    parser.add_argument("--log", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level. Default: INFO")
    parser.add_argument("--testnet", action=argparse.BooleanOptionalAction, default=True, 
                         help="Use Binance Testnet (default: True). Use --no-testnet for live.") # Changed to boolean flag
    parser.add_argument("--aws-secret-name", type=str, default=None,
                        help="Name of the secret in AWS Secrets Manager.")
    parser.add_argument("--aws-region", type=str, default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"), # Default from env or us-east-1
                        help="AWS region for Secrets Manager.")
    parser.add_argument("--runtime-config", type=str, default=DEFAULT_RUNTIME_CONFIG,
                        help=f"Path to the runtime configuration YAML file (default: {DEFAULT_RUNTIME_CONFIG}).")
    # Add args for strategy parameters if needed, or configure them below

    args = parser.parse_args()

    # Set logging level from args
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    logger.info(f"Starting trader with args: {args}")
    logger.warning("Using LIVE trading requires real funds and carries significant risk. Ensure API keys are for the correct environment (Live/Testnet).") if not args.testnet else None

    # --- Load API Keys --- 
    api_key: Optional[ApiKey] = None
    secret_key: Optional[SecretKey] = None
    aws_secrets: Optional[Dict[str, str]] = None

    # 1. Try AWS Secrets Manager if specified
    if args.aws_secret_name:
        aws_secrets = load_secrets_from_aws(args.aws_secret_name, args.aws_region)
        if aws_secrets: # If secrets were loaded successfully from AWS
            key_name = "BINANCE_API_KEY" if not args.testnet else "BINANCE_TESTNET_API_KEY"
            secret_name = "BINANCE_SECRET_KEY" if not args.testnet else "BINANCE_TESTNET_SECRET_KEY"
            api_key = aws_secrets.get(key_name)
            secret_key = aws_secrets.get(secret_name)
            if api_key and secret_key:
                 logger.info(f"Loaded API keys from AWS Secret: {args.aws_secret_name}")
            else:
                 logger.warning(f"Could not find expected keys ('{key_name}', '{secret_name}') within AWS Secret: {args.aws_secret_name}")
        else:
            logger.warning(f"Failed to load secrets from AWS: {args.aws_secret_name}. Falling back to environment variables.")

    # 2. Fallback to Environment Variables if not loaded from AWS
    if not api_key or not secret_key:
        logger.info("Attempting to load API keys from environment variables...")
        if args.testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY")
            secret_key = os.getenv("BINANCE_TESTNET_SECRET_KEY")
            env_var_names = "BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET_KEY"
        else:
            api_key = os.getenv("BINANCE_API_KEY")
            secret_key = os.getenv("BINANCE_SECRET_KEY")
            env_var_names = "BINANCE_API_KEY, BINANCE_SECRET_KEY"
        
        if api_key and secret_key:
            logger.info(f"Loaded API keys from environment variables ({env_var_names})")
        else:
             logger.warning(f"Could not find API keys in environment variables ({env_var_names}). Live trading/order execution will be disabled.")
             # Removed the hardcoded fallback - enforce secure practices

    # --- Strategy Selection and Initialization ---
    strategy_instance: Optional[Strategy] = None
    if args.strategy == "LongShort":
        # Define params for LongShort (could also come from config or args)
        ls_return_thresh: Tuple[float, float] = (-0.0001, 0.0001)
        ls_volume_thresh: Tuple[float, float] = (-3, 3)
        strategy_instance = LongShortStrategy(return_thresh=ls_return_thresh, volume_thresh=ls_volume_thresh)
    # Add elif blocks for other strategies (MACross, RSIReversion, BBReversion)
    # elif args.strategy == "MACross":
    #     strategy_instance = MovingAverageCrossoverStrategy(fast_period=9, slow_period=21) # Example
    # elif args.strategy == "RSIReversion":
    #     strategy_instance = RsiMeanReversionStrategy(rsi_period=14, oversold_threshold=30, overbought_threshold=70) # Example
    # elif args.strategy == "BBReversion":
    #     strategy_instance = BollingerBandReversionStrategy(bb_period=20, bb_std_dev=2.0) # Example
    else:
        logger.critical(f"Strategy '{args.strategy}' is not implemented in the main execution block.")
        exit(1) # Exit if strategy cannot be initialized

    # --- Trader Initialization ---
    if not api_key or not secret_key:
         logger.warning("API Key or Secret Key is missing. Check environment variables (BINANCE_TESTNET_API_KEY, BINANCE_TESTNET_SECRET_KEY) or script defaults. Live trading disabled.")
         # Decide if you want to exit or run without trading capabilities
         # exit(1)
         
    trader = Trader(symbol=args.symbol,
                     bar_length=args.interval,
                     strategy=strategy_instance,
                     units=args.units,
                     api_key=api_key, # Pass potentially None keys
                     secret_key=secret_key,
                     testnet=args.testnet,
                     runtime_config_path=args.runtime_config) # Pass runtime config path

    # --- Start Trading ---
    logger.info(f"Starting trader for {args.symbol} with strategy {args.strategy}...")
    try:
        trader.start_trading(historical_days=args.hist_days)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected. Stopping trader...")
        if trader.twm:
            trader.twm.stop()
        trader._close_open_position("STOPPED MANUALLY (Keyboard Interrupt)")
    except Exception as e:
        logger.critical(f"An critical error occurred during trading: {e}", exc_info=True)
        if trader.twm:
             trader.twm.stop()
        trader._close_open_position("STOPPED DUE TO CRITICAL ERROR")

    logger.info("Trading script finished.")
