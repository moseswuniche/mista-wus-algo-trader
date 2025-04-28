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
                 runtime_config_path: str = DEFAULT_RUNTIME_CONFIG,
                 stop_loss_pct: Optional[float] = None, # Added Stop Loss percentage
                 take_profit_pct: Optional[float] = None, # Added Take Profit percentage
                 trailing_stop_loss_pct: Optional[float] = None, # Added Trailing SL percentage
                 max_cumulative_loss: Optional[float] = None # Added Max Cumulative Loss threshold
                 ) -> None: 
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
            stop_loss_pct: Optional percentage for stop loss (e.g., 0.02 for 2%).
            take_profit_pct: Optional percentage for take profit (e.g., 0.04 for 4%).
            trailing_stop_loss_pct: Optional percentage for trailing stop loss (e.g., 0.01 for 1%).
            max_cumulative_loss: Optional maximum absolute cumulative loss allowed (e.g., 100.0). Bot stops if cum_profits drops below -max_cumulative_loss.
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

        # --- SL/TP Attributes ---
        self.stop_loss_pct: Optional[float] = stop_loss_pct
        self.take_profit_pct: Optional[float] = take_profit_pct
        self.trailing_stop_loss_pct: Optional[float] = trailing_stop_loss_pct # Store TSL %
        self.entry_price: Optional[float] = None
        self.current_stop_loss: Optional[float] = None
        self.current_take_profit: Optional[float] = None
        self.tsl_peak_price: Optional[float] = None # Track peak price for TSL
        # --- End SL/TP Attributes ---

        # --- Bot Stop Attributes ---
        self.max_cumulative_loss: Optional[float] = abs(max_cumulative_loss) if max_cumulative_loss is not None else None # Store absolute value
        self.max_loss_stop_triggered: bool = False
        # --- End Bot Stop Attributes ---

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
        if not self.client:
            logger.error("Binance client not initialized. Cannot start trading session.")
            return
        if not self.api_key or not self.secret_key:
             logger.error("API key/secret missing. Cannot start WebSocket Manager for trading.")
             return
             
        try:
            self.twm = ThreadedWebsocketManager(api_key=self.api_key, api_secret=self.secret_key)
            # Corrected: Add testnet=True if using testnet keys
            if self.testnet:
                # Assuming python-binance >= 1.0.17 where start supports testnet directly
                # For older versions, you might need to set api_url/ws_url manually
                self.twm.start(testnet=True) 
            else:
        self.twm.start()
            logger.info("Threaded WebSocket Manager started.")
        except Exception as e:
             logger.error(f"Failed to start ThreadedWebSocketManager: {e}", exc_info=True)
             return # Exit if TWM fails to start

        if self.bar_length in self.available_intervals:
            self.get_most_recent(symbol=self.symbol, interval=self.bar_length,
                                 days=historical_days)
            # Start socket - Indentation corrected here
            try:
                stream_name = self.twm.start_kline_socket(callback=self.stream_candles,
                                            symbol=self.symbol, interval=self.bar_length)
                if stream_name: # Check if stream started successfully
                     logger.info(f"Started Kline socket for {self.symbol} with interval {self.bar_length} (Stream: {stream_name})")
                     self.twm.join() # Wait for TWM to finish
                     logger.info("Threaded WebSocket Manager stopped gracefully.")
            else:
                    logger.error(f"Failed to start Kline socket for {self.symbol}.")
                    if self.twm: self.twm.stop() # Stop TWM if socket failed
            except Exception as e:
                 logger.error(f"Error during Kline socket operation or joining: {e}", exc_info=True)
                 # Ensure TWM is stopped on exception during socket lifetime
                 if self.twm: self.twm.stop()
        else:
            logger.error(f"Interval {self.bar_length} not supported.")
            if self.twm: self.twm.stop() # Stop TWM if interval is bad

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
        """Applies the loaded runtime configuration to the trader.
        Note: SL/TP percentage changes only apply to *new* trades entered after the config change.
        """
        # Check if bot stop triggered
        if self.max_loss_stop_triggered:
             logger.warning("Max loss stop triggered. Runtime config changes ignored.")
             return False
             
        new_strategy_name = config.get('strategy_name')
        new_params = config.get('strategy_params')
        new_stop_loss_pct = config.get('stop_loss_pct')
        new_take_profit_pct = config.get('take_profit_pct')
        new_trailing_stop_loss_pct = config.get('trailing_stop_loss_pct')
        new_max_cumulative_loss = config.get('max_cumulative_loss') # Get max loss

        # --- Basic Config Validation --- 
        if not new_strategy_name:
            logger.error("Runtime config missing 'strategy_name'.")
            return False 
        if not isinstance(new_params, dict):
            logger.error("Runtime config 'strategy_params' is not a dictionary.")
            return False
        if new_stop_loss_pct is not None and not isinstance(new_stop_loss_pct, (int, float)):
             logger.error("Runtime config 'stop_loss_pct' must be a number or null.")
             return False
        if new_take_profit_pct is not None and not isinstance(new_take_profit_pct, (int, float)):
             logger.error("Runtime config 'take_profit_pct' must be a number or null.")
             return False
        if new_trailing_stop_loss_pct is not None and not isinstance(new_trailing_stop_loss_pct, (int, float)):
             logger.error("Runtime config 'trailing_stop_loss_pct' must be a number or null.")
             return False
        if new_max_cumulative_loss is not None and not isinstance(new_max_cumulative_loss, (int, float)):
             logger.error("Runtime config 'max_cumulative_loss' must be a number or null.")
             return False
             
        current_strategy_name = self.strategy.__class__.__name__
        config_applied = False

        # --- Apply Strategy/Param Changes (only if flat) ---
        if new_strategy_name != current_strategy_name:
            if self.position == 0: 
                if new_strategy_name in STRATEGY_CLASS_MAP:
                    try:
                        logger.info(f"Runtime config change: Switching strategy to {new_strategy_name}")
                        self.strategy = STRATEGY_CLASS_MAP[new_strategy_name](**new_params)
                        logger.info(f"Successfully switched strategy with params: {new_params}")
                        config_applied = True
                    except Exception as e:
                         logger.error(f"Error instantiating new strategy {new_strategy_name}: {e}", exc_info=True)
                         logger.warning(f"Keeping old strategy {current_strategy_name}.")
                else:
                    logger.error(f"Unknown strategy '{new_strategy_name}' in runtime config.")
            else:
                logger.warning(f"Runtime config wants strategy switch, but position not flat ({self.position}). Deferring.")
        elif self.strategy.params != new_params: # Strategy same, params changed
            if self.position == 0:
                try:
                    logger.info(f"Runtime config change: Updating parameters for {current_strategy_name}")
                    self.strategy = STRATEGY_CLASS_MAP[current_strategy_name](**new_params)
                    logger.info(f"Successfully updated params for {current_strategy_name} to: {new_params}")
                    config_applied = True
                except Exception as e:
                     logger.error(f"Error updating strategy parameters for {current_strategy_name}: {e}", exc_info=True)
                     logger.warning(f"Keeping old strategy parameters.")
            else:
                 logger.warning(f"Runtime config wants param update, but position not flat ({self.position}). Deferring.")

        # --- Apply SL/TP Changes (can apply anytime, affects next trade) ---
        state_changed = False
        if new_stop_loss_pct != self.stop_loss_pct:
             logger.info(f"Runtime config change: Updating stop_loss_pct from {self.stop_loss_pct} to {new_stop_loss_pct}")
             self.stop_loss_pct = new_stop_loss_pct
             state_changed = True
             
        if new_take_profit_pct != self.take_profit_pct:
             logger.info(f"Runtime config change: Updating take_profit_pct from {self.take_profit_pct} to {new_take_profit_pct}")
             self.take_profit_pct = new_take_profit_pct
             state_changed = True
             
        if new_trailing_stop_loss_pct != self.trailing_stop_loss_pct:
             logger.info(f"Runtime config change: Updating trailing_stop_loss_pct from {self.trailing_stop_loss_pct} to {new_trailing_stop_loss_pct}")
             self.trailing_stop_loss_pct = new_trailing_stop_loss_pct
             state_changed = True
             
        if state_changed:
             logger.info("Trader state percentages/limits updated. Changes apply going forward.")
             
        # Update Max Cumulative Loss (apply absolute value)
        new_max_loss_abs = abs(new_max_cumulative_loss) if new_max_cumulative_loss is not None else None
        if new_max_loss_abs != self.max_cumulative_loss:
            logger.info(f"Runtime config change: Updating max_cumulative_loss from {self.max_cumulative_loss} to {new_max_loss_abs}")
            self.max_cumulative_loss = new_max_loss_abs
            state_changed = True
             
        return config_applied or state_changed

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
        # --- Stop Check --- 
        if self.max_loss_stop_triggered:
             # Silently return or log minimally, as TWM should be stopped already
             logger.debug("Max loss stop triggered. Ignoring incoming candle data.")
             return 
             
        try: 
        if 'e' in msg and msg['e'] == 'kline' and 'k' in msg:
            kline = msg['k']
            event_time = pd.to_datetime(msg["E"], unit="ms")
            start_time = pd.to_datetime(kline["t"], unit="ms")
                first = float(kline["o"])
                high = float(kline["h"])
                low = float(kline["l"])
                close = float(kline["c"])
                volume = float(kline["v"])
                complete = bool(kline["x"])

                # --- SL/TP Check on Each Tick --- 
                # Check SL/TP using the current tick's close price, even if bar isn't complete.
                # This allows faster reaction to SL/TP breaches.
                sl_tp_closed = self._check_sl_tp(close)
                if sl_tp_closed:
                    logger.debug(f"Position closed by SL/TP at price {close}. Skipping further processing for this tick.")
                    return # Exit callback early if SL/TP triggered a close
                # --- End SL/TP Check ---

                logger.debug(f"Tick: {self.symbol} | Start: {start_time} | Close: {close} | Vol: {volume} | Complete: {complete}")

                # --- Stop condition check --- 
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

                # --- Actions on Completed Bar --- 
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
        """Executes trades based on the latest signal. SL/TP calculation happens in _execute_order."""
        # --- Stop Check --- 
        if self.max_loss_stop_triggered:
             logger.warning("Max loss stop triggered. Skipping trade execution.")
             return 
             
        if self.prepared_data.empty:
            logger.warning("Prepared data is empty, cannot execute trades.")
            return
        try:
        latest_signal = self.prepared_data["position"].iloc[-1]
            current_position = self.position # Store current position before potential modification by _execute_order

            order_executed = False

        if latest_signal == 1: # Signal: Long
                if current_position == 0:
                self._execute_order("BUY", "GOING LONG")
                    order_executed = True
                    # self.position updated inside _calculate_set_sl_tp called by _execute_order
                elif current_position == -1:
                    self._execute_order("BUY", "GOING NEUTRAL FROM SHORT") # This resets SL/TP
                    order_executed = True
                    time.sleep(0.1)
                    self._execute_order("BUY", "GOING LONG") # This calculates new SL/TP
                    # self.position updated inside _calculate_set_sl_tp
        elif latest_signal == 0: # Signal: Neutral
                if current_position == 1:
                self._execute_order("SELL", "GOING NEUTRAL FROM LONG")
                    order_executed = True
                    # self.position updated inside _reset_sl_tp called by _execute_order
                elif current_position == -1:
                self._execute_order("BUY", "GOING NEUTRAL FROM SHORT")
                    order_executed = True
                    # self.position updated inside _reset_sl_tp called by _execute_order
        elif latest_signal == -1: # Signal: Short
                if current_position == 0:
                self._execute_order("SELL", "GOING SHORT")
                    order_executed = True
                    # self.position updated inside _calculate_set_sl_tp
                elif current_position == 1:
                    self._execute_order("SELL", "GOING NEUTRAL FROM LONG") # This resets SL/TP
                    order_executed = True
                    time.sleep(0.1)
                    self._execute_order("SELL", "GOING SHORT") # This calculates new SL/TP
                    # self.position updated inside _calculate_set_sl_tp
            
            if order_executed:
                 logger.debug(f"Trade execution attempted based on signal {latest_signal}. Final position: {self.position}")
                 
        except IndexError:
            logger.warning("Could not get latest signal from prepared data.")
        except Exception as e:
            logger.error(f"Error during trade execution logic: {e}", exc_info=True)

    def _execute_order(self, side: str, context_message: str) -> Optional[Dict[str, Any]]:
        """Executes a market order, reports the trade, and manages SL/TP state."""
        # --- Stop Check --- 
        if self.max_loss_stop_triggered:
             logger.warning(f"Max loss stop triggered. Skipping order execution: {side} {self.units} {self.symbol}")
             return None
             
        if not self.client:
            logger.error(f"Cannot execute {side} order. Binance client not initialized.")
            return None
        
        # Store position *before* executing the order to check for entry/exit later
        position_before_order = self.position 
        
        logger.info(f"Attempting to execute order: {side} {self.units} {self.symbol} ({context_message}) | Position Before: {position_before_order}")
        try:
            order = cast(Dict[str, Any], self.client.create_order(symbol=self.symbol, side=side, type="MARKET", quantity=self.units))
            
            # Process SL/TP based on whether it was an entry or exit
            if order.get('status') == 'FILLED':
            self.report_trade(order, context_message)
                
                # Check if it was an ENTRY trade (moving from flat or reversing)
                is_entry = (position_before_order == 0) or \
                           (position_before_order == 1 and side == 'SELL') or \
                           (position_before_order == -1 and side == 'BUY')
                           
                is_closing_leg_of_reversal = (position_before_order == 1 and side == 'SELL') or \
                                             (position_before_order == -1 and side == 'BUY')
                                             
                if is_entry and not is_closing_leg_of_reversal:
                    # This is the final order that establishes the new position (either from flat or second leg of reversal)
                    self._calculate_set_sl_tp(order)
                elif is_closing_leg_of_reversal:
                     # This is the first leg of a reversal (closing the old position)
                     self._reset_sl_tp() # Reset SL/TP from the old position
                     self.position = 0 # Temporarily mark as flat before the next entry order
                elif position_before_order != 0 : # This must be closing a position to flat
                     self._reset_sl_tp()
                     self.position = 0 # Mark as flat
            else:
                 logger.warning(f"Order status was not FILLED: {order.get('status')}. SL/TP state not updated.")
                 # Report trade anyway? Might be PARTIALLY_FILLED etc.
                 self.report_trade(order, context_message + f" (Status: {order.get('status')})")
                 
            return order
        except BinanceAPIException as bae:
             logger.error(f"Binance API Error executing {side} order for {self.symbol}: Code={bae.code}, Message={bae.message}")
             return None
        except Exception as e:
            logger.error(f"Error executing {side} order for {self.symbol}: {e}", exc_info=True)
            return None

    def _close_open_position(self, context_message: str) -> None:
        """Closes any open long or short position and resets SL/TP."""
        # ... (existing logging and order execution logic) ...
        # Resetting SL/TP is now handled within _execute_order when a closing trade is detected
        # We still need to ensure position is 0 here unconditionally
        logger.info(f"Closing position ({context_message}). Current position: {self.position}")
        if self.position == 1:
            logger.info("Executing SELL to close LONG position.")
            self._execute_order("SELL", context_message)
        elif self.position == -1:
            logger.info("Executing BUY to close SHORT position.")
            self._execute_order("BUY", context_message)
        else:
            logger.info("No open position to close.")
        # Explicitly set position to 0 and reset SL/TP here in case _execute_order failed or wasn't called
        self.position = 0
        self._reset_sl_tp()

    def report_trade(self, order: Dict[str, Any], going: str) -> None:
        """Formats and prints trade details, updates profit calculations, and checks max cumulative loss."""
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
            if self.trades % 2 == 0 and len(self.trade_values) >= 2:
                real_profit = round(sum(self.trade_values[-2:]), 3)

            self.cum_profits = round(sum(self.trade_values), 3)

            # Log trade details
            report_lines = []
            report_lines.append(100 * "-")
            report_lines.append(f"{time_dt} | {going}")
            report_lines.append(f"{time_dt} | Base_Units = {base_units} | Quote_Units = {quote_units} | Price = {price}")
            report_lines.append(f"{time_dt} | Trade Profit = {real_profit} | Cumulative Profits = {self.cum_profits}")
            report_lines.append(100 * "-")
            logger.info("\n" + "\n".join(report_lines))

            # --- Check Max Cumulative Loss --- 
            if self.max_cumulative_loss is not None and not self.max_loss_stop_triggered:
                if self.cum_profits < -self.max_cumulative_loss:
                    logger.critical(f"CRITICAL: Max Cumulative Loss limit reached! Cum Profits ({self.cum_profits:.3f}) < Limit (-{self.max_cumulative_loss:.3f}).")
                    self.max_loss_stop_triggered = True
                    logger.critical("Triggering immediate position closure and bot stop.")
                    self._close_open_position("MAX CUMULATIVE LOSS LIMIT REACHED") # Close position
                    if self.twm:
                        logger.info("Stopping WebSocket Manager due to max loss.")
                        try:
                            self.twm.stop()
                        except Exception as e:
                            logger.error(f"Error stopping TWM after max loss: {e}", exc_info=True)
                    # Optional: Add further shutdown logic here if needed (e.g., notifications)

        except KeyError as e:
            logger.error(f"Error processing order report: Missing key {e} in order object: {order}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during trade reporting: {e}", exc_info=True)

    # --- SL/TP Helper Methods --- 
    def _calculate_set_sl_tp(self, entry_order: Dict[str, Any]):
        """Calculates and sets initial SL/TP and TSL peak based on entry order price and updates position."""
        if not self.stop_loss_pct and not self.take_profit_pct:
            logger.debug("SL/TP percentages not configured. Skipping calculation.")
            # Still need to set position if entering
            side = entry_order["side"]
            new_position = 1 if side == 'BUY' else -1
            self.position = new_position
            return 

        try:
            side = entry_order["side"]
            new_position = 1 if side == 'BUY' else -1
            if self.position != 0 and self.position != -new_position: 
                logger.warning(f"Calculating SL/TP, but current position ({self.position}) is unexpected for a {side} entry.")
            
            entry_base_units = float(entry_order["executedQty"])
            entry_quote_units = float(entry_order["cummulativeQuoteQty"])
            
            if entry_base_units == 0:
                logger.warning("Entry order executed quantity is zero. Cannot calculate SL/TP.")
                # Set position even if SL/TP fails
                self.position = new_position
                return
                 
            self.entry_price = round(entry_quote_units / entry_base_units, 5)
            logger.info(f"Calculating SL/TP based on Entry Price: {self.entry_price}")
            
            # Initialize TSL peak price to entry price
            self.tsl_peak_price = self.entry_price 
            logger.debug(f"Initialized TSL peak price: {self.tsl_peak_price}")

            # Set Initial SL (will be potentially overridden by TSL later)
            if self.stop_loss_pct:
                if new_position == 1:
                    self.current_stop_loss = self.entry_price * (1 - self.stop_loss_pct)
                elif new_position == -1:
                    self.current_stop_loss = self.entry_price * (1 + self.stop_loss_pct)
                else:
                    self.current_stop_loss = None
                if self.current_stop_loss is not None:
                    logger.info(f"Initial Stop Loss set at: {self.current_stop_loss:.5f}")
                
            # Set TP (remains fixed)
            if self.take_profit_pct:
                if new_position == 1:
                    self.current_take_profit = self.entry_price * (1 + self.take_profit_pct)
                elif new_position == -1:
                    self.current_take_profit = self.entry_price * (1 - self.take_profit_pct)
                else:
                     self.current_take_profit = None
                if self.current_take_profit is not None:
                    logger.info(f"Take Profit set at: {self.current_take_profit:.5f}")
                    
            # Update trader's position state
            self.position = new_position

        except KeyError as e:
             logger.error(f"Error calculating SL/TP: Missing key {e} in order object: {entry_order}", exc_info=True)
             self._reset_sl_tp() 
             # Try to set position anyway? Risky if entry price unknown.
             # side = entry_order.get("side")
             # if side: self.position = 1 if side == 'BUY' else -1
        except Exception as e:
             logger.error(f"Error calculating or setting SL/TP: {e}", exc_info=True)
             self._reset_sl_tp() 

    def _reset_sl_tp(self):
        """Resets SL/TP and TSL tracking variables."""
        if self.entry_price is not None or self.tsl_peak_price is not None:
            logger.debug("Resetting SL/TP levels, entry price, and TSL peak price.")
            self.entry_price = None
            self.current_stop_loss = None
            self.current_take_profit = None
            self.tsl_peak_price = None # Reset TSL peak

    def _check_sl_tp(self, current_price: float) -> bool:
        """Checks if TSL needs updating, updates SL, checks if SL/TP hit, and triggers close. Returns True if close triggered."""
        if self.position == 0 or self.entry_price is None:
            return False 
            
        sl_hit = False
        tp_hit = False
        reason = ""
        initial_sl = self.current_stop_loss # Store initial SL before potential TSL update

        # --- Trailing Stop Loss Logic --- 
        if self.trailing_stop_loss_pct is not None and self.tsl_peak_price is not None:
            potential_tsl = None
            # Update peak price
            if self.position == 1: # Long
                self.tsl_peak_price = max(self.tsl_peak_price, current_price)
                potential_tsl = self.tsl_peak_price * (1 - self.trailing_stop_loss_pct)
            elif self.position == -1: # Short
                self.tsl_peak_price = min(self.tsl_peak_price, current_price)
                potential_tsl = self.tsl_peak_price * (1 + self.trailing_stop_loss_pct)
            
            # Update current_stop_loss if TSL is more favorable
            if potential_tsl is not None:
                new_stop_loss = None
                if self.position == 1:
                    # For long, higher SL is more favorable. Use initial SL if TSL is lower.
                    new_stop_loss = max(initial_sl if initial_sl is not None else -np.inf, potential_tsl)
                elif self.position == -1:
                    # For short, lower SL is more favorable. Use initial SL if TSL is higher.
                    new_stop_loss = min(initial_sl if initial_sl is not None else np.inf, potential_tsl)
                
                # Check if the stop loss actually changed
                if new_stop_loss is not None and new_stop_loss != self.current_stop_loss:
                     # Check against entry price to prevent TSL from moving stop to unprofitable level immediately
                     if (self.position == 1 and new_stop_loss > self.entry_price) or \
                        (self.position == -1 and new_stop_loss < self.entry_price): 
                         logger.info(f"Trailing Stop Loss updated. Peak: {self.tsl_peak_price:.5f} -> New SL: {new_stop_loss:.5f}")
                         self.current_stop_loss = new_stop_loss
                     else:
                          logger.debug(f"Potential TSL update ({new_stop_loss:.5f}) ignored as it's not yet profitable relative to entry ({self.entry_price:.5f}).")
        # --- End Trailing Stop Loss Logic ---

        # --- Check for SL/TP Hit (using potentially updated current_stop_loss) --- 
        # Check Stop Loss
        if self.current_stop_loss is not None:
            if self.position == 1 and current_price <= self.current_stop_loss:
                sl_hit = True
                reason = f"STOP LOSS triggered (Long) at {current_price:.5f} <= {self.current_stop_loss:.5f}"
            elif self.position == -1 and current_price >= self.current_stop_loss:
                sl_hit = True
                reason = f"STOP LOSS triggered (Short) at {current_price:.5f} >= {self.current_stop_loss:.5f}"
        
        # Check Take Profit (only if SL not already hit)
        if not sl_hit and self.current_take_profit is not None:
             if self.position == 1 and current_price >= self.current_take_profit:
                 tp_hit = True
                 reason = f"TAKE PROFIT triggered (Long) at {current_price:.5f} >= {self.current_take_profit:.5f}"
             elif self.position == -1 and current_price <= self.current_take_profit:
                 tp_hit = True
                 reason = f"TAKE PROFIT triggered (Short) at {current_price:.5f} <= {self.current_take_profit:.5f}"
                 
        # Execute closing order if SL or TP hit
        if sl_hit or tp_hit:
            logger.info(reason)
            self._close_open_position(f"Closing position due to: {reason}") 
            return True 
            
        return False 
    # --- End SL/TP Helper Methods ---

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
    parser.add_argument("--stop-loss", type=float, default=None, 
                        help="Stop loss percentage (e.g., 0.02 for 2%). Default: None")
    parser.add_argument("--take-profit", type=float, default=None, 
                        help="Take profit percentage (e.g., 0.04 for 4%). Default: None")
    parser.add_argument("--trailing-stop-loss", type=float, default=None, 
                        help="Trailing stop loss percentage (e.g., 0.01 for 1%). Overrides fixed SL if price moves favorably. Default: None")
    parser.add_argument("--max-cum-loss", type=float, default=None, 
                        help="Maximum absolute cumulative loss allowed (e.g., 100.0). Bot stops if reached. Default: None")

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
                     runtime_config_path=args.runtime_config,
                     stop_loss_pct=args.stop_loss, # Pass SL arg
                     take_profit_pct=args.take_profit, # Pass TP arg
                     trailing_stop_loss_pct=args.trailing_stop_loss, # Pass TSL arg
                     max_cumulative_loss=args.max_cum_loss # Pass Max Loss arg
                     )

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
