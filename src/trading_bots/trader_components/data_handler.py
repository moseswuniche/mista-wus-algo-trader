"""Handles fetching, processing, and streaming live market data."""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Callable, Dict, Any, List
from datetime import timedelta, datetime
from binance.client import Client  # Needed for historical fetch

# Import dependent components
from .client_manager import ClientManager
# Import StateManager for type hinting
from .state_manager import StateManager

# Import indicators if preparation happens here (might move later)
from ..technical_indicators import calculate_atr, calculate_sma

logger = logging.getLogger(__name__)

# --- Type Aliases --- (Consider moving)
Symbol = str
Interval = str
HistoricalDays = float
# --- End Type Aliases ---


class DataHandler:
    """Handles fetching historical data, processing incoming candles, and preparing data for strategies."""

    available_intervals: List[str] = [
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

    def __init__(
        self,
        symbol: Symbol,
        bar_length: Interval,  # Renamed from interval
        client_manager: ClientManager,
        strategy: Any,  # BaseStrategy or specific strategy instance
        closed_bar_callback: Callable[[pd.DataFrame], None],  # Callback for Trader
        required_lookback: int,  # Max lookback needed
        state_manager: 'StateManager', # Use forward reference string literal
        indicator_config: Optional[Dict[str, Any]] = None,
        # Removed strategy_lookback_period, filter params (use indicator_config)
    ):
        if bar_length not in self.available_intervals:
            raise ValueError(f"Interval {bar_length} not supported by Binance API.")

        self.symbol = symbol
        self.bar_length = bar_length
        self.client_manager = client_manager
        self.strategy = strategy
        self.closed_bar_callback = closed_bar_callback
        self.required_lookback = required_lookback
        self.state_manager = state_manager  # Store state manager if needed
        self.indicator_config = indicator_config or {}  # Store config for _prepare_data

        # Filter-related properties needed for data preparation
        # Removed direct filter attributes - use indicator_config
        # self.apply_atr_filter = apply_atr_filter
        # self.atr_filter_period = atr_filter_period
        # self.atr_filter_sma_period = atr_filter_sma_period

        self.data: pd.DataFrame = pd.DataFrame()  # Stores the main OHLCV data
        self.prepared_data: pd.DataFrame = pd.DataFrame()  # Data with indicators
        # Removed duplicate assignments and _determine_required_bars call
        # self.required_historical_bars = self._determine_required_bars()
        # self.symbol = symbol
        # self.bar_length = bar_length
        # self.required_lookback = required_lookback
        # self.closed_bar_callback = closed_bar_callback
        # self.indicator_config = indicator_config or {}
        # self.data: pd.DataFrame = pd.DataFrame()
        # self.prepared_data: pd.DataFrame = pd.DataFrame()

        logger.info(
            f"DataHandler initialized for {symbol} interval {bar_length}. Required lookback: {required_lookback}"
        )

    def get_latest_prepared_data(self) -> pd.DataFrame:
        """Returns the most recent prepared data frame."""
        return self.prepared_data.copy()  # Return a copy

    def fetch_historical_data(self, days: HistoricalDays) -> bool:
        """Fetches historical klines to initialize the DataFrame. Returns True on success."""
        now = datetime.utcnow()
        past = now - timedelta(days=days)
        start_str = past.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Fetching historical data for {self.symbol} from {start_str}...")

        bars = self.client_manager.get_historical_klines(
            symbol=self.symbol,
            interval=self.bar_length,
            start_str=start_str,
            end_str=None,
            limit=1000,
        )

        if bars is None:
            logger.error("Failed to fetch historical klines (API call returned None).")
            return False

        try:
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

            self.data = df[["Open", "High", "Low", "Close", "Volume"]]
            # Ensure minimum required bars are loaded
            if len(self.data) < self.required_lookback:
                logger.warning(
                    f"Loaded historical data ({len(self.data)} bars) is less than required lookback ({self.required_lookback}). Strategy might be delayed."
                )
            else:
                logger.info(f"Initialized data with {len(self.data)} bars.")

            self._prepare_data()  # Prepare initial data
            return True
        except Exception as e:
            logger.error(
                f"Error processing fetched historical klines: {e}", exc_info=True
            )
            self.data = pd.DataFrame()  # Clear data on error
            self.prepared_data = pd.DataFrame()
            return False

    # Renamed from stream_candles_callback
    def process_kline_message(self, msg: Dict[str, Any]) -> None:
        """Handles incoming candlestick data from the WebSocket (TWM callback)."""
        if msg.get("e") == "error":
            logger.error(f"WebSocket Error Message: {msg.get('m')}")
            # Consider adding logic to signal a restart is needed
            return

        if msg.get("stream") and "kline" not in msg["stream"]:
            # logger.debug(f"Ignoring non-kline message type: {msg.get('e')}")
            return

        if "k" not in msg.get("data", {}):
            logger.warning(
                f"Kline data missing ('k') in message data: {msg.get('data')}"
            )
            return

        candle = msg["data"]["k"]
        is_closed = candle.get("x", False)
        start_time_ms = candle.get("t")

        # --- Process Closed Candle --- #
        try:
            start_time = pd.to_datetime(start_time_ms, unit="ms")
            # Ensure data uses consistent types
            open_price = float(candle["o"])
            high_price = float(candle["h"])
            low_price = float(candle["l"])
            close_price = float(candle["c"])
            volume = float(candle["v"])
            # logger.debug(f"Bar closed: {start_time} | Close: {close_price}")

            new_data = pd.DataFrame(
                {
                    "Open": open_price,
                    "High": high_price,
                    "Low": low_price,
                    "Close": close_price,
                    "Volume": volume,
                },
                index=[start_time],
            )

            # Append new candle
            self.data = pd.concat([self.data, new_data])
            # Optional: Drop duplicates based on index, keeping the last occurrence
            self.data = self.data[~self.data.index.duplicated(keep="last")]
            # Keep only necessary bars (lookback + buffer)
            self.data = self.data.iloc[-(self.required_lookback + 50) :]  # Keep buffer

            # Prepare data with indicators AND STRATEGY SIGNALS
            self._prepare_data()

            # Call the main callback to notify trader loop
            if not self.prepared_data.empty:
                self.closed_bar_callback(
                    self.prepared_data.copy()
                )  # Pass the prepared data
            else:
                logger.warning(
                    "Prepared data is empty after processing closed bar. Callback not called."
                )
        # --- Add except blocks here, aligned with the try --- #
        except KeyError as e:
            logger.error(f"KeyError processing candle message: {e}. Message: {msg}")
        except (ValueError, TypeError) as e:
            logger.error(
                f"Data type error processing candle message: {e}. Message: {msg}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in process_kline_message: {e}", exc_info=True
            )

    def _prepare_data(self) -> None:
        """Prepares data by calculating necessary indicators based on config.
        This might be simplified if the strategy itself calculates indicators.
        """
        if self.data.empty or len(self.data) < 2:  # Need at least 2 rows for some calcs
            self.prepared_data = pd.DataFrame()
            return

        df = self.data.copy()

        # --- Calculate indicators based on indicator_config --- #
        # Example: Add indicators needed for filters
        if self.indicator_config.get("apply_atr_filter", False):
            atr_period = self.indicator_config.get("atr_filter_period", 14)
            sma_period = self.indicator_config.get("atr_filter_sma_period", 100)
            if len(df) >= atr_period:
                df["atr"] = calculate_atr(df, period=atr_period)
                if (
                    sma_period > 0 and len(df) >= atr_period + sma_period - 1
                ):  # Check length for SMA
                    df["atr_sma"] = calculate_sma(df["atr"], period=sma_period)
                else:
                    df["atr_sma"] = pd.NA  # Not enough data for SMA
            else:
                df["atr"] = pd.NA  # Not enough data for ATR
                df["atr_sma"] = pd.NA
        # --- End Indicator Calculation --- #

        self.prepared_data = df.copy()
        # logger.debug(f"Prepared data updated. Shape: {self.prepared_data.shape}")
