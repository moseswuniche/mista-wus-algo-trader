"""Manages connection and interaction with the Binance API client and WebSocket."""

import logging
import time
from typing import Optional, List, Dict, Any, Callable, Union, cast
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException

# --- Type Aliases --- (Consider moving to a shared types module later)
ApiKey = str
SecretKey = str
Symbol = str
Interval = str
# --- End Type Aliases ---

logger = logging.getLogger(__name__)


class ClientManager:
    def __init__(
        self,
        api_key: Optional[ApiKey] = None,
        secret_key: Optional[SecretKey] = None,
        testnet: bool = True,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.client: Optional[Client] = None
        self.twm: Optional[ThreadedWebsocketManager] = None
        self._initialize_client()

    def get_client(self) -> Optional[Client]:
        """Returns the initialized Binance client instance."""
        if not self.client:
            logger.error("Client accessed before initialization or after failure.")
        return self.client

    def _initialize_client(self) -> None:
        """Initializes the Binance client using stored credentials."""
        if not self.api_key or not self.secret_key:
            logger.error("Cannot initialize client: API key or secret key is missing.")
            self.client = None  # Ensure client is None if keys missing
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
                    f"Failed to initialize or test Binance client: {e}",
                    exc_info=True,
                )
                self.client = None  # Ensure client is None if init fails
        else:
            logger.debug("Binance Client already initialized.")

    def is_connected(self) -> bool:
        """Check if the client is initialized and responsive."""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.warning(f"Ping failed during is_connected check: {e}")
            return False

    def start_websocket(
        self,
        symbol: Symbol,
        interval: Interval,
        callback: Callable[[dict], None],
        restart_on_error: bool = True,
    ) -> bool:
        """Initializes and starts the ThreadedWebsocketManager and subscribes to klines."""
        if not self.api_key or not self.secret_key:
            logger.error("API key/secret missing. Cannot start WebSocket Manager.")
            return False
        if not self.client:
            logger.error(
                "Binance client not initialized. Cannot start WebSocket Manager."
            )
            return False

        # Stop existing TWM if it's running
        if self.twm and self.twm.is_alive():
            logger.info("Stopping existing TWM before starting new one...")
            self.stop_websocket()

        try:
            self.twm = ThreadedWebsocketManager(
                api_key=self.api_key, api_secret=self.secret_key
            )
            self.twm.start(testnet=self.testnet)
            logger.info(
                f"Threaded WebSocket Manager initialized and started (Testnet: {self.testnet})."
            )

            # Subscribe to the kline stream
            stream_name = self.twm.start_kline_socket(
                callback=callback, symbol=symbol, interval=interval
            )
            if stream_name:
                logger.info(
                    f"Started Kline socket for {symbol} with interval {interval} (Stream: {stream_name})"
                )
                return True
            else:
                logger.error(
                    f"Failed to start Kline socket for {symbol}. Stream name was {stream_name}"
                )
                self.stop_websocket()  # Stop TWM if socket failed
                return False

        except Exception as e:
            logger.error(
                f"Failed to initialize/start TWM or Kline socket: {e}",
                exc_info=True,
            )
            self.twm = None  # Ensure twm is None on failure
            return False

    def stop_websocket(self) -> None:
        """Stops the WebSocket connection."""
        if self.twm:
            logger.info("Stopping Threaded WebSocket Manager...")
            try:
                self.twm.stop()
                self.twm = None  # Clear TWM instance after stopping
                logger.info("Threaded WebSocket Manager stopped.")
            except Exception as e:
                logger.error(f"Error stopping TWM: {e}", exc_info=True)
                self.twm = None  # Still clear instance on error

    def is_websocket_alive(self) -> bool:
        """Checks if the TWM is initialized and running."""
        return bool(self.twm and self.twm.is_alive())

    def join_websocket(self) -> None:
        """Joins the TWM thread, blocking until it stops."""
        if self.twm:
            logger.info("Waiting for WebSocket Manager to join...")
            self.twm.join()
            logger.info("WebSocket Manager joined.")

    def retry_api_call(
        self,
        api_func: Callable,
        *args: Any,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
        **kwargs: Any,
    ) -> Any:
        """Attempts to call a Binance API function with retries on specific errors."""
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            try:
                # Attempt the API call
                return api_func(*args, **kwargs)
            except BinanceAPIException as bae:
                logger.warning(
                    f"API Error on attempt {retries + 1}/{max_retries} for {api_func.__name__}: Code={bae.code}, Msg={bae.message}. Retrying..."
                )
                retries += 1
                # Specific handling for rate limits (longer wait)
                if bae.code == -1003:  # IP Rate Limit
                    wait_time = float(60)  # Wait 60 seconds for rate limit
                    logger.warning(f"Rate limit hit (-1003). Waiting {wait_time}s...")
                elif bae.code == -1021:  # Timestamp error
                    logger.warning(
                        "Timestamp error (-1021). Assuming temporary sync issue, quick retry."
                    )
                    wait_time = float(min(delay, 5))  # Shorter wait, max 5s
                else:
                    wait_time = float(delay)
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
                    f"Unexpected error on attempt {retries + 1}/{max_retries} for {api_func.__name__}: {e}. Retrying..."
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
        raise RuntimeError(
            f"Retry mechanism failed for {api_func.__name__} after {max_retries} retries."
        )

    def get_historical_klines(
        self,
        symbol: Symbol,
        interval: Interval,
        start_str: str,
        end_str: Optional[str] = None,
        limit: int = 1000,
    ) -> Optional[List[List[Union[int, str, float]]]]:
        """Gets historical klines with retries. Return type reflects typical Binance Kline data.
        [open_time, open, high, low, close, volume, close_time, quote_asset_vol, num_trades, taker_buy_base_vol, taker_buy_quote_vol, ignore]
        """
        if not self.client:
            return None
        # We cast the result of retry_api_call, although mypy might not fully track it.
        result = self.retry_api_call(
            self.client.get_historical_klines,
            symbol=symbol,
            interval=interval,
            start_str=start_str,
            end_str=end_str,
            limit=limit,
        )
        # Explicitly cast or assert type if needed, but for now rely on caller handling
        return cast(Optional[List[List[Union[int, str, float]]]], result)

    def create_order(
        self, symbol: Symbol, side: str, type: str, quantity: float
    ) -> Optional[Dict[str, Any]]:
        """Creates an order with retries."""
        if not self.client:
            return None
        # Cast the result
        result = self.retry_api_call(
            self.client.create_order,
            symbol=symbol,
            side=side,
            type=type,
            quantity=quantity,
        )
        return cast(Optional[Dict[str, Any]], result)

    def get_asset_balance(self, asset: str) -> Optional[Dict[str, Any]]:
        """Gets asset balance with retries."""
        if not self.client:
            return None
        # Cast the result
        result = self.retry_api_call(self.client.get_asset_balance, asset=asset)
        return cast(Optional[Dict[str, Any]], result)

    # Add other necessary API call wrappers here (e.g., get_account)
