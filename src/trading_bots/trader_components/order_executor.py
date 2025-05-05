"""Responsible for executing trade orders via the exchange client."""

import logging
from typing import Optional, Dict, Any, cast
from binance.exceptions import BinanceAPIException

# Assuming ClientManager provides the retry mechanism and client access
from .client_manager import ClientManager

logger = logging.getLogger(__name__)

Symbol = str
Side = str  # "BUY" or "SELL"
Quantity = float


class OrderExecutor:
    """Handles the execution of trade orders via the Binance API."""

    def __init__(self, client_manager: ClientManager):
        self.client_manager = client_manager

    def execute_market_order(
        self,
        symbol: Symbol,
        side: Side,
        quantity: Quantity,
        context_message: str = "",  # Optional context for logging
    ) -> Optional[Dict[str, Any]]:
        """Executes a market order using the client manager's retry mechanism.
        Returns the order dictionary on success (FILLED or PARTIALLY_FILLED with qty > 0),
        None on failure or rejection.
        """
        client = self.client_manager.get_client()
        if not client:
            logger.error("Cannot execute order: Client not available.")
            return None

        log_prefix = f"Order Attempt ({context_message}) - Symbol: {symbol}, Side: {side}, Qty: {quantity}"
        logger.info(log_prefix)

        # Basic validation
        if side not in ["BUY", "SELL"]:
            logger.error(
                f"{log_prefix} - Invalid side: {side}. Must be 'BUY' or 'SELL'."
            )
            return None
        if quantity <= 0:
            logger.error(
                f"{log_prefix} - Invalid quantity: {quantity}. Must be positive."
            )
            return None

        try:
            order = self.client_manager.retry_api_call(
                client.create_order,
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
            )
            logger.info(f"Order submitted via API for {symbol}: {order}")

            # --- Process Order Result --- #
            if order and isinstance(order, dict):
                order_status = order.get("status")
                order_id = order.get("orderId")
                filled_qty = float(order.get("executedQty", 0))

                if order_status in ["FILLED", "PARTIALLY_FILLED"]:
                    if filled_qty > 0:
                        logger.info(
                            f"Order {order_id} ({symbol} {side} {quantity}) successful (Status: {order_status}, Filled Qty: {filled_qty})."
                        )
                        return cast(Dict[str, Any], order)
                    else:
                        logger.warning(
                            f"Order {order_id} has status {order_status} but filled quantity is 0. Treating as failed."
                        )
                        return None
                elif order_status in ["REJECTED", "EXPIRED", "CANCELED"]:
                    logger.error(
                        f"Order {order_id} failed or did not execute. Status: {order_status}."
                    )
                    return None
                elif order_status == "NEW":
                    # Should not persist for MARKET orders but handle defensively
                    logger.warning(
                        f"Order {order_id} has status NEW. Market order should fill immediately. Treating as potentially failed for now."
                    )
                    return None  # Consider None unless explicitly confirmed later
                else:
                    logger.warning(
                        f"Order {order_id} has unexpected status: {order_status}. Treating as potentially failed."
                    )
                    return None
            else:
                logger.error(
                    f"API call for {side} {quantity} {symbol} returned invalid/unexpected data: {order}"
                )
                return None

        except BinanceAPIException as bae:
            logger.error(
                f"Binance API Exception during order execution ({log_prefix}): Code={bae.code}, Message={bae.message}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected Exception during order execution ({log_prefix}): {e}",
                exc_info=True,
            )
            return None

    # Logic will be moved here
    pass
