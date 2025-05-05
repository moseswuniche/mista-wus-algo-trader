"""Handles logging executed trades to a CSV file."""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Define trade log directory constant
TRADE_LOG_DIR = Path("results/live_trades")


class CsvTradeLogger:
    """Logs trade entry and exit events to a CSV file."""

    def __init__(self, symbol: str, log_file_name: Optional[str] = None):
        self.symbol = symbol
        TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_name = (
            log_file_name
            or f"live_trades_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        self.trades_csv_path = TRADE_LOG_DIR / file_name
        self._ensure_header()
        logger.info(
            f"CsvTradeLogger initialized. Logging trades to: {self.trades_csv_path}"
        )

    def _ensure_header(self) -> None:
        """Writes the CSV header if the file doesn't exist or is empty."""
        try:
            if (
                not self.trades_csv_path.is_file()
                or self.trades_csv_path.stat().st_size == 0
            ):
                header_df = pd.DataFrame(
                    columns=[
                        "Symbol",
                        "EntryTime",
                        "EntryPrice",
                        "ExitTime",
                        "ExitPrice",
                        "PositionType",
                        "Quantity",
                        "GrossPnL",
                        "Commission",
                        "NetPnL",
                        "ExitReason",
                        "EntryOrderID",
                        "ExitOrderID",
                        "CumulativeProfit",
                    ]
                )
                header_df.to_csv(self.trades_csv_path, index=False)
                logger.debug(f"CSV header written to {self.trades_csv_path}")
        except Exception as e:
            logger.error(
                f"Failed to write CSV header to {self.trades_csv_path}: {e}",
                exc_info=True,
            )

    def log_trade(
        self,
        entry_time: Optional[pd.Timestamp],
        entry_price: Optional[float],
        exit_time: Optional[pd.Timestamp],
        exit_price: Optional[float],
        position_type: str,  # e.g., "Long", "Short", "Trade Open", "Trade Close"
        executed_qty: float,
        gross_pnl: Optional[float],
        commission_paid: Optional[float],
        net_pnl: Optional[float],
        exit_reason: Optional[str],
        entry_order_id: Optional[str],
        exit_order_id: Optional[str],
        cumulative_profit: Optional[float],
    ) -> None:
        """Appends a trade event record to the CSV file."""
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
            "CumulativeProfit": cumulative_profit,
        }
        try:
            # file_exists = self.trades_csv_path.is_file() # Header check done at init
            df_trade = pd.DataFrame([trade_data])
            df_trade.to_csv(self.trades_csv_path, mode="a", header=False, index=False)
            logger.debug(
                f"Trade event logged to {self.trades_csv_path}: {position_type} Qty {executed_qty}"
            )
        except Exception as e:
            logger.error(
                f"Failed to log trade to CSV {self.trades_csv_path}: {e}", exc_info=True
            )
