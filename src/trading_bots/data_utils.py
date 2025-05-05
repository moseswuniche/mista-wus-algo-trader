import pandas as pd
import logging
from typing import Optional
from pathlib import Path

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_csv_data(
    file_path: str, symbol: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Loads historical data from a CSV file.
    Assumes CSV has columns like 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
    Sets 'Date' as a DatetimeIndex.

    Args:
        file_path: Path to the CSV file.
        symbol: Optional symbol name for context/logging.

    Returns:
        A pandas DataFrame with the loaded data.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        ValueError: If required columns are missing or 'Date' cannot be parsed.
    """
    path = Path(file_path)
    if not path.is_file():
        logger.error(f"Data file not found: {file_path}")
        return None

    try:
        # Specify low_memory=False to potentially handle mixed types better
        # Specify dtype explicitly if known issues with types, e.g., dtype={'Volume': float}
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date", low_memory=False)

        # >>> Convert columns to lowercase immediately after loading <<<
        df.columns = df.columns.str.lower()

        # Ensure OHLCV columns exist (check lowercase names now)
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(
                f"Missing required columns (lowercase) in {file_path}: {missing_cols}"
            )
            return None

        df.sort_index(inplace=True)

        # Optional: Log loaded columns for verification
        logger.debug(f"Loaded columns from {file_path}: {df.columns.tolist()}")

        logger.info(f"Loaded {len(df)} data points from {file_path}.")
        return df

    except FileNotFoundError:
        # This case is already handled by the initial path check, but kept for robustness
        logger.error(f"Data file not found during read: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}", exc_info=True)
        return None
