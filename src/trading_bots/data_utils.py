import pandas as pd
import logging
from typing import Optional

# Setup basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_csv_data(file_path: str, symbol: Optional[str] = None) -> pd.DataFrame:
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
    print(f"Loading data for {symbol or 'symbol not specified'} from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        raise

    required_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    try:
        # Attempt to parse the 'Date' column - adjust format if needed
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception as e:
        raise ValueError(
            f"Error parsing 'Date' column: {e}. Ensure it's in a recognizable format."
        )

    df.set_index("Date", inplace=True)

    # Ensure numeric types for OHLCV columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaNs potentially introduced by coercion
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    print(f"Loaded {len(df)} data points.")
    return df
