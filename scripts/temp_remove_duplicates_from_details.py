import pandas as pd
import argparse
import json # Keep json import for potential future use, though not strictly needed now
import os
import logging # Use logging instead of print for consistency

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def remove_internal_duplicates(filepath: str):
    """
    Reads a CSV file containing optimization details, removes rows with
    duplicate parameter sets based *only* on the 'parameters' column
    (keeping the first occurrence), and overwrites the original file.

    Args:
        filepath: The path to the CSV details file.
    """
    if not os.path.exists(filepath):
        logger.error(f"Error: File not found at {filepath}")
        return

    logger.info(f"Processing file for internal duplicates: {filepath}")
    try:
        # Read the CSV
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(f"Read {len(df)} rows.")

        if 'parameters' not in df.columns:
            logger.error("Error: 'parameters' column not found in the CSV.")
            return

        # Store original row count
        original_count = len(df)

        # Drop duplicates based *only* on the 'parameters' column, keeping the first
        df.drop_duplicates(subset=['parameters'], keep='first', inplace=True)

        # Calculate removed count
        removed_count = original_count - len(df)

        if removed_count > 0:
            logger.info(f"Found and removed {removed_count} internal duplicate parameter sets (kept first occurrence)." )
            logger.info(f"Writing {len(df)} unique rows back to {filepath}...")
            # Overwrite the original file with the deduplicated data
            df.to_csv(filepath, index=False)
            logger.info("File successfully updated.")
        else:
            logger.info("No internal duplicate parameter sets found within the file.")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove internal duplicate parameter combinations from an optimization details CSV file."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the optimization details CSV file.",
    )
    args = parser.parse_args()

    remove_internal_duplicates(args.filepath) 