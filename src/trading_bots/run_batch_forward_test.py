#!/usr/bin/env python
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union

import yaml  # Requires PyYAML to be installed

# Assuming helper is in the same directory or accessible
from .optimization.parameter_utils import sanitize_filename  # Use updated path

# Setup logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Default paths (can be overridden by args if needed)
DEFAULT_OPTIMIZE_RESULTS_DIR = Path("results") / "optimize"
DEFAULT_FORWARD_TEST_RESULTS_DIR = Path("results") / "forward_test"
DEFAULT_DATA_DIR = Path("data")


def parse_best_params_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parses metric, symbol, and strategy CLASS NAME from a best_params filename.
    Expected format: {metric}_best_params_{SYMBOL}_{StrategyClassName}.yaml
    """
    marker = "_best_params_"
    marker_index = filename.find(marker)

    # Check 1: Basic format checks
    if not filename.endswith(".yaml") or marker_index == -1:
        logger.debug(
            f"Filename '{filename}' missing '.yaml' suffix or '{marker}' marker."
        )
        return None

    # Extract metric (part before the marker)
    metric = filename[:marker_index]
    if not metric:
        logger.warning(f"Could not extract metric part from filename: {filename}")
        return None

    # Extract the part after the marker
    remaining_part = filename[marker_index + len(marker) :].removesuffix(".yaml")
    if not remaining_part:
        logger.warning(
            f"Could not extract symbol/strategy part from filename: {filename}"
        )
        return None

    # Split the remaining part by underscore
    symbol_strategy_parts = remaining_part.split("_")

    # Symbol should be the first part
    if not symbol_strategy_parts:
        logger.warning(
            f"No symbol/strategy parts found after splitting: {remaining_part}"
        )
        return None
    symbol = symbol_strategy_parts[0]

    # Strategy Class Name is the rest, joined back together (handles underscores in strategy names)
    if len(symbol_strategy_parts) < 2:
        logger.warning(f"No strategy part found after symbol in: {remaining_part}")
        return None
    strategy_class_name = "_".join(symbol_strategy_parts[1:])

    # Final validation
    if not symbol or not strategy_class_name:
        logger.warning(
            f"Parsed empty symbol or strategy from filename: {filename} (symbol='{symbol}', strategy='{strategy_class_name}')"
        )
        return None

    return {"metric": metric, "symbol": symbol, "strategy": strategy_class_name}


def main():
    parser = argparse.ArgumentParser(
        description="Run forward tests for all optimized strategies found in results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_OPTIMIZE_RESULTS_DIR,
        help="Directory containing optimization results (best param YAML files).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Base directory containing historical data CSV files.",
    )
    parser.add_argument(
        "--start-date", required=True, help="Forward test start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date", help="Optional forward test end date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--balance",
        type=float,
        required=True,
        help="Initial balance for forward tests.",
    )
    parser.add_argument(
        "--commission",
        type=float,
        required=True,
        help="Commission per trade in basis points.",
    )
    parser.add_argument(
        "--units", type=float, default=1.0, help="Default units/quantity per trade."
    )
    parser.add_argument(
        "--interval",
        required=True,
        help="Time interval for data files (e.g., 1h, 1d).",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--target-metric",
        help="Optional: Only run tests for params optimized for this specific metric.",
    )
    parser.add_argument(
        "--optimization-metric-prefix",
        type=str,
        default=None,
        help="Optional prefix for report filenames, typically the optimization metric.",
    )
    parser.add_argument(
        "--include-symbols",
        type=str,
        default="",
        help="Optional: Comma-separated list of symbols to INCLUDE for testing (if empty, includes all).",
    )

    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.log.upper())
    logger.setLevel(log_level)
    # Optionally set levels for other imported modules if needed
    logging.getLogger("src.trading_bots.forward_test").setLevel(log_level)

    logger.info(f"Starting batch forward test run with args: {args}")

    if not args.results_dir.is_dir():
        logger.critical(f"Optimization results directory not found: {args.results_dir}")
        sys.exit(1)

    if not args.data_dir.is_dir():
        logger.critical(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Find all best param YAML files
    param_files = list(args.results_dir.glob("*_best_params_*.yaml"))

    if not param_files:
        logger.warning(f"No '*_best_params_*.yaml' files found in {args.results_dir}")
        sys.exit(0)

    # Filter by target metric if provided
    if args.target_metric:
        target_metric_safe = sanitize_filename(args.target_metric)
        logger.info(
            f"Filtering parameter files for target metric: {args.target_metric} (Sanitized: {target_metric_safe})"
        )
        filtered_files = [
            f
            for f in param_files
            if f.name.startswith(f"{target_metric_safe}_best_params_")
        ]
        if not filtered_files:
            logger.warning(
                f"No parameter files found matching metric '{args.target_metric}'. Exiting."
            )
            sys.exit(0)
        param_files = filtered_files  # Use the filtered list
    else:
        logger.info("No target metric specified, processing all found parameter files.")

    logger.info(f"Found {len(param_files)} best parameter files to process.")

    # Parse included symbols
    included_symbols_set = set()
    if args.include_symbols:
        included_symbols_set = {
            s.strip().upper() for s in args.include_symbols.split(",") if s.strip()
        }
        logger.info(f"Including only symbols: {included_symbols_set}")
    else:
        logger.info("No specific symbols requested, processing all found symbols.")

    success_count = 0
    skipped_count = 0
    error_count = 0

    for yaml_file in param_files:
        logger.info(f"--- Processing: {yaml_file.name} ---")

        # Parse filename
        parsed_info = parse_best_params_filename(yaml_file.name)
        if not parsed_info:
            logger.warning(f"Could not parse filename {yaml_file.name}. Skipping.")
            skipped_count += 1
            continue

        symbol = parsed_info["symbol"]
        strategy_class_name = parsed_info["strategy"]
        metric = parsed_info["metric"]  # Metric used for optimization (for info)

        # Skip if a specific list was provided and this symbol is NOT in it
        if included_symbols_set and symbol.upper() not in included_symbols_set:
            logger.info(f"  Symbol {symbol} is not in the include list. Skipping.")
            skipped_count += 1
            continue

        logger.info(
            f"  Parsed: Symbol={symbol}, Strategy={strategy_class_name}, Metric={metric}"
        )

        # Check data file existence
        data_file = args.data_dir / args.interval / f"{symbol}_{args.interval}.csv"
        if not data_file.is_file():
            logger.warning(f"  Data file not found: {data_file}. Skipping.")
            skipped_count += 1
            continue
        logger.debug(f"  Using data file: {data_file}")

        # Build command for forward_test.py
        cmd = [
            sys.executable,  # Use the same python executable
            "-m",
            "src.trading_bots.forward_test",
            "--strategy",
            strategy_class_name,
            "--symbol",
            symbol,
            "--file",
            str(data_file),
            "--param-config",
            str(yaml_file),
            "--fwd-start",
            args.start_date,
            "--balance",
            str(args.balance),
            "--commission",
            str(args.commission),
            "--units",
            str(args.units),
            "--log",
            args.log,
        ]

        # Add the report prefix argument based on the parsed metric
        cmd.extend(["--optimization-metric-prefix", metric])

        if args.end_date:
            cmd.extend(["--fwd-end", args.end_date])

        # Execute the command
        logger.info(f"  Running forward test command: {' '.join(cmd)}")
        try:
            # Use poetry run if poetry environment detected? Or rely on sys.executable?
            # Assuming direct execution works if environment is activated.
            # For robustness, might prepend with `poetry run` if needed.
            process = subprocess.run(
                cmd,
                check=True,  # Raise exception on non-zero exit code
                capture_output=True,  # Capture stdout/stderr
                text=True,  # Decode stdout/stderr as text
                encoding="utf-8",
            )
            logger.info(
                f"  Forward test completed successfully for {symbol}/{strategy_class_name}."
            )
            # Log stdout/stderr at DEBUG level if needed
            logger.debug(f"  stdout:\n%s", process.stdout)
            logger.debug(f"  stderr:\n%s", process.stderr)
            success_count += 1
        except subprocess.CalledProcessError as e:
            logger.error(
                f"  Forward test FAILED for {symbol}/{strategy_class_name}. Exit code: {e.returncode}"
            )
            # Use %-formatting for potentially large/multiline stdout/stderr
            logger.error("  stdout:\n%s", e.stdout)
            logger.error("  stderr:\n%s", e.stderr)
            error_count += 1
        except FileNotFoundError:
            logger.error(
                f"  Error: Could not find Python executable '{sys.executable}' or forward_test module."
            )
            error_count += 1
            # Consider stopping the whole batch run if python itself fails
            break
        except Exception as e:
            logger.error(
                f"  An unexpected error occurred running forward test for {symbol}/{strategy_class_name}: {e}",
                exc_info=True,
            )
            error_count += 1

        logger.info("--- Finished Processing ---")
        print("")  # Add a blank line for readability

    logger.info("=" * 30)
    logger.info("Batch Forward Test Summary:")
    logger.info(f"  Total Files Processed: {len(param_files)}")
    logger.info(f"  Successful Tests:      {success_count}")
    logger.info(f"  Skipped Tests:         {skipped_count}")
    logger.info(f"  Failed Tests:          {error_count}")
    logger.info("=" * 30)

    if error_count > 0:
        sys.exit(1)  # Exit with error code if any tests failed
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
