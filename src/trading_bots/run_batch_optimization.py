#!/usr/bin/env python
import argparse
import subprocess
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
import shlex
from typing import List, Tuple, Optional

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---
# DEFAULT_PROCESSES = max(1, cpu_count() - 1)  # Old default
DEFAULT_PROCESSES = 6  # Optimized default for M3 Pro (typically 6 Performance cores)


def run_single_optimization(args_tuple: Tuple) -> Tuple[str, int, str, str]:
    """
    Worker function to run a single optimization command using subprocess.

    Args:
        args_tuple: A tuple containing (strategy, symbol, file_path_str,
                     common_args_str, job_index, total_jobs).

    Returns:
        A tuple containing (job_description, return_code, stdout, stderr).
    """
    (
        strategy,
        symbol,
        file_path_str,
        common_args_str,
        job_index,
        total_jobs,
    ) = args_tuple

    job_description = f"Optimize {strategy} {symbol} ({job_index}/{total_jobs})"
    logger.info(f"--- Starting: {job_description} ---")

    # Construct the specific arguments for this run
    specific_args = (
        f"--strategy {strategy} --symbol {symbol} --file {shlex.quote(file_path_str)}"
    )

    # Combine specific and common arguments
    full_args_str = f"{specific_args} {common_args_str}"

    # Construct the full command using poetry run
    # Ensure the command is correctly tokenized for subprocess
    command = [
        "poetry",
        "run",
        "python",
        "-m",
        "src.trading_bots.optimize",
    ] + shlex.split(
        full_args_str
    )  # Split the args string safely

    logger.debug(
        f"Executing command: {' '.join(command)}"
    )  # Log the command for debugging

    try:
        # Execute the command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
        )

        stdout_log = result.stdout.strip()
        stderr_log = result.stderr.strip()

        if result.returncode == 0:
            logger.info(f"--- Finished successfully: {job_description} ---")
            if stdout_log:
                logger.info(f"Stdout: \n{stdout_log}")  # Log stdout only if not empty
            if stderr_log:
                logger.warning(
                    f"Stderr: \n{stderr_log}"
                )  # Log stderr as warning if not empty
        else:
            logger.error(
                f"--- Failed: {job_description} (Exit Code: {result.returncode}) ---"
            )
            if stdout_log:
                logger.error(f"Stdout: \n{stdout_log}")
            if stderr_log:
                logger.error(f"Stderr: \n{stderr_log}")

        return (job_description, result.returncode, stdout_log, stderr_log)

    except Exception as e:
        logger.error(
            f"--- Exception during: {job_description} - {e} ---", exc_info=True
        )
        return (job_description, -1, "", str(e))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch strategy optimization in parallel."
    )

    # Arguments mirroring Makefile variables
    parser.add_argument(
        "--strategies",
        type=str,
        required=True,
        help="Space-separated list of strategy names.",
    )
    parser.add_argument(
        "--symbols", type=str, required=True, help="Space-separated list of symbols."
    )
    parser.add_argument(
        "--interval", type=str, required=True, help="Data interval (e.g., 1h)."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Optimization start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Optimization end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission in basis points (e.g., 7.5 for 0.075%).",
    )
    parser.add_argument(
        "--balance", type=float, default=10000.0, help="Initial balance for backtests."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cumulative_profit",
        help="Metric for optimization.",
    )
    parser.add_argument(
        "--save-details",
        action="store_true",
        help="Save detailed optimization results.",
    )
    parser.add_argument(
        "--details-file",
        type=str,
        default=None,
        help="Optional path for detailed results CSV.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Base directory for CSV data files.",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=DEFAULT_PROCESSES,
        help=f"Number of parallel processes to use (default: {DEFAULT_PROCESSES}).",
    )

    args = parser.parse_args()

    strategies = args.strategies.split()
    symbols = args.symbols.split()
    data_dir = Path(args.data_dir)
    interval_dir = data_dir / args.interval

    # --- Construct common arguments string ---
    common_args_list = [
        f"--opt-start {args.start_date}",
        f"--opt-end {args.end_date}",
        f"--commission {args.commission}",
        f"--balance {args.balance}",
        f"--metric {args.metric}",
    ]
    if args.save_details:
        common_args_list.append("--save-details")
    if args.details_file:
        common_args_list.append(f"--details-file {shlex.quote(args.details_file)}")

    common_args_str = " ".join(common_args_list)

    # --- Prepare list of jobs ---
    jobs_to_run: List[Tuple] = []
    job_index = 1

    # Calculate total jobs first for logging
    total_potential_jobs = len(strategies) * len(symbols)
    actual_jobs_count = 0
    tasks_to_submit = []

    logger.info(f"Preparing {total_potential_jobs} potential optimization jobs...")
    for strategy in strategies:
        for symbol in symbols:
            file_path = interval_dir / f"{symbol}_{args.interval}.csv"
            if file_path.is_file():
                actual_jobs_count += 1
                # We'll pass the total count later when submitting
                tasks_to_submit.append(
                    (strategy, symbol, str(file_path), common_args_str)
                )
            else:
                logger.warning(
                    f"--- Skipping Strategy: {strategy}, Symbol: {symbol} - File not found: {file_path} ---"
                )

    if not tasks_to_submit:
        logger.error("No valid optimization jobs found (check file paths). Exiting.")
        return

    logger.info(f"Found {actual_jobs_count} valid optimization jobs to run.")

    # Add job index and total count to each task
    jobs_with_indices = [
        task + (idx + 1, actual_jobs_count) for idx, task in enumerate(tasks_to_submit)
    ]

    # --- Run jobs in parallel ---
    logger.info(f"Starting parallel optimization using {args.processes} processes...")

    results = []
    try:
        with Pool(processes=args.processes) as pool:
            # Use imap_unordered for potentially better memory usage and responsiveness
            for result in pool.imap_unordered(
                run_single_optimization, jobs_with_indices
            ):
                results.append(result)
                # Log completion (can be out of order)
                job_desc, ret_code, _, _ = result
                status = "succeeded" if ret_code == 0 else f"failed (code: {ret_code})"
                logger.info(f"Completed: {job_desc} - Status: {status}")

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt received. Terminating running jobs...")
        # Pool context manager handles cleanup
        return  # Exit gracefully
    except Exception as e:
        logger.error(f"An error occurred during multiprocessing: {e}", exc_info=True)
        return  # Exit on unexpected error

    # --- Summarize Results ---
    logger.info("--- Batch Optimization Summary ---")
    successful_jobs = sum(1 for _, rc, _, _ in results if rc == 0)
    failed_jobs = len(results) - successful_jobs

    logger.info(f"Total Jobs Run: {len(results)}")
    logger.info(f"Successful Jobs: {successful_jobs}")
    logger.info(f"Failed Jobs: {failed_jobs}")

    if failed_jobs > 0:
        logger.warning("Failed jobs details:")
        for desc, rc, _, stderr in results:
            if rc != 0:
                logger.warning(
                    f"  - {desc}: Exit Code {rc} {'; Stderr: ' + stderr if stderr else ''}"
                )

    logger.info("Parallel batch optimization finished.")


if __name__ == "__main__":
    main()
