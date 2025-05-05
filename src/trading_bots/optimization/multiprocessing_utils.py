import pandas as pd
import logging
import sys
import multiprocessing
from logging.handlers import QueueHandler
from typing import Dict, Any, Optional

# --- Worker Globals ---
# These are intended to be populated by the initializer in each worker process.
worker_log_queue: Optional[multiprocessing.Queue] = None
worker_data: Optional[pd.DataFrame] = None
worker_shared_args: Dict[str, Any] = {}
WORKER_STRATEGY_MAP: Dict[str, type] = {}  # Map populated in initializer
# --- End Worker Globals ---


def worker_log_configurer(log_queue: multiprocessing.Queue):
    """Configures logging for a worker process to send ONLY to the queue."""
    root = logging.getLogger()
    # Ensure root logger exists
    if not root:
        print(
            f"Worker {multiprocessing.current_process().pid}: Root logger not found!",
            file=sys.stderr,
        )
        return

    # *** IMPORTANT: Remove existing handlers from worker's root logger ***
    if root.hasHandlers():
        print(
            f"Worker {multiprocessing.current_process().pid}: Removing {len(root.handlers)} existing handler(s) from root logger.",
            file=sys.stderr,
        )
        root.handlers.clear()

    # Add ONLY the QueueHandler
    queue_handler = QueueHandler(log_queue)
    root.addHandler(queue_handler)
    # Set the level for this specific handler - messages BELOW this level won't be put on the queue
    # The overall root level will be set by the main process setup.
    root.setLevel(logging.DEBUG)  # Send DEBUG and higher from workers to the queue


def pool_worker_initializer_with_data(
    log_q: multiprocessing.Queue, shared_worker_args_for_init: Dict
):
    """Initializer for worker processes.
    Sets up logging and stores shared data in global variables.
    """
    global worker_log_queue, worker_data, worker_shared_args, WORKER_STRATEGY_MAP

    # Print directly to stderr during initialization as logging might not be fully set up
    print(
        f"Initializing worker {multiprocessing.current_process().pid} with queue {id(log_q)}...",
        file=sys.stderr,
    )

    # 1. Configure Logging
    worker_log_queue = log_q
    worker_log_configurer(worker_log_queue)

    # Use a distinct logger name for worker init messages if needed
    worker_logger = logging.getLogger(
        f"WorkerInit.{multiprocessing.current_process().pid}"
    )

    # 2. Store Shared Data/Args
    # Make a copy to ensure each worker has its own reference
    worker_shared_args = shared_worker_args_for_init.copy()
    worker_data = worker_shared_args.pop(
        "data", None
    )  # Extract data to separate global

    if not isinstance(worker_data, pd.DataFrame) or worker_data.empty:
        print(
            f"CRITICAL ERROR in worker {multiprocessing.current_process().pid}: Did not receive valid data in initializer! Type: {type(worker_data)}",
            file=sys.stderr,
        )
        worker_logger.error(
            "Worker did not receive data DataFrame during initialization."
        )
        # Attempting to log to queue if possible
        if worker_log_queue:
            try:
                # Create record manually if logger might not be fully functional yet
                err_record = logging.LogRecord(
                    name=worker_logger.name,
                    level=logging.CRITICAL,
                    pathname="",
                    lineno=0,
                    msg="Worker data is None or empty",
                    args=(),
                    exc_info=None,
                    func="pool_worker_initializer_with_data",
                )
                worker_log_queue.put(err_record)
            except Exception as e:
                print(
                    f"WorkerInitError: Failed to put critical log record in queue: {e}",
                    file=sys.stderr,
                )
        # Consider exiting or raising if data is absolutely essential
        # sys.exit(1) # Or let the pool handle the worker failure

    # 3. Populate Worker Strategy Map (avoids repeated imports in worker function)
    # Note: This relies on the calling context ensuring strategies are importable
    # from the worker's perspective.
    try:
        # Dynamically import strategies based on the current package structure
        # Adjust the relative import path as necessary if this file moves
        from trading_bots.strategies import (
            MovingAverageCrossoverStrategy,
            BollingerBandReversionStrategy,
            ScalpingStrategy,
            MomentumStrategy,
            MeanReversionStrategy,
            BreakoutStrategy,
            HybridStrategy,
        )

        WORKER_STRATEGY_MAP.update(
            {
                "MovingAverageCrossoverStrategy": MovingAverageCrossoverStrategy,
                "BollingerBandReversionStrategy": BollingerBandReversionStrategy,
                "ScalpingStrategy": ScalpingStrategy,
                "MomentumStrategy": MomentumStrategy,
                "MeanReversionStrategy": MeanReversionStrategy,
                "BreakoutStrategy": BreakoutStrategy,
                "HybridStrategy": HybridStrategy,
            }
        )
        worker_logger.info(
            f"Successfully imported and mapped strategies: {list(WORKER_STRATEGY_MAP.keys())}"
        )
    except ImportError as e:
        print(
            f"CRITICAL ERROR in worker {multiprocessing.current_process().pid}: Could not import strategies: {e}",
            file=sys.stderr,
        )
        worker_logger.critical(f"Worker could not import strategies: {e}")
        if worker_log_queue:
            try:
                err_record = logging.LogRecord(
                    name=worker_logger.name,
                    level=logging.CRITICAL,
                    pathname="",
                    lineno=0,
                    msg=f"Strategy import failed: {e}",
                    args=(),
                    exc_info=None,
                    func="pool_worker_initializer_with_data",
                )
                worker_log_queue.put(err_record)
            except Exception as log_e:
                print(
                    f"WorkerInitError: Failed to put log record in queue: {log_e}",
                    file=sys.stderr,
                )
        # sys.exit(1) # Consider exiting

    print(
        f"Worker {multiprocessing.current_process().pid} initialized. Data shape: {worker_data.shape if worker_data is not None else 'None'}",
        file=sys.stderr,
    )
    worker_logger.info(
        f"Initialization complete. Data shape: {worker_data.shape if worker_data is not None else 'None'}"
    )
