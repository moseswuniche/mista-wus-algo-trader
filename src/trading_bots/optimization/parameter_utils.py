import logging
import string
import re
import itertools
import sys
from typing import Dict, Any, List, Tuple, Optional, Set
from ..config_models import OptimizeParamsConfig  # Import main config model if needed

logger = logging.getLogger(__name__)  # Module-level logger


def params_to_tuple_rep(params: Dict[str, Any], precision: int = 8) -> tuple:
    """Creates a stable, hashable tuple representation of a parameter dictionary,
       normalizing numeric types, sorting keys, and handling complex types.
    Args:
        params: The parameter dictionary.
        precision: The number of decimal places to round floats to.
    Returns:
        A hashable tuple representation.
    """
    items: List[Tuple[str, Any]] = []  # Explicitly type items list
    # Use the module-level logger
    for k, v in sorted(params.items()):
        # Handle floats specifically for rounding
        if isinstance(v, float):
            try:
                rounded_v = round(v, precision)
                if rounded_v == -0.0:
                    rounded_v = 0.0
                items.append((k, rounded_v))
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not round float parameter '{k}' value {repr(v)}: {e}. Using original value."
                )
                items.append((k, v))
        # Keep ints as ints
        elif isinstance(v, int):
            items.append((k, v))
        elif isinstance(v, tuple):
            # Round float elements within tuples
            try:
                rounded_tuple_elements = []
                for elem in v:
                    if isinstance(elem, float):
                        rounded_elem = round(elem, precision)
                        if rounded_elem == -0.0:
                            rounded_elem = 0.0
                        rounded_tuple_elements.append(rounded_elem)
                    else:
                        rounded_tuple_elements.append(elem)  # Keep non-floats as is
                items.append((k, tuple(rounded_tuple_elements)))
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not round float elements in tuple parameter '{k}' value {repr(v)}: {e}. Using original value."
                )
                items.append((k, v))  # Use original tuple on error
        elif isinstance(v, list):
            # Convert lists to tuples for hashability, round floats within
            try:
                rounded_list_elements = []
                for elem in v:
                    if isinstance(elem, float):
                        rounded_elem = round(elem, precision)
                        if rounded_elem == -0.0:
                            rounded_elem = 0.0
                        rounded_list_elements.append(rounded_elem)
                    else:
                        rounded_list_elements.append(elem)
                # Correctly indented items.append block inside the try block
                items.append(
                    (
                        k,
                        tuple(
                            sorted(
                                rounded_list_elements,
                                key=lambda x: (
                                    (isinstance(x, type(None)), x)
                                    if isinstance(
                                        x, (int, float, str, bool, type(None))
                                    )
                                    else (2, type(x).__name__)
                                ),
                            )
                        ),
                    )
                )  # Sort list elements for stability
            except (ValueError, TypeError) as e:
                logger.warning(
                    f"Could not process list parameter '{k}' value {repr(v)}: {e}. Converting to tuple directly."
                )
                items.append((k, tuple(v)))  # Basic tuple conversion on error
        else:
            # For other hashable types (bool, str, None), append directly
            items.append((k, v))

    return tuple(items)


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """Removes or replaces characters invalid for filenames."""
    # Allow alphanumeric, underscore, hyphen, period
    valid_chars = set(string.ascii_letters + string.digits + "_-.")
    # Replace invalid characters with the replacement character
    sanitized = "".join(c if c in valid_chars else replacement for c in filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Remove consecutive replacements (e.g., ___)
    sanitized = re.sub(f"{re.escape(replacement)}+", replacement, sanitized)
    # Remove leading/trailing replacements
    sanitized = sanitized.strip(replacement)
    # Limit length (optional, but good practice)
    max_len = 200
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized


def _load_and_prepare_param_grid(
    full_config: Dict[
        str, Any
    ],  # This should ideally be the parsed OptimizeParamsConfig model
    symbol: str,
    strategy_class_name: str,
    cmd_apply_atr_filter: bool,  # Keep these flags for specific overrides if needed
    cmd_apply_seasonality_filter: bool,
) -> Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Loads grid for symbol/strategy, merges global filters, pre-filters based on flags, normalizes.
    Returns:
        Tuple[Dict[str, List[Any]], Dict[str, List[Any]]]:
            - Processed parameter grid ready for combination generation.
            - Original unprocessed grid for type reference (after global merge).
    Raises:
        ValueError: If symbol/strategy not found or grid format is invalid.
    """
    logger.debug(f"Preparing param grid for {symbol}/{strategy_class_name}")

    if symbol not in full_config:
        raise ValueError(f"Symbol '{symbol}' not found in validated config.")

    symbol_config = full_config[symbol]
    if strategy_class_name not in symbol_config:
        raise ValueError(
            f"Strategy '{strategy_class_name}' not found for symbol '{symbol}' in config."
        )

    loaded_grid_raw = symbol_config[strategy_class_name]
    if not isinstance(loaded_grid_raw, dict):
        raise ValueError(
            f"Expected dict for grid {symbol}/{strategy_class_name}, got {type(loaded_grid_raw)}"
        )

    # --- Merge Global Filters into Raw Grid --- #
    global_filters = full_config.get("filters", {})
    merged_grid_raw = loaded_grid_raw.copy()

    # 1. Seasonality / Time Window
    global_apply_seasonality = global_filters.get("apply_seasonality", False)
    global_seasonality_window = global_filters.get(
        "seasonality_window"
    )  # Can be list or string

    # If global seasonality is on AND a window is defined,
    # add/overwrite 'time_window' in the strategy grid *unless* it's already present.
    # The cmd_apply_seasonality_filter flag acts as a global kill switch AFTER this merge.
    if global_apply_seasonality and global_seasonality_window:
        # Convert list window to string if necessary
        if isinstance(global_seasonality_window, list):
            time_window_str = (
                global_seasonality_window[0] if global_seasonality_window else None
            )
        else:
            time_window_str = global_seasonality_window

        if time_window_str and "time_window" not in merged_grid_raw:
            logger.debug(
                f"Applying global time_window '{time_window_str}' to {symbol}/{strategy_class_name}"
            )
            merged_grid_raw["time_window"] = [
                time_window_str
            ]  # Wrap in list for grid format
        elif "time_window" in merged_grid_raw:
            logger.debug(
                f"Strategy {symbol}/{strategy_class_name} already has 'time_window', skipping global merge."
            )

    # 2. Liquidity Threshold
    global_liquidity_reqs = global_filters.get("liquidity_requirements", {})
    symbol_base = symbol.replace(
        "USDT", ""
    )  # Simple way to get base currency (e.g., BTC from BTCUSDT)
    global_liquidity_threshold = global_liquidity_reqs.get(symbol_base)

    # If global liquidity is defined for the symbol,
    # add/overwrite 'liquidity_threshold' *unless* already present.
    if (
        global_liquidity_threshold is not None
        and "liquidity_threshold" not in merged_grid_raw
    ):
        logger.debug(
            f"Applying global liquidity_threshold {global_liquidity_threshold} for {symbol_base} to {symbol}/{strategy_class_name}"
        )
        merged_grid_raw["liquidity_threshold"] = [global_liquidity_threshold]
    elif "liquidity_threshold" in merged_grid_raw:
        logger.debug(
            f"Strategy {symbol}/{strategy_class_name} already has 'liquidity_threshold', skipping global merge."
        )

    # --- Use the merged grid from now on --- #
    source_param_grid_for_types = merged_grid_raw.copy()  # Save the merged version
    loaded_grid = merged_grid_raw.copy()  # Work on a copy

    # --- Pre-filter Grid based on Command-Line Flags --- #
    # (Keep existing logic for cmd_apply_atr_filter and cmd_apply_seasonality_filter)
    # Note: cmd_apply_seasonality_filter will remove 'time_window' if it was added globally above.
    if not cmd_apply_atr_filter:
        atr_params_to_remove = [
            "apply_atr_filter",
            "atr_filter_period",
            "atr_filter_multiplier",
            "atr_filter_threshold",
            "atr_filter_sma_period",
        ]
        removed_atr_count = 0
        for param in atr_params_to_remove:
            if loaded_grid.pop(param, None) is not None:
                removed_atr_count += 1
        if removed_atr_count > 0:
            logger.info(
                f"CMD Flag --apply-atr-filter=OFF: Removed {removed_atr_count} ATR params."
            )

    if not cmd_apply_seasonality_filter:
        # This will now correctly remove 'time_window' regardless of source
        seasonality_params_to_remove = [
            "apply_seasonality_filter",
            "apply_seasonality",
            "allowed_trading_hours_utc",
            "apply_seasonality_to_symbols",
            "seasonality_start_hour",
            "seasonality_end_hour",
            "time_window",  # Added time_window here
        ]
        removed_season_count = 0
        for param in seasonality_params_to_remove:
            if loaded_grid.pop(param, None) is not None:
                removed_season_count += 1
        if removed_season_count > 0:
            logger.info(
                f"CMD Flag --apply-seasonality-filter=OFF: Removed {removed_season_count} seasonality params."
            )
    # --- End Pre-filter Grid ---

    # --- Process Remaining Parameter Grid (Normalize, Dedupe, Sort) --- #
    param_grid: Dict[str, List[Any]] = {}
    if not loaded_grid:
        logger.warning(
            f"Parameter grid for {symbol}/{strategy_class_name} is empty after command-line filtering."
        )
        # Return empty grid and the original source grid
        return {}, source_param_grid_for_types

    for key, value in loaded_grid.items():
        if not isinstance(value, list):
            logger.error(
                f"Unexpected non-list value for parameter '{key}' in {symbol}/{strategy_class_name} after validation. Value: {value}"
            )
            continue

        processed_values_dict = {}
        for item in value:
            processed_item = item
            key_for_dedupe: Any = item
            if isinstance(item, str) and item.lower() == "none":
                processed_item = None
                key_for_dedupe = None
            elif isinstance(item, (int, float)):
                key_for_dedupe = float(item)

            if key_for_dedupe not in processed_values_dict:
                processed_values_dict[key_for_dedupe] = processed_item

        unique_values_list = list(processed_values_dict.values())

        try:

            def sort_key(x):
                if x is None:
                    return (0, None)
                if isinstance(x, bool):
                    return (1, x)
                if isinstance(x, (int, float)):
                    return (2, x)
                if isinstance(x, str):
                    return (3, x)
                else:
                    return (4, str(x))

            sorted_unique_values = sorted(unique_values_list, key=sort_key)
            param_grid[key] = sorted_unique_values
            logger.debug(
                f"Processed param '{key}': Original={value}, UniqueSorted={sorted_unique_values}"
            )
        except TypeError as e:
            logger.warning(
                f"Sorting failed for parameter '{key}' values ({unique_values_list}): {e}. Using original order of unique items."
            )
            param_grid[key] = unique_values_list

    return (
        param_grid,
        source_param_grid_for_types,
    )


def _generate_combinations(
    param_grid: Dict[str, List[Any]],
    strategy_class_name: str,
    completed_param_reps: Optional[Set[tuple]] = None,  # Use Set[tuple] for specificity
) -> List[Dict[str, Any]]:
    """Generates unique parameter combinations, handling flags/dependencies and resuming.
    Args:
        param_grid: The processed parameter grid (output from _load_and_prepare_param_grid).
        strategy_class_name: Name of the strategy (for potential special handling).
        completed_param_reps: Set of tuple representations of previously completed params.
    Returns:
        List of unique parameter dictionaries to run backtests for.
    """
    # Use the module-level logger
    completed_param_reps = completed_param_reps or set()

    if not param_grid:
        logger.warning("Parameter grid is empty, no combinations to generate.")
        return []

    # --- Define Flags and Dependencies --- #
    flags_and_deps = {
        "apply_seasonality": [
            "seasonality_start_hour",
            "seasonality_end_hour",
        ],
    }

    # Separate grid into flags, dependents, and independents
    flag_grid = {k: param_grid.pop(k) for k in flags_and_deps if k in param_grid}
    dependent_params = {
        dep: param_grid.pop(dep)
        for flag, deps in flags_and_deps.items()
        for dep in deps
        if dep in param_grid
    }
    independent_params = param_grid  # Remaining params

    # --- Helper Function for Smart Combination Generation --- #
    def generate_smart_combinations(
        flag_grid, dependent_params, independent_params, flags_and_deps
    ):
        independent_names = list(independent_params.keys())
        independent_values = list(independent_params.values())
        independent_combinations = list(itertools.product(*independent_values))

        flag_names = list(flag_grid.keys())
        flag_values = list(flag_grid.values())
        flag_combinations = list(itertools.product(*flag_values))

        # total_combinations = 0 # Keep track outside if needed
        for flag_combo_values in flag_combinations:
            flag_combo_dict = dict(zip(flag_names, flag_combo_values))

            active_dependent_params_for_combo = {}
            fixed_dependent_params_for_combo = {}
            for flag_name, is_active in flag_combo_dict.items():
                deps = flags_and_deps.get(flag_name, [])
                for dep_name in deps:
                    if dep_name in dependent_params:
                        if is_active:
                            active_dependent_params_for_combo[dep_name] = (
                                dependent_params[dep_name]
                            )
                        else:
                            # Default to the first value (or None) if flag is inactive
                            if (
                                dep_name in dependent_params
                                and dependent_params[dep_name]
                            ):
                                fixed_dependent_params_for_combo[dep_name] = (
                                    dependent_params[dep_name][0]
                                )
                            else:
                                fixed_dependent_params_for_combo[dep_name] = None

            active_dep_names = list(active_dependent_params_for_combo.keys())
            active_dep_values = list(active_dependent_params_for_combo.values())

            dependent_combinations = (
                itertools.product(*active_dep_values)
                if active_dependent_params_for_combo
                else [()]
            )

            for dep_combo_values in dependent_combinations:
                active_dep_combo_dict = dict(zip(active_dep_names, dep_combo_values))

                for indep_combo_values in independent_combinations:
                    indep_combo_dict = dict(zip(independent_names, indep_combo_values))
                    final_params = {
                        **flag_combo_dict,
                        **fixed_dependent_params_for_combo,
                        **active_dep_combo_dict,
                        **indep_combo_dict,
                    }
                    # total_combinations += 1
                    yield final_params

    # --- End Inner Smart Combination Generator --- #

    # --- Generate Unique Parameter Combinations Using Smart Generator --- #
    unique_rep_to_params_map: Dict[tuple, Dict[str, Any]] = {}
    smart_combination_generator = generate_smart_combinations(
        flag_grid, dependent_params, independent_params, flags_and_deps
    )
    generated_count = 0

    # Use logger.info instead of print to stderr for consistency
    logger.info("Generating and deduplicating unique parameter combinations...")
    for i, params in enumerate(smart_combination_generator):
        generated_count += 1

        # Special handling for LongShortStrategy tuple params (if needed)
        # This might be better placed elsewhere or made more generic
        if strategy_class_name == "LongShortStrategy":
            rt_low = params.pop("return_thresh_low", None)
            rt_high = params.pop("return_thresh_high", None)
            vt_low = params.pop("volume_thresh_low", None)
            vt_high = params.pop("volume_thresh_high", None)
            if rt_low is not None and rt_high is not None:
                if "return_thresh" not in params:
                    params["return_thresh"] = (rt_low, rt_high)
            if vt_low is not None and vt_high is not None:
                if "volume_thresh" not in params:
                    params["volume_thresh"] = (vt_low, vt_high)

        rep = params_to_tuple_rep(params, precision=8)

        if i < 5:
            logger.debug(f"Generated Param Set [{i}]: Params={repr(params)}, Rep={rep}")
        log_interval = 10000
        if generated_count % log_interval == 0:
            logger.debug(f"Generated {generated_count} combinations so far...")

        if rep in unique_rep_to_params_map:
            continue
        unique_rep_to_params_map[rep] = params

    total_unique_combinations = len(unique_rep_to_params_map)
    logger.info(f"Smart generator yielded {generated_count} combinations.")
    logger.info(
        f"Deduplication complete: {total_unique_combinations} unique combinations identified initially."
    )
    logger.info(
        f"Total unique combinations calculated from CURRENT config: {total_unique_combinations}"
    )
    duplicate_count = generated_count - total_unique_combinations
    if duplicate_count > 0:
        logger.info(f"  ({duplicate_count} duplicate combinations ignored)")

    # --- Filter based on completed_param_reps (Resume) --- #
    unique_params_list = list(unique_rep_to_params_map.values())
    if not completed_param_reps:
        logger.info(f"Running all {total_unique_combinations} unique combinations.")
        # DEBUG logging - use logger.debug
        for i, p in enumerate(unique_params_list[:5]):
            logger.debug(
                f"DEBUG GENERATE[{i}]: Generated rep = {repr(params_to_tuple_rep(p))}"
            )
        return unique_params_list
    else:
        params_to_run = []
        skipped_count = 0
        logger.debug(
            f"--- Resuming: Comparing {total_unique_combinations} generated params against {len(completed_param_reps)} loaded reps ---"
        )
        for i, params in enumerate(unique_params_list):
            rep = params_to_tuple_rep(params)
            if i < 5:
                logger.debug(f"DEBUG GENERATE[{i}]: Generated rep = {repr(rep)}")
            if rep in completed_param_reps:
                skipped_count += 1
            else:
                params_to_run.append(params)
        logger.debug(f"--- Resume Check Finished: Total Skipped={skipped_count} ---")

        remaining_count = len(params_to_run)
        logger.info(f"Resume: Skipped {skipped_count} already completed parameters.")
        logger.info(f"Resume: {remaining_count} parameters remaining to run.")
        return params_to_run
