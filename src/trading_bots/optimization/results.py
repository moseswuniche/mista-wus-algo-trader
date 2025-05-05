import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Assuming sanitize_filename is available from parameter_utils
from .parameter_utils import sanitize_filename

logger = logging.getLogger(__name__)


def _find_overall_best_from_csv(
    details_filepath: Path,
    optimization_metric: str,
    source_param_grid: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Reads the full optimization details CSV and finds the overall best parameters."""
    logger.info(
        f"Analyzing full results file to determine overall best params: {details_filepath}"
    )
    minimize_metric = optimization_metric in ["max_drawdown"]

    try:
        if not details_filepath.is_file() or details_filepath.stat().st_size == 0:
            logger.warning(
                f"Details file {details_filepath} not found or is empty. Cannot determine overall best."
            )
            return None, None

        df_full_results = pd.read_csv(details_filepath, low_memory=False)

        if df_full_results.empty:
            logger.warning(
                f"Details file {details_filepath} is empty. Cannot determine overall best."
            )
            return None, None

        metric_col = f"result_{optimization_metric}"
        if metric_col not in df_full_results.columns:
            logger.warning(
                f"Metric column '{metric_col}' not found in {details_filepath}. Cannot determine overall best."
            )
            return None, None

        df_valid_metric = df_full_results.dropna(subset=[metric_col])
        if df_valid_metric.empty:
            logger.warning(
                f"No valid (non-NaN) values found for metric '{metric_col}' in {details_filepath}. Cannot determine overall best."
            )
            return None, None

        if minimize_metric:
            best_idx = df_valid_metric[metric_col].idxmin()
        else:
            best_idx = df_valid_metric[metric_col].idxmax()
        best_row = df_valid_metric.loc[best_idx]

        param_cols = [c for c in df_full_results.columns if c.startswith("param_")]
        csv_best_params_raw = {
            col.replace("param_", "", 1): best_row[col] for col in param_cols
        }

        type_map_from_source: Dict[str, Optional[type]] = {}
        if source_param_grid:
            for key, value_list in source_param_grid.items():
                if value_list:
                    type_map_from_source[key] = type(value_list[0])
                else:
                    type_map_from_source[key] = None
        else:
            logger.warning(
                "Source parameter grid not provided for CSV analysis. Type conversion might be inaccurate."
            )

        overall_best_params_typed: Dict[str, Any] = {}
        for k, v in csv_best_params_raw.items():
            if pd.isna(v):
                overall_best_params_typed[k] = None
                continue
            expected_type = type_map_from_source.get(k)
            csv_value_str = str(v).strip()
            if csv_value_str.lower() == "none":
                overall_best_params_typed[k] = None
                continue

            converted_value = None
            conversion_successful = False

            # --- Attempt 1: Convert using expected_type if available --- #
            if expected_type:
                try:
                    if expected_type == bool:
                        lower_val = csv_value_str.lower()
                        if lower_val in ["true", "1", "t", "y", "yes"]:
                            converted_value = True
                            conversion_successful = True
                        elif lower_val in ["false", "0", "f", "n", "no"]:
                            converted_value = False
                            conversion_successful = True
                        else:
                             logger.warning(f"Param '{k}' expected bool, got ambiguous value '{v}'. Treating as None.")
                             converted_value = None # Assign None for ambiguous bool
                             # Do not mark successful, let fallback handle if NOT expected bool
                             # But if expected bool, this None is the final value.
                             if expected_type == bool:
                                conversion_successful = True # Final value is None for expected bool
                    elif expected_type == int:
                        converted_value = int(float(csv_value_str)) # type: ignore[assignment]
                        conversion_successful = True
                    elif expected_type == float:
                        converted_value = float(csv_value_str) # type: ignore[assignment]
                        conversion_successful = True
                    else:
                        # Try direct conversion for other types (e.g., str)
                        converted_value = expected_type(v)
                        conversion_successful = True
                except (ValueError, TypeError, Exception) as e:
                     logger.warning(f"Conversion using expected type {expected_type} failed for '{k}'='{v}': {e}. Trying fallback inference.")

            # --- Attempt 2: Fallback Inference (if expected_type missing or conversion failed) --- #
            if not conversion_successful:
                # --- IMPORTANT: Do NOT perform fallback if expected_type was bool --- #
                if expected_type == bool:
                    # We already assigned True, False, or None in Attempt 1
                    # If conversion_successful is False here, it means value was ambiguous
                    # and converted_value is already None. Do nothing further.
                    pass
                else:
                    # Perform fallback inference only if bool was NOT expected
                    lower_val = csv_value_str.lower()
                    if lower_val in ["true", "1", "t", "y", "yes"]:
                        converted_value = True
                    elif lower_val in ["false", "0", "f", "n", "no"]:
                        converted_value = False
                    else:
                        # Try numeric/string if bool inference failed
                        try:
                            # The ignores below address cases where fallback assigns int/float to Optional[bool]
                            converted_value = int(float(v)) # type: ignore[assignment]
                        except (ValueError, TypeError):
                            try:
                                converted_value = float(v) # type: ignore[assignment]
                            except (ValueError, TypeError):
                                converted_value = csv_value_str # type: ignore[assignment]
                                logger.debug(f"Fallback inference: Could not infer bool/numeric for '{k}'='{v}', keeping as string.")

            # --- Final Assignment & Type Check --- #
            final_value = converted_value
            # If bool was expected, ensure final value is bool or None
            if expected_type == bool and not isinstance(final_value, bool) and final_value is not None:
                 logger.warning(f"Param '{k}' expected bool but final value is {type(final_value)} ('{final_value}'). Assigning None.")
                 final_value = None

            overall_best_params_typed[k] = final_value

            # Log if the final type STILL doesn't match expected_type (and expected_type existed)
            # This is mostly for non-bool types where fallback might result in string
            if expected_type and expected_type != bool and not isinstance(final_value, expected_type) and final_value is not None:
                 logger.warning(f"Param '{k}' final type {type(final_value)} differs from expected {expected_type}. Value: '{final_value}'. Check source data/grid.")

        logger.info(f"Overall best parameters found: {overall_best_params_typed}")
        return overall_best_params_typed, {optimization_metric: best_row[metric_col]}

    except pd.errors.EmptyDataError:
        logger.warning(f"Details file {details_filepath} is empty during analysis.")
        return None, None
    except FileNotFoundError:
        logger.error(f"Details file {details_filepath} not found for final analysis.")
        return None, None
    except Exception as e:
        logger.error(
            f"Error reading or analyzing full results file {details_filepath}: {e}.",
            exc_info=True,
        )
        return None, None


def adjust_params_for_printing(
    params: Dict[str, Any], strategy_class_name: str
) -> Dict[str, Any]:
    """Converts specific strategy param representations for printing/saving."""
    printable_params = params.copy()
    if strategy_class_name == "LongShortStrategy":
        if (
            "return_thresh" in printable_params
            and isinstance(printable_params["return_thresh"], tuple)
            and len(printable_params["return_thresh"]) == 2
        ):
            rt = printable_params.pop("return_thresh")
            printable_params["return_thresh_low"] = rt[0]
            printable_params["return_thresh_high"] = rt[1]
        if (
            "volume_thresh" in printable_params
            and isinstance(printable_params["volume_thresh"], tuple)
            and len(printable_params["volume_thresh"]) == 2
        ):
            vt = printable_params.pop("volume_thresh")
            printable_params["volume_thresh_low"] = vt[0]
            printable_params["volume_thresh_high"] = vt[1]
    return printable_params


def save_best_params(
    best_params: Dict[str, Any],
    strategy_class_name: str,
    details_filepath: Optional[Path],  # Keep for context, though not directly used
    optimization_metric: str,
    best_metrics: Dict[str, Any],
    output_filepath: Path,
    cmd_apply_atr_filter: bool,  # Added flag
    cmd_apply_seasonality_filter: bool,  # Added flag
):
    """Saves the best parameters and metrics to a YAML file."""
    # Note: The original implementation was within optimize_strategy.
    # This function replicates that saving logic.
    try:
        # Ensure the output directory exists
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Prepare metrics for saving (handle NaN, numpy types)
        serializable_metrics: Dict[str, Any] = {}
        for k, v in best_metrics.items():
            if pd.isna(v):
                serializable_metrics[k] = None
            elif isinstance(v, (np.integer, int)):
                serializable_metrics[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                serializable_metrics[k] = round(float(v), 6)
            elif isinstance(v, (bool, str)):
                serializable_metrics[k] = v
            else:
                serializable_metrics[k] = str(v)

        # Adjust parameters for saving/printing
        printable_best_params = adjust_params_for_printing(
            best_params, strategy_class_name
        )

        # Ensure filter enablement flags reflect the optimization context
        if "apply_atr_filter" not in printable_best_params:
            printable_best_params["apply_atr_filter"] = cmd_apply_atr_filter
            logger.debug(
                "Adding cmd_apply_atr_filter=%s to saved params", cmd_apply_atr_filter
            )
        if (
            "apply_seasonality_filter" not in printable_best_params
            and "apply_seasonality" not in printable_best_params
        ):
            printable_best_params["apply_seasonality_filter"] = (
                cmd_apply_seasonality_filter
            )
            logger.debug(
                "Adding cmd_apply_seasonality_filter=%s to saved params",
                cmd_apply_seasonality_filter,
            )

        data_to_save = {
            "parameters": printable_best_params,
            "metrics": serializable_metrics,
            "optimized_metric": optimization_metric,
        }

        with open(output_filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                data_to_save,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
        logger.info(f"Saved best parameters to: {output_filepath}")

    except Exception as e:
        logger.error(
            f"Failed to save best parameters file to {output_filepath}: {e}",
            exc_info=True,
        )


def get_details_filepath(
    strategy_short_name: str, symbol: str, start_date_str: str, end_date_str: str
) -> Path:
    """Generates a default filepath for the optimization details CSV."""
    safe_strategy = sanitize_filename(strategy_short_name)
    safe_symbol = sanitize_filename(symbol)
    safe_start = sanitize_filename(start_date_str)
    safe_end = sanitize_filename(end_date_str)
    details_dir = Path("results") / "optimize" / "details"
    filename = f"{safe_strategy}_{safe_symbol}_{safe_start}_{safe_end}_details.csv"
    return details_dir / filename


# Helper function to format values nicely
def _format_value(cell: Any) -> Any:
    """Formats a value for display, handling None, bool, floats, etc."""
    if cell is None:
        return cell
    elif isinstance(cell, (bool, str)):
        return cell
    elif isinstance(cell, (np.floating, float)):
        return round(cell, 6)
    elif isinstance(cell, (np.integer, int)):
        return int(cell)
    else:
        # Convert pandas Timestamp if present
        try:
            return cell.strftime("%Y-%m-%d %H:%M:%S")
        except AttributeError:
            # Fallback for other types
            return str(cell)


def _highlight_best(s, best_value):
    """
    Highlights the best value in a Series.
    """
    is_max = s == best_value
    return ["background-color: yellow" if v else "" for v in is_max]
