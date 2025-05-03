"""Pydantic models for configuration validation."""

from pydantic import BaseModel, Field, ValidationError, field_validator, RootModel
from typing import Dict, List, Optional, Any, Union
import re

# --- Runtime Config Model ---


class RuntimeConfig(BaseModel):
    """Pydantic model for runtime_config.yaml."""

    symbol: str
    strategy_name: str
    strategy_params: Dict[
        str, Any
    ]  # Strategy params can be diverse, basic validation here

    # Optional Trader-level overrides
    units: Optional[float] = Field(default=None, gt=0)
    stop_loss_pct: Optional[float] = Field(default=None, ge=0, le=1)
    take_profit_pct: Optional[float] = Field(default=None, ge=0)
    trailing_stop_loss_pct: Optional[float] = Field(default=None, ge=0, le=1)
    max_cumulative_loss: Optional[float] = None  # Can be negative, store absolute later
    apply_atr_filter: Optional[bool] = None
    apply_seasonality_filter: Optional[bool] = None
    allowed_trading_hours_utc: Optional[str] = None  # Add validator?
    apply_seasonality_to_symbols: Optional[str] = None

    # TODO: Add validator for allowed_trading_hours_utc format if needed


# --- Optimization Params Models ---

# Define a type for the possible values within a parameter grid list
ParamValue = Union[str, int, float, bool, None]


# Model for a single strategy's parameter grid
# Use RootModel for Pydantic v2 compatibility when the model is just a dict
class StrategyOptimizeParams(RootModel[Dict[str, List[ParamValue]]]):
    """Represents the parameter grid for a single strategy using RootModel."""

    # The actual data is accessed via self.root

    @field_validator("root", mode="before")  # Validate the root dictionary
    @classmethod
    def check_param_lists(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Strategy parameters must be a dictionary")
        for param_name, v in data.items():
            if not isinstance(v, list):
                raise ValueError(
                    f"Parameter grid value for '{param_name}' must be a list"
                )
            # Optional: Add checks for list contents if needed
            # e.g., ensure values are of compatible types
        return data


# Model for the entire optimize_params.yaml structure
# Uses a dictionary where keys are symbol strings (like "BTCUSDT")
# and values are dictionaries where keys are strategy class names
# and values are the parameter grids (validated by StrategyOptimizeParams type).
# Using RootModel for top-level dictionary validation
class OptimizeParamsConfig(
    RootModel[Dict[str, Dict[str, Dict[str, List[ParamValue]]]]]
):
    """Pydantic model for the overall optimize_params.yaml structure using RootModel."""

    # The actual data is accessed via self.root

    @field_validator("root", mode="before")  # Validate the root dictionary structure
    @classmethod
    def check_structure(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Optimize params config must be a dictionary of symbols")

        for symbol, strategy_dict in data.items():
            if not isinstance(strategy_dict, dict):
                raise ValueError(
                    f"Entry for symbol '{symbol}' must be a dictionary of strategies"
                )

            for strategy_name, params_dict in strategy_dict.items():
                if not isinstance(params_dict, dict):
                    raise ValueError(
                        f"Strategy entry '{strategy_name}' for symbol '{symbol}' must be a dictionary"
                    )
                # Validate the inner parameter grid structure
                for param_key, param_list in params_dict.items():
                    if not isinstance(param_list, list):
                        raise ValueError(
                            f"Parameter grid for '{param_key}' in strategy '{strategy_name}' (symbol '{symbol}') must be a list"
                        )
        return data


# --- Backtest Run Configuration Model ---


class BacktestRunConfig(BaseModel):
    """Configuration for a single backtest run."""

    # --- Core Settings ---
    symbol: str
    initial_balance: float = Field(gt=0)
    commission_bps: float = Field(ge=0)  # Commission in basis points
    units: float = Field(gt=0)  # Fixed units for backtest trade sizing

    # --- Strategy ---
    # Store strategy name and params separately for clarity
    strategy_short_name: str  # Short name like "MACross", "BBReversion"
    strategy_params: Dict[
        str, Any
    ]  # The specific parameters for this strategy instance

    # --- Risk Management (Optional, applied within backtest if provided) ---
    stop_loss_pct: Optional[float] = Field(default=None, ge=0, le=1)
    take_profit_pct: Optional[float] = Field(default=None, ge=0)
    trailing_stop_loss_pct: Optional[float] = Field(default=None, ge=0, le=1)

    # --- Filters (Optional, applied within backtest if enabled) ---
    # Global flags indicating if a filter *type* should be applied
    apply_atr_filter: bool = False
    apply_seasonality_filter: bool = False

    # ATR Filter parameters (used only if apply_atr_filter is True)
    atr_filter_period: int = Field(default=14, gt=0)
    atr_filter_multiplier: float = Field(default=1.5, gt=0)
    atr_filter_sma_period: int = Field(default=100, ge=0)  # 0 means don't use SMA

    # Seasonality Filter parameters (used only if apply_seasonality_filter is True)
    allowed_trading_hours_utc: Optional[str] = None  # e.g., "5-17"
    apply_seasonality_to_symbols: Optional[str] = (
        None  # Comma-separated symbols, empty means apply to main symbol
    )

    @field_validator("allowed_trading_hours_utc")
    def validate_trading_hours(cls, v, values):
        data = values.data  # Access other field values if needed
        if data.get("apply_seasonality_filter") and v is None:
            raise ValueError(
                "allowed_trading_hours_utc must be set if apply_seasonality_filter is True"
            )
        if v is not None:
            # Basic format check (doesn't check hour validity extensively)
            if not re.match(r"^\d{1,2}-\d{1,2}$", v):
                raise ValueError("allowed_trading_hours_utc must be in 'HH-HH' format")
        return v

    # --- Add Strategy Class Mapping for Validation/Instantiation ---
    # Could potentially add validation here to ensure strategy_short_name is known
    # and strategy_params match expected types, but that might be complex.
    # Keeping it simpler for now.


# Example Usage (for testing models):
if __name__ == "__main__":

    # Example runtime_config.yaml data
    runtime_data = {
        "symbol": "BTCUSDT",
        "strategy_name": "MACross",
        "strategy_params": {"fast_period": 10, "slow_period": 50},
        "stop_loss_pct": 0.05,
        "apply_atr_filter": True,
    }
    try:
        validated_runtime = RuntimeConfig.model_validate(runtime_data)
        print("Runtime Config Valid:")
        print(validated_runtime.model_dump_json(indent=2))
    except ValidationError as e:
        print("Runtime Config Invalid:")
        print(e)

    print("\\n" + "=" * 20 + "\\n")

    # Example optimize_params.yaml data
    optimize_data = {
        "BTCUSDT": {
            "MovingAverageCrossoverStrategy": {
                "fast_period": [5, 9, 13],
                "slow_period": [21, 34, 50],
                "stop_loss_pct": [None, 0.02, 0.03],
                "ma_type": ["SMA", "EMA"],
            },
            "RsiMeanReversionStrategy": {
                "rsi_period": [14, 21],
                "oversold_threshold": [20, 30],
                "overbought_threshold": [70, 80],
            },
        },
        "ETHUSDT": {
            "MovingAverageCrossoverStrategy": {
                "fast_period": [8, 10],
                "slow_period": [40, 50],
                "stop_loss_pct": [0.025],  # Still a list
            }
        },
        # Example of invalid structure:
        # "XRPUSDT": "invalid_entry"
        # "LTCUSDT": {
        #      "MACross": { "fast_period": 10 } # Not a list
        # }
    }
    try:
        # Note: Pydantic v2 uses model_validate
        # Using the specific validator logic within OptimizeParamsConfig now
        validated_optimize = OptimizeParamsConfig.model_validate(optimize_data)
        print("Optimize Params Config Valid:")
        # print(validated_optimize.model_dump_json(indent=2)) # Dump the RootModel content
    except ValidationError as e:
        print("Optimize Params Config Invalid:")
        print(e)

    # Test invalid data
    invalid_optimize_data = {"LTCUSDT": {"MACross": {"fast_period": 10}}}  # Not a list
    try:
        # When using RootModel, validate the data directly, not wrapped in {'__root__': ...}
        OptimizeParamsConfig.model_validate(invalid_optimize_data)
    except ValidationError as e:
        print("\\nOptimize Params Config Invalid (Test):")
        print(e)
