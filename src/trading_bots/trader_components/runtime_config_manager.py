"""Manages loading and checking runtime configuration changes."""

import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import time

# Import the Pydantic model for validation
from ..config_models import RuntimeConfig, ValidationError

logger = logging.getLogger(__name__)

# Define constants
DEFAULT_RUNTIME_CONFIG = "config/runtime_config.yaml"
CONFIG_CHECK_INTERVAL_BARS = (
    5  # Check config file every N closed bars (or equivalent trigger)
)


class RuntimeConfigManager:
    """Loads, validates, and checks for updates in the runtime config file."""

    def __init__(
        self,
        config_path: str = DEFAULT_RUNTIME_CONFIG,
        symbol: str = "",  # Store symbol to validate config against
        update_callback: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,  # Callback to apply updates
    ):
        self.runtime_config_path = Path(config_path)
        self.symbol = symbol  # Used to ignore config if symbol mismatch
        self.update_callback = update_callback
        self.last_config_mtime: Optional[float] = None
        self.config_check_counter: int = 0

        self._update_config_mtime()  # Get initial mtime
        logger.info(
            f"RuntimeConfigManager initialized. Watching: {self.runtime_config_path}"
        )

    def _update_config_mtime(self):
        """Reads and stores the modification time of the config file."""
        try:
            if self.runtime_config_path.is_file():
                self.last_config_mtime = self.runtime_config_path.stat().st_mtime
            else:
                self.last_config_mtime = None
        except OSError as e:
            logger.error(
                f"Error accessing runtime config stats ({self.runtime_config_path}): {e}"
            )
            self.last_config_mtime = None

    def load_and_validate_config(self) -> Optional[Dict[str, Any]]:
        """Loads and validates the runtime configuration from the YAML file."""
        if not self.runtime_config_path.is_file():
            # logger.debug(f"Runtime config file not found: {self.runtime_config_path}")
            return None
        try:
            with open(self.runtime_config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if not isinstance(config_data, dict):
                    logger.error(
                        f"Invalid runtime config format in {self.runtime_config_path}. Expected dict."
                    )
                    return None

                validated_config = RuntimeConfig.model_validate(config_data)
                logger.debug(f"Runtime config {self.runtime_config_path} validated.")

                # Check if symbol matches if self.symbol is set
                if self.symbol and validated_config.symbol != self.symbol:
                    logger.warning(
                        f"Symbol in runtime config ({validated_config.symbol}) != trader symbol ({self.symbol}). Ignoring."
                    )
                    return None

                return validated_config.model_dump()  # Return validated data as dict

        except ValidationError as e:
            logger.error(
                f"Runtime config validation failed: {self.runtime_config_path}\n{e}"
            )
            return None
        except yaml.YAMLError as e:
            logger.error(
                f"Error parsing runtime config YAML {self.runtime_config_path}: {e}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error reading/processing runtime config {self.runtime_config_path}: {e}",
                exc_info=True,
            )
            return None

    def check_for_updates(self) -> None:
        """Checks the runtime config file for modifications and applies updates via callback."""
        self.config_check_counter += 1
        if self.config_check_counter < CONFIG_CHECK_INTERVAL_BARS:
            return
        self.config_check_counter = 0

        if not self.runtime_config_path.is_file():
            return

        try:
            current_mtime = self.runtime_config_path.stat().st_mtime
            if self.last_config_mtime is None or current_mtime > self.last_config_mtime:
                logger.info(
                    f"Runtime configuration change detected: {self.runtime_config_path}"
                )
                config = self.load_and_validate_config()
                if config:
                    if self.update_callback:
                        logger.info("Applying runtime configuration updates...")
                        self.update_callback(config)  # Call the provided callback
                        self._update_config_mtime()  # Update mtime after successful apply
                    else:
                        logger.warning(
                            "Runtime config changed, but no update_callback set."
                        )
                        self._update_config_mtime()  # Update mtime anyway
                else:
                    # Config invalid or symbol mismatch
                    logger.warning(
                        "Loaded runtime config was invalid or mismatched. Not applying."
                    )
                    self._update_config_mtime()  # Update mtime so we don't retry bad config

        except OSError as e:
            logger.error(
                f"Error checking runtime config file stats ({self.runtime_config_path}): {e}"
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during runtime config check: {e}", exc_info=True
            )
