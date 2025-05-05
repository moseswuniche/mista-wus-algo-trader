"""Handles loading and saving the bot's persistent state."""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import components whose state needs managing
from .position_manager import PositionManager
from .risk_manager import RiskManager

# Import other stateful components as needed (e.g., RuntimeConfigManager?)

logger = logging.getLogger(__name__)

# Define state file directory constant
STATE_DIR = Path("results/state")


class StateManager:
    """Handles saving and loading the combined state of various trader components."""

    def __init__(
        self,
        symbol: str,
        components: List[Any],  # List of components with get_state/load_state methods
        state_file_name: Optional[str] = None,
    ):
        self.symbol = symbol
        # Store references to components that need state management
        # Filter components to ensure they have the required methods
        self.managed_components = {
            comp.__class__.__name__: comp
            for comp in components
            if hasattr(comp, "get_state")
            and callable(getattr(comp, "get_state"))
            and hasattr(comp, "load_state")
            and callable(getattr(comp, "load_state"))
        }

        if not self.managed_components:
            logger.warning(
                "StateManager initialized, but no manageable components provided."
            )

        # Determine state file path
        file_name = state_file_name or f"trader_state_{self.symbol}.json"
        self.state_file_path = STATE_DIR / file_name
        logger.info(
            f"StateManager initialized. Managing state for: {list(self.managed_components.keys())}. State file: {self.state_file_path}"
        )

    def save_state(self) -> None:
        """Gathers state from all managed components and saves it to the JSON file."""
        combined_state: Dict[str, Any] = {}
        logger.debug(f"Saving state for {len(self.managed_components)} components...")
        for comp_name, component in self.managed_components.items():
            try:
                component_state = component.get_state()
                if isinstance(component_state, dict):
                    combined_state[comp_name] = component_state
                    logger.debug(f"  - Got state from {comp_name}")
                else:
                    logger.warning(
                        f"Component {comp_name} get_state() did not return a dict. Skipping."
                    )
            except Exception as e:
                logger.error(
                    f"Error getting state from component {comp_name}: {e}",
                    exc_info=True,
                )

        if not combined_state:
            logger.warning(
                "No state gathered from components. State file will not be written."
            )
            return

        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.state_file_path, "w") as f:
                json.dump(
                    combined_state, f, indent=4, default=str
                )  # Use default=str for non-serializable types like Timestamps
            logger.info(f"Combined trader state saved to {self.state_file_path}")
        except TypeError as te:
            logger.error(
                f"TypeError saving state to {self.state_file_path}. Check for non-serializable data: {te}",
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                f"Failed to save combined trader state to {self.state_file_path}: {e}",
                exc_info=True,
            )

    def load_state(self) -> bool:
        """Loads combined state from the JSON file and distributes it to managed components."""
        if not self.state_file_path.is_file():
            logger.info(
                f"No existing state file found ({self.state_file_path}). Components will use initial state."
            )
            return False  # Indicate state was not loaded

        try:
            logger.info(f"Loading combined state from {self.state_file_path}...")
            with open(self.state_file_path, "r") as f:
                combined_state = json.load(f)

            if not isinstance(combined_state, dict):
                logger.error(
                    f"Invalid state format in {self.state_file_path}. Expected dictionary. Loading aborted."
                )
                return False

            loaded_something = False
            for comp_name, component in self.managed_components.items():
                if comp_name in combined_state:
                    component_state = combined_state[comp_name]
                    if isinstance(component_state, dict):
                        try:
                            component.load_state(component_state)
                            logger.debug(f"  - Loaded state into {comp_name}")
                            loaded_something = True
                        except Exception as e:
                            logger.error(
                                f"Error loading state into component {comp_name}: {e}",
                                exc_info=True,
                            )
                    else:
                        logger.warning(
                            f"State data for {comp_name} in file is not a dict. Skipping load for this component."
                        )
                else:
                    logger.warning(
                        f"No state found in file for component: {comp_name}. It will use initial state."
                    )

            logger.info("State loading process complete.")
            return (
                loaded_something  # Return True if at least one component loaded state
            )

        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding state file {self.state_file_path}: {e}. Components will use initial state."
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to load state from {self.state_file_path}: {e}. Components will use initial state.",
                exc_info=True,
            )
            return False

    # Logic will be moved here
    pass
