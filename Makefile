# Makefile for Trading Bots project

# Use bash as the shell
SHELL := /bin/bash

# Define default data directory
DATA_DIR ?= data

# Define the virtual environment activation
# This assumes a standard .venv directory created by Poetry
# Adjust if your venv is located elsewhere
ACTIVATE = . .venv/bin/activate

# Define Python executable within the venv
PYTHON = .venv/bin/python

# Define Poetry executable (assuming it's in PATH or use full path)
POETRY = poetry

# Define the python command to be used with poetry run
PYTHON_CMD = python

# Variables for batch optimization (override on command line)
# Use Full Class Names for Strategies
STRATEGIES ?= BollingerBandReversionStrategy # Space-separated strategy list (e.g., "BollingerBandReversionStrategy MovingAverageCrossoverStrategy")
SYMBOLS    ?= XRPUSDT BTCUSDT # Space-separated symbol list (e.g., "XRPUSDT BTCUSDT ETHUSDT")
INTERVAL   ?= 1h
START_DATE ?= 2017-01-01
END_DATE   ?= 2023-08-17
COMMISSION ?= 7.5      # Commission in basis points (e.g., 7.5 for 0.075%)

# Forward Test Period
FWD_START_DATE ?= 2023-01-01
FWD_END_DATE   ?= 2025-04-27

# Common Backtest/Optimization Parameters
BALANCE        ?= 10000   # Initial balance for backtests
METRIC         ?= cumulative_profit # Metric for optimization (cumulative_profit, sharpe_ratio, profit_factor, max_drawdown, win_rate)
SAVE_DETAILS   ?= false    # Save detailed optimization results (true/false)
DETAILS_FILE   ?=          # Optional path for detailed results CSV (default is auto-generated)
PROCESSES      ?= 12 # Default number of processes for backtesting within optimize
CONFIG         ?= config/optimize_params.yaml # Default config file path for optimize
# BEST_PARAMS_FILE is no longer used as input argument, output path is now fixed within optimize.py
# BEST_PARAMS_FILE ?= results/optimize/best_params.yaml

# === Additions for Threaded Strategy Optimizer ===

# Define the lists of symbols and strategies for threaded-strategy-optimizer
# Use Full Class Names for Strategies
TSO_SYMBOLS ?= BTCUSDT ETHUSDT ADAUSDT SOLUSDT # Example List for threaded optimizer
TSO_STRATEGIES ?= MovingAverageCrossoverStrategy ScalpingStrategy BollingerBandReversionStrategy MomentumStrategy MeanReversionStrategy BreakoutStrategy HybridStrategy # Use full class names

# Define other parameters used by 'make optimize' for the threaded-strategy-optimizer target - allow overrides
# ... (TSO vars remain mostly the same, BEST_PARAMS_FILE removed from here too) ...
TSO_START_DATE ?= 2023-01-01
TSO_END_DATE ?= 2023-12-31
TSO_INTERVAL ?= 1h
TSO_COMMISSION ?= 7.5 # Example in bps
TSO_BALANCE ?= 10000 # Default balance
TSO_PROCESSES ?= $(shell python -c 'import multiprocessing; print(multiprocessing.cpu_count())') # Default processes for inner optimize runs
TSO_METRIC ?= cumulative_profit # Default optimization metric
TSO_CONFIG ?= config/optimize_params.yaml # Default config file
TSO_SAVE_DETAILS ?= true # Default to save details
TSO_DETAILS_FILE ?= # Default to auto-generated details file name
# TSO_BEST_PARAMS_FILE removed
# TSO Filter Vars REMOVED

# Phony target to run optimization for all combinations via repeated 'make optimize' calls
.PHONY: trigger-threaded-optimizer
trigger-threaded-optimizer:
	@echo "Starting Threaded Strategy Optimizer for Symbols: [$(TSO_SYMBOLS)] and Strategies: [$(TSO_STRATEGIES)]..."
	@for strategy in $(TSO_STRATEGIES); do \
		for symbol in $(TSO_SYMBOLS); do \
			DATA_FILE="$(DATA_DIR)/$(TSO_INTERVAL)/$${symbol}_$(TSO_INTERVAL).csv"; \
			if [ ! -f "$${DATA_FILE}" ]; then \
				echo "Data file $${DATA_FILE} not found for $${symbol}. Attempting to fetch..."; \
				$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.fetch_data --symbols $${symbol} --interval $(TSO_INTERVAL) --start $(TSO_START_DATE); \
				if [ ! -f "$${DATA_FILE}" ]; then \
					echo "!!! Fetching failed or data file still not found: $${DATA_FILE}. Skipping task. !!!"; \
					continue; \
				fi; \
			else \
				echo "Data file $${DATA_FILE} found for $${symbol}."; \
			fi; \
			echo ""; \
			echo "--- Running Optimizer Task: Strategy=$${strategy}, Symbol=$${symbol} ---"; \
			$(MAKE) optimize \
				STRATEGY="$${strategy}" \ /* Pass full class name */
				SYMBOL="$${symbol}" \
				INTERVAL="$(TSO_INTERVAL)" \
				START_DATE="$(TSO_START_DATE)" \
				END_DATE="$(TSO_END_DATE)" \
				COMMISSION="$(TSO_COMMISSION)" \
				BALANCE="$(TSO_BALANCE)" \
				PROCESSES="$(TSO_PROCESSES)" \
				METRIC="$(TSO_METRIC)" \
				CONFIG="$(TSO_CONFIG)" \
				SAVE_DETAILS="$(TSO_SAVE_DETAILS)" \
				$(if $(TSO_DETAILS_FILE),DETAILS_FILE="$(TSO_DETAILS_FILE)") \
				$(if $(RESUME),RESUME=true) \ # Pass RESUME if set for TSO
				LOG_LEVEL=DEBUG; \
			if [ $$? -ne 0 ]; then \
				echo "!!! Optimizer Task failed for Strategy=$${strategy}, Symbol=$${symbol} !!!"; \
			fi; \
			echo "--- Finished Optimizer Task: Strategy=$${strategy}, Symbol=$${symbol} ---"; \
		done; \
	done
	@echo ""
	@echo "Threaded Strategy Optimizer finished."

# === End of Additions ===

.PHONY: install run lint test optimize optimize-batch fetch-data forward-test forward-test-batch all clean help trader analyze-details format trigger-threaded-optimizer analyze-forward-trades forward-test-all

# Default target
all: install lint

## Install dependencies
install:
	@echo "Installing dependencies using Poetry..."
	$(POETRY) install --with dev
	@echo "Installation complete. Ensure backtrader and matplotlib are included in pyproject.toml"

## Run the live/simulated trading bot
trader:
	@echo "Running the trading bot (use TRADER_ARGS=... for options)"
	# Assumes trader.py loads config/params similarly to forward_test.py
	# Example:
	#   make trader TRADER_ARGS="--strategy MovingAverageCrossoverStrategy --symbol ETHUSDT --interval 5m --units 0.01 --no-testnet"
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.trader $(TRADER_ARGS)

## Run static type checking with mypy
lint:
	@echo "Running mypy for static type checking..."
	$(POETRY) run mypy src

## Run Black code formatter
format:
	@echo "Running black code formatter on src directory..."
	$(POETRY) run black src

## Run tests (placeholder)
test:
	@echo "Running tests... (No tests implemented yet)"

## Run strategy optimization (single run)
# Updated to use full class name for --strategy
# Removed --output-config argument
optimize:
	@echo "Running strategy optimization: Strategy=$(STRATEGY), Symbol=$(SYMBOL), Interval=$(INTERVAL)"
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.optimize \
		--strategy "$(STRATEGY)" \ /* Use full class name */
		--symbol "$(SYMBOL)" \
		--file "data/$(INTERVAL)/$(SYMBOL)_$(INTERVAL).csv" \
		--config "$(CONFIG)" \
		--opt-start "$(START_DATE)" \
		--opt-end "$(END_DATE)" \
		--balance $(BALANCE) \
		--commission $(COMMISSION) \
		--metric "$(METRIC)" \
		$(if $(filter true,$(SAVE_DETAILS)),--save-details) \
		$(if $(DETAILS_FILE),--details-file "$(DETAILS_FILE)") \
		--processes $(PROCESSES) \
		$(if $(filter true,$(RESUME)),--resume) \
		--log $(or $(LOG_LEVEL),INFO)

## Run strategy optimization for multiple strategies and symbols
# No changes needed here if run_batch_optimization.py accepts args as before
optimize-batch:
	@echo "Running parallel batch strategy optimization..."
	@echo "Strategies: $(STRATEGIES), Symbols: $(SYMBOLS), Interval: $(INTERVAL)"
	$(POETRY) run python -m src.trading_bots.run_batch_optimization \
		--strategies "$(STRATEGIES)" \
		--symbols "$(SYMBOLS)" \
		--interval "$(INTERVAL)" \
		--start-date "$(START_DATE)" \
		--end-date "$(END_DATE)" \
		--commission $(COMMISSION) \
		--balance $(BALANCE) \
		--metric "$(METRIC)" \
		--config "$(CONFIG)" \
		--processes $(PROCESSES) \
		$(if $(filter true,$(SAVE_DETAILS)),--save-details) \
		$(if $(DETAILS_FILE),--details-file "$(DETAILS_FILE)") \
		$(if $(filter true,$(RESUME)),--resume) \
		$(if $(TARGET_METRIC),--target-metric "$(TARGET_METRIC)") \
		$(if $(OPTIMIZATION_METRIC_PREFIX),--optimization-metric-prefix "$(OPTIMIZATION_METRIC_PREFIX)") \
		$(if $(INCLUDE_SYMBOLS),--include-symbols "$(INCLUDE_SYMBOLS)")
	@echo "Parallel batch optimization finished."

## Fetch historical data from Binance
fetch-data:
	@echo "Fetching historical data from Binance (use FETCH_ARGS=...)"
	# Example: make fetch-data FETCH_ARGS="--symbols 'BTCUSDT ETHUSDT XRPUSDT' --interval 1h --start 2020-01-01 --processes 4"
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.fetch_data $(FETCH_ARGS)

## Run forward test (single run)
# Updated to use full class name for --strategy
# Removed filter arguments
forward-test:
	@echo "Running forward test: Strategy=$(STRATEGY), Symbol=$(SYMBOL), ParamFile=$(PARAM_CONFIG)"
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.forward_test \
		--strategy "$(STRATEGY)" \ /* Use full class name */
		--symbol "$(SYMBOL)" \
		--param-config "$(PARAM_CONFIG)" \ /* Requires path to specific best params file */
		--file "$(DATA_DIR)/$(INTERVAL)/$(SYMBOL)_$(INTERVAL).csv" \
		--fwd-start "$(FWD_START_DATE)" \
		$(if $(FWD_END_DATE),--fwd-end "$(FWD_END_DATE)") \
		--balance $(BALANCE) \
		--commission $(COMMISSION) \
		--units $(or $(UNITS),1.0) \
		--log $(or $(LOG_LEVEL),INFO)

## Run forward tests for all optimized results
forward-test-batch:
	@echo "Running batch forward tests..."
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.run_batch_forward_test \
		--results-dir "$(or $(OPTIMIZE_RESULTS_DIR),results/optimize)" \
		--data-dir "$(DATA_DIR)" \
		--start-date "$(FWD_START_DATE)" \
		$(if $(FWD_END_DATE),--end-date "$(FWD_END_DATE)") \
		--balance $(BALANCE) \
		--commission $(COMMISSION) \
		--units $(or $(UNITS),1.0) \
		--interval "$(INTERVAL)" \
		--log $(or $(LOG_LEVEL),INFO) \
		$(if $(TARGET_METRIC),--target-metric "$(TARGET_METRIC)") \
		$(if $(OPTIMIZATION_METRIC_PREFIX),--optimization-metric-prefix "$(OPTIMIZATION_METRIC_PREFIX)") \
		$(if $(INCLUDE_SYMBOLS),--include-symbols "$(INCLUDE_SYMBOLS)")

## Analyze optimization details CSV (if saved)
analyze-details:
	@echo "Analyzing optimization details CSV... (Requires --details-file argument)"
	# Add implementation or call to analysis script here
	@echo "(Not implemented yet)"

## Clean up generated files
clean:
	@echo "Cleaning up generated files..."
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete
	rm -rf .mypy_cache .pytest_cache .coverage
	rm -rf results/optimize/*.yaml results/optimize/*.csv results/optimize/plots/*.png results/optimize/reports/*.html
	rm -rf results/forward_test/*.yaml results/forward_test/trades/*.csv results/forward_test/plots/*.png results/forward_test/reports/*.html
	# Removed analyze_trades related cleaning
	@echo "Cleanup finished."

## Show help messages for targets
help:
	@echo "Available targets:"
	@grep -E '^##' Makefile | sed -e 's/## //g' | column -t -s ':' | sed -e 's/^/    /'

# Removed analyze-trades target 