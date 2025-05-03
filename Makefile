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
# This avoids the need to specify the full path to the venv python
PYTHON_CMD = python

# Variables for batch optimization (override on command line)
STRATEGIES ?= BBReversion # Space-separated strategy list (e.g., "BBReversion RSIReversion")
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
BEST_PARAMS_FILE ?= results/optimize/best_params.yaml # Default best params file path for optimize

# --- Filter Parameters (Defaults) ---
# These can be overridden for optimize, optimize-batch, forward-test, etc.
APPLY_ATR_FILTER ?= false
ATR_FILTER_PERIOD ?= 14
ATR_FILTER_MULTIPLIER ?= 1.5
ATR_FILTER_SMA_PERIOD ?= 100
APPLY_SEASONALITY_FILTER ?= false
ALLOWED_TRADING_HOURS_UTC ?= "" # e.g., '5-17'
APPLY_SEASONALITY_TO_SYMBOLS ?=  # Ensure this defaults to completely empty
# --- End Filter Parameters ---

# Default number of processes for fetching data
FETCH_PROCESSES ?= 6

# Arguments for single runs (override completely on command line)
# OPTIMIZE_ARGS is no longer used by the optimize target directly, but kept for reference/manual runs if needed
OPTIMIZE_ARGS ?= --strategy RSIReversion --symbol BTCUSDT --file data/1h/BTCUSDT_1h.csv --metric cumulative_profit --balance 10000 --commission 7.5 --opt-start 2021-01-01 --opt-end 2022-12-31 # Default example
FWD_ARGS      ?= --strategy RsiMeanReversionStrategy --symbol BTCUSDT --file data/1h/BTCUSDT_1h.csv --fwd-start 2021-01-01 --balance 10000 --commission 7.5 \
                 # --apply-atr-filter --atr-filter-period 14 --atr-filter-multiplier 1.5 --atr-filter-sma-period 100 \
                 # --apply-seasonality-filter --allowed-trading-hours-utc '5-17' --apply-seasonality-to-symbols 'XRPUSDT,SOLUSDT'
FETCH_ARGS    ?= # Default empty, provide args like --symbols BTCUSDT --interval 1h
TRADER_ARGS   ?= --strategy RsiMeanReversionStrategy --symbol XRPUSDT --interval 1h --units 0.1 --testnet \
                 # --stop-loss 0.02 --take-profit 0.05 --trailing-stop-loss 0.01 --max-cum-loss 100 \
                 # --apply-atr-filter --atr-filter-period 14 --atr-filter-multiplier 1.5 --atr-filter-sma-period 100 \
                 # --apply-seasonality-filter --allowed-trading-hours-utc '5-17' --apply-seasonality-to-symbols 'XRPUSDT,SOLUSDT'

# === Additions for Threaded Strategy Optimizer ===

# Define the lists of symbols and strategies for threaded-strategy-optimizer
# Override these on the command line if needed, e.g.:
# make threaded-strategy-optimizer TSO_SYMBOLS="BTCUSDT SOLUSDT" TSO_STRATEGIES="MACross"
TSO_SYMBOLS ?= BTCUSDT ETHUSDT ADAUSDT SOLUSDT # Example List for threaded optimizer
TSO_STRATEGIES ?= LongShort MACross RSIReversion BBReversion # Example List for threaded optimizer

# Define other parameters used by 'make optimize' for the threaded-strategy-optimizer target - allow overrides
# Ensure these match the variables your 'make optimize' target expects
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
TSO_BEST_PARAMS_FILE ?= results/optimize/best_params.yaml # Default output for best params
# Add TSO Filter Vars (using defaults from above if not set)
TSO_APPLY_ATR_FILTER ?= $(APPLY_ATR_FILTER)
TSO_ATR_FILTER_PERIOD ?= $(ATR_FILTER_PERIOD)
TSO_ATR_FILTER_MULTIPLIER ?= $(ATR_FILTER_MULTIPLIER)
TSO_ATR_FILTER_SMA_PERIOD ?= $(ATR_FILTER_SMA_PERIOD)
TSO_APPLY_SEASONALITY_FILTER ?= $(APPLY_SEASONALITY_FILTER)
TSO_ALLOWED_TRADING_HOURS_UTC ?= $(ALLOWED_TRADING_HOURS_UTC)
TSO_APPLY_SEASONALITY_TO_SYMBOLS ?= $(APPLY_SEASONALITY_TO_SYMBOLS)

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
				STRATEGY=$${strategy} \
				SYMBOL=$${symbol} \
				INTERVAL=$(TSO_INTERVAL) \
				START_DATE=$(TSO_START_DATE) \
				END_DATE=$(TSO_END_DATE) \
				COMMISSION=$(TSO_COMMISSION) \
				BALANCE=$(TSO_BALANCE) \
				PROCESSES=$(TSO_PROCESSES) \
				METRIC=$(TSO_METRIC) \
				CONFIG=$(TSO_CONFIG) \
				SAVE_DETAILS=$(TSO_SAVE_DETAILS) \
				DETAILS_FILE=$(TSO_DETAILS_FILE) \
				BEST_PARAMS_FILE=$(TSO_BEST_PARAMS_FILE) \
				APPLY_ATR_FILTER=$(TSO_APPLY_ATR_FILTER) \
				ATR_FILTER_PERIOD=$(ATR_FILTER_PERIOD) \
				ATR_FILTER_MULTIPLIER=$(ATR_FILTER_MULTIPLIER) \
				ATR_FILTER_SMA_PERIOD=$(ATR_FILTER_SMA_PERIOD) \
				APPLY_SEASONALITY_FILTER=$(TSO_APPLY_SEASONALITY_FILTER) \
				ALLOWED_TRADING_HOURS_UTC='$(TSO_ALLOWED_TRADING_HOURS_UTC)' \
				APPLY_SEASONALITY_TO_SYMBOLS='$(TSO_APPLY_SEASONALITY_TO_SYMBOLS)' \
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

.PHONY: install run lint test optimize optimize-batch fetch-data forward-test forward-test-batch all clean help trader analyze-trades analyze-details format trigger-threaded-optimizer analyze-forward-trades

# Default target
all: install lint

## Install dependencies
install:
	@echo "Installing dependencies using Poetry..."
	$(POETRY) install --with dev

## Run the live/simulated trading bot
trader:
	@echo "Running the trading bot (use TRADER_ARGS=... for options)"
	# Add --no-testnet for live trading! Add SL/TP args as needed.
	# Example:
	#   make trader TRADER_ARGS=" \
	#     --symbol ETHUSDT --interval 5m --units 0.01 \
	#     --strategy MACross --stop-loss 0.02 --trailing-stop-loss 0.01 \
	#     --apply-atr-filter --atr-filter-multiplier 2.0 \
	#     --apply-seasonality-filter --allowed-trading-hours-utc '9-21' --apply-seasonality-to-symbols 'ETHUSDT' \
	#     --no-testnet" # (Note: TRADER_ARGS uses a single string, internal newlines might need adjusting based on shell)
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
# All parameters are passed directly from Make variables to the script using native Make conditionals
# Removed check-data-file prerequisite (handled in trigger-threaded-optimizer)
optimize:
	@echo "Running strategy optimization: Strategy=$(STRATEGY), Symbol=$(SYMBOL), Interval=$(INTERVAL)"
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.optimize \
		--strategy "$(STRATEGY)" \
		--symbol "$(SYMBOL)" \
		--file "data/$(INTERVAL)/$(SYMBOL)_$(INTERVAL).csv" \
		--config "$(CONFIG)" \
		--output-config "$(BEST_PARAMS_FILE)" \
		--opt-start "$(START_DATE)" \
		--opt-end "$(END_DATE)" \
		--balance $(BALANCE) \
		--commission $(COMMISSION) \
		--metric "$(METRIC)" \
		$(if $(filter true,$(SAVE_DETAILS)),--save-details) \
		$(if $(DETAILS_FILE),--details-file "$(DETAILS_FILE)") \
		--processes $(PROCESSES) \
		$(if $(filter true,$(APPLY_ATR_FILTER)),--apply-atr-filter) \
		--atr-filter-period $(ATR_FILTER_PERIOD) \
		--atr-filter-multiplier $(ATR_FILTER_MULTIPLIER) \
		--atr-filter-sma-period $(ATR_FILTER_SMA_PERIOD) \
		$(if $(filter true,$(APPLY_SEASONALITY_FILTER)),--apply-seasonality-filter) \
		--allowed-trading-hours-utc "$(ALLOWED_TRADING_HOURS_UTC)" \
		--apply-seasonality-to-symbols "$(APPLY_SEASONALITY_TO_SYMBOLS)" \
		$(if $(filter true,$(RESUME)),--resume) \
		--log $(or $(LOG_LEVEL),INFO)

## Run strategy optimization for multiple strategies and symbols
# Passes Make variables to the batch runner script using native Make conditionals
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
		$(if $(filter true,$(APPLY_ATR_FILTER)),--apply-atr-filter) \
		--atr-filter-period $(ATR_FILTER_PERIOD) \
		--atr-filter-multiplier $(ATR_FILTER_MULTIPLIER) \
		--atr-filter-sma-period $(ATR_FILTER_SMA_PERIOD) \
		$(if $(filter true,$(APPLY_SEASONALITY_FILTER)),--apply-seasonality-filter) \
		--allowed-trading-hours-utc "$(ALLOWED_TRADING_HOURS_UTC)" \
		--apply-seasonality-to-symbols "$(APPLY_SEASONALITY_TO_SYMBOLS)"
	@echo "Parallel batch optimization finished."

## Fetch historical data from Binance
fetch-data:
	@echo "Fetching historical data from Binance (use FETCH_ARGS=...)"
	# Example: make fetch-data FETCH_ARGS=" \
	#   --symbols 'BTCUSDT ETHUSDT XRPUSDT' \
	#   --interval 1d \
	#   --start '2017-01-01' \
	#   --end '2024-01-01' \
	#   --data-dir ./historical_data" # (Note: FETCH_ARGS uses a single string)
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.fetch_data $(FETCH_ARGS) --processes $(FETCH_PROCESSES)

## Run simulated forward test (single run)
forward-test:
	@echo "Running simulated forward test (use FWD_ARGS=...)"
	# Example: make forward-test FWD_ARGS=\" \
	#   --strategy BBReversion \
	#   --symbol ETHUSDT \
	#   --file data/1h/ETHUSDT_1h.csv \
	#   --fwd-start 2023-08-18 \
	#   --commission 7.5 \
	#   --balance 1000 \
	#   --param-config results/optimize/best_params.yaml \
	#   --apply-atr-filter \
	#   --apply-seasonality-filter \" # (Note: FWD_ARGS uses a single string)
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.forward_test $(FWD_ARGS)

## Run forward testing for multiple strategies and symbols
# Passes Make variables explicitly in the loop using native Make conditionals
forward-test-batch:
	@echo "Running batch forward testing..."; \
	echo "Strategies: $(STRATEGIES)"; \
	echo "Symbols   : $(SYMBOLS)"; \
	echo "Interval  : $(INTERVAL)"; \
	echo "Fwd Dates : $(FWD_START_DATE) to $(FWD_END_DATE)"; \
	echo "Commission: $(COMMISSION) bps"; \
	echo "Balance   : $(BALANCE)"; \
	for strategy in $(STRATEGIES); do \
	    for symbol in $(SYMBOLS); do \
	        file_path=\"data/$(INTERVAL)/$${symbol}_$(INTERVAL).csv\"; \
	        if [ -f \"$${file_path}\" ]; then \
	            echo \"\"; \
	            echo \"--- Fwd Testing: Strategy=$${strategy}, Symbol=$${symbol}, File=$${file_path} ---\"; \
	            $(POETRY) run $(PYTHON_CMD) -m src.trading_bots.forward_test \
	                --strategy $${strategy} \
	                --symbol $${symbol} \
	                --file $${file_path} \
	                --param-config \"$(BEST_PARAMS_FILE)\" \
	                --fwd-start $(FWD_START_DATE) \
	                --fwd-end $(FWD_END_DATE) \
	                --commission $(COMMISSION) \
	                --balance $(BALANCE) \
	                --units 1.0 \
	                $(if $(filter true,$(APPLY_ATR_FILTER)),--apply-atr-filter) \
	                --atr-filter-period $(ATR_FILTER_PERIOD) \
	                --atr-filter-multiplier $(ATR_FILTER_MULTIPLIER) \
	                --atr-filter-sma-period $(ATR_FILTER_SMA_PERIOD) \
	                $(if $(filter true,$(APPLY_SEASONALITY_FILTER)),--apply-seasonality-filter) \
	                --allowed-trading-hours-utc '$(ALLOWED_TRADING_HOURS_UTC)' \
	                --apply-seasonality-to-symbols '$(APPLY_SEASONALITY_TO_SYMBOLS)'; \
	        else \
	            echo ""; \
	            echo "--- Skipping Fwd Test: Strategy=$${strategy}, Symbol=$${symbol} - File not found: $${file_path} ---"; \
	        fi; \
	    done; \
	done; \
	echo ""; \
	echo "Batch forward testing finished."

## Clean up Python bytecode and caches
clean:
	@echo "Cleaning up Python bytecode and cache files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	# Optionally clean results directories
	# rm -rf results/

## Display help information
help:
	@echo "Available commands:"
	@echo "  make install        - Install project dependencies"
	@echo "  make trader         - Run the live/simulated trading bot (use TRADER_ARGS=...)"
	@echo "                      Example: make trader TRADER_ARGS=\"\"
	@echo "                      \\  --symbol ETHUSDT --interval 5m --units 0.01 \\"
	@echo "                      \\  --strategy MACross --stop-loss 0.02 --trailing-stop-loss 0.01 \\"
	@echo "                      \\  --apply-atr-filter --atr-filter-multiplier 2.0 \\"
	@echo "                      \\  --apply-seasonality-filter --allowed-trading-hours-utc '9-21' --apply-seasonality-to-symbols 'ETHUSDT' \\"
	@echo "                      \\  --no-testnet \""
	@echo "  make lint           - Run mypy static type checker"
	@echo "  make test           - Run tests (placeholder)"
	@echo "  make optimize       - Run single strategy optimization (use make VAR=value overrides)"
	@echo "                      Example: make optimize STRATEGY=BBReversion SYMBOL=SOLUSDT METRIC=sharpe_ratio SAVE_DETAILS=true"
	@echo "  make optimize-batch - Run batch optimization for multiple strategies/symbols (use make VAR=value overrides)"
	@echo "                      Example: make optimize-batch STRATEGIES=\"BBReversion MACross\" SYMBOLS=\"SOLUSDT ADAUSDT\" METRIC=profit_factor SAVE_DETAILS=true"
	@echo "  make fetch-data     - Fetch historical data from Binance (use FETCH_ARGS=... and FETCH_PROCESSES=N)"
	@echo "                      Example: make fetch-data FETCH_ARGS=\"--symbols 'BTCUSDT ETHUSDT' --interval 1d --start '2017-01-01'\" FETCH_PROCESSES=8"
	@echo "  make forward-test   - Run single simulated forward test (use FWD_ARGS=...)"
	@echo "                      Example: make forward-test FWD_ARGS=\"--strategy MACross --balance 25000 --apply-atr-filter\""
	@echo "  make forward-test-batch - Run batch forward testing for multiple strategies/symbols (use make VAR=value overrides)"
	@echo "                      Example: make forward-test-batch STRATEGIES=\"RSIReversion\" SYMBOLS=\"XRPUSDT LTCUSDT\" APPLY_ATR_FILTER=true"
	@echo "  make all            - Install dependencies and run linter (default)"
	@echo "  make clean          - Remove Python bytecode and cache files"
	@echo "  make help           - Show this help message"
	@echo "  make analyze-trades - Analyze trade logs and generate reports/plots"
	@echo "  make analyze-details - Analyze optimization detail summary files"

# Analyze individual trade logs (_trades.csv) and generate full reports/plots
analyze-trades:
	@echo "--- Running Analysis on Trade Logs (results/optimize/trades/) ---"
	$(POETRY) run python -m src.trading_bots.analyze_trades --results-dir results/optimize --plotting
	@echo "--- Finished Trade Log Analysis ---"

# Analyze optimization detail summaries (_optimize_details_*.csv) 
analyze-details:
	@echo "--- Running Analysis on Optimization Details (results/optimize/) ---"
	$(POETRY) run python -m src.trading_bots.analyze_trades --analyze-details --plotting
	@echo "--- Finished Optimization Detail Analysis ---"

# Analyze forward test trade logs (_trades.csv)
analyze-forward-trades:
	@echo "--- Running Analysis on Forward Test Trade Logs (results/forward_test/trades/) ---"
	$(POETRY) run python -m src.trading_bots.analyze_trades --results-dir results/forward_test --plotting
	@echo "--- Finished Forward Test Trade Log Analysis ---"

# --- Helper Targets ---
# Internal target to check if data file exists for optimize/forward-test
# If not found, attempts to fetch data for the specific symbol/interval
.PHONY: check-data-file
check-data-file:
	@# Logic moved to trigger-threaded-optimizer loop
	@echo "Data check bypassed (handled in loop)." 