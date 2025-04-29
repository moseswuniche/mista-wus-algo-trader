# Makefile for Trading Bots project

# Use bash as the shell
SHELL := /bin/bash

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
PROCESSES      ?= 1         # Default number of processes for backtesting within optimize
CONFIG         ?= config/optimize_params.yaml # Default config file path for optimize
BEST_PARAMS_FILE ?= results/optimize/best_params.yaml # Default best params file path for optimize

# Arguments for single runs (override completely on command line)
# OPTIMIZE_ARGS is no longer used by the optimize target directly, but kept for reference/manual runs if needed
OPTIMIZE_ARGS ?= --strategy RSIReversion --symbol BTCUSDT --file data/1h/BTCUSDT_1h.csv --metric cumulative_profit --balance 10000 --commission 7.5 --opt-start 2021-01-01 --opt-end 2022-12-31 # Default example
FWD_ARGS      ?= --strategy RSIReversion --symbol BTCUSDT --file data/1h/BTCUSDT_1h.csv --fwd-start 2021-01-01 --balance 10000 --commission 7.5 # Default example
FETCH_ARGS    ?= # Default empty, provide args like --symbols BTCUSDT --interval 1h
TRADER_ARGS   ?= --strategy LongShort --symbol BTCUSDT --interval 1m --units 0.001 --testnet # Default example

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

# Phony target to run optimization for all combinations via repeated 'make optimize' calls using shell loops
.PHONY: trigger-threaded-optimizer
trigger-threaded-optimizer:
	@echo "Starting Threaded Strategy Optimizer for Symbols: [$(TSO_SYMBOLS)] and Strategies: [$(TSO_STRATEGIES)]..."
	@for strategy in $(TSO_STRATEGIES); do \
		for symbol in $(TSO_SYMBOLS); do \
			echo ""; \
			echo "--- Running Optimizer Task: Strategy=$${strategy}, Symbol=$${symbol} ---"; \
			$(MAKE) optimize \
				STRATEGY=$${strategy} \
				SYMBOL=$${symbol} \
				START_DATE=$(TSO_START_DATE) \
				END_DATE=$(TSO_END_DATE) \
				INTERVAL=$(TSO_INTERVAL) \
				COMMISSION=$(TSO_COMMISSION) \
				BALANCE=$(TSO_BALANCE) \
				PROCESSES=$(TSO_PROCESSES) \
				METRIC=$(TSO_METRIC) \
				CONFIG=$(TSO_CONFIG) \
				SAVE_DETAILS=$(TSO_SAVE_DETAILS) \
				DETAILS_FILE=$(TSO_DETAILS_FILE) \
				BEST_PARAMS_FILE=$(TSO_BEST_PARAMS_FILE); \
			if [ $$? -ne 0 ]; then \
				echo "!!! Optimizer Task failed for Strategy=$${strategy}, Symbol=$${symbol} !!!"; \
			fi; \
			echo "--- Finished Optimizer Task: Strategy=$${strategy}, Symbol=$${symbol} ---"; \
		done; \
	done
	@echo ""
	@echo "Threaded Strategy Optimizer finished."

# === End of Additions ===

.PHONY: install run lint test optimize optimize-batch fetch-data forward-test forward-test-batch all clean help trader analyze-trades analyze-details format trigger-threaded-optimizer

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
# This target now builds arguments from individual make variables passed to it.
optimize:
	@echo "Running strategy optimization for Strategy=$(STRATEGY), Symbol=$(SYMBOL)"; \
	file_path="data/$(INTERVAL)/$(SYMBOL)_$(INTERVAL).csv"; \
	if [ ! -f "$${file_path}" ]; then \
		echo "Error: Data file not found: $${file_path}"; \
		exit 1; \
	fi; \
	save_details_arg=$$(if [ "$(SAVE_DETAILS)" = "true" ]; then echo "--save-details"; else echo ""; fi); \
	details_file_arg=$$(if [ -n "$(DETAILS_FILE)" ]; then echo "--details-file \"$(DETAILS_FILE)\""; else echo ""; fi); \
	config_arg=$$(if [ -n "$(CONFIG)" ]; then echo "--config $(CONFIG)"; else echo ""; fi); \
	processes_arg=$$(if [ -n "$(PROCESSES)" ]; then echo "-p $(PROCESSES)"; else echo ""; fi); \
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.optimize \
		--strategy "$(STRATEGY)" \
		--symbol "$(SYMBOL)" \
		--file "$${file_path}" \
		--opt-start "$(START_DATE)" \
		--opt-end "$(END_DATE)" \
		--balance $(BALANCE) \
		--commission $(COMMISSION) \
		--metric "$(METRIC)" \
		$${config_arg} \
		$${save_details_arg} \
		$${details_file_arg} \
		$${processes_arg}

## Run strategy optimization for multiple strategies and symbols
optimize-batch:
	@echo "Running parallel batch strategy optimization..."
	@echo "Strategies: $(STRATEGIES)"
	@echo "Symbols   : $(SYMBOLS)"
	@echo "Interval  : $(INTERVAL)"
	@echo "Date Range: $(START_DATE) to $(END_DATE)"
	@echo "Commission: $(COMMISSION) bps"
	@echo "Balance   : $(BALANCE)"
	@echo "Metric    : $(METRIC)"
	@echo "SaveDetails: $(SAVE_DETAILS)"
	# Construct arguments for the Python script
	save_details_arg=$$(if [ "$(SAVE_DETAILS)" = "true" ]; then echo "--save-details"; else echo ""; fi);
	details_file_arg=$$(if [ -n "$(DETAILS_FILE)" ]; then echo "--details-file \"$(DETAILS_FILE)\""; else echo ""; fi);
	# Execute the parallel runner script
	$(POETRY) run python -m src.trading_bots.run_batch_optimization \
		--strategies "$(STRATEGIES)" \
		--symbols "$(SYMBOLS)" \
		--interval "$(INTERVAL)" \
		--start-date "$(START_DATE)" \
		--end-date "$(END_DATE)" \
		--commission $(COMMISSION) \
		--balance $(BALANCE) \
		--metric "$(METRIC)" \
		$${save_details_arg} \
		$${details_file_arg}
	@echo "Parallel batch optimization finished."

## Fetch historical data from Binance
fetch-data:
	@echo "Fetching historical data from Binance (use FETCH_ARGS=...)"
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.fetch_data $(FETCH_ARGS)

## Run simulated forward test (single run)
forward-test:
	@echo "Running simulated forward test (use FWD_ARGS=...)"
	# Example: make forward-test FWD_ARGS="--strategy BBReversion --symbol ETHUSDT --file data/1h/ETHUSDT_1h.csv --fwd-start 2023-08-18 --commission 7.5 --balance 1000"
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.forward_test $(FWD_ARGS)

## Run forward testing for multiple strategies and symbols
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
	         file_path="data/$(INTERVAL)/$${symbol}_$(INTERVAL).csv"; \
	         if [ -f "$${file_path}" ]; then \
	             echo "--- Forward Testing Strategy: $${strategy}, Symbol: $${symbol}, File: $${file_path} ---"; \
	             args="--strategy $${strategy} --symbol $${symbol} --file $${file_path} --fwd-start $(FWD_START_DATE) --fwd-end $(FWD_END_DATE) --commission $(COMMISSION) --balance $(BALANCE)"; \
	             echo "Running: $(POETRY) run $(PYTHON_CMD) -m src.trading_bots.forward_test $${args}"; \
	             $(POETRY) run $(PYTHON_CMD) -m src.trading_bots.forward_test $${args}; \
	         else \
	             echo "--- Skipping Strategy: $${strategy}, Symbol: $${symbol} - File not found: $${file_path} ---"; \
	         fi; \
	     done; \
	 done; \
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
	@echo "                      Example: make trader TRADER_ARGS=\"--symbol ETHUSDT --interval 5m --units 0.01 --strategy MACross --stop-loss 0.02 --trailing-stop-loss 0.01 --no-testnet\""
	@echo "  make lint           - Run mypy static type checker"
	@echo "  make test           - Run tests (placeholder)"
	@echo "  make optimize       - Run single strategy optimization (use OPTIMIZE_ARGS=...)"
	@echo "                      Example: make optimize OPTIMIZE_ARGS=\"--strategy RSIReversion --metric sharpe_ratio --save-details\""
	@echo "  make optimize-batch - Run batch optimization for multiple strategies/symbols"
	@echo "                      Override vars: make optimize-batch STRATEGIES=... SYMBOLS=... METRIC=... SAVE_DETAILS=true ..."
	@echo "  make fetch-data     - Fetch historical data from Binance (use FETCH_ARGS=...)"
	@echo "  make forward-test   - Run single simulated forward test (use FWD_ARGS=...)"
	@echo "                      Example: make forward-test FWD_ARGS=\"--strategy MACross --balance 25000\""
	@echo "  make forward-test-batch - Run batch forward testing for multiple strategies/symbols"
	@echo "                      Override vars: make forward-test-batch STRATEGIES=... SYMBOLS=... BALANCE=..."
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