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
START_DATE ?= 2018-01-01
END_DATE   ?= 2021-01-01
COMMISSION ?= 7.5      # Commission in basis points (e.g., 7.5 for 0.075%)

# Forward Test Period
FWD_START_DATE ?= 2021-01-01
FWD_END_DATE   ?= 2025-04-27

# Common Backtest/Optimization Parameters
BALANCE        ?= 10000   # Initial balance for backtests
METRIC         ?= cumulative_profit # Metric for optimization (cumulative_profit, sharpe_ratio, profit_factor, max_drawdown, win_rate)
SAVE_DETAILS   ?= false    # Save detailed optimization results (true/false)
DETAILS_FILE   ?=          # Optional path for detailed results CSV (default is auto-generated)

# Arguments for single runs (override completely on command line)
OPTIMIZE_ARGS ?= --strategy RSIReversion --symbol BTCUSDT --file data/1h/BTCUSDT_1h.csv --metric cumulative_profit --balance 10000 --commission 7.5 # Default example
FWD_ARGS      ?= --strategy RSIReversion --symbol BTCUSDT --file data/1h/BTCUSDT_1h.csv --fwd-start 2021-01-01 --balance 10000 --commission 7.5 # Default example
FETCH_ARGS    ?= # Default empty, provide args like --symbols BTCUSDT --interval 1h
TRADER_ARGS   ?= --strategy LongShort --symbol BTCUSDT --interval 1m --units 0.001 --testnet # Default example

.PHONY: install run lint test optimize optimize-batch fetch-data forward-test forward-test-batch all clean help trader

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

## Run tests (placeholder)
test:
	@echo "Running tests... (No tests implemented yet)"

## Run strategy optimization (single run)
optimize:
	@echo "Running strategy optimization (use OPTIMIZE_ARGS=...)"
	# Example: make optimize OPTIMIZE_ARGS="--strategy RSIReversion --symbol ALGOUSDT --file data/1h/ALGOUSDT_1h.csv --opt-start 2021-01-01 --opt-end 2022-12-31 --commission 7.5 --metric sharpe_ratio --balance 5000 --save-details"
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.optimize $(OPTIMIZE_ARGS)

## Run strategy optimization for multiple strategies and symbols
optimize-batch:
	@echo "Running batch strategy optimization..."; \
	 echo "Strategies: $(STRATEGIES)"; \
	 echo "Symbols   : $(SYMBOLS)"; \
	 echo "Interval  : $(INTERVAL)"; \
	 echo "Date Range: $(START_DATE) to $(END_DATE)"; \
	 echo "Commission: $(COMMISSION) bps"; \
	 echo "Balance   : $(BALANCE)"; \
	 echo "Metric    : $(METRIC)"; \
	 echo "SaveDetails: $(SAVE_DETAILS)"; \
	 save_details_arg=$$(if [ "$(SAVE_DETAILS)" = "true" ]; then echo "--save-details"; else echo ""; fi); \
	 details_file_arg=$$(if [ -n "$(DETAILS_FILE)" ]; then echo "--details-file $(DETAILS_FILE)"; else echo ""; fi); \
	 for strategy in $(STRATEGIES); do \
	     for symbol in $(SYMBOLS); do \
	         file_path="data/$(INTERVAL)/$${symbol}_$(INTERVAL).csv"; \
	         if [ -f "$${file_path}" ]; then \
	             echo "--- Optimizing Strategy: $${strategy}, Symbol: $${symbol}, File: $${file_path} ---"; \
	             args="--strategy $${strategy} --symbol $${symbol} --file $${file_path} --opt-start $(START_DATE) --opt-end $(END_DATE) --commission $(COMMISSION) --balance $(BALANCE) --metric $(METRIC) $${save_details_arg} $${details_file_arg}"; \
	             echo "Running: $(POETRY) run $(PYTHON_CMD) -m src.trading_bots.optimize $${args}"; \
	             $(POETRY) run $(PYTHON_CMD) -m src.trading_bots.optimize $${args}; \
	         else \
	             echo "--- Skipping Strategy: $${strategy}, Symbol: $${symbol} - File not found: $${file_path} ---"; \
	         fi; \
	     done; \
	 done; \
	 echo "Batch optimization finished."

## Fetch historical data from Binance
fetch-data:
	@echo "Fetching historical data from Binance (use FETCH_ARGS=...)"
	$(POETRY) run $(PYTHON_CMD) -m src.trading_bots.fetch_data $(FETCH_ARGS)

## Run simulated forward test (single run)
forward-test:
	@echo "Running simulated forward test (use FWD_ARGS=...)"
	# Example: make forward-test FWD_ARGS="--strategy BBReversion --symbol ETHUSDT --file data/1h/ETHUSDT_1h.csv --fwd-start 2022-01-01 --commission 5 --balance 20000"
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