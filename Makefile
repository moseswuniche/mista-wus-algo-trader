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

.PHONY: install run lint test optimize fetch-data all clean help

# Default target
all: install lint

## Install dependencies
install:
	@echo "Installing dependencies using Poetry..."
	$(POETRY) install --with dev

## Run the main trading bot script
run:
	@echo "Running the trading bot..."
	# Run as a module to handle relative imports correctly
	$(POETRY) run $(PYTHON) -m src.trading_bots.trader

## Run static type checking with mypy
lint:
	@echo "Running mypy for static type checking..."
	$(POETRY) run mypy src

## Run tests (placeholder)
test:
	@echo "Running tests... (No tests implemented yet)"
	# Add your test command here when tests are available, e.g.:
	# $(POETRY) run pytest tests/

## Run strategy optimization (placeholder/example)
optimize:
	@echo "Running strategy optimization..."
	# Example: Optimize MACross strategy using default bitcoin.csv
	# Modify --strategy, --file, --units, --metric, --symbol as needed
	# Use OPTIMIZE_ARGS make variable to pass arguments: make optimize OPTIMIZE_ARGS="--strategy RSIReversion --file data/ALGOUSDT.csv --symbol ALGOUSDT"
	$(POETRY) run $(PYTHON) -m src.trading_bots.optimize --strategy MACross $(OPTIMIZE_ARGS)

## Fetch historical data from Binance
fetch-data:
	@echo "Fetching historical data from Binance..."
	# Example: Fetch default symbols (BTC, XRP, ALGO), daily, from 2017 to now
	# Modify args as needed: make fetch-data FETCH_ARGS="--symbols ETHUSDT --interval 1h --start 2020-01-01"
	$(POETRY) run $(PYTHON) -m src.trading_bots.fetch_data $(FETCH_ARGS)

## Clean up Python bytecode and caches
clean:
	@echo "Cleaning up Python bytecode and cache files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .mypy_cache
	rm -rf .pytest_cache

## Display help information
help:
	@echo "Available commands:"
	@echo "  make install    - Install project dependencies including dev dependencies"
	@echo "  make run        - Run the main trading bot script"
	@echo "  make lint       - Run mypy static type checker"
	@echo "  make test       - Run tests (placeholder)"
	@echo "  make optimize   - Run strategy optimization (use OPTIMIZE_ARGS=... for options)"
	@echo "  make fetch-data - Fetch historical data from Binance (use FETCH_ARGS=... for options)"
	@echo "  make all        - Install dependencies and run linter (default)"
	@echo "  make clean      - Remove Python bytecode and cache files"
	@echo "  make help       - Show this help message" 