# Trading Bots

This project contains various deployable and runtime configurable trading strategies for cryptocurrencies using the Binance API.

Features include:
- Strategy modularity (easily add new strategies)
- Live trading via WebSockets (`trader.py`)
- Backtesting engine (`backtest.py`)
- Grid search parameter optimization (`optimize.py`)
- Simulated forward testing (`forward_test.py`)
- Historical data fetching (`fetch_data.py`)
- Type hints and MyPy checking
- Configuration via YAML for optimization parameters
- Logging
- Runtime configuration reloading (strategy & parameters)

## Project Structure

```
.
├── .gitignore
├── Makefile
├── README.md
├── config
│   ├── optimize_params.yaml  # Parameter ranges for optimization (COMMITTED - edit ranges here)
│   ├── best_params.yaml      # Stores best parameters found (IGNORED - generated)
│   └── runtime_config.yaml   # Runtime configuration for the live trader (COMMITTED - edit for live changes)
├── data                      # Default directory for CSV data (IGNORED)
│   └── ... (CSV files)
├── poetry.lock
├── pyproject.toml
└── src
    └── trading_bots
        ├── __init__.py
        ├── backtest.py         # Backtesting logic
        ├── data_utils.py       # Data loading utilities
        ├── fetch_data.py       # Script to fetch historical data
        ├── optimize.py         # Strategy parameter optimization script
        ├── forward_test.py     # Script for simulated forward testing
        ├── trader.py           # Main live trading bot class & execution
        └── strategies
            ├── __init__.py
            ├── base_strategy.py
            ├── bb_reversion_strategy.py
            ├── long_short_strategy.py
            ├── ma_crossover_strategy.py
            └── rsi_reversion_strategy.py
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd trading_bots
    ```
2.  **Install Dependencies:** Requires Python >= 3.11 and Poetry.
    ```bash
    poetry install --with dev
    ```
3.  **API Keys & AWS Credentials:**
    *   Live trading (`trader.py`) requires API keys.
    *   Using AWS Secrets Manager (`--aws-secret-name`) requires configured AWS credentials (e.g., via environment variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, or an IAM role attached to the compute instance).
    *   **Environment Variables (Recommended):** Set `BINANCE...` keys as previously described.
    *   **AWS Secrets Manager (Optional):** Create a secret in AWS Secrets Manager containing a JSON structure with the necessary `BINANCE...` keys (see `src/trading_bots/config_utils.py` for expected format). Provide `--aws-secret-name` and optionally `--aws-region` when running `trader.py`.

## Usage (Makefile)

Use `make help` to see available commands.

*   **Install/Update Dependencies:**
    ```bash
    make install
    ```
*   **Fetch Historical Data:** Fetches daily data for BTC, XRP, ALGO, ETH, HBAR, ADA, SOL, LTC, SUI (all USDT pairs) from 2017 to now into the `data/` directory by default.
    ```bash
    make fetch-data
    # Fetch hourly ETH data from 2020:
    make fetch-data FETCH_ARGS="--symbols ETHUSDT --interval 1h --start 2020-01-01"
    # Set log level to DEBUG:
    make fetch-data FETCH_ARGS="--log DEBUG"
    ```
*   **Run Strategy Optimization:** Optimizes parameters using grid search based on `config/optimize_params.yaml`. Saves the best results to `config/best_params.yaml` by default.
    ```bash
    # Optimize MACross for BTCUSDT using daily data from 2020-01-01 to 2022-12-31
    make optimize OPTIMIZE_ARGS="--strategy MACross --symbol BTCUSDT --file data/BTCUSDT_1d.csv --opt-start 2020-01-01 --opt-end 2022-12-31 --units 0.01 --commission 7.5"
    # Optimize RSIReversion for ALGOUSDT using 1h data for all of 2023
    make optimize OPTIMIZE_ARGS="--strategy RSIReversion --symbol ALGOUSDT --file data/ALGOUSDT_1h.csv --opt-start 2023-01-01 --opt-end 2023-12-31 --units 100 --metric win_rate --commission 7.5"
    ```
    *   Configure parameter search ranges in `config/optimize_params.yaml`.
    *   Use the **required** `--file` argument to specify the data file for optimization.
    *   Use `--opt-start` and `--opt-end` (YYYY-MM-DD format) to specify the date range of the data used for optimization. If omitted, the entire dataset specified by `--file` is used.
    *   Best parameters found are saved to the file specified by `--output-config` (default: `config/best_params.yaml`).
    *   `--commission` is in basis points (e.g., 7.5 for 0.075%).
*   **Run Simulated Forward Test:** Runs the backtester using parameters from a specified file (e.g., `config/best_params.yaml`) on a *different* historical data period.
    ```bash
    # Example: Test optimized MACross for BTCUSDT on data from 2023-01-01 onwards
    make forward-test FWD_ARGS="--strategy MACross --symbol BTCUSDT --file data/BTCUSDT_1d.csv --fwd-start 2023-01-01 --param-config config/best_params.yaml --units 0.01 --commission 7"
    # Test optimized RSIReversion for ALGOUSDT on data from 2023-06-01 to 2023-12-31
    make forward-test FWD_ARGS="--strategy RSIReversion --symbol ALGOUSDT --file data/ALGOUSDT_1d.csv --fwd-start 2023-06-01 --fwd-end 2023-12-31 --param-config config/best_params.yaml --units 100 --commission 7"
    ```
    *   Requires optimized parameters to exist in the file specified by `--param-config`.
    *   Uses the *full* historical data specified by `--file` and slices it based on `--fwd-start` and `--fwd-end`.
*   **Run Live Trader (Testnet by Default):**
    ```bash
    # Run default BTCUSDT with strategy defined in runtime_config.yaml (initially LongShort)
    make run
    # Run with specific initial args, still monitors runtime_config.yaml for changes
    make run RUN_ARGS="--symbol XRPUSDT --strategy MACross --units 100 --runtime-config config/my_runtime.yaml"
    # Run using AWS Secrets Manager for keys
    make run RUN_ARGS="--aws-secret-name my-binance-secrets --aws-region eu-west-1"
    ```
    *   Requires API keys (via env vars or AWS).
    *   **Runtime Configuration:** While running, edit the file specified by `--runtime-config` (default: `config/runtime_config.yaml`) to change the `strategy_name` or `strategy_params`. The trader will check this file periodically (every 5 bars by default) and attempt to apply changes if the bot is currently flat (position = 0).
*   **Linting:**
    ```bash
    make lint
    ```
*   **Clean:** Removes cache files.
    ```bash
    make clean
    ```

## Adding New Strategies

1.  Create a new Python file in `src/trading_bots/strategies/` (e.g., `my_strategy.py`).
2.  Define a class inheriting from `src.trading_bots.strategies.base_strategy.Strategy`.
3.  Implement the `__init__` method to accept parameters.
4.  Implement the `generate_signals(self, data: pd.DataFrame) -> pd.Series` method. It should return a pandas Series with the same index as the input data, containing integers: `1` for long, `-1` for short, `0` for neutral.
5.  Add the new strategy class to `src/trading_bots/strategies/__init__.py` (imports and `__all__`).
6.  Add parameter search ranges to `config/optimize_params.yaml` for the new strategy under the desired symbols.
7.  (Optional) Add the strategy choice to the `

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.