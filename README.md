# Trading Bots

This project contains various deployable and runtime configurable trading strategies for cryptocurrencies using the Binance API.

Features include:
- Strategy modularity (easily add new strategies)
- Live trading via WebSockets (`trader.py`)
- Configurable Stop Loss, Take Profit, and Trailing Stop Loss (`trader.py` + Backtests)
- Configurable Max Cumulative Loss Stop for the bot (`trader.py`)
- Backtesting engine with intra-bar SL/TP/TSL simulation (`backtest.py`)
- Calculation of standard performance metrics (Profit, Win Rate, Sharpe Ratio, Max Drawdown, Profit Factor)
- **Parallel** Grid search parameter optimization (`optimize.py`)
- **Parallel** Batch optimization for multiple strategies/symbols (`run_batch_optimization.py`)
- Optimization based on selectable performance metrics
- Simulated forward testing (`forward_test.py`) with HTML reports
- Saving of detailed optimization results to CSV
- Analysis of trade logs and optimization detail results (`analyze_trades.py`)
  - Generation of performance summary plots and HTML reports
- Historical data fetching (`fetch_data.py`)
- Type hints and MyPy checking
- Code formatting using Black
- Configuration via YAML for optimization and runtime parameters
- Logging
- Runtime configuration reloading (strategy, parameters, SL/TP, max loss)

## Project Structure

```
.
├── .gitignore
├── Makefile
├── README.md
├── config
│   ├── optimize_params.yaml  # Parameter ranges for optimization (incl. SL/TP/TSL)
│   ├── best_params.yaml      # Stores best parameters found (incl. SL/TP/TSL, metric) (IGNORED - generated)
│   └── runtime_config.yaml   # Runtime configuration for the live trader (strategy, params, SL/TP)
├── data                      # Default directory for CSV data (IGNORED)
│   └── ... (CSV files)
├── poetry.lock
├── pyproject.toml
├── results                   # Default directory for generated results (IGNORED)
│   ├── analysis              # Results from analyze_trades.py
│   │   ├── plots             # Overall performance plots
│   │   ├── reports           # HTML reports (per-group and detail summary)
│   │   └── summary           # Aggregated summary CSV
│   ├── optimization          # Detailed optimization CSVs
│   │   └── ...
│   └── forward_test          # Forward test results (plots, reports, trades)
│       ├── plots
│       ├── reports
│       └── trades
└── src
    └── trading_bots
        ├── __init__.py
        ├── analyze_trades.py   # Script to analyze trade logs or optimization details
        ├── backtest.py         # Backtesting logic
        ├── config_utils.py     # Configuration loading utilities (e.g., AWS Secrets)
        ├── fetch_data.py       # Script to fetch historical data
        ├── optimize.py         # Strategy parameter optimization script (single run, parallel backtests)
        ├── forward_test.py     # Script for simulated forward testing
        ├── reporting_utils.py  # Shared plotting and reporting functions
        ├── run_batch_optimization.py # Script to run multiple optimizations in parallel
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

Use `make help` to see available commands and default variable values.

*   **Install/Update Dependencies:**
    ```bash
    make install
    ```
*   **Fetch Historical Data:**
    ```bash
    make fetch-data # Fetch default symbols/interval
    # Fetch hourly ETH data from 2020:
    make fetch-data FETCH_ARGS="--symbols ETHUSDT --interval 1h --start 2020-01-01"
    ```
*   **Run Strategy Optimization:** Optimizes parameters (including optional SL/TP/TSL) using grid search based on `config/optimize_params.yaml`. Saves the best parameters to `config/best_params.yaml`.
    *   **Single Run (Parallel Backtests):** Uses multiprocessing *within* the `optimize.py` script to speed up testing parameter combinations for *one* strategy/symbol pair.
        ```bash
        # Optimize MACross for BTCUSDT daily data, maximizing Sharpe Ratio, saving details, using 6 cores
        make optimize OPTIMIZE_ARGS="--strategy MACross --symbol BTCUSDT --file data/1d/BTCUSDT_1d.csv --metric sharpe_ratio --save-details --balance 10000 --commission 7.5 -p 6"
        ```
        *   Use the `-p N` argument within `OPTIMIZE_ARGS` to specify the number of cores for backtesting *within* this single optimization run.
    *   **Batch Run (Parallel Optimizations - `optimize-batch`):** Runs *multiple independent optimization tasks* (for different strategies and/or symbols) *concurrently* using multiprocessing managed by `run_batch_optimization.py`. This is generally faster for optimizing many pairs if you have sufficient CPU cores.
        ```bash
        # Batch optimize RSIReversion and BBReversion for XRP/ETH hourly, minimizing Max Drawdown
        make optimize-batch STRATEGIES="RSIReversion BBReversion" SYMBOLS="XRPUSDT ETHUSDT" INTERVAL=1h METRIC=max_drawdown START_DATE=2020-01-01 END_DATE=2022-12-31 BALANCE=25000 COMMISSION=5 SAVE_DETAILS=true
        ```
        *   The number of parallel optimization *processes* is controlled by the `--processes` argument passed to `src/trading_bots/run_batch_optimization.py` (defaults set in that script, check it for details or override with `-p N` in the script call if needed).
    *   **Batch Run (Sequential Trigger - `trigger-threaded-optimizer`):** Runs optimization for multiple strategy/symbol combinations *sequentially*. For each combination, it calls `make optimize`, which in turn runs `optimize.py` potentially using multiple cores for its internal backtests (controlled by `TSO_PROCESSES` variable). This approach is simpler but slower than `optimize-batch` as the strategy/symbol pairs are not run concurrently.
        ```bash
        # Sequentially optimize RSIReversion and BBReversion for XRP/ETH hourly, using TSO_ variables
        make trigger-threaded-optimizer TSO_STRATEGIES="RSIReversion BBReversion" TSO_SYMBOLS="XRPUSDT ETHUSDT" TSO_INTERVAL=1h TSO_METRIC=sharpe_ratio TSO_BALANCE=5000 TSO_SAVE_DETAILS=true TSO_PROCESSES=4
        ```
        *   The `TSO_PROCESSES` variable controls the number of cores used by the underlying `optimize.py` script for *each* sequential optimization task.
    *   Configure parameter search ranges, including `stop_loss_pct`, `take_profit_pct`, `trailing_stop_loss_pct` (use `None` to test without them), in `config/optimize_params.yaml`.
    *   Specify the optimization target using `METRIC=` or `TSO_METRIC=` (Makefile variables) or `--metric` (`OPTIMIZE_ARGS`). Choices: `cumulative_profit`, `final_balance`, `sharpe_ratio`, `profit_factor`, `max_drawdown`, `win_rate`.
    *   Use `SAVE_DETAILS=true` or `TSO_SAVE_DETAILS=true` (Makefile variables) or `--save-details` (`OPTIMIZE_ARGS`) to save all tested combinations to a CSV file in `results/optimization/`.
    *   Set initial balance using `BALANCE=` or `TSO_BALANCE=` or `--balance`.
    *   Best parameters (including SL/TP/TSL) are saved to `--output-params` (default: `config/best_params.yaml`).
*   **Run Simulated Forward Test:** Runs the backtester using the best parameters found during optimization (from `config/best_params.yaml`) on a *different* historical data period. Generates reports and saves trade logs.
    *   **Single Run:**
        ```bash
        # Test optimized MACross for BTCUSDT on data from 2023-01-01 onwards
        make forward-test FWD_ARGS="--strategy MACross --symbol BTCUSDT --file data/1d/BTCUSDT_1d.csv --fwd-start 2023-01-01 --balance 10000 --commission 7"
        ```
    *   **Batch Run (Sequential):**
        ```bash
        # Batch test RSIReversion for multiple symbols sequentially
        make forward-test-batch STRATEGIES="RSIReversion" SYMBOLS="BTCUSDT ETHUSDT" FWD_START_DATE=2023-01-01 BALANCE=50000
        ```
    *   Requires optimized parameters (incl. SL/TP/TSL) to exist in `--param-config` (default: `config/best_params.yaml`).
    *   Set initial balance using `BALANCE=` or `--balance`.
    *   Generates HTML reports (with metrics and plots) in `results/forward_test/reports/`.
    *   Saves detailed trade logs (`_trades.csv`) in `results/forward_test/trades/`.
*   **Analyze Results:** Analyze trade logs or optimization detail files.
    ```bash
    # Analyze all trade logs found in results/optimization/trades/ and generate reports/plots
    make analyze-trades 
    # Analyze optimization detail CSVs found in results/optimization/ and generate summary report/plot
    make analyze-details
    # Analyze only a specific strategy/symbol trade log
    make analyze-trades ANALYZE_ARGS="--strategy RSIReversion --symbol BTCUSDT"
    ```
    *   Output saved to `results/analysis/`.
*   **Run Live/Simulated Trader (Testnet by Default):**
    ```bash
    # Run default strategy/symbol defined in TRADER_ARGS
    make trader 
    # Run ETHUSDT with 5m interval, testnet, specific SL/TP/TSL/MaxLoss
    make trader TRADER_ARGS="--symbol ETHUSDT --strategy MACross --interval 5m --units 0.01 --stop-loss 0.015 --trailing-stop-loss 0.01 --max-cum-loss 500 --testnet"
    # Run LIVE (USE WITH EXTREME CAUTION)
    make trader TRADER_ARGS="--symbol BTCUSDT --strategy RSIReversion --interval 1h --units 0.001 --no-testnet --stop-loss 0.02"
    ```
    *   Requires API keys (via env vars or AWS).
    *   Use `--stop-loss`, `--take-profit`, `--trailing-stop-loss`, `--max-cum-loss` to enable risk management.
    *   **Runtime Configuration:** Edit the `--runtime-config` file (default: `config/runtime_config.yaml`) to change `strategy_name`, `strategy_params`, `stop_loss_pct`, `take_profit_pct`, `trailing_stop_loss_pct`, `max_cumulative_loss`. Changes are checked periodically and applied if the bot is flat.
*   **Linting:** `make lint`
*   **Formatting:** `make format` (Runs Black code formatter)
*   **Clean:** `make clean`

## Output Files

*   `config/best_params.yaml`: Stores the best parameters found by optimization for each symbol/strategy combination.
*   `results/optimization/`: Contains detailed CSV logs of *all* combinations tested during optimization if `SAVE_DETAILS=true` is used.
*   `results/forward_test/reports/`: Contains HTML reports summarizing forward test performance, including metrics and equity curve plots.
*   `results/forward_test/plots/`: PNG images of equity curves from forward tests.
*   `results/forward_test/trades/`: CSV files detailing the trades made during each forward test.
*   `results/analysis/`: Contains aggregated summaries, plots, and reports generated by `analyze-trades` and `analyze-details`.

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