# WAX CPU Keeper

The WAX CPU Keeper is a Python-based automation tool designed to monitor and manage the CPU resources of a WAX blockchain account. It prevents transaction failures due to insufficient CPU by automatically powering up the account when its CPU usage exceeds a predefined threshold. This tool is ideal for developers and users who need to ensure their WAX account remains operational without manual intervention.

The keeper can be configured to run in two modes:
1.  **One-Time Execution**: Performs a single check and powers up the CPU if necessary.
2.  **Monitoring Mode**: Continuously monitors the account's CPU usage at regular intervals and powers up as needed.

## Key Features

*   **Automated CPU Management**: Proactively monitors and maintains WAX account CPU & NET levels.
*   **Intelligent Power-Up Logic**: Calculates the required CPU and NET stake based on current network conditions and your target levels.
*   **Flexible Configuration**: All settings, including account names, private keys, and API endpoints, are managed via environment variables for security and ease of use.
*   **Resilient Error Handling**: Implements retry logic with exponential back-off for handling temporary network or transaction failures.
*   **Configurable Retry Parameters**: Tune `MAX_RETRIES`, `RETRY_DELAY` and `CPU_EXCEEDED_WAIT` without touching the code.
*   **Advanced Fee Calculation**: Includes a sophisticated fee calculator (`fee_rent_calculator.py`) that determines the precise cost of a power-up transaction based on the current state of the WAX power-up market.
*   **Informative Logging**: Provides detailed, colour-coded logs for clear visibility into the keeper’s operations.

## How It Works

The WAX CPU Keeper periodically checks the target account’s CPU (`cpu_weight`) **and NET (`net_weight`)**.  
If either value falls below the configured `TARGET_CPU_STAKE` **or** `TARGET_NET_STAKE`, the keeper initiates a power-up transaction.

The core logic is handled by the `WaxCpuKeeper` class in `main.py`, which orchestrates the following steps:
1.  **Initialization**: The keeper loads configuration from environment variables and initializes the WAX API connection.
2.  **CPU Check**: It fetches the current CPU state of the `RECEIVER_NAME` account.
3.  **Power-Up Calculation**: If a power-up is needed, it uses the `PowerupCalculator` to determine the transaction parameters, including the required fee to reach the configured CPU / NET stake targets.
4.  **Transaction Execution**: It builds and executes a `powerup` transaction using the `PAYER_NAME` account to pay the fees.
5.  **Monitoring**: In monitoring mode, the process repeats at a specified interval (`CHECK_INTERVAL`).

## Getting Started

### Prerequisites

*   Python 3.8+
*   A WAX account to be monitored (`RECEIVER_NAME`)
*   A WAX account to pay for the power-up transactions (`PAYER_NAME`) with sufficient funds and its private key.

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/wax-cpu-keeper.git
    cd wax-cpu-keeper
    ```

2.  **Install dependencies**:

   If you don’t have **uv** yet, install it first:

   ```bash
   # Option A: install globally
   pip install uv
   # Option B: isolated install
   pipx install uv
   ```

   Then sync project dependencies:

   ```bash
   uv sync pyproject.toml
   ```

### Configuration

Copy the `env.example` file to a new file named `.env` and add your specific values:

```env
# WAX API endpoint
WAX_API_ENDPOINT="https://wax.greymass.com"

# The account that will receive the CPU power-up
RECEIVER_NAME="your_receiver_account"

# The account that will pay for the power-up transaction
PAYER_NAME="your_payer_account"

# The active private key of the payer account
PAYER_PRIVATE_KEY="your_payer_private_key"

# Desired CPU stake in whole WAX tokens (e.g., 1000 => 1000.00000000 WAX)
TARGET_CPU_STAKE="1000"

# Desired NET stake in whole WAX tokens (e.g., 1)
TARGET_NET_STAKE="1"

# Set to "true" to run in continuous monitoring mode
MONITOR_MODE="false"

# Time in seconds between checks in monitoring mode (default: 300)
CHECK_INTERVAL="300"
```

**Security Note**: Never commit your `.env` file to version control. Ensure it is listed in your `.gitignore` file.

### Usage

To run the WAX CPU Keeper, execute the `main.py` script from your terminal:

```bash
python main.py
```

*   If `MONITOR_MODE` is set to `"false"` (the default), the script will perform a one-time check and exit.
*   If `MONITOR_MODE` is set to `"true"`, the script will run continuously, checking the CPU at the interval specified by `CHECK_INTERVAL`.

### Run in one line

```bash
# Monitoring mode every 2 minutes without редактирования .env
MONITOR_MODE=true CHECK_INTERVAL=120 python main.py
```

### Environment variables reference

| Variable | Default | Description |
|----------|---------|-------------|
| `WAX_API_ENDPOINT` | — | RPC endpoint URL |
| `RECEIVER_NAME` | — | Account receiving resources |
| `PAYER_NAME` | — | Account paying the fee |
| `PAYER_PRIVATE_KEY` | — | Active private key for payer |
| `TARGET_CPU_STAKE` | — | Desired CPU stake in WAX |
| `TARGET_NET_STAKE` | — | Desired NET stake in WAX |
| `MONITOR_MODE` | `false` | Continuous monitoring if `true` |
| `CHECK_INTERVAL` | `300` | Seconds between checks |

## Scripts Overview

*   `main.py`: The main entry point of the application. It contains the `WaxCpuKeeper` class, which orchestrates the monitoring and power-up logic.
*   `fee_rent_calculator.py`: A sophisticated calculator for determining power-up fees and building transaction parameters. It can also be run standalone to calculate fees for given amounts.
*   `utils/custom_logging.py`: Configures custom, color-coded logging for improved readability.

## Troubleshooting

| Error | Likely Cause | Fix |
|-------|--------------|-----|
| `tx_cpu_usage_exceeded` | Network congestion or low CPU | Increase `TARGET_CPU_STAKE` or wait/retry |
| `insufficient_ram` | Not enough RAM on payer | Buy more RAM or lower stake amounts |
| `expired_tx_exception` | Slow network / RPC timeout | Switch RPC endpoint and retry |

## Disclaimer

This tool interacts directly with the WAX blockchain and executes transactions that have a real financial cost. Use it at your own risk. Ensure your configuration is correct and that you understand the implications of running this script. It is highly recommended to test on a WAX testnet before deploying to the mainnet.

