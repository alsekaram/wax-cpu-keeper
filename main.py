from decimal import Decimal
from eosapi import EosApi
from eosapi.exceptions import TransactionException
from dotenv import load_dotenv
import os
import logging
import time
from typing import Optional, Dict, Any

from fee_rent_calculator import PowerupCalculator, PowerupState
from utils.custom_logging import configure_color_logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)  # Production level
configure_color_logging()

load_dotenv()


class WaxKeeperError(Exception):
    """Base exception for powerup-related errors."""

    pass


class ValidationError(WaxKeeperError):
    """Raised when input validation fails."""

    pass


class TransactionError(WaxKeeperError):
    """Raised when transaction fails."""

    pass


class WaxCpuKeeper:
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds
    CPU_EXCEEDED_WAIT = 30  # seconds to wait when CPU is exceeded

    def __init__(
        self,
        endpoint: str,
        account: str,
        payer: str,
        private_key: str,
        target_cpu_stake: int,
        target_net_stake: int,
    ):
        self.wax_api = EosApi(rpc_host=endpoint)
        self.account = account
        self.payer = payer
        self.private_key = private_key
        self.target_cpu_stake = target_cpu_stake
        self.target_net_stake = target_net_stake
        self.powerup_calculator = PowerupCalculator()
        self._initialized = False

    def initialize(self) -> None:
        """Initialize API with private key."""
        try:
            self.wax_api.import_key(self.payer, self.private_key, "active")
            self._initialized = True
            log.info("WAX API initialized successfully for payer: %s", self.payer)
        except Exception as e:
            log.error("Failed to initialize WAX API: %s", e)
            raise WaxKeeperError(f"Initialization failed: {e}")

    def get_account(self, account_name: str) -> Optional[Dict[str, Any]]:
        """Get account information from blockchain."""
        try:
            url = self.wax_api._build_url("get_account")
            post_data = {"account_name": account_name}
            resp = self.wax_api._post(url, post_data)
            return resp.json()
        except Exception as e:
            log.error("Failed to get account data for %s: %s", account_name, e)
            return None

    def get_cpu_info(self) -> Optional[int]:
        """Get current CPU weight for the account."""
        try:
            account_info = self.get_account(self.account)
            if account_info is None:
                return None

            if "cpu_weight" not in account_info:
                log.warning("cpu_weight field missing in node response")
                return None

            cpu_weight = account_info["cpu_weight"]
            return int(cpu_weight)
        except (ValueError, TypeError) as e:
            log.error("Invalid cpu_weight format: %s", cpu_weight)
            return None
        except Exception as e:
            log.error("Error getting CPU info: %s", e)
            return None

    def is_stake_needed(self) -> Optional[bool]:
        """Check if stake is needed based on current CPU weight."""
        cpu_info = self.get_cpu_info()
        if cpu_info is None:
            return None

        cpu_info = cpu_info / 1e8  # Теперь безопасно делить
        needs_stake = (cpu_info + 1) < self.target_cpu_stake
        log.info(
            "Current CPU: %.4f, Target: %d, Stake needed: %s",  # ✅ %.4f для float
            cpu_info,
            self.target_cpu_stake,
            needs_stake,
        )
        return needs_stake

    def get_powerup_state(self) -> Optional[PowerupState]:
        """Get current powerup state from blockchain."""
        try:
            payload = {
                "code": "eosio",
                "scope": "",
                "table": "powup.state",
                "limit": 1,
                "json": True,
            }
            data = self.wax_api.get_table_rows(payload)

            if not data or not data.get("rows"):
                raise ValueError("No powerup state data available")

            state_data = data["rows"][0]
            return PowerupState.from_dict(state_data)
        except Exception as e:
            log.error("Error getting powerup state: %s", e)
            return None

    def build_transaction(
        self,
        payer: str,
        receiver: str,
        cpu_amount: int,
        net_amount: int,
        powerup_state: PowerupState,
    ) -> Optional[Dict[str, Any]]:
        """Build powerup transaction."""
        try:
            self._validate_account_name(payer)
            self._validate_account_name(receiver)

            params = self.powerup_calculator.calculate_parameters(
                powerup_state,
                cpu_amount,
                net_amount,
            )

            action_data = {
                "payer": payer,
                "receiver": receiver,
                "days": params.days,
                "cpu_frac": str(params.cpu_frac),
                "net_frac": str(params.net_frac),
                "max_payment": f"{params.max_payment:.{self.powerup_calculator.precision}f} WAX",
            }

            actions = {
                "account": "eosio",
                "name": "powerup",
                "authorization": [
                    {
                        "actor": self.payer,
                        "permission": "active",
                    },
                ],
                "data": action_data,
            }

            return actions
        except Exception as e:
            log.error("Error building transaction: %s", e)
            return None

    def _execute_transaction(self, actions: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transaction with error handling."""
        trx = {"actions": [actions]}

        try:
            transaction = self.wax_api.make_transaction(trx)
            result = self.wax_api.push_transaction(transaction)
            return result
        except TransactionException as e:
            error_str = str(e)

            # Check for specific error types
            if "tx_cpu_usage_exceeded" in error_str:
                raise TransactionError(f"CPU usage exceeded, need to wait: {e}")
            elif "insufficient_ram" in error_str:
                raise TransactionError(f"Insufficient RAM: {e}")
            elif "expired_tx_exception" in error_str:
                raise TransactionError(f"Transaction expired: {e}")
            else:
                raise TransactionError(f"Transaction failed: {e}")

    def powerup_cpu(self) -> bool:
        """Execute powerup transaction with retry logic."""
        if not self._initialized:
            raise WaxKeeperError(
                "WaxCpuKeeper not initialized. Call initialize() first."
            )

        # Check if stake is actually needed
        stake_needed = self.is_stake_needed()
        if stake_needed is False:
            log.info("Stake not needed, current CPU is sufficient")
            return True
        elif stake_needed is None:
            log.error("Could not determine if stake is needed")
            return False

        # Get powerup state
        powerup_state = self.get_powerup_state()
        if powerup_state is None:
            log.error("Could not get powerup state")
            return False

        # Build transaction
        actions = self.build_transaction(
            payer=self.payer,
            receiver=self.account,
            cpu_amount=self.target_cpu_stake,
            net_amount=self.target_net_stake,
            powerup_state=powerup_state,
        )

        if actions is None:
            log.error("Could not build transaction")
            return False

        # Execute with retries
        for attempt in range(self.MAX_RETRIES):
            try:
                log.info(
                    "Attempting powerup transaction (attempt %d/%d)",
                    attempt + 1,
                    self.MAX_RETRIES,
                )

                result = self._execute_transaction(actions)

                log.info(
                    "Transaction successful: %s",
                    result.get("transaction_id", "unknown"),
                )
                return True

            except TransactionError as e:
                error_msg = str(e)

                if "CPU usage exceeded" in error_msg:
                    wait_time = self.CPU_EXCEEDED_WAIT * (attempt + 1)
                    log.warning(
                        "CPU exceeded, waiting %d seconds before retry", wait_time
                    )
                    time.sleep(wait_time)

                elif "insufficient_ram" in error_msg:
                    log.error("Insufficient RAM for transaction, cannot retry")
                    return False

                elif "expired" in error_msg:
                    log.warning("Transaction expired, retrying immediately")
                    continue

                else:
                    if attempt < self.MAX_RETRIES - 1:
                        log.warning(
                            "Transaction failed: %s, retrying in %d seconds",
                            e,
                            self.RETRY_DELAY,
                        )
                        time.sleep(self.RETRY_DELAY)
                    else:
                        log.error(
                            "Transaction failed after %d attempts: %s",
                            self.MAX_RETRIES,
                            e,
                        )
                        return False

            except Exception as e:
                log.error("Unexpected error during transaction: %s", e)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                else:
                    return False

        return False

    def monitor_and_maintain(self, check_interval: int = 300) -> None:
        """Monitor CPU levels and maintain stake as needed.

        Args:
            check_interval: Seconds between checks (default 5 minutes)
        """
        log.info("Starting CPU monitoring with %d second intervals", check_interval)

        while True:
            try:
                stake_needed = self.is_stake_needed()

                if stake_needed is True:
                    log.info("Stake needed, initiating powerup")
                    success = self.powerup_cpu()

                    if success:
                        log.info("Powerup successful")
                    else:
                        log.error("Powerup failed, will retry on next cycle")

                elif stake_needed is False:
                    log.debug("CPU levels adequate, no action needed")

                else:
                    log.warning("Could not determine stake status")

            except Exception as e:
                log.error("Error in monitoring loop: %s", e)

            time.sleep(check_interval)

    def _validate_account_name(self, account: str) -> None:
        """Validate EOS account name format."""
        if not account or len(account) > 12:
            raise ValidationError(f"Invalid account name length: {account}")

        valid_chars = set("abcdefghijklmnopqrstuvwxyz12345.")
        if not all(c in valid_chars for c in account):
            raise ValidationError(f"Invalid account name characters: {account}")


def main():
    """Main entry point for the application."""
    # Load configuration
    endpoint_url = os.getenv("WAX_API_ENDPOINT")
    receiver = os.getenv("RECEIVER_NAME")
    payer = os.getenv("PAYER_NAME")  # Fixed typo
    private_key = os.getenv("PAYER_PRIVATE_KEY")  # Fixed typo

    # Validate required environment variables
    required_vars = {
        "WAX_API_ENDPOINT": endpoint_url,
        "RECEIVER_NAME": receiver,
        "PAYER_NAME": payer,
        "PAYER_PRIVATE_KEY": private_key,
    }

    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        log.error("Missing required environment variables: %s", missing_vars)
        return 1

    try:
        cpu_amount = int(os.getenv("TARGET_CPU_STAKE", "0"))
        net_amount = int(os.getenv("TARGET_NET_STAKE", "0"))
    except ValueError as e:
        log.error("Invalid stake amounts in environment variables: %s", e)
        return 1

    if cpu_amount <= 0 and net_amount <= 0:
        log.error("At least one of TARGET_CPU_STAKE or TARGET_NET_STAKE must be > 0")
        return 1

    log.info("Starting WAX CPU Keeper")
    log.info("Receiver: %s, Payer: %s", receiver, payer)
    log.info("Target CPU: %d, Target NET: %d", cpu_amount, net_amount)

    # Initialize keeper
    keeper = WaxCpuKeeper(
        endpoint_url, receiver, payer, private_key, cpu_amount, net_amount
    )

    try:
        # Initialize API
        keeper.initialize()

        # Check if one-time execution or monitoring mode
        monitor_mode = os.getenv("MONITOR_MODE", "false").lower() == "true"

        if monitor_mode:
            check_interval = int(os.getenv("CHECK_INTERVAL", "300"))
            keeper.monitor_and_maintain(check_interval)
        else:
            # One-time execution
            success = keeper.powerup_cpu()
            return 0 if success else 1

    except KeyboardInterrupt:
        log.info("Shutdown requested by user")
        return 0
    except Exception as e:
        log.error("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    exit(main())
