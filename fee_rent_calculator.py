"""
WAX Blockchain Powerup Calculator
Production-ready implementation for calculating powerup fees and building transactions.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, getcontext
from enum import Enum
from typing import Dict, Optional, Any, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from utils.custom_logging import configure_color_logging

# Set decimal precision for financial calculations
getcontext().prec = 28


class PowerupError(Exception):
    """Base exception for powerup-related errors."""

    pass


class ValidationError(PowerupError):
    """Raised when input validation fails."""

    pass


class APIError(PowerupError):
    """Raised when API communication fails."""

    pass


class CalculationError(PowerupError):
    """Raised when calculation errors occur."""

    pass


class ResourceType(Enum):
    """Resource types for powerup."""

    CPU = "cpu"
    NET = "net"


@dataclass
class ResourceState:
    """State of a powerup resource (CPU or NET)."""

    version: int
    weight: Decimal
    weight_ratio: Decimal
    assumed_stake_weight: Decimal
    initial_weight_ratio: Decimal
    target_weight_ratio: Decimal
    initial_timestamp: str
    target_timestamp: str
    exponent: Decimal
    decay_secs: int
    min_price: Decimal
    max_price: Decimal
    utilization: Decimal
    adjusted_utilization: Decimal
    utilization_timestamp: str

    def __post_init__(self):
        """Convert string values to Decimal for precise calculations."""
        decimal_fields = [
            "weight",
            "weight_ratio",
            "assumed_stake_weight",
            "initial_weight_ratio",
            "target_weight_ratio",
            "exponent",
            "min_price",
            "max_price",
            "utilization",
            "adjusted_utilization",
        ]
        for field_name in decimal_fields:
            value = getattr(self, field_name)

            # Preserve None (or empty string) – the API may omit a field.
            if value is None or (isinstance(value, str) and value.strip() == ""):
                setattr(self, field_name, None)
                continue

            # If it’s already a Decimal we’re done.
            if isinstance(value, Decimal):
                continue

            # ---- NEW: очистка строки от токен‑суффикса ----
            if isinstance(value, str):
                # Удаляем всё, что не является частью числа (например " WAX").
                # Оставляем только первую часть, разделённую пробелом.
                # Также убираем запятые, если они встречаются.
                cleaned = value.split()[0].replace(",", "")
                value = cleaned
            # --------------------------------------------

            # Try to coerce to Decimal; give a clear error if it fails.
            try:
                setattr(self, field_name, Decimal(str(value)))
            except Exception as exc:
                raise ValidationError(
                    f"Invalid numeric value for '{field_name}': {value!r}"
                ) from exc


@dataclass
class PowerupState:
    """Complete powerup state containing CPU and NET resources."""

    cpu: ResourceState
    net: ResourceState
    powerup_days: int = 1
    min_powerup_fee: Optional[Decimal] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PowerupState":
        """Create PowerupState from dictionary."""
        cpu_data = data.get("cpu", {})
        net_data = data.get("net", {})

        # ---- NEW: clean and convert min_powerup_fee safely ----
        raw_fee = data.get("min_powerup_fee")
        if raw_fee is None:
            min_fee = None
        else:
            # If the fee comes as a string with a token suffix (e.g. "0.001 WAX")
            # or contains commas, strip those parts before converting.
            if isinstance(raw_fee, str):
                cleaned = raw_fee.split()[0].replace(",", "")
                min_fee = Decimal(cleaned)
            else:
                # Fallback for numeric types (int, float, Decimal, etc.)
                min_fee = Decimal(str(raw_fee))

        return cls(
            cpu=ResourceState(**cpu_data),
            net=ResourceState(**net_data),
            powerup_days=data.get("powerup_days", 1),
            min_powerup_fee=min_fee,
        )


@dataclass
class PowerupParameters:
    """Calculated powerup parameters."""

    cpu_fee: Decimal
    net_fee: Decimal
    total_fee: Decimal
    cpu_frac: int
    net_frac: int
    max_payment: Decimal
    days: int


@dataclass
class PowerupTransaction:
    """Complete powerup transaction data."""

    transaction: Dict[str, Any]
    parameters: PowerupParameters
    action_data: Dict[str, Any]


class PowerupCalculator:
    """Calculator for WAX blockchain powerup fees and transactions."""

    POWERUP_FRAC = Decimal("1000000000000000")
    DEFAULT_PRECISION = 8
    MAX_PAYMENT_BUFFER = Decimal("0.1")  # 10% buffer for max payment

    def __init__(self, precision: int = DEFAULT_PRECISION):
        """
        Initialize the PowerupCalculator.

        Args:
            precision: Token precision (default: 8 for WAX)
        """
        self.precision = precision
        self.precision_multiplier = Decimal(10) ** precision

    def price_function(self, state: ResourceState, utilization: Decimal) -> Decimal:
        """
        Calculate price based on utilization.

        Args:
            state: Resource state
            utilization: Current utilization value

        Returns:
            Calculated price
        """
        try:
            price = state.min_price
            new_exponent = state.exponent - Decimal("1.0")

            if new_exponent <= 0:
                return state.max_price

            utilization_ratio = utilization / state.weight
            price_range = state.max_price - state.min_price
            price += price_range * (utilization_ratio**new_exponent)

            return price
        except Exception as e:
            raise CalculationError(f"Error in price calculation: {e}")

    def price_integral_delta(
        self, state: ResourceState, start_utilization: Decimal, end_utilization: Decimal
    ) -> Decimal:
        """
        Calculate the integral of the price function between two utilization points.

        Args:
            state: Resource state
            start_utilization: Starting utilization
            end_utilization: Ending utilization

        Returns:
            Integral value
        """
        try:
            coefficient = (state.max_price - state.min_price) / state.exponent
            start_u = start_utilization / state.weight
            end_u = end_utilization / state.weight

            linear_term = state.min_price * (end_u - start_u)
            exponential_term = coefficient * (
                (end_u**state.exponent) - (start_u**state.exponent)
            )

            return linear_term + exponential_term
        except Exception as e:
            raise CalculationError(f"Error in integral calculation: {e}")

    def calculate_fee(
        self,
        state: ResourceState,
        utilization_increase: Decimal,
        min_fee: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate fee for a given utilization increase.

        Args:
            state: Resource state
            utilization_increase: Amount of utilization increase
            min_fee: Minimum fee threshold

        Returns:
            Calculated fee
        """
        try:
            fee = Decimal("0")
            start_utilization = state.utilization
            end_utilization = start_utilization + utilization_increase

            # Handle adjusted utilization
            if start_utilization < state.adjusted_utilization:
                adjustment_increase = min(
                    utilization_increase, state.adjusted_utilization - start_utilization
                )
                fee += (
                    self.price_function(state, state.adjusted_utilization)
                    * adjustment_increase
                ) / state.weight
                start_utilization = state.adjusted_utilization

            # Calculate remaining fee
            if start_utilization < end_utilization:
                fee += self.price_integral_delta(
                    state, start_utilization, end_utilization
                )

            # Apply minimum fee if specified
            if min_fee and fee > 0 and fee < min_fee:
                fee = min_fee

            return fee
        except Exception as e:
            raise CalculationError(f"Error calculating fee: {e}")

    def calculate_parameters(
        self, powerup_state: PowerupState, cpu_amount: Decimal, net_amount: Decimal
    ) -> PowerupParameters:
        """
        Calculate all powerup parameters.

        Args:
            powerup_state: Current powerup state
            cpu_amount: CPU amount to powerup (in WAX)
            net_amount: NET amount to powerup (in WAX)

        Returns:
            Calculated powerup parameters
        """
        try:
            # Validate inputs
            self._validate_amounts(cpu_amount, net_amount)

            # Calculate CPU fee
            cpu_utilization = cpu_amount * self.precision_multiplier
            cpu_fee = self.calculate_fee(
                powerup_state.cpu, cpu_utilization, powerup_state.min_powerup_fee
            )
            cpu_fee = self._round_fee(cpu_fee)

            # Calculate NET fee
            net_utilization = net_amount * self.precision_multiplier
            net_fee = self.calculate_fee(
                powerup_state.net, net_utilization, powerup_state.min_powerup_fee
            )
            net_fee = self._round_fee(net_fee)

            # Total fee
            total_fee = self._round_fee(cpu_fee + net_fee)

            # Calculate fractions
            cpu_frac = int(
                (
                    cpu_utilization * self.POWERUP_FRAC / powerup_state.cpu.weight
                ).quantize(Decimal("1"))
            )
            net_frac = int(
                (
                    net_utilization * self.POWERUP_FRAC / powerup_state.net.weight
                ).quantize(Decimal("1"))
            )

            # Max payment with buffer
            max_payment = self._round_fee(
                total_fee * (Decimal("1") + self.MAX_PAYMENT_BUFFER)
            )

            return PowerupParameters(
                cpu_fee=cpu_fee,
                net_fee=net_fee,
                total_fee=total_fee,
                cpu_frac=cpu_frac,
                net_frac=net_frac,
                max_payment=max_payment,
                days=powerup_state.powerup_days,
            )
        except Exception as e:
            raise CalculationError(f"Error calculating parameters: {e}")

    def build_transaction(
        self,
        payer: str,
        receiver: str,
        cpu_amount: Decimal,
        net_amount: Decimal,
        powerup_state: PowerupState,
    ) -> PowerupTransaction:
        """
        Build a complete powerup transaction.

        Args:
            payer: Account paying for powerup
            receiver: Account receiving powerup
            cpu_amount: CPU amount (in WAX)
            net_amount: NET amount (in WAX)
            powerup_state: Current powerup state

        Returns:
            Complete transaction data
        """
        try:
            # Validate account names
            self._validate_account_name(payer)
            self._validate_account_name(receiver)

            # Calculate parameters
            params = self.calculate_parameters(powerup_state, cpu_amount, net_amount)

            # Build action data
            action_data = {
                "payer": payer,
                "receiver": receiver,
                "days": params.days,
                "cpu_frac": str(params.cpu_frac),
                "net_frac": str(params.net_frac),
                "max_payment": f"{params.max_payment:.{self.precision}f} WAX",
            }

            # Build transaction
            transaction = {
                "actions": [
                    {
                        "account": "eosio",
                        "name": "powerup",
                        "authorization": [{"actor": payer, "permission": "active"}],
                        "data": action_data,
                    }
                ]
            }

            return PowerupTransaction(
                transaction=transaction, parameters=params, action_data=action_data
            )
        except Exception as e:
            raise PowerupError(f"Error building transaction: {e}")

    def _validate_amounts(self, cpu_amount: Decimal, net_amount: Decimal) -> None:
        """Validate CPU and NET amounts."""
        if cpu_amount < 0 or net_amount < 0:
            raise ValidationError("Amounts must be non-negative")
        if cpu_amount == 0 and net_amount == 0:
            raise ValidationError("At least one amount must be greater than zero")

    def _validate_account_name(self, account: str) -> None:
        """Validate EOS account name format."""
        if not account or len(account) > 12:
            raise ValidationError(f"Invalid account name: {account}")
        if not all(c in "abcdefghijklmnopqrstuvwxyz12345." for c in account):
            raise ValidationError(f"Invalid account name characters: {account}")

    def _round_fee(self, fee: Decimal) -> Decimal:
        """Round fee to the specified precision."""
        quantizer = Decimal(f'0.{"0" * self.precision}')
        return fee.quantize(quantizer, rounding=ROUND_HALF_UP)


class PowerupAPIClient:
    """Client for interacting with WAX blockchain API."""

    DEFAULT_ENDPOINT = "https://wax.greymass.com"
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3

    def __init__(self, endpoint: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
        """
        Initialize API client.

        Args:
            endpoint: API endpoint URL
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint or os.getenv("WAX_API_ENDPOINT", self.DEFAULT_ENDPOINT)
        self.timeout = timeout
        self.session = self._create_session()

        # Validate endpoint
        self._validate_endpoint()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        retry = Retry(
            total=self.MAX_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _validate_endpoint(self) -> None:
        """Validate the API endpoint URL."""
        try:
            result = urlparse(self.endpoint)
            if not all([result.scheme, result.netloc]):
                raise ValidationError(f"Invalid API endpoint: {self.endpoint}")
        except Exception as e:
            raise ValidationError(f"Invalid API endpoint: {e}")

    def get_powerup_state(self) -> PowerupState:
        """
        Fetch current powerup state from blockchain.

        Returns:
            Current powerup state

        Raises:
            APIError: If API request fails
        """
        try:
            response = self.session.post(
                f"{self.endpoint}/v1/chain/get_table_rows",
                json={
                    "code": "eosio",
                    "scope": "",
                    "table": "powup.state",
                    "limit": 1,
                    "json": True,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            if not data.get("rows"):
                raise APIError("No powerup state found in response")

            state_data = data["rows"][0]
            return PowerupState.from_dict(state_data)

        except requests.RequestException as e:
            log.error(f"API request failed: {e}")
            raise APIError(f"Failed to fetch powerup state: {e}")
        except (KeyError, IndexError, ValueError) as e:
            log.error(f"Failed to parse API response: {e}")
            raise APIError(f"Invalid API response format: {e}")

    def get_account_resources(self, account: str) -> Dict[str, Any]:
        """
        Get account resource usage.

        Args:
            account: Account name

        Returns:
            Account resource data
        """
        try:
            response = self.session.post(
                f"{self.endpoint}/v1/chain/get_account",
                json={"account_name": account},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            log.error(f"Failed to get account resources: {e}")
            raise APIError(f"Failed to get account resources: {e}")


def format_wax_amount(amount: Decimal, precision: int = 8) -> str:
    """Format amount as WAX string."""
    return f"{amount:.{precision}f} WAX"


def main():
    """Example usage of the PowerupCalculator."""
    try:
        # Initialize components
        calculator = PowerupCalculator()
        client = PowerupAPIClient()

        # Get current state from blockchain
        log.info("Fetching current powerup state...")
        powerup_state = client.get_powerup_state()

        # User parameters
        payer = "getdropsfast"
        receiver = "getdropsfast"
        cpu_amount = Decimal("4000000")  # 4M WAX for CPU
        net_amount = Decimal("2000")  # 2K WAX for NET

        # Build transaction
        log.info("Building powerup transaction...")
        result = calculator.build_transaction(
            payer=payer,
            receiver=receiver,
            cpu_amount=cpu_amount,
            net_amount=net_amount,
            powerup_state=powerup_state,
        )

        # Display results
        print("\n=== Powerup Transaction Details ===")
        print(f"Payer: {payer}")
        print(f"Receiver: {receiver}")
        print(f"\n=== Resource Amounts ===")
        print(f"CPU: {cpu_amount:,.0f} WAX")
        print(f"NET: {net_amount:,.0f} WAX")
        print(f"\n=== Calculated Fees ===")
        print(f"CPU Fee: {result.parameters.cpu_fee:.8f} WAX")
        print(f"NET Fee: {result.parameters.net_fee:.8f} WAX")
        print(f"Total Fee: {result.parameters.total_fee:.8f} WAX")
        print(f"\n=== Transaction Parameters ===")
        print(f"CPU Fraction: {result.parameters.cpu_frac:,}")
        print(f"NET Fraction: {result.parameters.net_frac:,}")
        print(f"Max Payment: {result.parameters.max_payment:.8f} WAX")
        print(f"Duration: {result.parameters.days} day(s)")
        print(f"\n=== Transaction JSON ===")
        print(json.dumps(result.transaction, indent=2))

    except PowerupError as e:
        log.error(f"Powerup error: {e}")
        print(f"Error: {e}")
        return 1
    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Configure logging
    # logging.basicConfig(
    #     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )
    # logger = logging.getLogger(__name__)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    configure_color_logging()
    exit(main())
