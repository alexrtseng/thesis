"""Deterministic battery optimization package.

Re-exports common types for convenience:
        - BatteryParams: dataclass of battery parameters
        - DEFAULT_BATTERY: a default BatteryParams instance
"""

from .single_market_battery import DEFAULT_BATTERY, BatteryParams

__all__ = ["BatteryParams", "DEFAULT_BATTERY"]

# Reference to mark imports as used for linters
_exported = (BatteryParams, DEFAULT_BATTERY)
