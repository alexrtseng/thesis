"""Convenient public API for forecasting package.

Import primary APIs directly:
    from forecasting import ModelName, make_registry, run_sweep_for_node
"""

from .model_zoo import (
    ModelCategory,
    ModelName,
    get_global_registry,
    get_statistical_registry,
    make_registry,
)
from .sweep_runner import run_sweep_for_node

__all__ = [
    "ModelName",
    "ModelCategory",
    "make_registry",
    "get_statistical_registry",
    "get_global_registry",
    "run_sweep_for_node",
]
