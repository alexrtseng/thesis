from __future__ import annotations

import os

import torch


def configure_fp32_precision(default: str = "high") -> None:
    """Configure matmul precision using the unified modern PyTorch API.

    Accepts env var TORCH_MATMUL_PRECISION overriding `default`.
    Supported canonical values for torch.set_float32_matmul_precision:
      - "high"   (fast, uses TF32 on Ampere+)
      - "medium" (intermediate; potentially reduced convergence variance)
      - "highest" (full IEEE FP32)

    Aliases handled:
      tf32 -> high
      ieee -> highest

    We deliberately avoid mixing legacy backend attribute mutation with the new API
    to prevent the RuntimeError about mixed precision configuration.
    """
    raw = os.getenv("TORCH_MATMUL_PRECISION", default).strip().lower()
    if raw in {"tf32"}:
        precision = "high"
    elif raw in {"ieee"}:
        precision = "highest"
    elif raw in {"high", "medium", "highest"}:
        precision = raw
    else:
        precision = "high"  # safe default favoring speed

    try:
        torch.set_float32_matmul_precision(precision)  # PyTorch >=1.12 / 2.x
    except Exception:
        # Older versions: silently ignore; TF32 may be controlled by allow_tf32 flags.
        pass
