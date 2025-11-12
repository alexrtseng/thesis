from __future__ import annotations

import os

import torch


def configure_fp32_precision(default: str = "high") -> None:
    """Configure FP32 matmul/conv precision using new PyTorch APIs when available.

    Maps legacy levels to new settings:
      - "high"/"tf32"   -> tf32 (fast, lower precision)
      - "highest"/"ieee" -> ieee (strict FP32)
      - "medium"         -> tf32 (closest equivalent for speed)

    Honors env var TORCH_MATMUL_PRECISION if set; otherwise uses `default`.
    Falls back to torch.set_float32_matmul_precision on older versions.
    """
    desired = os.getenv("TORCH_MATMUL_PRECISION", default).lower()

    # Normalize aliases
    if desired in {"tf32", "high"}:
        matmul_setting = "tf32"
        conv_setting = "tf32"
        legacy_setting = "high"
    elif desired in {"ieee", "highest"}:
        matmul_setting = "ieee"
        conv_setting = "ieee"
        legacy_setting = "highest"
    elif desired in {"medium"}:
        # No direct analog in the new API; prefer speed with tf32
        matmul_setting = "tf32"
        conv_setting = "tf32"
        legacy_setting = "medium"
    else:
        # Safe fallback
        matmul_setting = "ieee"
        conv_setting = "ieee"
        legacy_setting = "highest"

    # Try new APIs first
    try:
        # Matmul precision (CUDA)
        if (
            hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "matmul")
            and hasattr(torch.backends.cuda.matmul, "fp32_precision")
        ):
            torch.backends.cuda.matmul.fp32_precision = matmul_setting  # type: ignore[attr-defined]

        # cuDNN conv precision
        if (
            hasattr(torch.backends, "cudnn")
            and hasattr(torch.backends.cudnn, "conv")
            and hasattr(torch.backends.cudnn.conv, "fp32_precision")
        ):
            torch.backends.cudnn.conv.fp32_precision = conv_setting  # type: ignore[attr-defined]

        return
    except Exception:
        # Fall through to legacy below
        pass

    # Legacy fallback to keep behavior consistent on older versions
    try:
        torch.set_float32_matmul_precision(legacy_setting)  # type: ignore[attr-defined]
    except Exception:
        pass
