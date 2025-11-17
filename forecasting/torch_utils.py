from __future__ import annotations

import os

import torch


def configure_fp32_precision(default: str = "high") -> None:
    """
    Configure TF32/FP32 behavior using the new PyTorch API:

      - torch.backends.cuda.matmul.fp32_precision = {'tf32'|'ieee'}
      - torch.backends.cudnn.conv.fp32_precision   = {'tf32'|'ieee'}

    Input mapping (backwards compat with older envs):
      - 'high' / 'medium' / 'tf32'  -> use TF32 (fast on Ampere+)
      - 'highest' / 'ieee' / '32'   -> strict IEEE FP32

    Respects env TORCH_MATMUL_PRECISION if set; otherwise uses `default`.
    No-ops safely on CPU/MPS or older PyTorch without these attributes.
    """
    raw = (os.getenv("TORCH_MATMUL_PRECISION", default) or "").strip().lower()

    # Normalize to new-backend choices
    if raw in {"high", "medium", "tf32"}:
        mm_prec = "tf32"
        conv_prec = "tf32"
    elif raw in {"highest", "ieee", "32", "fp32"}:
        mm_prec = "ieee"
        conv_prec = "ieee"
    else:
        # Favor speed unless explicitly overridden
        mm_prec = "tf32"
        conv_prec = "tf32"

    # Apply to CUDA backends if available; guard everything so CPU/MPS won’t complain.
    try:
        matmul = getattr(torch.backends.cuda, "matmul", None)
        if matmul is not None and hasattr(matmul, "fp32_precision"):
            matmul.fp32_precision = mm_prec
    except Exception:
        pass

    try:
        conv = getattr(torch.backends.cudnn, "conv", None)
        if conv is not None and hasattr(conv, "fp32_precision"):
            conv.fp32_precision = conv_prec
    except Exception:
        pass

    # IMPORTANT: do not call torch.set_float32_matmul_precision() here,
    # as it now emits a deprecation-style warning about legacy APIs.
