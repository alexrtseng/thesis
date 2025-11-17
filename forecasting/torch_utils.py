from __future__ import annotations

import os

import torch


def configure_fp32_precision(default: str = "high") -> None:
    """
    Configure TF32/FP32 behavior using the new PyTorch API:

      - torch.backends.cuda.matmul.fp32_precision = {'tf32'|'ieee'}
      - torch.backends.cudnn.conv.fp32_precision   = {'tf32'|'ieee'}

    Input mapping (backwards compat with older envs):
      - 'high' / 'medium' / 'tf32'  -> TF32 (faster)
      - 'highest' / 'ieee' / '32'   -> strict IEEE FP32

    Respects env TORCH_MATMUL_PRECISION if set; otherwise uses `default`.
    No-ops safely on CPU/MPS or if attributes are missing.
    """
    raw = (os.getenv("TORCH_MATMUL_PRECISION", default) or "").strip().lower()

    if raw in {"high", "medium", "tf32"}:
        mm_prec = "tf32"
        conv_prec = "tf32"
    elif raw in {"highest", "ieee", "32", "fp32"}:
        mm_prec = "ieee"
        conv_prec = "ieee"
    else:
        mm_prec = "tf32"
        conv_prec = "tf32"

    try:
        matmul = getattr(torch.backends.cuda, "matmul", None)
        if matmul is not None and hasattr(matmul, "fp32_precision"):
            matmul.fp32_precision = mm_prec
    except Exception:
        print("Could not set CUDA matmul fp32_precision")
        pass

    try:
        conv = getattr(torch.backends.cudnn, "conv", None)
        if conv is not None and hasattr(conv, "fp32_precision"):
            conv.fp32_precision = conv_prec
    except Exception:
        print("Could not set cuDNN conv fp32_precision")
        pass

    # Do NOT call torch.set_float32_matmul_precision() here (legacy).
