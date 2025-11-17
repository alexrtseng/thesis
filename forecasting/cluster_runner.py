"""Cluster parallel sweep runner.

Launch multiple model sweeps concurrently using Python multiprocessing.
Designed for HPC clusters where each process can be mapped to a node or GPU.

Usage (example):
    python -m forecasting.cluster_runner --pnode 2156113094 --models RNNModel,TCNModel --runs-per-model 5 --max-proc 4

Environment variables (override defaults):
    WANDB_API_KEY          : W&B authentication.
    TORCH_ACCELERATOR      : "cpu" | "gpu" | "mps" etc.
    TORCH_DEVICES          : integer number of devices per process (default 1).

Strategies:
 - Each process runs a sweep for a single model (count runs-per-model).
 - Staggered start to reduce I/O bursts.
 - Optional GPU assignment via CUDA_VISIBLE_DEVICES round-robin.
 - When --use-gpus is true, only GPU-capable models (ModelSpec.uses_gpu) are launched.
 - When --use-gpus is false, only CPU-only models (uses_gpu == False) are launched.

Notes:
 - W&B agents internally spawn runs; avoid too many processes * number of runs.
 - If dataset is large, consider subset_data_size to reduce per-run time.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from multiprocessing import Process, Queue
from typing import List

import pandas as pd
import torch

from forecasting.model_zoo import ModelName, make_registry
from forecasting.sweep_runner import build_series_for_node, run_sweep_for_node
from forecasting.torch_utils import configure_fp32_precision


def _parse_models(raw: str) -> List[ModelName]:
    reg = make_registry()
    token = raw.strip()
    if token.upper() in {"ALL", "*", ""}:
        return list(reg.keys())
    out: List[ModelName] = []
    for part in raw.split(","):
        t = part.strip().lower()
        if not t:
            continue
        matched = None
        for m in reg.keys():
            if m.value.lower() == t:
                matched = m
                break
        if not matched:
            raise ValueError(f"Unknown model name '{part}'")
        out.append(matched)
    return out


def _worker(
    model_name: ModelName,
    pnode_id: int,
    feature_df: pd.DataFrame,
    project: str,
    count: int,
    subset_data_size: float,
    gpu_id: int | None,
    delay_start: float,
    status_queue: Queue,
):
    try:
        if gpu_id is not None:
            # Isolate a single GPU per process and hint Lightning to use it
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            os.environ.setdefault("TORCH_ACCELERATOR", "gpu")
            os.environ.setdefault("TORCH_DEVICES", "1")
            # Quick diagnostics
            try:
                print(
                    f"[worker] assigned GPU={gpu_id} cuda_available={torch.cuda.is_available()} visible_count={torch.cuda.device_count()}"
                )
            except Exception:
                pass
        time.sleep(delay_start)
        status_queue.put((model_name.value, "starting"))
        run_sweep_for_node(
            model_name=model_name,
            pnode_id=pnode_id,
            feature_df=feature_df,
            project=project,
            count=count,
            subset_data_size=subset_data_size,
        )
        status_queue.put((model_name.value, "completed"))
    except Exception as e:
        status_queue.put((model_name.value, f"error: {e}"))


class GracefulTerminator:
    def __init__(self):
        self._terminate = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        self._terminate = True

    @property
    def terminated(self) -> bool:
        return self._terminate


def run_parallel(
    pnode_id: int,
    project: str,
    models: List[ModelName],
    runs_per_model: int,
    max_processes: int,
    subset_data_size: float,
    use_gpus: bool,
):
    feature_df = build_series_for_node(pnode_id)
    status_queue: Queue = Queue()
    # Simple GPU round robin
    available_gpus = []
    if use_gpus:
        n = torch.cuda.device_count()
        if n > 0:
            available_gpus = list(range(n))
        else:
            print(
                "[warn] --use-gpus requested, but no CUDA GPUs are visible. Falling back to CPU."
            )
            use_gpus = False

    terminator = GracefulTerminator()

    # Limit number of concurrent processes
    # Build queue of (pnode, model) tasks
    launch_queue = models.copy()
    active: List[Process] = []
    while launch_queue or active:
        # Launch new processes if slots available
        while launch_queue and len(active) < max_processes:
            model = launch_queue.pop(0)
            gpu_id = None
            if available_gpus:
                gpu_id = available_gpus[len(active) % len(available_gpus)]
            delay = random.uniform(0.5, 2.5)  # stagger start
            p = Process(
                target=_worker,
                args=(
                    model,
                    pnode_id,
                    feature_df,
                    project,
                    runs_per_model,
                    subset_data_size,
                    gpu_id,
                    delay,
                    status_queue,
                ),
                daemon=True,
            )
            p.start()
            active.append(p)
            print(
                f"[launch] pnode={pnode_id} model={model.value} pid={p.pid} gpu={gpu_id} delay={delay:.2f}s"
            )

        # Check status messages
        while not status_queue.empty():
            m, msg = status_queue.get()
            print(f"[status] {m}: {msg}")

        # Cull finished processes
        still_active = []
        for p in active:
            if p.is_alive():
                still_active.append(p)
            else:
                print(f"[exit] pid={p.pid} code={p.exitcode}")
        active = still_active

        if terminator.terminated:
            print("[terminate] Signal received; terminating child processes...")
            for p in active:
                p.terminate()
            break

        time.sleep(1.0)

    print("[done] All model sweeps finished or terminated.")


def main():
    parser = argparse.ArgumentParser(description="Parallel cluster sweep runner")
    parser.add_argument(
        "--pnode",
        type=str,
        default="2156113094",
        help="Single PJM node id or comma-separated list (e.g. 111,222,333)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ALL",
        help="Comma-separated model names. Use 'ALL' (default) to run every registered model.",
    )
    parser.add_argument(
        "--runs-per-model", type=int, default=2, help="Sweep runs per model"
    )
    parser.add_argument(
        "--max-proc", type=int, default=2, help="Maximum concurrent processes"
    )
    parser.add_argument(
        "--subset-data-size",
        type=float,
        default=0.05,
        help="Fraction of most recent data to keep (0<x<=1)",
    )
    # Proper boolean flags for GPU usage
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--use-gpus",
        dest="use_gpus",
        action="store_true",
        help="Enable GPU round-robin assignment",
    )
    group.add_argument(
        "--cpu-only", dest="use_gpus", action="store_false", help="Force CPU-only runs"
    )
    # Backward-friendly alias for CPU flag
    group.add_argument(
        "--use-cpu", dest="use_gpus", action="store_false", help="Alias for --cpu-only"
    )
    parser.set_defaults(use_gpus=torch.cuda.is_available())

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Start method selection: use 'spawn' for GPU runs to avoid
    # "Cannot re-initialize CUDA in forked subprocess" errors.
    # Keep default (often 'fork' on Linux) for CPU-only for lower overhead.
    # Allow override via env THESIS_MP_START_METHOD.
    # ------------------------------------------------------------------
    requested_method = os.getenv(
        "THESIS_MP_START_METHOD", "spawn" if args.use_gpus else "fork"
    )
    current = mp.get_start_method(allow_none=True)
    if current is None:
        try:
            mp.set_start_method(requested_method, force=True)
            print(
                f"[mp] start method set to '{requested_method}' (use_gpus={args.use_gpus})"
            )
        except RuntimeError as e:
            print(f"[mp] could not set start method '{requested_method}': {e}")
    else:
        if args.use_gpus and current != "spawn":
            print(
                f"[mp][warn] current start method '{current}' may cause CUDA fork issues; recommend 'spawn'."
            )
        else:
            print(f"[mp] start method already '{current}'")

    # Configure precision AFTER setting start method so child processes inherit cleanly
    configure_fp32_precision()

    try:
        model_list = _parse_models(args.models)
    except ValueError as e:
        print(f"Error parsing models: {e}")
        sys.exit(1)

    # GPU/CPU filtering based on ModelSpec.uses_gpu
    registry = make_registry()
    if args.use_gpus:
        filtered = [m for m in model_list if registry[m].uses_gpu]
        if not filtered:
            print("No GPU-capable models selected after filtering. Exiting.")
            sys.exit(0)
        model_list = filtered
    else:
        filtered = [m for m in model_list if not registry[m].uses_gpu]
        if not filtered:
            print("No CPU-only models selected after filtering. Exiting.")
            sys.exit(0)
        model_list = filtered

    # Parse pnodes
    pnode_id = int(args.pnode)

    print(
        f"Launching parallel sweeps: pnode={pnode_id} models={[m.value for m in model_list]} runs_per_model={args.runs_per_model} max_proc={args.max_proc} subset_data_size={args.subset_data_size} use_gpus={args.use_gpus} (filtered by uses_gpu)"
    )

    run_parallel(
        pnode_id=pnode_id,
        project="Thesis",
        models=model_list,
        runs_per_model=args.runs_per_model,
        max_processes=args.max_proc,
        subset_data_size=args.subset_data_size,
        use_gpus=args.use_gpus,
    )


if __name__ == "__main__":
    main()
