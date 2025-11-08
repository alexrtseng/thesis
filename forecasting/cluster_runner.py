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
import os
import random
import signal
import sys
import time
from multiprocessing import Process, Queue
from typing import List

import torch

from forecasting.model_zoo import ModelName, make_registry
from forecasting.sweep_runner import run_sweep_for_node


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
    project: str,
    count: int,
    subset_data_size: float,
    gpu_id: int | None,
    delay_start: float,
    status_queue: Queue,
):
    try:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        time.sleep(delay_start)
        status_queue.put((model_name.value, "starting"))
        run_sweep_for_node(
            model_name=model_name,
            pnode_id=pnode_id,
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
    pnode_ids: List[int],
    project: str,
    models: List[ModelName],
    runs_per_model: int,
    max_processes: int,
    subset_data_size: float,
    use_gpus: bool,
):
    status_queue: Queue = Queue()
    # Track processes explicitly via 'active' list below

    # Simple GPU round robin
    available_gpus = []
    if use_gpus:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            n = torch.cuda.device_count()
            available_gpus = list(range(n))

    terminator = GracefulTerminator()

    # Limit number of concurrent processes
    # Build queue of (pnode, model) tasks
    launch_queue = [(pid, m) for pid in pnode_ids for m in models]
    active: List[Process] = []

    while launch_queue or active:
        # Launch new processes if slots available
        while launch_queue and len(active) < max_processes:
            pnode_id, model = launch_queue.pop(0)
            gpu_id = None
            if available_gpus:
                gpu_id = available_gpus[len(active) % len(available_gpus)]
            delay = random.uniform(0.5, 2.5)  # stagger start
            p = Process(
                target=_worker,
                args=(
                    model,
                    pnode_id,
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
        required=True,
        help="Single PJM node id or comma-separated list (e.g. 111,222,333)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="ALL",
        help="Comma-separated model names. Use 'ALL' (default) to run every registered model.",
    )
    parser.add_argument(
        "--project", type=str, default="Thesis", help="W&B project name"
    )
    parser.add_argument(
        "--runs-per-model", type=int, default=20, help="Sweep runs per model"
    )
    parser.add_argument(
        "--max-proc", type=int, default=2, help="Maximum concurrent processes"
    )
    parser.add_argument(
        "--subset-data-size",
        type=float,
        default=1.0,
        help="Fraction of most recent data to keep (0<x<=1)",
    )
    parser.add_argument(
        "--use-gpus", type=bool, default=True, help="Enable GPU round-robin assignment"
    )

    args = parser.parse_args()

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
    try:
        pnode_ids = [int(tok.strip()) for tok in args.pnode.split(",") if tok.strip()]
    except ValueError:
        print("Invalid --pnode list; must be integers")
        sys.exit(1)

    print(
        f"Launching parallel sweeps: pnodes={pnode_ids} models={[m.value for m in model_list]} runs_per_model={args.runs_per_model} max_proc={args.max_proc} subset_data_size={args.subset_data_size} use_gpus={args.use_gpus} (filtered by uses_gpu)"
    )

    run_parallel(
        pnode_ids=pnode_ids,
        project=args.project,
        models=model_list,
        runs_per_model=args.runs_per_model,
        max_processes=args.max_proc,
        subset_data_size=args.subset_data_size,
        use_gpus=args.use_gpus,
    )


if __name__ == "__main__":
    main()
