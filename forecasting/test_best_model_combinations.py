import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.evaluate_hf_lf_pair_ensemble import evaluate_hf_lf_pair_ensemble
from forecasting.store_test_forecasts import default_cache_path, populate_cache

# Hardcoded per-evaluation wall-clock timeout.
# Prevents a single stuck eval (solver/network/IO) from keeping the whole job alive.
PER_EVAL_TIMEOUT_S = 60 * 30  # 30 minutes

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
TEST_HF_MODEL_RUNS = [
    "watt-our/thesis-hf-forecasters/tks540ei",  # transformer
    "watt-our/thesis-hf-forecasters/obvw0nqv",  # transformer
    "watt-our/thesis-hf-forecasters/ttzome1t",  # nhits
    "watt-our/thesis-hf-forecasters/j7oviesl",  # transformer
    "watt-our/thesis-hf-forecasters/mh9t0r9v",  # tsmixer
    "watt-our/thesis-hf-forecasters/wtdafv56",  # transformer
    "watt-our/thesis-hf-forecasters/2keubomv",  # blockrnn
    "watt-our/thesis-hf-forecasters/2ydmchx4",  # tsmixer
    "watt-our/thesis-hf-forecasters/xl7ier6h",  # dlinear
    "watt-our/thesis-hf-forecasters/lkujq05c",  # nhits
    "watt-our/thesis-hf-forecasters/aclpbztf",  # nbeats
    "watt-our/thesis-hf-forecasters/y3s9jm3u",  # tide
    "watt-our/thesis-hf-forecasters/zu09p08u",  # RNN
    "watt-our/thesis-hf-forecasters/qm3no329",  # TCN
    "watt-our/thesis-hf-forecasters/d7odgxif",  # RNN
    "watt-our/thesis-hf-forecasters/wqipc7pb",  # xgb
    "watt-our/thesis-hf-forecasters/a2t6syeh",  # tcn
    "watt-our/thesis-hf-forecasters/hrx8q1e6",  # xgb
    "watt-our/thesis-hf-forecasters/wa2txvsd",  # tft
    "watt-our/thesis-hf-forecasters/n9468o4a",  # autoarima
]

TEST_LF_MODEL_RUNS = [
    "watt-our/thesis-lf-forecasters/tbk5xmvg",  # tft
    "watt-our/thesis-lf-forecasters/a3ilhshd",  # XGB
    "watt-our/thesis-lf-forecasters/4b1o2o11",  # tsmixer
    "watt-our/thesis-lf-forecasters/9je5svmu",  # xgboost
    "watt-our/thesis-lf-forecasters/60zx0lhp",  # xgboost
    "watt-our/thesis-lf-forecasters/epkknrba",  # xgboost
    "watt-our/thesis-lf-forecasters/k9jkje7f",  # xgboost
    "watt-our/thesis-lf-forecasters/hk4zyw7a",  # xgboost
    "watt-our/thesis-lf-forecasters/gwwjxk6g",  # rnn
    "watt-our/thesis-lf-forecasters/1grl6iaf",  # xgb
    "watt-our/thesis-lf-forecasters/j0r01ir1",  # tcn
    "watt-our/thesis-lf-forecasters/lx82mfuf",  # tide
    "watt-our/thesis-lf-forecasters/zqd9zf36",  # rnn
    "watt-our/thesis-lf-forecasters/75610i3y",  # tft
    "watt-our/thesis-lf-forecasters/2jcdrsn8",  # nhits
    "watt-our/thesis-lf-forecasters/db1wpxb6",  # blockrnn
    "watt-our/thesis-lf-forecasters/bpp4kwd9",  # dlinear
    "watt-our/thesis-lf-forecasters/iwvd9kfh",  # nbeats
    "watt-our/thesis-lf-forecasters/fl8aigij",  # transformer
    "watt-our/thesis-lf-forecasters/0c7iapks",  # autoarima
]


def _short_run_id(run_path: str) -> str:
    # W&B run paths end with an 8-char run id (e.g. .../tks540ei)
    tail = run_path.split("/")[-1]
    return tail[-8:]


def _eval_one_ensemble(
    pnode_id, hf_sel, lf_sel, test_size, hf_horizon, cache_file=None
) -> dict:
    """Worker: evaluate one randomly-sampled HF/LF ensemble combination."""
    metrics, _ = evaluate_hf_lf_pair_ensemble(
        pnode_id=pnode_id,
        hf_run_paths=hf_sel,
        lf_run_paths=lf_sel,
        test_size=test_size,
        hf_horizon=hf_horizon,
        cache_file=cache_file,
    )

    return {
        "hf_forecasters": " ".join(_short_run_id(p) for p in hf_sel),
        "lf_forecasters": " ".join(_short_run_id(p) for p in lf_sel),
        "num_hf_forecasters": int(len(hf_sel)),
        "num_lf_forecasters": int(len(lf_sel)),
        "pct_perf": float(metrics.get("pct_perf", float("nan"))),
        "pred_start": metrics["common_start"],
        "pred_end": metrics["common_end"],
    }


def _available_cpu_count() -> int:
    """Best-effort CPU count respecting Slurm/cgroup affinity."""
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except Exception:
        pass
    slurm = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm and slurm.isdigit():
        return max(1, int(slurm))
    return max(1, os.cpu_count() or 1)


def _eval_one_ensemble_child(
    q: mp.queues.Queue,
    idx: int,
    pnode_id: int,
    hf_sel: list[str],
    lf_sel: list[str],
    test_size: int,
    hf_horizon: int,
    cache_file: str | None,
) -> None:
    """Run one eval in a child process and report result via a queue."""
    try:
        res = _eval_one_ensemble(
            pnode_id,
            hf_sel,
            lf_sel,
            test_size,
            hf_horizon,
            cache_file,
        )
        q.put(("ok", res))
    except BaseException:
        q.put(("err", f"idx={idx}\n{traceback.format_exc()}"))


def _run_one_process(
    args: tuple[int, int, list[str], list[str], int, int, str | None],
) -> dict:
    """Process worker: evaluate one task. Must be top-level for spawn pickling."""
    idx, pnode_id, hf_sel, lf_sel, test_size, hf_horizon, cache_file = args
    try:
        # Hard timeout: run the eval in a subprocess so we can terminate it even if
        # it hangs inside native code (solver/network/IO).
        ctx = mp.get_context("spawn")
        q = ctx.Queue(maxsize=1)
        p = ctx.Process(
            target=_eval_one_ensemble_child,
            args=(q, idx, pnode_id, hf_sel, lf_sel, test_size, hf_horizon, cache_file),
            daemon=True,
        )
        p.start()
        p.join(PER_EVAL_TIMEOUT_S)

        if p.is_alive():
            p.terminate()
            p.join(30)
            raise TimeoutError(
                f"Evaluation idx={idx} exceeded timeout ({PER_EVAL_TIMEOUT_S}s)"
            )

        if q.empty():
            raise RuntimeError(f"Evaluation idx={idx} finished but produced no result")

        status, payload = q.get()
        if status == "ok":
            return payload
        raise RuntimeError(payload)
    except BaseException:
        # If the worker is dying due to a normal Python exception (not segfault/OOM),
        # this prints a full traceback into the Slurm log.
        print(
            "Worker exception:\n" + traceback.format_exc(),
            flush=True,
        )
        raise


def random_test_hf_lf_ensemble_combinations(
    pnode_id: int,
    hf_model_run_paths: list[str],
    lf_model_run_paths: list[str],
    num_evals: int = 25,
    min_hf: int = 1,
    max_hf: int = 3,
    min_lf: int = 1,
    max_lf: int = 3,
    test_size: int = 300,
    hf_horizon: int = 6,
    seed: int | None = None,
    max_workers: int | None = None,
    parallel: bool = True,
) -> pd.DataFrame:
    """Randomly sample HF/LF ensembles and evaluate via two-stage stochastic MPC.

    Returns a DataFrame with columns:
    - hf_forecasters: space-separated last-8-char run ids
    - lf_forecasters: space-separated last-8-char run ids
    - num_hf_forecasters
    - num_lf_forecasters
    - test_size
    - pct_perf
    - pred_start
    - pred_end
    """
    if num_evals <= 0:
        raise ValueError("num_evals must be positive")
    if min_hf <= 0 or min_lf <= 0:
        raise ValueError("min_hf and min_lf must be >= 1")
    if max_hf < min_hf or max_lf < min_lf:
        raise ValueError("max_hf/max_lf must be >= min_hf/min_lf")
    if max_hf > len(hf_model_run_paths):
        max_hf = len(hf_model_run_paths)
    if max_lf > len(lf_model_run_paths):
        max_lf = len(lf_model_run_paths)

    populate_cache(pnode_id, test_size, hf_horizon)
    cache_file = str(default_cache_path(pnode_id, test_size, hf_horizon))

    rng = np.random.default_rng(seed)

    tasks: list[tuple[int, list[str], list[str]]] = []
    for idx in range(num_evals):
        n_hf = int(rng.integers(min_hf, max_hf + 1))
        n_lf = int(rng.integers(min_lf, max_lf + 1))
        hf_sel = rng.choice(hf_model_run_paths, size=n_hf, replace=False).tolist()
        lf_sel = rng.choice(lf_model_run_paths, size=n_lf, replace=False).tolist()
        tasks.append((idx, hf_sel, lf_sel))

    rows: list[dict] = []
    num_successful = 0
    num_failed = 0

    if parallel and num_evals > 1:
        workers = max_workers
        if workers is None:
            # Default: use all visible CPUs (Slurm affinity aware).
            workers = _available_cpu_count()

        print(f"Running {num_evals} evaluations with {workers} workers...")

        proc_tasks = [
            (
                idx,
                pnode_id,
                hf_sel,
                lf_sel,
                test_size,
                hf_horizon,
                cache_file,
            )
            for (idx, hf_sel, lf_sel) in tasks
        ]
        # On many HPC clusters, using the default 'fork' start method can cause
        # native libraries (notably Gurobi) to crash in child processes.
        ctx = mp.get_context("spawn")
        ex_kwargs = {"max_workers": workers, "mp_context": ctx}
        # Python 3.11+ supports recycling workers to avoid memory growth.
        if sys.version_info >= (3, 11):
            ex_kwargs["max_tasks_per_child"] = 10

        with ProcessPoolExecutor(**ex_kwargs) as ex:
            futures = [ex.submit(_run_one_process, t) for t in proc_tasks]
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                    num_successful += 1
                    print(f"Completed evaluations: {num_successful + num_failed}")
                except Exception as e:
                    num_failed += 1
                    print(f"Error in evaluation: {e}")
                    print(f"Completed evaluations: {num_successful + num_failed}")
    else:
        for task in tasks:
            try:
                idx, hf_sel, lf_sel = task
                rows.append(
                    _run_one_process(
                        (
                            idx,
                            pnode_id,
                            hf_sel,
                            lf_sel,
                            test_size,
                            hf_horizon,
                            cache_file,
                        )
                    )
                )
                num_successful += 1
            except Exception as e:
                num_failed += 1
                print(f"Error in evaluation: {e}")
                print(f"Completed evaluations: {num_successful + num_failed}")
    print(f"Successful iterations: {num_successful}, Failed iterations: {num_failed}")
    return pd.DataFrame(rows)


def test_and_write_ensemble_combos(
    pnode_id: int,
    num_evals,
    min_hf,
    max_hf,
    min_lf,
    max_lf,
    test_size,
    hf_horizon,
    seed,
    run_name: str = None,
    max_workers: int | None = None,
):
    start = time.perf_counter()
    df = random_test_hf_lf_ensemble_combinations(
        pnode_id=pnode_id,
        hf_model_run_paths=TEST_HF_MODEL_RUNS,
        lf_model_run_paths=TEST_LF_MODEL_RUNS,
        num_evals=num_evals,
        min_hf=min_hf,
        max_hf=max_hf,
        min_lf=min_lf,
        max_lf=max_lf,
        test_size=test_size,  # should be 4320 or None for full runs
        hf_horizon=hf_horizon,
        seed=seed,
        max_workers=max_workers,
        parallel=True,
    )
    elapsed = time.perf_counter() - start
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path("forecasting") / "outputs" / "tests" / "ensemble_combinations" / ts
        if run_name is None
        else Path("forecasting")
        / "outputs"
        / "tests"
        / "ensemble_combinations"
        / run_name
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"random_ensemble_{len(df)}_{ts}.csv"
    df.to_csv(out_path, index=False)
    out_path = out_dir / "info.txt"
    with open(out_path, "w") as f:
        f.write(f"Run datetime: {ts}\n")
        f.write(f"Total runtime (s): {elapsed:.2f}\n")
        f.write(f"pnode_id: {pnode_id}\n")
        f.write(f"num_evals: {num_evals}\n")
        f.write(f"min_hf: {min_hf}\n")
        f.write(f"max_hf: {max_hf}\n")
        f.write(f"min_lf: {min_lf}\n")
        f.write(f"max_lf: {max_lf}\n")
        f.write(f"test_size: {test_size}\n")
        f.write(f"hf_horizon: {hf_horizon}\n")
    print(f"Wrote results to {out_path}")
    print(f"Total runtime (s): {elapsed:.2f}")
    if df.empty:
        print("No successful evaluations; results DataFrame is empty.")
    elif "pct_perf" not in df.columns:
        print(
            f"Results DataFrame missing 'pct_perf' column. Columns: {list(df.columns)}"
        )
    else:
        print(df.sort_values("pct_perf", ascending=False).head(10))


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    return float(raw)


def _env_opt_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name, "" if default is None else str(default)).strip()
    if raw.lower() in {"", "none", "null"}:
        return None
    return int(raw)


def _env_opt_str(name: str, default: str | None) -> str | None:
    raw = os.getenv(name, "" if default is None else str(default)).strip()
    if raw.lower() in {"", "none", "null"}:
        return None
    return raw


def main(argv: list[str] | None = None) -> None:
    """Entry point.

    - If CLI args are provided, uses argparse (backward compatible).
    - If no CLI args are provided, reads configuration from environment variables.
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 0:
        pnode_id = _env_int("PNODE_ID", 2156113094)
        num_evals = _env_int("NUM_EVALS", 5)
        min_hf = _env_int("MIN_HF", 1)
        max_hf = _env_int("MAX_HF", 10)
        min_lf = _env_int("MIN_LF", 1)
        max_lf = _env_int("MAX_LF", 10)
        test_size = _env_opt_int("TEST_SIZE", None)
        hf_horizon = _env_int("HF_HORIZON", 6)
        seed = _env_opt_int("SEED", None)
        max_workers = _env_opt_int("MAX_WORKERS", None)
        run_name = _env_opt_str("RUN_NAME", None)
    else:
        parser = argparse.ArgumentParser(description="Test best model combinations")
        parser.add_argument("--pnode-id", type=int, required=True, help="Power node ID")
        parser.add_argument(
            "--num-evals", type=int, required=True, help="Number of evaluations"
        )
        parser.add_argument(
            "--min-hf", type=int, default=1, help="Minimum number of HF forecasters"
        )
        parser.add_argument(
            "--max-hf", type=int, default=10, help="Maximum number of HF forecasters"
        )
        parser.add_argument(
            "--min-lf", type=int, default=1, help="Minimum number of LF forecasters"
        )
        parser.add_argument(
            "--max-lf", type=int, default=10, help="Maximum number of LF forecasters"
        )
        parser.add_argument(
            "--test-size",
            type=lambda x: int(x) if x.lower() != "none" else None,
            default=None,
            help="Test size",
        )
        parser.add_argument("--hf-horizon", type=int, required=True, help="HF horizon")
        parser.add_argument(
            "--seed",
            type=lambda x: int(x) if x.lower() != "none" else None,
            default=None,
            help="Random seed",
        )
        parser.add_argument(
            "--max-workers",
            type=lambda x: int(x) if x.lower() != "none" else None,
            default=None,
            help="Maximum workers",
        )

        parser.add_argument(
            "--run-name",
            type=lambda x: x if x.lower() != "none" else None,
            default=None,
            help="Run name for output directory",
        )

        args = parser.parse_args(argv)
        pnode_id = args.pnode_id
        num_evals = args.num_evals
        min_hf = args.min_hf
        max_hf = args.max_hf
        min_lf = args.min_lf
        max_lf = args.max_lf
        test_size = args.test_size
        hf_horizon = args.hf_horizon
        seed = args.seed
        max_workers = args.max_workers
        run_name = args.run_name

    test_and_write_ensemble_combos(
        pnode_id=pnode_id,
        num_evals=num_evals,
        min_hf=min_hf,
        max_hf=max_hf,
        min_lf=min_lf,
        max_lf=max_lf,
        test_size=test_size,
        hf_horizon=hf_horizon,
        seed=seed,
        max_workers=max_workers,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
