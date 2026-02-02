import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import wandb
from forecasting.test_forecaster_combo import evaluate_hf_lf_pair_ensemble
from forecasting.train import WANDB_API_KEY

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


def _eval_one_ensemble(args: tuple) -> dict:
    """Worker: evaluate one randomly-sampled HF/LF ensemble combination."""
    (
        pnode_id,
        hf_sel,
        lf_sel,
        test_size,
        hf_horizon,
    ) = args

    metrics, _two_stage_df = evaluate_hf_lf_pair_ensemble(
        pnode_id=pnode_id,
        hf_run_paths=hf_sel,
        lf_run_paths=lf_sel,
        test_size=test_size,
        hf_horizon=hf_horizon,
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
    num_workers: int = 4,
    seed: int | None = None,
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
    wandb.login(key=WANDB_API_KEY)
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

    rng = np.random.default_rng(seed)

    # Pre-sample tasks in the parent process for reproducibility.
    tasks: list[tuple] = []
    for _ in range(num_evals):
        n_hf = int(rng.integers(min_hf, max_hf + 1))
        n_lf = int(rng.integers(min_lf, max_lf + 1))
        hf_sel = rng.choice(hf_model_run_paths, size=n_hf, replace=False).tolist()
        lf_sel = rng.choice(lf_model_run_paths, size=n_lf, replace=False).tolist()
        tasks.append((pnode_id, hf_sel, lf_sel, test_size, hf_horizon))

    rows: list[dict] = []
    num_successful = 0
    num_failed = 0

    num_workers = max(1, min(int(num_workers), len(tasks)))
    if num_workers == 1:
        for t in tasks:
            try:
                rows.append(_eval_one_ensemble(t))
                num_successful += 1
            except Exception as e:
                num_failed += 1
                print(f"Error in evaluation: {e}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(_eval_one_ensemble, t): t for t in tasks}
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                    num_successful += 1
                except Exception as e:
                    num_failed += 1
                    print(f"Error in evaluation: {e}")

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
    num_workers,
    seed,
    run_name: str = None,
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
        num_workers=num_workers,
        seed=seed,
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
        f.write(f"num_workers: {num_workers}\n")
    print(f"Wrote results to {out_path}")
    print(f"Total runtime (s): {elapsed:.2f}")
    print(df.sort_values("pct_perf", ascending=False).head(10))


if __name__ == "__main__":
    pnode_id = 2156113094
    num_evals = 5
    min_hf = 1
    max_hf = 3
    min_lf = 1
    max_lf = 3
    test_size = 800  # should be 4320 or None for full runs
    hf_horizon = 6
    num_workers = 4
    seed = None

    test_and_write_ensemble_combos(
        pnode_id=pnode_id,
        num_evals=num_evals,
        min_hf=min_hf,
        max_hf=max_hf,
        min_lf=min_lf,
        max_lf=max_lf,
        test_size=test_size,
        hf_horizon=hf_horizon,
        num_workers=num_workers,
        seed=seed,
    )
