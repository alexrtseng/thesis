import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import wandb
from forecasting.test_forecaster_combo import evaluate_hf_lf_pair
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


def _eval_one_hf(args):
    """Worker: evaluate one HF run against PJM DA and all LF runs."""
    hf_model_path, lf_model_run_paths, test_size, pnode_id = args
    local_runs = {}

    # HF + PJM DA
    metrics, hf_run_name, _ = evaluate_hf_lf_pair(
        pnode_id=pnode_id,
        hf_run_path=hf_model_path,
        lf_run_path=None,
        test_size=test_size,
        pjm_da_preds=True,
    )
    best_perf = max(
        metrics.get("pct_perf_hor_3", float("-inf")),
        metrics.get("pct_perf_hor_6", float("-inf")),
        metrics.get("pct_perf_hor_12", float("-inf")),
    )
    combo_name = f"{hf_run_name} & PJMDa"
    local_runs[combo_name] = best_perf

    # HF + each LF
    for lf_model_path in lf_model_run_paths:
        metrics, hf_run_name, lf_run_name = evaluate_hf_lf_pair(
            pnode_id=pnode_id,
            hf_run_path=hf_model_path,
            lf_run_path=lf_model_path,
            test_size=test_size,
            pjm_da_preds=False,
        )
        best_perf = max(
            metrics.get("pct_perf_hor_3", float("-inf")),
            metrics.get("pct_perf_hor_6", float("-inf")),
            metrics.get("pct_perf_hor_12", float("-inf")),
        )
        combo_name = f"{hf_run_name} & {lf_run_name}"
        local_runs[combo_name] = best_perf

    return local_runs


def test_hf_lf_model_combinations(
    pnode_id: int,
    hf_model_run_paths,
    lf_model_run_paths,
    test_size: int | None = None,
    num_workers: int = 4,
):
    wandb.login(key=WANDB_API_KEY)
    start = time.perf_counter()

    runs = {}
    tasks = [
        (hf_path, lf_model_run_paths, test_size, pnode_id)
        for hf_path in hf_model_run_paths
    ]
    num_workers = max(1, min(num_workers, len(tasks)))

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_eval_one_hf, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            runs.update(fut.result())

    elapsed = time.perf_counter() - start
    num_combos = len(hf_model_run_paths) * (
        len(lf_model_run_paths) + 1
    )  # +1 for PJM DA

    # Ensure output directory exists
    out_dir = os.path.join("forecasting", "outputs", "tests", str(pnode_id))
    os.makedirs(out_dir, exist_ok=True)

    # Filename: {num_combos_tested}_{datetime}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"{num_combos}_{ts}.txt")

    # Sort by best performance desc
    sorted_runs = sorted(runs.items(), key=lambda kv: kv[1], reverse=True)

    with open(out_path, "w") as f:
        f.write(
            f"Total runtime (HH:MM:SS): {time.strftime('%H:%M:%S', time.gmtime(elapsed))}\n"
        )
        f.write(f"Number of combinations tested: {num_combos}\n")
        f.write(f"Workers used: {num_workers}\n")
        f.write("\n")
        # Pretty column formatting
        name_width = max(len(name) for name, _ in sorted_runs) if sorted_runs else 12
        f.write(f"{'Combination':<{name_width}} | {'Best Perf':>10}\n")
        f.write("-" * (name_width + 13) + "\n")

        for combo_name, best_perf in sorted_runs:
            f.write(f"{combo_name:<{name_width}} | {best_perf:>10.6f}\n")


if __name__ == "__main__":
    pnode_id = 2156113094  # Example pnode_id
    test_hf_lf_model_combinations(
        pnode_id=pnode_id,
        hf_model_run_paths=TEST_HF_MODEL_RUNS,
        lf_model_run_paths=TEST_LF_MODEL_RUNS,
        test_size=300,
        num_workers=4,  # adjust concurrency here
    )
