import os

import wandb

from forecasting.test_best_models import TEST_HF_MODEL_RUNS, TEST_LF_MODEL_RUNS

api = wandb.Api(api_key=os.environ.get("WANDB_API_KEY"))


def print_run_names(label: str, run_paths: list[str]) -> None:
    print(f"\n{label}:")
    for path in run_paths:
        run = api.run(path)
        print(f"  {path}  ->  {run.name}")


print_run_names("HF Model Runs", TEST_HF_MODEL_RUNS)
print_run_names("LF Model Runs", TEST_LF_MODEL_RUNS)
