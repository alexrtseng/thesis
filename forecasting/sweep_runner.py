import pandas as pd

import wandb
from forecasting.train import (
    WANDB_API_KEY,
    WANDB_PROJECT_NAME,
    build_series_for_node,
    train_hf_model,
)

from .model_zoo import ModelName, make_registry


def run_sweep_for_node(
    model_name: ModelName,
    pnode_id: int,
    feature_df: pd.DataFrame,
    project: str,
    count: int = 50,
    subset_data_size: float = 1.0,
):
    if model_name == ModelName.KALMANFORECASTER:
        subset_data_size = min(0.2, subset_data_size)
    total = len(feature_df)
    keep = max(1, int(total * subset_data_size))
    df = feature_df[-keep:].copy()
    reg = make_registry()
    assert model_name in reg, f"Unknown model: {model_name}"
    spec = reg[model_name]
    count = min(count, spec.max_needed_runs)

    sweep_cfg = spec.sweep_config()
    wandb.login(key=WANDB_API_KEY)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=project)

    def _fn(config=None):
        train_hf_model(
            feature_df=df,
            model_name=model_name,
            pnode_id=pnode_id,
            config=config or {},
            post_run_logging=True,
        )

    wandb.agent(sweep_id, function=_fn, project=project, count=count)


if __name__ == "__main__":
    subset_data_size = 0.01
    feature_df = build_series_for_node(2156113094)

    run_sweep_for_node(
        model_name=ModelName.TCNMODEL,
        pnode_id=2156113094,
        feature_df=feature_df,
        project=WANDB_PROJECT_NAME,
        count=10,
    )
