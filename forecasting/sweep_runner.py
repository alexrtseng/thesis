import inspect
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import wandb
from data.data_output_functions import read_rt_da_with_weather
from forecasting.transforms import (
    align_on_overlap,
    delayed_past_covariates,
    drop_nan_rows,
)

from .metrics import per_step_metrics
from .model_zoo import ModelName, make_registry
import os

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
WANDB_PROJECT_NAME = os.environ.get("WANDB_PROJECT_NAME", "Thesis")

FEATURE_COLUMNS = [
    # "lmp_rt", this is the target
    "lmp_da",
    "temperature_2m",
    "relative_humidity_2m",
    # "dew_point_2m",
    # "apparent_temperature",
    "precipitation",
    # "rain",
    "snowfall",
    # "surface_pressure",
    "cloud_cover",
    "windspeed_10m",
    # "winddirection_10m",
    "shortwave_radiation",
    # "direct_radiation",
    # "diffuse_radiation",
    # "global_tilted_irradiance",
]


def _latest_weather_csv(pnode_id: int) -> Optional[Path]:
    d = Path(f"data/weather/node_{pnode_id}")
    if not d.exists():
        return None
    files = sorted(d.glob("*.csv"))
    return files[-1] if files else None


def build_series_for_node(pnode_id: int) -> tuple[TimeSeries, Optional[TimeSeries]]:
    feature_df = read_rt_da_with_weather(
        rt_dir=Path("data/pjm_lmps"),
        da_dir=Path("data/pjm_lmps_da"),
        weather_file=_latest_weather_csv(pnode_id),
        pnode_id=int(pnode_id),
    )
    # Ensure index is a DatetimeIndex so TimeSeries will use it as the time column
    feature_df.index = pd.to_datetime(feature_df.index)
    # Cast all numeric inputs to float32 for MPS compatibility
    num_cols = feature_df.select_dtypes(include=["number"]).columns
    feature_df[num_cols] = feature_df[num_cols].astype("float32")
    # Ensure the index is timezone-naive (drop any tz info if present)
    idx = pd.to_datetime(feature_df.index)
    if getattr(idx, "tz", None) is not None:
        try:
            idx = idx.tz_convert(None)
        except Exception:
            idx = idx.tz_localize(None)
    else:
        try:
            idx = idx.tz_localize(None)
        except Exception:
            pass
    feature_df.index = idx

    target_series = TimeSeries.from_dataframe(
        feature_df,
        time_col=None,
        value_cols="lmp_rt",
        freq="5min",
    ).astype(np.float32)
    future_covariates = TimeSeries.from_dataframe(
        feature_df,
        time_col=None,
        value_cols=FEATURE_COLUMNS,
        freq="5min",
    ).astype(np.float32)
    return target_series, future_covariates


def train_once(
    model_name: ModelName,
    config: Dict[str, Any],
    pnode_id: int,
    project: str,
    entity: Optional[str] = None,
    subset_data_size: float = 1.0,
):
    reg = make_registry()
    spec = reg[model_name]
    model = spec.builder(config)

    y, fut = build_series_for_node(pnode_id)
    print(f"Total data points used for training {pnode_id}: {len(y)}")
    # Keep only the most recent fraction of the data if requested
    if subset_data_size <= 0:
        raise ValueError("subset_data_size must be > 0")
    if subset_data_size < 1.0:
        total = len(y)
        keep = max(1, int(total * subset_data_size))
        y = y[-keep:]
        if fut is not None:
            fut = fut[-keep:]
    split_idx = int(len(y) * 0.8)
    train_y, val_y = y[:split_idx], y[split_idx:]
    train_fut, val_fut = fut[:split_idx], fut[split_idx:]

    wandb.login(key=WANDB_API_KEY)
    with wandb.init(
        project=project, config=config, name=f"{model_name.value}-{pnode_id}"
    ):
        # Normalize target and covariates using Darts Scaler (fit on training only)
        y_scaler = Scaler()
        train_y_s = y_scaler.fit_transform(train_y)
        val_y_s = y_scaler.transform(val_y)

        if spec.supports_future and train_fut is not None and val_fut is not None:
            fut_scaler = Scaler()
            train_fut_s = fut_scaler.fit_transform(train_fut)
            val_fut_s = fut_scaler.transform(val_fut)
        else:
            train_fut_s = None
            val_fut_s = None

        # Optionally create delayed past covariates from future covariates
        include_delayed = (
            bool(config.get("include_delayed_covariates", False)) and spec.supports_past
        )

        delay_steps = 0
        train_past_s = None
        val_past_s = None

        if include_delayed and (train_fut_s is not None and val_fut_s is not None):
            # Resolve output_chunk_length robustly
            ocl_attr = getattr(model, "output_chunk_length", None)
            cfg_ocl = config.get("output_chunk_length", None)
            if ocl_attr is not None:
                resolved_ocl = int(ocl_attr)
            elif cfg_ocl is not None:
                resolved_ocl = int(cfg_ocl)
            else:
                resolved_ocl = 1  # safe fallback for models without OCL

            extra_delay = int(config.get("delayed_steps", 0))
            if extra_delay < 0:
                extra_delay = 0
            delay_steps = resolved_ocl + extra_delay

            try:
                train_past_s = delayed_past_covariates(train_fut_s, delay_steps)
                val_past_s = delayed_past_covariates(val_fut_s, delay_steps)

                # Align partitions and drop NaNs from the shift
                train_y_s, train_past_s = align_on_overlap(train_y_s, train_past_s)
                train_y_s, train_past_s = drop_nan_rows(train_past_s, train_y_s)

                val_y_s, val_past_s = align_on_overlap(val_y_s, val_past_s)
                val_y_s, val_past_s = drop_nan_rows(val_past_s, val_y_s)

                # Log traceability for delayed covariates
                wandb.summary["delayed_base_output_chunk_length"] = int(resolved_ocl)
                wandb.summary["delayed_extra_offset"] = int(extra_delay)
                wandb.summary["delayed_steps_used"] = int(delay_steps)
            except Exception as e:
                wandb.summary["delayed_covariates_error"] = str(e)
                train_past_s = None
                val_past_s = None
                delay_steps = 0

        # Build fit args dynamically based on model.fit signature to avoid passing
        # unsupported parameters (e.g., ARIMA doesn't take val_series/future_covariates)
        fit_sig = inspect.signature(model.fit)
        fit_args: Dict[str, Any] = {"series": train_y_s}
        if "val_series" in fit_sig.parameters:
            fit_args["val_series"] = val_y_s
        if train_past_s is not None and "past_covariates" in fit_sig.parameters:
            fit_args["past_covariates"] = train_past_s

        if (
            spec.supports_future
            and train_fut_s is not None
            and "future_covariates" in fit_sig.parameters
        ):
            fit_args["future_covariates"] = train_fut_s
            if (
                "val_future_covariates" in fit_sig.parameters
                and val_fut_s is not None
                and "val_series" in fit_sig.parameters
            ):
                fit_args["val_future_covariates"] = val_fut_s
        if (
            train_past_s is not None
            and "val_past_covariates" in fit_sig.parameters
            and val_past_s is not None
            and "val_series" in fit_sig.parameters
        ):
            fit_args["val_past_covariates"] = val_past_s
        # Some models accept verbose
        if "verbose" in fit_sig.parameters:
            fit_args["verbose"] = False

        model.fit(**fit_args)

        # After training: clear callbacks if they were added so saving works reliably
        try:
            model.trainer_params["callbacks"] = []
            model._model_params["pl_trainer_kwargs"]["callbacks"] = []
        except Exception:
            pass

        # Compute validation predictions & metrics (handle models without val_series)
        try:
            # Decide horizon based on model capability and validation length
            horizon = 24 * 5  # 24 hours of 5-min steps

            pred_sig = inspect.signature(model.predict)
            predict_kwargs: Dict[str, Any] = {}
            if "series" in pred_sig.parameters:
                predict_kwargs["series"] = train_y_s
            if (
                train_past_s is not None
                and val_past_s is not None
                and "past_covariates" in pred_sig.parameters
            ):
                predict_kwargs["past_covariates"] = train_past_s.append(val_past_s)
            if (
                spec.supports_future
                and train_fut_s is not None
                and val_fut_s is not None
                and "future_covariates" in pred_sig.parameters
            ):
                predict_kwargs["future_covariates"] = train_fut_s.append(val_fut_s)
            preds_s = model.predict(n=len(val_y_s), **predict_kwargs)
            print(f"Length of preds: {len(preds_s)}")

            # Inverse transform predictions back to original scale for metrics/plots
            preds = y_scaler.inverse_transform(preds_s)

            # Compute per-step metrics up to "horizon"
            metrics = per_step_metrics(val_y, preds, max_steps=horizon, prefix="val")
            wandb.summary.update(metrics)

            # Provide fallback best_val_loss for models lacking internal callback logging
            if "val_series" not in inspect.signature(model.fit).parameters:
                # Use RMSE (val_rmse_full) as stand-in
                if "val_rmse_full" in metrics:
                    wandb.summary["best_val_loss"] = metrics["val_rmse_full"]
                elif "val_mse_full" in metrics:
                    wandb.summary["best_val_loss"] = metrics["val_mse_full"] ** 0.5

            # Generate and log 3 random daily plots of predictions vs actuals
            # log_random_day_plots(
            #     actual=val_y,
            #     predicted=preds,
            #     num_days=3,
            #     title_prefix=f"{model_name.value} - node {pnode_id}",
            #     wandb_key="val_random_day_plots",
            # )
        except Exception as e:
            wandb.summary["metrics_error"] = str(e)

        # Save model using sweep + run name for traceability: <sweep>__<run>.pkl
        out_dir = Path("forecasting/outputs") / str(pnode_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        sweep_id = getattr(wandb.run, "sweep_id", None) or "nosweep"
        run_name = wandb.run.name or wandb.run.id

        def _sanitize(s: str) -> str:
            # Allow alnum, dot, dash, underscore; collapse others to _
            return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._-") or "run"

        base_name = f"{_sanitize(sweep_id)}__{_sanitize(run_name)}"
        save_path = out_dir / f"{base_name}.pkl"
        # If collision (unlikely), append short run id
        if save_path.exists():
            short_id = (wandb.run.id or "dup")[:8]
            save_path = out_dir / f"{base_name}__{short_id}.pkl"

        model.save(str(save_path))
        wandb.summary["model_path"] = str(save_path)
        wandb.summary["model_filename"] = save_path.name


def run_sweep_for_node(
    model_name: ModelName,
    pnode_id: int,
    *,
    project: str,
    count: int = 50,
    subset_data_size: float = 1.0,
):
    reg = make_registry()
    assert model_name in reg, f"Unknown model: {model_name}"
    spec = reg[model_name]

    sweep_cfg = spec.sweep_config()
    wandb.login(key=WANDB_API_KEY)
    sweep_id = wandb.sweep(sweep=sweep_cfg, project=project)

    def _fn(config=None):
        train_once(
            model_name,
            config or {},
            pnode_id,
            project=project,
            subset_data_size=subset_data_size,
        )

    wandb.agent(sweep_id, function=_fn, project=project, count=count)


def run_all_models_for_node(
    pnode_id: int,
    *,
    project: str,
    count: int = 10,
    subset_data_size: float = 1.0,
    include: Optional[list[ModelName]] = None,
    exclude: Optional[list[ModelName]] = None,
):
    """Run a basic sweep for every registered model for a single node.

    Parameters
    ----------
    pnode_id : int
        The PJM node id.
    project : str
        Weights & Biases project name.
    count : int
        Number of runs (configs) to try for each model.
    subset_data_size : float
        Fraction of most recent data to keep (0 < x <= 1).
    include : Optional[list[ModelName]]
        If provided, limit to this subset of models.
    exclude : Optional[list[ModelName]]
        If provided, skip these models.
    """
    reg = make_registry()
    names = list(reg.keys())
    if include:
        names = [n for n in names if n in set(include)]
    if exclude:
        names = [n for n in names if n not in set(exclude)]

    # Run sweeps sequentially, isolating failures per model
    for name in names:
        try:
            print(f"Starting sweep for {name.value} (count={count}) on node {pnode_id}")
            run_sweep_for_node(
                model_name=name,
                pnode_id=pnode_id,
                project=project,
                count=count,
                subset_data_size=subset_data_size,
            )
        except Exception as e:
            print(f"Sweep failed for {name.value}: {e}")


if __name__ == "__main__":
    # Example: run a brief sweep on a single node for a selected model
    run_sweep_for_node(
        model_name=ModelName.TCNMODEL,
        pnode_id=2156113094,
        project=WANDB_PROJECT_NAME,
        count=10,
        subset_data_size=0.01,
    )
