import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.forecasting_model import (
    TransferableFutureCovariatesLocalForecastingModel,
)
from darts.models.forecasting.sf_model import StatsForecastModel
from darts.models.forecasting.torch_forecasting_model import (
    PastCovariatesTorchModel,
    TorchForecastingModel,
)
from darts.models.forecasting.xgboost import XGBModel

# Removed unused torchmetrics import; losses resolved via model builders.
import wandb
from data.data_output_functions import read_rt_da_with_weather
from forecasting.graphing import plot_opt_vs_perf_samples
from forecasting.metrics import calculate_metrics, short_horizon_pred_performance
from forecasting.model_zoo import ModelName, make_registry
from forecasting.transforms import name_to_transformer

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


def build_series_for_node(pnode_id: int) -> pd.DataFrame:
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

    return feature_df


def _prep_data(transformer_cls, feature_df: pd.DataFrame):
    df = feature_df.copy()
    df["lmp_rt"] = transformer_cls.transform(df["lmp_rt"])
    df["lmp_da"] = transformer_cls.transform(df["lmp_da"])
    target_series = TimeSeries.from_dataframe(
        df,
        time_col=None,
        value_cols="lmp_rt",
        freq="5min",
    ).astype(np.float32)
    future_covariates = TimeSeries.from_dataframe(
        df,
        time_col=None,
        value_cols=FEATURE_COLUMNS,
        freq="5min",
    ).astype(np.float32)
    split_idx = int(len(target_series) * 0.8)
    train_y, val_y = target_series[:split_idx], target_series[split_idx:]
    train_fut, val_fut = (
        future_covariates[:split_idx],
        future_covariates[split_idx:],
    )
    return train_y, val_y, train_fut, val_fut


def _get_preds_df(
    actual_val: TimeSeries,
    preds: list[TimeSeries],
) -> pd.DataFrame:
    # Initialize DataFrame index with actual validation timestamps
    all_index = pd.Index(actual_val.time_index)
    # Also include any future timestamps beyond validation (union with predictions)
    for pred in preds:
        all_index = all_index.union(pd.Index(pred.time_index))
    all_index = all_index.sort_values()
    df = pd.DataFrame(index=all_index)
    df["actual"] = np.nan
    # Fill actual where available
    df.loc[actual_val.time_index, "actual"] = actual_val.values().reshape(-1)

    # Allocate horizon columns
    for h in range(1, 25):
        df[f"h_{h}"] = np.nan

    # Insert shifted predictions: each pred_inv has timestamps t+1..t+24 after origin t
    for pred in preds:
        for h, ts in enumerate(pred.time_index, start=1):
            col = f"h_{h}"
            # Only set if empty to preserve earliest origin (optional policy)
            if pd.isna(df.at[ts, col]):
                df.at[ts, col] = pred.values()[h - 1]
            else:
                raise ValueError(f"Conflict at timestamp {ts} for horizon {col}")
    return df


def _post_run_logging(
    preds,
    actual_val,
    model,
    pnode_id,
    day_start: pd.Timestamp = None,
    week_start: pd.Timestamp = None,
    show_graphs: bool = False,
):
    print("Getting preds df")
    t0 = time.perf_counter()
    preds_df = _get_preds_df(
        actual_val,
        preds,
    )
    print(f"Preds df conversion took {time.perf_counter() - t0:.3f}s")

    print("Calculating metrics")
    t0 = time.perf_counter()
    metrics = calculate_metrics(preds_df)
    print(f"Calculating metrics took {time.perf_counter() - t0:.3f}s")

    print("Running opt metrics")
    t0 = time.perf_counter()
    opt_results = short_horizon_pred_performance(preds, preds_df)
    print(f"Running opt metrics took {time.perf_counter() - t0:.3f}s")

    # Save model using sweep + run name for traceability: <sweep>__<run>.pkl
    opt_dict = opt_results[1]
    if isinstance(opt_dict, dict):
        metrics.update({f"opt/{k}": v for k, v in opt_dict.items()})

    t0 = time.perf_counter()
    out_dir = Path("forecasting/outputs") / str(pnode_id) / model.__class__.__name__
    sweep_id = getattr(wandb.run, "sweep_id", None) or "nosweep"
    run_name = wandb.run.name or wandb.run.id

    def _sanitize(s: str) -> str:
        # Allow alnum, dot, dash, underscore; collapse others to _
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._-") or "run"

    base_name = f"{_sanitize(sweep_id)}__{_sanitize(run_name)}"
    save_dir = out_dir / f"{base_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(save_dir / "model.pkl"))
    plot_opt_vs_perf_samples(
        opt_results=opt_results,  # (combined_decisions_df, pct_dict)
        preds_df=preds_df,
        day=day_start,
        week_start=week_start,
        save_dir=save_dir,
        show=show_graphs,
    )

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model saving and graph generation took {time.perf_counter() - t0:.3f}s")

    # --- Log all metrics to W&B ---
    # Flatten already-flat dict; add a prefix for organization
    t0 = time.perf_counter()
    metrics_prefixed = {f"metrics/{k}": v for k, v in metrics.items()}
    wandb.summary.update(metrics_prefixed)
    wandb.summary["save_dir"] = str(save_dir)
    wandb.finish()
    print(f"W&B logging took {time.perf_counter() - t0:.3f}s")

    return metrics, opt_results, preds_df, save_dir


def train_hf_model(
    feature_df: pd.DataFrame,
    model_name: ModelName,
    config: Dict[str, Any],
    pnode_id: int,
    verbose: bool = False,
    post_run_logging: bool = False,
    day_start: pd.Timestamp = None,
    week_start: pd.Timestamp = None,
    show_graphs: bool = False,
):
    """Train model and build a horizon-aligned prediction DataFrame.

    Returns a DataFrame whose rows are target timestamps. Columns:
    - 'actual': inverse-transformed actual RT LMP (only where available in validation segment)
    - 'h_1'..'h_24': For a timestamp t, column h_k holds the k-step ahead
      prediction that was MADE at time t - k*freq. (Shifted so predictions
      align with their target timestamps.)
    Missing predictions remain NaN.
    """
    run_name = f"{model_name.value}-{pnode_id}-{wandb.util.generate_id()}"
    with wandb.init(
        project=WANDB_PROJECT_NAME,
        config=config,
        name=run_name,
    ):
        reg = make_registry()
        spec = reg[model_name]
        model = spec.builder(config)

        tt_name = str(config.get("target_transform", "NoneTransform"))
        transformer_cls = name_to_transformer(tt_name)
        train_y, val_y, train_fut, val_fut = _prep_data(transformer_cls, feature_df)

        # Normalize target (fit on transformed train segment only)
        y_transformer = Scaler()
        train_y_s = y_transformer.fit_transform(train_y)
        val_y_s = y_transformer.transform(val_y)
        feat_transformer = Scaler()
        train_fut_s = feat_transformer.fit_transform(train_fut)
        val_fut_s = feat_transformer.transform(val_fut)

        if isinstance(model, PastCovariatesTorchModel):
            if config.get("include_delayed_covariates", False):
                delay = model.output_chunk_length + config.get(
                    "covariate_delay_steps", 0
                )
                train_train_s = train_fut_s.shift(-delay)
                val_train_s = val_fut_s.shift(-delay)
                model.fit(
                    series=train_y_s,
                    past_covariates=train_train_s,
                    val_series=val_y_s,
                    val_past_covariates=val_train_s,
                    verbose=verbose,
                )
            else:
                model.fit(
                    series=train_y_s,
                    val_series=val_y_s,
                    verbose=verbose,
                )
        elif isinstance(
            model,
            (StatsForecastModel, TransferableFutureCovariatesLocalForecastingModel),
        ):
            model.fit(
                series=train_y_s,
                verbose=verbose,
            )
        elif isinstance(model, (TorchForecastingModel, XGBModel)):
            model.fit(
                series=train_y_s,
                future_covariates=train_fut_s,
                val_series=val_y_s,
                val_future_covariates=val_fut_s,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        # After training: clear callbacks if they were added so saving works reliably
        try:
            model.trainer_params["callbacks"] = []
            model._model_params["pl_trainer_kwargs"]["callbacks"] = []
        except Exception:
            pass

        t0 = time.perf_counter()
        if isinstance(model, PastCovariatesTorchModel):
            if config.get("include_delayed_covariates", False):
                raw_preds = model.predict(
                    n=24,
                    series=[
                        val_y_s[:i]
                        for i in range(model.input_chunk_length, len(val_y_s) - 24)
                    ],
                    past_covariates=[
                        val_train_s[: i + 24 - model.output_chunk_length]
                        for i in range(model.input_chunk_length, len(val_train_s) - 24)
                    ],
                    n_jobs=-1,
                )
            else:
                raw_preds = model.predict(
                    n=24,
                    series=[
                        val_y_s[:i]
                        for i in range(model.input_chunk_length, len(val_y_s) - 24)
                    ],
                    n_jobs=-1,
                )
        elif isinstance(
            model,
            (StatsForecastModel, TransferableFutureCovariatesLocalForecastingModel),
        ):
            raw_preds = []
            for i in range(10, len(val_y_s) - 24):
                raw_preds.append(
                    model.predict(
                        n=24,
                        series=val_y_s[:i],
                    )
                )
        elif isinstance(model, TorchForecastingModel):
            raw_preds = model.predict(
                n=24,
                series=[
                    val_y_s[:i]
                    for i in range(model.input_chunk_length, len(val_y_s) - 24)
                ],
                future_covariates=[
                    val_fut_s[: i + 24]
                    for i in range(model.input_chunk_length, len(val_fut_s) - 24)
                ],
                n_jobs=-1,
            )
        elif isinstance(model, XGBModel):
            lags = config.get("lags", 24)
            if not isinstance(lags, int):
                lags = max(lags)
            lags_future_covariates_p = config.get("lags_future_covariates_p", 12)
            lag = max(lags, lags_future_covariates_p)
            raw_preds = model.predict(
                n=24,
                series=[val_y_s[:i] for i in range(lag, len(val_y_s) - 24)],
                future_covariates=[
                    val_fut_s[: i + 24] for i in range(lag, len(val_fut_s) - 24)
                ],
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
        time_taken = time.perf_counter() - t0
        print(f"Predicting validation for model {run_name} took {time_taken:.3f}s")

        # Inverse transform actual validation series correctly (undo scaler then target transform)
        t0 = time.perf_counter()
        actual_val_norm = y_transformer.inverse_transform(val_y_s)
        actual_val = transformer_cls.inverse_transform_darts_timeseries(actual_val_norm)
        preds: list[TimeSeries] = []
        for pred in raw_preds:
            inv_norm = y_transformer.inverse_transform(pred)
            preds.append(transformer_cls.inverse_transform_darts_timeseries(inv_norm))
        time_taken = time.perf_counter() - t0
        print(
            f"Inverse transforming predictions for model {run_name} took {time_taken:.3f}s"
        )
        if post_run_logging:
            return _post_run_logging(
                preds,
                actual_val,
                model,
                pnode_id,
                day_start=day_start,
                week_start=week_start,
                show_graphs=show_graphs,
            )

        return preds, actual_val


def test_fut_cov_train():
    for model_name in [
        # ModelName.DLINEARMODEL,
        # ModelName.NLINEARMODEL,
        # ModelName.TIDEMODEL,
        # ModelName.TSMIXERMODEL,
        ModelName.XGBMODEL,
    ]:
        print(f"Testing model: {model_name}")
        feature_df = build_series_for_node(2156113094)
        feature_df = feature_df[-10000:]  # smaller data for test speed
        config = {"target_transform": "Clip", "n_epochs": 1, "model": "GRU"}
        preds, actual_val = train_hf_model(
            feature_df,
            model_name,
            config,
            verbose=True,
            pnode_id=2156113094,
        )
        print("Getting df")
        t0 = time.perf_counter()
        preds_df = _get_preds_df(
            actual_val,
            preds,
        )
        print(f"Getting df took {time.perf_counter() - t0:.3f}s")

        print("Calculating metrics")
        t0 = time.perf_counter()
        metrics = calculate_metrics(preds_df)
        print(f"Calculating metrics took {time.perf_counter() - t0:.3f}s")

        print("Running opt metrics")
        t0 = time.perf_counter()
        opt_results = short_horizon_pred_performance(preds, preds_df)
        print(opt_results[1])
        print(f"Running opt metrics took {time.perf_counter() - t0:.3f}s")


def test_post_run_logging():
    for model_name in [
        # ModelName.AUTO_ARIMA,
        #ModelName.RNNMODEL,
        #ModelName.TCNMODEL,
        #ModelName.NLINEARMODEL,
        ModelName.XGBMODEL,
    ]:
        print(f"Testing model: {model_name}")
        feature_df = build_series_for_node(2156113094)
        feature_df = feature_df[-50000:]  # smaller data for test speed
        config = {"target_transform": "Clip", "n_epochs": 1, "model": "LSTM"}
        metrics, opt_results, preds_df, save_dir = train_hf_model(
            feature_df,
            model_name,
            config,
            verbose=True,
            pnode_id=2156113094,
            post_run_logging=True,
            # day_start=feature_df.index[-9000].normalize(),
            # week_start=feature_df.index[-9000].normalize(),
            show_graphs=True,
        )


if __name__ == "__main__":
    # test_fut_cov_train()
    test_post_run_logging()
