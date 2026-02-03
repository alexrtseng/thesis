import json
import time
from pathlib import Path

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

import wandb
from forecasting.metrics import (
    calculate_metrics,
    long_horizon_pred_performance,
    short_and_long_pred_performance,
    short_horizon_pred_performance,
)
from forecasting.model_zoo import model_name_to_class, model_name_to_enum
from forecasting.sweep_runner import _hf_slice_feature_df, _lf_slice_features
from forecasting.train import (
    WANDB_API_KEY,
    _get_preds_df,
    _prep_data,
    _prep_data_lf,
    build_series_for_node,
    predict_hf_model,
    predict_lf_model,
)
from forecasting.transforms import name_to_transformer

WANDB_ENTITY = "watt-our"

_model_dict = {}


def _load_model_from_outputs(run_path: str):
    # Cache loaded models

    if run_path not in _model_dict:
        api = wandb.Api(api_key=WANDB_API_KEY)
        run = api.run(run_path)
        cfg = run.config
        summ = run.summary
        model_class = summ["model_class"]
        ModelClass = model_name_to_class(model_class)

        model_dir = Path(summ["save_dir"])
        model_file = model_dir / "model.pkl"
        print(f"Loading model from {model_file}")
        # Darts models support classmethod `load`
        _model_dict[run_path] = ModelClass.load(str(model_file)), cfg, summ, run
    else:
        print(f"Using cached model for {run_path}")
    return _model_dict[run_path]


def _ind_test_logging(
    preds,
    actual_val,
    size,
    _out_dir: Path,
    day_start: pd.Timestamp = None,
    week_start: pd.Timestamp = None,
    show_graphs: bool = False,
    lf: bool = False,
    lmp_series: pd.Series = None,
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
    if lf:
        assert lmp_series is not None
        opt_results = long_horizon_pred_performance(preds, lmp_series)
    else:
        opt_results = short_horizon_pred_performance(preds, preds_df, True)
    print(f"Running opt metrics took {time.perf_counter() - t0:.3f}s")

    # Save model using sweep + run name for traceability: <sweep>__<run>.pkl
    opt_dict = opt_results[1]
    if isinstance(opt_dict, dict):
        metrics.update({f"opt/{k}": v for k, v in opt_dict.items()})

    t0 = time.perf_counter()
    if lf:
        out_dir = _out_dir / "lf"
    else:
        out_dir = _out_dir / "hf"

    out_dir.mkdir(parents=True, exist_ok=True)

    ## Plotting is too redundant for the combo tests
    # plot_opt_vs_perf_samples(
    #     opt_results=opt_results,  # (combined_decisions_df, pct_dict)
    #     preds_df=preds_df,
    #     day=day_start,
    #     week_start=week_start,
    #     save_dir=out_dir,
    #     show=show_graphs,
    # )

    metrics_path = out_dir / "metrics.json"
    metrics["test_size"] = size
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Model saving and graph generation took {time.perf_counter() - t0:.3f}s")

    ## Residual plots are too redundant for the combo tests
    # t0 = time.perf_counter()
    # plot_residuals_and_save(preds_df, out_dir)
    # print(f"Residual plotting took {time.perf_counter() - t0:.3f}s")

    return metrics, opt_results, preds_df, out_dir


def _create_transformers(cfg, summ, feature_df, hourly_feature_dfs, lf: bool = False):
    tt_name = str(cfg.get("target_transform", "NoneTransform"))
    transformer_cls = name_to_transformer(tt_name)
    subset_data_size = summ["subset_data_size"]
    test_size = summ["omitted_test_size"]
    if lf:
        _hourly_feature_dfs = _lf_slice_features(
            hourly_feature_dfs, subset_data_size=subset_data_size, test_size=test_size
        )
        train_y_series, _, train_fut_series, _ = _prep_data_lf(
            transformer_cls, feature_df, _hourly_feature_dfs
        )
    else:
        model_name = model_name_to_enum(summ["model_class"])
        _feature_df = _hf_slice_feature_df(
            model_name, feature_df, subset_data_size, test_size
        )
        train_y_series, _, train_fut_series, _ = _prep_data(
            transformer_cls, _feature_df
        )

    # Normalize target (fit on transformed train segment only)
    y_transformer = Scaler(global_fit=True)
    y_transformer.fit(train_y_series)
    feat_transformer = Scaler(global_fit=True)
    feat_transformer.fit(train_fut_series)

    return y_transformer, feat_transformer


def _generate_pjm_da_preds(feature_df: pd.DataFrame) -> pd.Series:
    """Generate deterministic arbitrage predictions for the test set."""
    prices_df = feature_df[["lmp_da"]].copy()

    preds: list[TimeSeries] = []
    for i in range(0, len(prices_df) - 24):
        da_prices_pred = prices_df["lmp_da"].iloc[i + 1 : i + 1 + 12 * 24 : 12]
        preds.append(TimeSeries.from_series(da_prices_pred))

    return preds


def evaluate_hf_lf_pair(
    pnode_id: int,
    hf_run_path: str,
    lf_run_path: str | None,
    test_size: int | None = None,
    pjm_da_preds: bool = False,
):
    t1 = time.perf_counter()
    feature_df = build_series_for_node(pnode_id)
    if pjm_da_preds:
        lf_run_path = None
    else:
        assert lf_run_path is not None, (
            "lf_run_path must be provided if pjm_da_preds is False"
        )

    hf_model, hf_cfg, hf_summ, hf_run = _load_model_from_outputs(hf_run_path)
    if test_size is None:
        test_size = hf_summ["omitted_test_size"] // 12  # convert to hourly

    # Load models
    if not pjm_da_preds:
        lf_model, lf_cfg, lf_summ, lf_run = _load_model_from_outputs(lf_run_path)
        feature_df["lmp_lf_avg"] = (
            feature_df["lmp_rt"].rolling(window=13, center=True, min_periods=1).mean()
        )
        # get scalars that would have been used during training
        hourly_feature_dfs = [feature_df.iloc[i::12].copy() for i in range(12)]
        lf_scalar_y, lf_scalar_feat = _create_transformers(
            lf_cfg, lf_summ, feature_df, hourly_feature_dfs, lf=True
        )
        # Slice test set
        for i in range(len(hourly_feature_dfs)):
            hourly_feature_dfs[i] = hourly_feature_dfs[i][
                : len(hourly_feature_dfs[11])
            ]  # align lengths
        for i in range(len(hourly_feature_dfs)):
            hourly_feature_dfs[i] = hourly_feature_dfs[i][-test_size:]

    hf_scalar_y, hf_scalar_feat = _create_transformers(
        hf_cfg, hf_summ, feature_df, None
    )

    feature_df = feature_df[-(test_size * 12) :]

    if not pjm_da_preds:
        # get lf series
        lf_tt_name = str(lf_cfg.get("target_transform", "NoneTransform"))
        lf_transformer_cls = name_to_transformer(lf_tt_name)
        _, lf_y_series, _, lf_fut_series = _prep_data_lf(
            lf_transformer_cls, feature_df, hourly_feature_dfs, train_size=0.0
        )
        lf_y_s = lf_scalar_y.transform(lf_y_series)
        lf_fut_s = lf_scalar_feat.transform(lf_fut_series)
        lf_past_s = None
        if lf_cfg.get("include_delayed_covariates", False):
            delay = lf_model.output_chunk_length + lf_cfg.get(
                "covariate_delay_steps", 0
            )
            lf_past_s = []
            for i in range(len(lf_fut_s)):
                lf_past_s.append(lf_fut_s[i].shift(-delay))

    # get hf series
    hf_tt_name = str(hf_cfg.get("target_transform", "NoneTransform"))
    hf_transformer_cls = name_to_transformer(hf_tt_name)
    _, hf_y_series, _, hf_fut_series = _prep_data(
        hf_transformer_cls, feature_df, train_size=0.0
    )
    hf_y_s = hf_scalar_y.transform(hf_y_series)
    hf_fut_s = hf_scalar_feat.transform(hf_fut_series)
    hf_past_s = None
    if hf_cfg.get("include_delayed_covariates", False):
        delay = hf_model.output_chunk_length + hf_cfg.get("covariate_delay_steps", 0)
        hf_past_s = hf_fut_s.shift(-delay)

    # get preds
    if not pjm_da_preds:
        lf_raw_preds = predict_lf_model(
            lf_model, lf_y_s, lf_fut_s, lf_past_s, lf_cfg, None
        )
    hf_raw_preds = predict_hf_model(hf_model, hf_y_s, hf_fut_s, hf_past_s, hf_cfg, None)

    # Inverse transform hf preds
    t0 = time.perf_counter()
    hf_actual_val = hf_transformer_cls.inverse_transform_darts_timeseries(hf_y_series)
    hf_preds: list[TimeSeries] = []
    for pred in hf_raw_preds:
        inv_norm = hf_scalar_y.inverse_transform(pred)
        hf_preds.append(hf_transformer_cls.inverse_transform_darts_timeseries(inv_norm))
    time_taken = time.perf_counter() - t0
    print(f"Inverse transforming hf predictions for model took {time_taken:.3f}s")

    # Inverse transform lf preds
    if not pjm_da_preds:
        t0 = time.perf_counter()
        lf_actual_val = feature_df["lmp_lf_avg"].loc[
            lf_raw_preds[0].time_index[0] : lf_raw_preds[-1].time_index[-1]
        ]
        lf_actual_val = TimeSeries.from_series(lf_actual_val)
        lmp_series = feature_df["lmp_rt"].loc[
            lf_raw_preds[0].time_index[0] - pd.Timedelta(minutes=5) : lf_raw_preds[
                -1
            ].time_index[0]
        ]
        lf_preds: list[TimeSeries] = []
        for pred in lf_raw_preds:
            inv_norm = lf_scalar_y.inverse_transform(pred)
            lf_preds.append(
                lf_transformer_cls.inverse_transform_darts_timeseries(inv_norm)
            )
        time_taken = time.perf_counter() - t0
        print(f"Inverse transforming hf predictions for model took {time_taken:.3f}s")

    out_dir = (
        Path("forecasting/outputs")
        / "tests"
        / str(pnode_id)
        / f"{hf_run.name}and{lf_run.name}"
        if not pjm_da_preds
        else Path("forecasting/outputs")
        / "tests"
        / str(pnode_id)
        / f"{hf_run.name}andPJMDa"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _, joint_opt_metrics = short_and_long_pred_performance(
        hf_preds,
        lf_preds if not pjm_da_preds else _generate_pjm_da_preds(feature_df),
        feature_df,
    )

    _ind_test_logging(
        hf_preds,
        hf_actual_val,
        test_size * 12,
        out_dir,
        lf=False,
    )

    if not pjm_da_preds:
        _ind_test_logging(
            lf_preds,
            lf_actual_val,
            test_size * 12,
            out_dir,
            lf=True,
            lmp_series=lmp_series,
        )

    time_taken = time.perf_counter() - t1
    joint_opt_metrics["total_time_taken_s"] = time_taken
    joint_opt_metrics["eval_start"] = feature_df.index[0].strftime("%Y-%m-%d %H:%M:%S")
    joint_opt_metrics["eval_end"] = feature_df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    joint_opt_metrics["Int_test_size"] = test_size

    metrics_path = out_dir / "joint_opt_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(joint_opt_metrics, f, indent=2)

    return joint_opt_metrics, hf_run.name, lf_run.name if not pjm_da_preds else "PJMDa"


def produce_forecasts_for_eval(
    pnode_id: int,
    hf_run_paths: list[str] | None,
    lf_run_paths: list[str] | None,
    test_size: int | None = None,
    hf_horizon: int = 6,
):
    hf_run_paths = hf_run_paths or []
    lf_run_paths = lf_run_paths or []
    if len(hf_run_paths) == 0 or len(lf_run_paths) == 0:
        raise ValueError(
            "Pass at least one HF and LF run path (or set pjm_da_preds=True)."
        )
    if hf_horizon <= 0:
        raise ValueError("hf_horizon must be positive")

    feature_df_full = build_series_for_node(pnode_id)
    feature_df_full["lmp_lf_avg"] = (
        feature_df_full["lmp_rt"].rolling(window=13, center=True, min_periods=1).mean()
    )

    # Load all models
    hf_loaded = [(_load_model_from_outputs(p), p) for p in hf_run_paths]
    lf_loaded = [(_load_model_from_outputs(p), p) for p in lf_run_paths]

    if test_size is None:
        # Prefer a conservative test_size that works across all HF runs.
        candidates: list[int] = []
        for (model, cfg, summ, run), _path in hf_loaded:
            candidates.append(int(summ["omitted_test_size"]) // 12)
        test_size = min(candidates)

    # Prepare LF hourly features on the *full* dataset (needed to fit scalers on train).
    hourly_feature_dfs_full = [feature_df_full.iloc[i::12].copy() for i in range(12)]
    for i in range(len(hourly_feature_dfs_full)):
        hourly_feature_dfs_full[i] = hourly_feature_dfs_full[i][
            : len(hourly_feature_dfs_full[11])
        ]

    # Slice to test window at 5-min resolution (used for prediction generation).
    feature_df = feature_df_full[-(test_size * 12) :].copy()
    hourly_feature_dfs = [feature_df.iloc[i::12].copy() for i in range(12)]
    for i in range(len(hourly_feature_dfs)):
        hourly_feature_dfs[i] = hourly_feature_dfs[i][: len(hourly_feature_dfs[11])]

    run_path_forecast_dict = {}
    # --- Generate HF predictions for each HF model ---
    for (hf_model, hf_cfg, hf_summ, hf_run), _path in hf_loaded:
        # Fit scalers on training portion derived from the full feature dataframe.
        hf_scalar_y, hf_scalar_feat = _create_transformers(
            hf_cfg, hf_summ, feature_df_full, None
        )

        hf_tt_name = str(hf_cfg.get("target_transform", "NoneTransform"))
        hf_transformer_cls = name_to_transformer(hf_tt_name)
        _, hf_y_series, _, hf_fut_series = _prep_data(
            hf_transformer_cls, feature_df, train_size=0.0
        )
        hf_y_s = hf_scalar_y.transform(hf_y_series)
        hf_fut_s = hf_scalar_feat.transform(hf_fut_series)
        hf_past_s = None
        if hf_cfg.get("include_delayed_covariates", False):
            delay = hf_model.output_chunk_length + hf_cfg.get(
                "covariate_delay_steps", 0
            )
            hf_past_s = hf_fut_s.shift(-delay)

        hf_raw_preds = predict_hf_model(
            hf_model, hf_y_s, hf_fut_s, hf_past_s, hf_cfg, None
        )
        preds: list[TimeSeries] = []
        for pred in hf_raw_preds:
            inv_norm = hf_scalar_y.inverse_transform(pred)
            preds.append(
                hf_transformer_cls.inverse_transform_darts_timeseries(inv_norm)
            )
        print(f"HF model {hf_run.name} generated {len(preds)} predictions.")

        run_path_forecast_dict[_path] = preds

    # --- Generate LF predictions for each LF model (or DA baseline) ---
    for (lf_model, lf_cfg, lf_summ, lf_run), _path in lf_loaded:
        # Fit scalers on training portion derived from the full feature dataframe.
        lf_scalar_y, lf_scalar_feat = _create_transformers(
            lf_cfg, lf_summ, feature_df_full, hourly_feature_dfs_full, lf=True
        )
        lf_tt_name = str(lf_cfg.get("target_transform", "NoneTransform"))
        lf_transformer_cls = name_to_transformer(lf_tt_name)
        _, lf_y_series, _, lf_fut_series = _prep_data_lf(
            lf_transformer_cls, feature_df, hourly_feature_dfs, train_size=0.0
        )
        lf_y_s = lf_scalar_y.transform(lf_y_series)
        lf_fut_s = lf_scalar_feat.transform(lf_fut_series)
        lf_past_s = None
        if lf_cfg.get("include_delayed_covariates", False):
            delay = lf_model.output_chunk_length + lf_cfg.get(
                "covariate_delay_steps", 0
            )
            lf_past_s = []
            for i in range(len(lf_fut_s)):
                lf_past_s.append(lf_fut_s[i].shift(-delay))

        lf_raw_preds = predict_lf_model(
            lf_model, lf_y_s, lf_fut_s, lf_past_s, lf_cfg, None
        )
        preds: list[TimeSeries] = []
        for pred in lf_raw_preds:
            inv_norm = lf_scalar_y.inverse_transform(pred)
            preds.append(
                lf_transformer_cls.inverse_transform_darts_timeseries(inv_norm)
            )
        print(f"LF model {lf_run.name} generated {len(preds)} predictions.")

        run_path_forecast_dict[_path] = preds

    return run_path_forecast_dict


# testing in evaluate hf lf pair ensemble
