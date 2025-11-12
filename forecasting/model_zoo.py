from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict

import torch
from darts.models import (
    ARIMA,
    FFT,
    BlockRNNModel,
    DLinearModel,
    ExponentialSmoothing,
    KalmanForecaster,
    NBEATSModel,
    NHiTSModel,
    NLinearModel,
    RNNModel,
    TCNModel,
    TFTModel,
    Theta,
    TiDEModel,
    TransformerModel,
    TSMixerModel,
    XGBModel,
)
from darts.utils.statistics import SeasonalityMode  # add this import

from forecasting.torch_utils import configure_fp32_precision
from forecasting.wandb_callback import wandb_logger

# Configure float32/TF32 precision using new APIs when available
configure_fp32_precision(os.getenv("TORCH_MATMUL_PRECISION", "high"))


class ModelName(Enum):
    ARIMA = "ARIMA"
    EXPONENTIALSMOOTHING = "ExponentialSmoothing"
    THETA = "Theta"
    FFT = "FFT"
    KALMANFORECASTER = "KalmanForecaster"
    RNNMODEL = "RNNModel"
    BLOCKRNNMODEL = "BlockRNNModel"
    NBEATSMODEL = "NBEATSModel"
    NHITSMODEL = "NHiTSModel"
    TCNMODEL = "TCNModel"
    TRANSFORMERMODEL = "TransformerModel"
    TFTMODEL = "TFTModel"
    DLINEARMODEL = "DLinearModel"
    NLINEARMODEL = "NLinearModel"
    TIDEMODEL = "TiDEModel"
    TSMIXERMODEL = "TSMixerModel"
    XGBMODEL = "XGBModel"


@dataclass(frozen=True)
class ModelSpec:
    name: ModelName
    builder: Callable[[Dict[str, Any]], Any]
    supports_past: bool
    supports_future: bool
    supports_static: bool
    default_params: Dict[str, Any]
    uses_gpu: bool = False  # True if the underlying implementation can leverage a GPU via PyTorch/Lightning

    def sweep_config(self) -> Dict[str, Any]:
        """
        Minimal-yet-effective sweep spaces tuned for ~20 trials.
        Assumes hourly data; if you're on 5-min, replace {24,168} with {288,2016}.
        """
        common = {
            "metric": {"name": "best_val_loss", "goal": "minimize"},
            "method": "bayes",
        }
        name = self.name.value.lower()

        if name in {"arima"}:
            params = {
                "p": {"distribution": "int_uniform", "min": 0, "max": 2},
                "d": {"distribution": "int_uniform", "min": 0, "max": 2},
                "q": {"distribution": "int_uniform", "min": 0, "max": 2},
            }
        elif name in {"exponentialsmoothing"}:
            params = {
                "trend": {"values": [None, "add"]},
                "seasonal": {"values": ["add"]},
                "seasonal_periods": {"values": [24, 168]},
            }
        elif name in {"theta"}:
            params = {
                "season_mode": {"values": ["additive", "multiplicative"]},
                "seasonality_period": {"values": [24, 168]},
            }
        elif name in {"fft"}:
            params = {
                "nr_freqs_to_keep": {
                    "distribution": "int_uniform",
                    "min": 6,
                    "max": 32,
                },
            }
        elif name in {"kalmanforecaster"}:
            params = {
                "dim_x": {"distribution": "int_uniform", "min": 2, "max": 8},
            }
        elif name in {"rnnmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 72,
                    "max": 192,
                },
                "output_chunk_length": {"values": [24]},
                "hidden_dim": {"distribution": "int_uniform", "min": 32, "max": 256},
                "n_rnn_layers": {"distribution": "int_uniform", "min": 1, "max": 3},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
                "model": {"values": ["LSTM", "GRU"]},
                "n_epochs": {"values": [20]},
            }
        elif name in {"blockrnnmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 72,
                    "max": 192,
                },
                "output_chunk_length": {"values": [24]},
                "hidden_dim": {"distribution": "int_uniform", "min": 32, "max": 256},
                "n_rnn_layers": {"distribution": "int_uniform", "min": 1, "max": 3},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
                "model": {"values": ["LSTM", "GRU"]},
                "n_epochs": {"values": [20]},
            }
        elif name in {"nbeatsmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 96,
                    "max": 168,
                },
                "output_chunk_length": {"values": [24]},
                "n_epochs": {"values": [20]},
                "include_delayed_covariates": {"values": [False, True]},
                "delayed_steps": {"distribution": "int_uniform", "min": 0, "max": 48},
            }
        elif name in {"nhitsmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 96,
                    "max": 168,
                },
                "output_chunk_length": {"values": [24]},
                "n_epochs": {"values": [20]},
                "include_delayed_covariates": {"values": [False, True]},
                "delayed_steps": {"distribution": "int_uniform", "min": 0, "max": 440},
            }
        elif name in {"tcnmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 72,
                    "max": 192,
                },
                "output_chunk_length": {"values": [24]},
                "kernel_size": {"values": [3, 5]},
                "num_filters": {"distribution": "int_uniform", "min": 16, "max": 64},
                "dilation_base": {"values": [2]},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
                "n_epochs": {"values": [20]},
                "include_delayed_covariates": {"values": [False, True]},
                "delayed_steps": {"distribution": "int_uniform", "min": 0, "max": 440},
            }
        elif name in {"transformermodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 96,
                    "max": 168,
                },
                "output_chunk_length": {"values": [24]},
                "d_model": {"distribution": "int_uniform", "min": 64, "max": 256},
                "nhead": {"distribution": "int_uniform", "min": 2, "max": 8},
                "num_encoder_layers": {
                    "distribution": "int_uniform",
                    "min": 2,
                    "max": 4,
                },
                "num_decoder_layers": {
                    "distribution": "int_uniform",
                    "min": 1,
                    "max": 3,
                },
                "n_epochs": {"values": [20]},
                "include_delayed_covariates": {"values": [False, True]},
                "delayed_steps": {"distribution": "int_uniform", "min": 0, "max": 440},
            }
        elif name in {"tftmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 72,
                    "max": 168,
                },
                "output_chunk_length": {"values": [24]},
                "hidden_size": {"distribution": "int_uniform", "min": 32, "max": 128},
                "lstm_layers": {"distribution": "int_uniform", "min": 1, "max": 3},
                "n_epochs": {"values": [20]},
            }
        elif name in {"dlinearmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 168,
                    "max": 336,
                },
                "output_chunk_length": {"values": [24]},
                "n_epochs": {"values": [20]},
            }
        elif name in {"nlinearmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 168,
                    "max": 336,
                },
                "output_chunk_length": {"values": [24]},
                "n_epochs": {"values": [20]},
            }
        elif name in {"tidemodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 168,
                    "max": 336,
                },
                "output_chunk_length": {"values": [24]},
                "hidden_size": {"distribution": "int_uniform", "min": 32, "max": 256},
                "num_layers": {"values": [2]},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.2},
                "n_epochs": {"values": [20]},
            }
        elif name in {"tsmixermodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 168,
                    "max": 336,
                },
                "output_chunk_length": {"values": [24]},
                "hidden_size": {"distribution": "int_uniform", "min": 32, "max": 256},
                "num_layers": {"values": [2]},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.2},
                "n_epochs": {"values": [20]},
            }
        elif name in {"xgbmodel"}:
            params = {
                "lags": {
                    "values": [
                        [1, 2, 24, 168],
                        [1, 2, 3, 24, 168],
                        [1, 24, 48, 168],
                    ]
                },
                "output_chunk_length": {"values": [24]},
                "max_depth": {"distribution": "int_uniform", "min": 3, "max": 10},
                "learning_rate": {
                    "distribution": "log_uniform",
                    "min": 0.01,
                    "max": 0.3,
                },
                "n_estimators": {"distribution": "int_uniform", "min": 200, "max": 800},
                "subsample": {"distribution": "uniform", "min": 0.5, "max": 1.0},
                "colsample_bytree": {"distribution": "uniform", "min": 0.5, "max": 1.0},
                "min_child_weight": {"distribution": "int_uniform", "min": 1, "max": 6},
                "reg_lambda": {"distribution": "log_uniform", "min": 1e-3, "max": 10.0},
                "reg_alpha": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-1},
            }
        else:
            params = {"n_epochs": {"values": [20]}}

        # Shared target transform hyperparameter across models
        params["target_transform"] = {
            "values": [
                "NoneTransform",
                "AsinhScaler",
                "SignedLogScaler",
                "ScaledTanhScaler",
            ]
        }
        # Optional numeric knobs (activated only if chosen transform referenced outside sweep)
        params["signed_power"] = {"values": [0.5, 0.75]}
        params["tanh_scale"] = {"values": [50.0, 100.0, 200.0]}
        return {**common, "parameters": params}


def _torch_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer explicit settings; otherwise auto-detect GPU/MPS and fall back to CPU
    accel_cfg = (
        config.get("accelerator") or os.getenv("TORCH_ACCELERATOR") or ""
    ).lower()
    if accel_cfg:
        accelerator = accel_cfg
    else:
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
        else:
            accelerator = "cpu"

    devices = int(config.get("devices", os.getenv("TORCH_DEVICES", 1)))
    precision = config.get("precision") or os.getenv("TORCH_PRECISION") or "16-mixed"
    # Allow disabling mixed precision explicitly
    if precision not in {"16-mixed", "bf16-mixed", "32"}:
        precision = "16-mixed"
    accumulate = int(config.get("accumulate_grad_batches", 1))
    grad_clip = float(config.get("gradient_clip_val", 0.0))
    trainer_kwargs = {
        "callbacks": [wandb_logger()],
        "accelerator": accelerator,
        "devices": devices,
        "enable_progress_bar": False,
        "precision": precision,
        "benchmark": True,  # enable cuDNN autotuner for speed on constant shapes
    }
    if accumulate > 1:
        trainer_kwargs["accumulate_grad_batches"] = accumulate
    if grad_clip > 0:
        trainer_kwargs["gradient_clip_val"] = grad_clip
    return {
        "random_state": int(config.get("random_state", 42)),
        "dropout": float(config.get("dropout", 0.0)),
        "n_epochs": int(config.get("n_epochs", 10)),
        "batch_size": int(config.get("batch_size", 32)),
        "pl_trainer_kwargs": trainer_kwargs,
        "save_checkpoints": False,
    }


def _filtered_torch_kwargs(model_cls, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter generic torch kwargs so we don't pass unsupported ones
    (e.g., dropout not in DLinearModel/NLinearModel).
    """
    base = _torch_kwargs(config)
    sig = inspect.signature(model_cls.__init__)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in base.items() if k in allowed}


def make_registry() -> Dict[ModelName, ModelSpec]:
    reg: Dict[ModelName, ModelSpec] = {}

    reg[ModelName.ARIMA] = ModelSpec(
        name=ModelName.ARIMA,
        supports_past=False,
        supports_future=False,
        supports_static=False,
        default_params={},
        builder=lambda cfg: ARIMA(
            p=int(cfg.get("p", 1)),
            d=int(cfg.get("d", 1)),
            q=int(cfg.get("q", 0)),
        ),
    )

    reg[ModelName.EXPONENTIALSMOOTHING] = ModelSpec(
        name=ModelName.EXPONENTIALSMOOTHING,
        supports_past=False,
        supports_future=False,
        supports_static=False,
        default_params={},
        builder=lambda cfg: ExponentialSmoothing(
            trend=cfg.get("trend"),
            seasonal=cfg.get("seasonal"),
            seasonal_periods=int(cfg.get("seasonal_periods", 288)),
        ),
    )

    reg[ModelName.THETA] = ModelSpec(
        name=ModelName.THETA,
        supports_past=False,
        supports_future=False,
        supports_static=False,
        default_params={},
        builder=lambda cfg: Theta(
            season_mode=(
                SeasonalityMode.MULTIPLICATIVE
                if str(cfg.get("season_mode", "additive")).lower() == "multiplicative"
                else SeasonalityMode.ADDITIVE
            ),
            seasonality_period=int(
                cfg.get("seasonality_period", cfg.get("season_period", 288))
            ),
        ),
    )

    reg[ModelName.FFT] = ModelSpec(
        name=ModelName.FFT,
        supports_past=False,
        supports_future=False,
        supports_static=False,
        default_params={},
        builder=lambda cfg: FFT(nr_freqs_to_keep=int(cfg.get("nr_freqs_to_keep", 10))),
    )

    reg[ModelName.KALMANFORECASTER] = ModelSpec(
        name=ModelName.KALMANFORECASTER,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        builder=lambda cfg: KalmanForecaster(dim_x=int(cfg.get("dim_x", 2))),
    )

    # Torch-based Global models
    reg[ModelName.RNNMODEL] = ModelSpec(
        name=ModelName.RNNMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: RNNModel(
            model=cfg.get("model", "LSTM"),
            input_chunk_length=int(cfg.get("input_chunk_length", 60)),
            output_chunk_length=int(cfg.get("output_chunk_length", 20)),
            training_length=int(cfg.get("input_chunk_length", 60))
            + int(cfg.get("output_chunk_length", 20)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            n_rnn_layers=int(cfg.get("n_rnn_layers", 1)),
            **_torch_kwargs(cfg),
        ),
    )

    reg[ModelName.BLOCKRNNMODEL] = ModelSpec(
        name=ModelName.BLOCKRNNMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        # does not need training_length
        builder=lambda cfg: BlockRNNModel(
            model=cfg.get("model", "LSTM"),
            input_chunk_length=int(cfg.get("input_chunk_length", 60)),
            output_chunk_length=int(cfg.get("output_chunk_length", 20)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            n_rnn_layers=int(cfg.get("n_rnn_layers", 1)),
            **_filtered_torch_kwargs(BlockRNNModel, cfg),
        ),
    )

    reg[ModelName.NBEATSMODEL] = ModelSpec(
        name=ModelName.NBEATSMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: NBEATSModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            **_filtered_torch_kwargs(NBEATSModel, cfg),
        ),
    )

    reg[ModelName.NHITSMODEL] = ModelSpec(
        name=ModelName.NHITSMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: NHiTSModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            **_filtered_torch_kwargs(NHiTSModel, cfg),
        ),
    )

    reg[ModelName.TCNMODEL] = ModelSpec(
        name=ModelName.TCNMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: TCNModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 60)),
            output_chunk_length=int(cfg.get("output_chunk_length", 20)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            num_filters=int(cfg.get("num_filters", 16)),
            dilation_base=int(cfg.get("dilation_base", 2)),
            **_filtered_torch_kwargs(TCNModel, cfg),
        ),
    )

    reg[ModelName.TRANSFORMERMODEL] = ModelSpec(
        name=ModelName.TRANSFORMERMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: TransformerModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            d_model=int(cfg.get("d_model", 128)),
            nhead=int(cfg.get("nhead", 4)),
            num_encoder_layers=int(cfg.get("num_encoder_layers", 2)),
            num_decoder_layers=int(cfg.get("num_decoder_layers", 2)),
            **_filtered_torch_kwargs(TransformerModel, cfg),
        ),
    )

    reg[ModelName.TFTMODEL] = ModelSpec(
        name=ModelName.TFTMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=True,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: TFTModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            hidden_size=int(cfg.get("hidden_size", 32)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
            **_filtered_torch_kwargs(TFTModel, cfg),
        ),
    )

    reg[ModelName.DLINEARMODEL] = ModelSpec(
        name=ModelName.DLINEARMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: DLinearModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            **_filtered_torch_kwargs(DLinearModel, cfg),
        ),
    )

    reg[ModelName.NLINEARMODEL] = ModelSpec(
        name=ModelName.NLINEARMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: NLinearModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            **_filtered_torch_kwargs(NLinearModel, cfg),
        ),
    )

    reg[ModelName.TIDEMODEL] = ModelSpec(
        name=ModelName.TIDEMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: TiDEModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            hidden_size=int(cfg.get("hidden_size", 64)),
            # removed num_layers (not in signature)
            **_filtered_torch_kwargs(TiDEModel, cfg),
        ),
    )

    reg[ModelName.TSMIXERMODEL] = ModelSpec(
        name=ModelName.TSMIXERMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        builder=lambda cfg: TSMixerModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            hidden_size=int(cfg.get("hidden_size", 64)),
            # removed num_layers (not in signature)
            **_filtered_torch_kwargs(TSMixerModel, cfg),
        ),
    )

    reg[ModelName.XGBMODEL] = ModelSpec(
        name=ModelName.XGBMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        builder=lambda cfg: XGBModel(
            lags=cfg.get("lags", [-1, -2, -24]),
            lags_future_covariates=list(
                range(0, int(cfg.get("output_chunk_length", 24)))
            ),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            max_depth=int(cfg.get("max_depth", 5)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            n_estimators=int(cfg.get("n_estimators", 300)),
            subsample=float(cfg.get("subsample", 1.0)),
            colsample_bytree=float(cfg.get("colsample_bytree", 1.0)),
            min_child_weight=int(cfg.get("min_child_weight", 1)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            reg_alpha=float(cfg.get("reg_alpha", 0.0)),
            random_state=int(cfg.get("random_state", 42)),
        ),
    )

    return reg
