from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict

import torch
from darts.dataprocessing.transformers import Scaler
from darts.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    BlockRNNModel,
    DLinearModel,
    KalmanForecaster,
    NBEATSModel,
    NHiTSModel,
    NLinearModel,
    RNNModel,
    TCNModel,
    TFTModel,
    TiDEModel,
    TransformerModel,
    TSMixerModel,
    XGBModel,
)
from darts.models.forecasting.torch_forecasting_model import (
    PastCovariatesTorchModel,
)
from pytorch_lightning.callbacks import EarlyStopping

# (Removed direct torchmetrics imports; using torch.nn losses via _resolve_loss())
from forecasting.wandb_callback import wandb_logger

# Configure float32/TF32 precision using new APIs when available
early_stopper = EarlyStopping(monitor="val_loss", patience=5, mode="min")


class ModelName(Enum):
    AUTO_ARIMA = "AutoARIMA"
    AUTO_EXPONENTIALSMOOTHING = "AutoETS"
    AUTO_THETA = "AutoTheta"
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


class ModelCategory(Enum):
    """High-level grouping of forecasting models.

    STATISTICAL: classic/time-series or tree/boosted models usually CPU-bound with few/no deep hyperparameters.
    GLOBAL: neural network (PyTorch Lightning) global models benefiting from GPU acceleration & richer sweeps.
    """

    STATISTICAL = "statistical"
    GLOBAL = "global"


@dataclass(frozen=True)
class ModelSpec:
    name: ModelName
    builder: Callable[[Dict[str, Any]], Any]
    supports_past: bool
    supports_future: bool
    supports_static: bool
    default_params: Dict[str, Any]
    uses_gpu: bool = False  # True if the underlying implementation can leverage a GPU via PyTorch/Lightning
    category: ModelCategory = (
        ModelCategory.STATISTICAL
    )  # default conservative; override for global/torch models
    max_needed_runs: int = 1000

    def sweep_config(self) -> Dict[str, Any]:
        """Return a W&B sweep configuration tailored to the model category.

        For GLOBAL (deep torch) models: keep richer Bayesian spaces (≈20 trials default).
        For STATISTICAL models: provide a tiny grid/random space (usually <5 parameters) to avoid over-sweeping.
        """
        common = {
            "metric": {"name": "best_val_loss", "goal": "minimize"},
            # Default method; statistical models may override below
            "method": "bayes",
        }
        name = self.name.value.lower()
        # ------------------------------
        # Statistical models (CPU, few params)
        # ------------------------------
        if self.category is ModelCategory.STATISTICAL:
            if name == "kalmanforecaster":
                params = {"dim_x": {"values": [2, 4, 6]}}
            elif name == "xgbmodel":
                # Retain modest space; XGB benefits from a couple knobs
                params = {
                    "lags": {
                        "values": [0, 1, 24, 72, [-1, -2, -4, -12, -24, -72, -168]]
                    },
                    # Use proper W&B distribution syntax (not nested under values)
                    "lags_future_covariates_p": {
                        "distribution": "int_uniform",
                        "min": 0,
                        "max": 36,
                    },
                    "lags_future_covariates_f": {
                        "distribution": "int_uniform",
                        "min": 0,
                        "max": 36,
                    },
                    "max_depth": {"values": [4, 6, 8]},
                    "learning_rate": {"values": [0.03, 0.1]},
                    "n_estimators": {"values": [300, 600]},
                }
            else:
                params = {}
            # Always include at least one parameter for a valid sweep config
            # Keep transforms optional but lightweight; choose NoneTransform by default
            params["target_transform"] = {
                "values": ["NoneTransform", "AsinhScaler", "Clip"]
            }
            return {**common, "parameters": params}

        # ------------------------------
        # Global (Torch) models (GPU capable, richer spaces)
        # ------------------------------
        if name in {"rnnmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 72,
                    "max": 192,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "hidden_dim": {"distribution": "int_uniform", "min": 32, "max": 256},
                "n_rnn_layers": {"distribution": "int_uniform", "min": 1, "max": 3},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
                "model": {"values": ["LSTM", "GRU", "RNN"]},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"blockrnnmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 72,
                    "max": 192,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "hidden_dim": {"distribution": "int_uniform", "min": 32, "max": 256},
                "n_rnn_layers": {"distribution": "int_uniform", "min": 1, "max": 3},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
                "model": {"values": ["LSTM", "GRU", "RNN"]},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"nbeatsmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 96,
                    "max": 168,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "include_delayed_covariates": {"values": [False, True]},
                "delayed_steps": {"distribution": "int_uniform", "min": 0, "max": 48},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"nhitsmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 96,
                    "max": 168,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "include_delayed_covariates": {"values": [False, True]},
                "delayed_steps": {"distribution": "int_uniform", "min": 0, "max": 440},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"tcnmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 72,
                    "max": 192,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "kernel_size": {"values": [3, 5]},
                "num_filters": {"distribution": "int_uniform", "min": 16, "max": 64},
                "dilation_base": {"values": [2]},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.3},
                "include_delayed_covariates": {"values": [False, True]},
                "delayed_steps": {"distribution": "int_uniform", "min": 0, "max": 440},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"transformermodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 96,
                    "max": 168,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
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
                "include_delayed_covariates": {"values": [False, True]},
                "delayed_steps": {"distribution": "int_uniform", "min": 0, "max": 440},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"tftmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 72,
                    "max": 168,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "hidden_size": {"distribution": "int_uniform", "min": 32, "max": 128},
                "lstm_layers": {"distribution": "int_uniform", "min": 1, "max": 3},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"dlinearmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 168,
                    "max": 336,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"nlinearmodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 168,
                    "max": 336,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"tidemodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 168,
                    "max": 336,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "hidden_size": {"distribution": "int_uniform", "min": 32, "max": 256},
                "num_layers": {"values": [2]},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.2},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"tsmixermodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 168,
                    "max": 336,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "hidden_size": {"distribution": "int_uniform", "min": 32, "max": 256},
                "num_layers": {"values": [2]},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.2},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        elif name in {"tsmixermodel"}:
            params = {
                "input_chunk_length": {
                    "distribution": "int_uniform",
                    "min": 168,
                    "max": 336,
                },
                "output_chunk_length": {"values": [1, 3, 6, 12, 24]},
                "hidden_size": {"distribution": "int_uniform", "min": 32, "max": 256},
                "num_layers": {"values": [2]},
                "dropout": {"distribution": "uniform", "min": 0.0, "max": 0.2},
                "loss_fn": {"values": ["MeanAbsoluteError", "MeanSquaredError"]},
            }
        else:  # fallback for any future GLOBAL model additions
            params = {}

        # Shared target transform hyperparameter across models
        params["target_transform"] = {
            "values": [
                "NoneTransform",
                "AsinhScaler",
                "Clip",
            ]
        }
        params["time_features"] = {
            "values": [
                "none",
                "all",
                "cyclical",
                "attributes",
            ]
        }
        # Optional numeric knobs (activated only if chosen transform referenced outside sweep)
        return {**common, "parameters": params}


def _torch_kwargs(model_cls, config: Dict[str, Any]) -> Dict[str, Any]:
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
        "callbacks": [wandb_logger(), early_stopper],
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
    encoder = {}
    time_features = config.get("time_features", "none")

    if time_features == "all":
        encoder = {
            "datetime_attribute": {
                "past": ["minute", "hour", "dayofweek", "month"],
                "future": ["minute", "hour", "dayofweek", "month"]
                if model_cls != PastCovariatesTorchModel
                else [],
                "transformer": Scaler(),
            },
            "cyclic": {
                "past": ["minute", "hour", "dayofweek", "month"],
                "future": ["minute", "hour", "dayofweek", "month"]
                if model_cls != PastCovariatesTorchModel
                else [],
                "transformer": Scaler(),
            },
        }
    if time_features == "cyclical":
        encoder = {
            "cyclic": {
                "past": ["minute", "hour", "dayofweek", "month"],
                "future": ["minute", "hour", "dayofweek", "month"]
                if model_cls != PastCovariatesTorchModel
                else [],
                "transformer": Scaler(),
            }
        }
    if time_features == "attributes":
        encoder = {
            "datetime_attribute": {
                "past": ["minute", "hour", "dayofweek", "month"],
                "future": ["minute", "hour", "dayofweek", "month"]
                if model_cls != PastCovariatesTorchModel
                else [],
                "transformer": Scaler(),
            }
        }

    args = {
        "random_state": int(config.get("random_state", 42)),
        "n_epochs": int(
            config.get("n_epochs", 100)
        ),  # should have an early stopper; override available
        "batch_size": int(config.get("batch_size", 64)),
        "pl_trainer_kwargs": trainer_kwargs,
        "add_encoders": encoder,
        "save_checkpoints": False,
    }
    if model_cls not in {DLinearModel, NLinearModel}:
        args["dropout"] = float(config.get("dropout", 0.0))

    return args


def make_registry() -> Dict[ModelName, ModelSpec]:
    reg: Dict[ModelName, ModelSpec] = {}

    # Helper to resolve loss function selection
    def _resolve_loss(cfg: Dict[str, Any]):
        name = str(cfg.get("loss_fn", "MeanAbsoluteError"))
        if name == "MeanSquaredError":
            return torch.nn.MSELoss()
        # default MAE
        return torch.nn.L1Loss() if name == "MeanAbsoluteError" else torch.nn.L1Loss()

    # ---------------- STATISTICAL MODELS -----------------
    reg[ModelName.AUTO_ARIMA] = ModelSpec(
        name=ModelName.AUTO_ARIMA,
        supports_past=False,
        supports_future=False,
        supports_static=False,
        default_params={},
        category=ModelCategory.STATISTICAL,
        builder=lambda cfg: AutoARIMA(),
        max_needed_runs=3,
    )

    reg[ModelName.AUTO_EXPONENTIALSMOOTHING] = ModelSpec(
        name=ModelName.AUTO_EXPONENTIALSMOOTHING,
        supports_past=False,
        supports_future=False,
        supports_static=False,
        default_params={},
        category=ModelCategory.STATISTICAL,
        builder=lambda cfg: AutoETS(),
        max_needed_runs=3,
    )

    reg[ModelName.AUTO_THETA] = ModelSpec(
        name=ModelName.AUTO_THETA,
        supports_past=False,
        supports_future=False,
        supports_static=False,
        default_params={},
        category=ModelCategory.STATISTICAL,
        builder=lambda cfg: AutoTheta(),
        max_needed_runs=3,
    )

    reg[ModelName.KALMANFORECASTER] = ModelSpec(
        name=ModelName.KALMANFORECASTER,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        category=ModelCategory.STATISTICAL,
        builder=lambda cfg: KalmanForecaster(dim_x=int(cfg.get("dim_x", 2))),
        max_needed_runs=15,
    )

    # Torch-based Global models
    # ---------------- GLOBAL (TORCH) MODELS -----------------
    reg[ModelName.RNNMODEL] = ModelSpec(
        name=ModelName.RNNMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: RNNModel(
            model=cfg.get("model", "LSTM"),
            input_chunk_length=int(cfg.get("input_chunk_length", 60)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            training_length=int(cfg.get("input_chunk_length", 60))
            + int(cfg.get("output_chunk_length", 20)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            n_rnn_layers=int(cfg.get("n_rnn_layers", 1)),
            **_torch_kwargs(RNNModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.BLOCKRNNMODEL] = ModelSpec(
        name=ModelName.BLOCKRNNMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        # does not need training_length
        builder=lambda cfg: BlockRNNModel(
            model=cfg.get("model", "LSTM"),
            input_chunk_length=int(cfg.get("input_chunk_length", 60)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            hidden_dim=int(cfg.get("hidden_dim", 64)),
            n_rnn_layers=int(cfg.get("n_rnn_layers", 1)),
            **_torch_kwargs(BlockRNNModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.NBEATSMODEL] = ModelSpec(
        name=ModelName.NBEATSMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: NBEATSModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            **_torch_kwargs(NBEATSModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.NHITSMODEL] = ModelSpec(
        name=ModelName.NHITSMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: NHiTSModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            **_torch_kwargs(NHiTSModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.TCNMODEL] = ModelSpec(
        name=ModelName.TCNMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: TCNModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 60)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            kernel_size=int(cfg.get("kernel_size", 3)),
            num_filters=int(cfg.get("num_filters", 16)),
            dilation_base=int(cfg.get("dilation_base", 2)),
            **_torch_kwargs(TCNModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.TRANSFORMERMODEL] = ModelSpec(
        name=ModelName.TRANSFORMERMODEL,
        supports_past=True,
        supports_future=False,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: TransformerModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            d_model=int(cfg.get("d_model", 128)),
            nhead=int(cfg.get("nhead", 4)),
            num_encoder_layers=int(cfg.get("num_encoder_layers", 2)),
            num_decoder_layers=int(cfg.get("num_decoder_layers", 2)),
            **_torch_kwargs(TransformerModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.TFTMODEL] = ModelSpec(
        name=ModelName.TFTMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=True,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: TFTModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            hidden_size=int(cfg.get("hidden_size", 32)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
            **_torch_kwargs(TFTModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.DLINEARMODEL] = ModelSpec(
        name=ModelName.DLINEARMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: DLinearModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            **_torch_kwargs(DLinearModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.NLINEARMODEL] = ModelSpec(
        name=ModelName.NLINEARMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: NLinearModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            **_torch_kwargs(NLinearModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.TIDEMODEL] = ModelSpec(
        name=ModelName.TIDEMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: TiDEModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            hidden_size=int(cfg.get("hidden_size", 64)),
            # removed num_layers (not in signature)
            **_torch_kwargs(TiDEModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    reg[ModelName.TSMIXERMODEL] = ModelSpec(
        name=ModelName.TSMIXERMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        uses_gpu=True,
        category=ModelCategory.GLOBAL,
        builder=lambda cfg: TSMixerModel(
            input_chunk_length=int(cfg.get("input_chunk_length", 72)),
            output_chunk_length=int(cfg.get("output_chunk_length", 24)),
            hidden_size=int(cfg.get("hidden_size", 64)),
            # removed num_layers (not in signature)
            **_torch_kwargs(TSMixerModel, cfg),
            loss_fn=_resolve_loss(cfg),
        ),
    )

    # XGB is tree-based & CPU; keep as STATISTICAL
    reg[ModelName.XGBMODEL] = ModelSpec(
        name=ModelName.XGBMODEL,
        supports_past=True,
        supports_future=True,
        supports_static=False,
        default_params={},
        category=ModelCategory.STATISTICAL,
        builder=lambda cfg: XGBModel(
            lags=cfg.get("lags", 24),
            lags_future_covariates=(
                cfg.get("lags_future_covariates_p", 12),
                cfg.get("lags_future_covariates_f", 12),
            ),
            output_chunk_length=24,
            max_depth=int(cfg.get("max_depth", 5)),
            learning_rate=float(cfg.get("learning_rate", 0.05)),
            n_estimators=int(cfg.get("n_estimators", 300)),
            subsample=float(cfg.get("subsample", 1.0)),
            colsample_bytree=float(cfg.get("colsample_bytree", 1.0)),
            min_child_weight=int(cfg.get("min_child_weight", 1)),
            early_stopping_rounds=10,
        ),
    )

    return reg


def get_statistical_registry() -> Dict[ModelName, ModelSpec]:
    """Convenience accessor returning only statistical (CPU-oriented) models."""
    return {
        k: v
        for k, v in make_registry().items()
        if v.category is ModelCategory.STATISTICAL
    }


def get_global_registry() -> Dict[ModelName, ModelSpec]:
    """Convenience accessor returning only global (Torch, GPU-capable) models."""
    return {
        k: v for k, v in make_registry().items() if v.category is ModelCategory.GLOBAL
    }
