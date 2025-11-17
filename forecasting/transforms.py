from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from darts import TimeSeries


# --- Delay / covariate utilities ---
def delayed_past_covariates(future_covs: TimeSeries, delay_steps: int) -> TimeSeries:
    """Return a TimeSeries of delayed covariates shifted by `delay_steps` steps.

    The first `delay_steps` rows become NaN. Caller should subsequently align and
    drop NaNs relative to the target series before model ingestion.
    """
    if delay_steps <= 0:
        return future_covs
    df = future_covs.pd_dataframe(copy=False)
    lagged = df.shift(delay_steps)
    # Rebuild TimeSeries (preserve freq if available)
    return TimeSeries.from_dataframe(
        lagged.reset_index(),
        time_col=lagged.index.name or "time",
        freq=future_covs.freq_str,
    )


def align_on_overlap(a: TimeSeries, b: TimeSeries) -> tuple[TimeSeries, TimeSeries]:
    """Slice both series to their common overlapping time interval."""
    idx_a = a.time_index
    idx_b = b.time_index
    common = idx_a.intersection(idx_b)
    if len(common) == 0:
        raise ValueError("No overlapping time index between series.")
    return a.slice_included(common[0], common[-1]), b.slice_included(
        common[0], common[-1]
    )


def drop_nan_rows(
    covs: TimeSeries, target: TimeSeries
) -> tuple[TimeSeries, TimeSeries]:
    """Drop rows where any covariate value is NaN; mirror drops on target."""
    df = covs.pd_dataframe(copy=False)
    mask = ~df.isna().any(axis=1)
    kept = df.index[mask]
    if len(kept) == 0:
        raise ValueError("All delayed covariate rows are NaN; reduce delay_steps.")
    return target.slice_included(kept[0], kept[-1]), covs.slice_included(
        kept[0], kept[-1]
    )


def name_to_transformer(name: str) -> "BaseTransformer":
    """
    Converts a string name to a corresponding BaseTransformer subclass.
    """
    transformers = {
        "NoneTransform": NoneTransform,
        "AsinhScaler": AsinhScaler,
        "SignedPowerScaler": SignedPowerScaler,
        "ScaledTanhScaler": ScaledTanhScaler,
        "Clip": Clip,
    }
    if name not in transformers:
        raise ValueError(f"Invalid transformer name: {name}")
    return transformers[name]


class BaseTransformer(ABC):
    @staticmethod
    @abstractmethod
    def transform(series: pd.Series, *args, **kwargs) -> pd.Series: ...

    @staticmethod
    @abstractmethod
    def inverse_transform(series: pd.Series, *args, **kwargs) -> pd.Series: ...

    @classmethod
    def inverse_transform_darts_timeseries(
        cls, series: TimeSeries, *args, **kwargs
    ) -> TimeSeries:
        df = series.to_dataframe()
        inv_df = df.apply(lambda col: cls.inverse_transform(col, *args, **kwargs))
        return TimeSeries.from_dataframe(inv_df, time_col=None)

    @classmethod
    def transform_darts_timeseries(
        cls, series: TimeSeries, *args, **kwargs
    ) -> TimeSeries:
        df = series.to_dataframe()
        tr_df = df.apply(lambda col: cls.transform(col, *args, **kwargs))
        return TimeSeries.from_dataframe(tr_df, time_col=None)


class NoneTransform(BaseTransformer):
    @staticmethod
    def transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        return pd.Series(series, index=series.index)

    @staticmethod
    def inverse_transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        return pd.Series(series, index=series.index)


class SignedLogScaler(BaseTransformer):
    epsilon = 1e-8

    @staticmethod
    def transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        result = np.sign(series) * np.log1p(np.abs(series) + SignedLogScaler.epsilon)
        return pd.Series(result, index=series.index)

    @staticmethod
    def inverse_transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        result = np.sign(series) * (np.expm1(np.abs(series)) - SignedLogScaler.epsilon)
        return pd.Series(result, index=series.index)


class AsinhScaler(BaseTransformer):
    @staticmethod
    def transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        result = np.arcsinh(series / 100.0)
        return pd.Series(result, index=series.index)

    @staticmethod
    def inverse_transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        result = 100.0 * np.sinh(series)
        return pd.Series(result, index=series.index)


class SignedPowerScaler(BaseTransformer):
    power = 0.5

    @staticmethod
    def transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        power = getattr(SignedPowerScaler, "power", 0.5)
        result = np.sign(series) * (np.abs(series) ** power)
        return pd.Series(result, index=series.index)

    @staticmethod
    def inverse_transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        power = getattr(SignedPowerScaler, "power", 0.5)
        result = np.sign(series) * (np.abs(series) ** (1 / power))
        return pd.Series(result, index=series.index)


class ScaledTanhScaler(BaseTransformer):
    scale = 100.0

    @staticmethod
    def transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        scale = getattr(ScaledTanhScaler, "scale", 100.0)
        result = np.tanh(series / scale)
        return pd.Series(result, index=series.index)

    @staticmethod
    def inverse_transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        scale = getattr(ScaledTanhScaler, "scale", 100.0)
        result = scale * np.arctanh(np.clip(series, -0.999999, 0.999999))
        return pd.Series(result, index=series.index)


class Clip(BaseTransformer):
    @staticmethod
    def transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        result = series.clip(upper=800)
        return pd.Series(result, index=series.index)

    @staticmethod
    def inverse_transform(series: pd.Series, *args, **kwargs) -> pd.Series:
        # Clipping is not invertible; return the series as is
        return pd.Series(series, index=series.index)
