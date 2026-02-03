import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from darts import TimeSeries

from forecasting.test_best_models import TEST_HF_MODEL_RUNS, TEST_LF_MODEL_RUNS
from forecasting.test_forecaster_combo import produce_forecasts_for_eval


def default_cache_path(pnode_id: int, test_size: int, hf_horizon: int) -> Path:
    cache_dir = Path("forecasting") / "outputs" / "forecast_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir
        / f"forecast_cache_pnode={pnode_id}_test={test_size}_hfhor={hf_horizon}.pkl"
    )


def load_forecast_cache(cache_path: str | Path) -> dict[str, Any]:
    """Load the forecast cache pickle created by this module."""
    cache_path = Path(cache_path)
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "forecasts" not in payload:
        raise ValueError(
            "Invalid forecast cache format: expected dict with a 'forecasts' key"
        )
    return payload


def get_cached_forecasts(
    cache_path: str | Path, run_paths: list[str] | None = None
) -> dict[str, list[TimeSeries]]:
    """Retrieve cached forecasts as lists of Darts TimeSeries.

    Args:
        cache_path: Path to the pickle produced by this module.
        run_paths: Optional subset of W&B run paths to retrieve. If None, returns all.

    Returns:
        Mapping from run_path -> list[TimeSeries]
    """
    payload = load_forecast_cache(cache_path)
    forecasts = payload["forecasts"]
    if not isinstance(forecasts, dict):
        raise ValueError("Invalid forecast cache: 'forecasts' is not a dict")

    selected = (
        forecasts if run_paths is None else {rp: forecasts[rp] for rp in run_paths}
    )

    out: dict[str, list[TimeSeries]] = {}
    for rp, preds in selected.items():
        if not isinstance(preds, list):
            raise TypeError(f"Forecasts for {rp} are not a list")
        if len(preds) > 0 and not isinstance(preds[0], TimeSeries):
            raise TypeError(
                f"Forecasts for {rp} are not list[TimeSeries] (got {type(preds[0])})"
            )
        out[str(rp)] = preds
    return out


def populate_cache(pnode_id: int, test_size: int, hf_horizon: int) -> None:
    out_path = default_cache_path(pnode_id, test_size, hf_horizon)

    if out_path.exists():
        print("already exists")
        return

    print(
        f"Generating forecast cache for pnode_id={pnode_id}, test_size={test_size}, hf_horizon={hf_horizon}"
    )
    print(f"HF runs: {len(TEST_HF_MODEL_RUNS)} | LF runs: {len(TEST_LF_MODEL_RUNS)}")

    run_path_forecast_dict = produce_forecasts_for_eval(
        pnode_id=pnode_id,
        hf_run_paths=TEST_HF_MODEL_RUNS,
        lf_run_paths=TEST_LF_MODEL_RUNS,
        test_size=test_size,
        hf_horizon=hf_horizon,
    )

    payload = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "pnode_id": pnode_id,
            "test_size": test_size,
            "hf_horizon": hf_horizon,
            "hf_run_paths": list(TEST_HF_MODEL_RUNS),
            "lf_run_paths": list(TEST_LF_MODEL_RUNS),
        },
        "forecasts": run_path_forecast_dict,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote forecast cache to: {out_path}")


if __name__ == "__main__":
    populate_cache(2156113094, 800, 6)
    forecasts = get_cached_forecasts(
        default_cache_path(2156113094, 800, 6), run_paths=[TEST_HF_MODEL_RUNS[0]]
    )
    print(
        f"Loaded {len(forecasts)} runs from cache, first run has {len(forecasts[TEST_HF_MODEL_RUNS[0]])} forecasts"
    )
