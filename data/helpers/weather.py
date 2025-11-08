import datetime as dt
import time
from pathlib import Path

import pandas as pd
import requests


def get_weather_last_5_years(
    latitude: float,
    longitude: float,
    hourly_vars=None,
    timezone: str = "auto",
) -> pd.DataFrame:
    """
    Fetch hourly weather data for the last 5 years for a given lat/lon
    using the Open-Meteo archive API (free, no API key).

    Parameters
    ----------
    latitude : float
        Latitude in decimal degrees.
    longitude : float
        Longitude in decimal degrees.
    hourly_vars : list[str], optional
        List of hourly variables to request. If None, a common default set is used.
        Examples (see Open-Meteo docs for full list):
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "apparent_temperature", "precipitation", "rain", "snowfall",
            "surface_pressure", "cloud_cover", "windspeed_10m", "winddirection_10m"
    timezone : str
        Timezone string, e.g. "auto", "UTC", "America/New_York".

    Returns
    -------
    pd.DataFrame
        DataFrame with a 'time' column (as pandas.DatetimeIndex-compatible)
        and one column per requested variable.
    """
    if hourly_vars is None:
        hourly_vars = [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "surface_pressure",
            "cloud_cover",
            "windspeed_10m",
            "winddirection_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "global_tilted_irradiance",
        ]

    end_date = dt.date.today()
    # crude 5-year window; adjust if you care about leap years
    start_date = end_date - dt.timedelta(days=5 * 365)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(hourly_vars),
        "timezone": timezone,
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Open-Meteo returns a dict like {"hourly": {"time": [...], "<var>": [...], ...}}
    hourly = data.get("hourly", {})
    if not hourly:
        raise ValueError(f"No hourly data returned for {latitude}, {longitude}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


if __name__ == "__main__":
    # Read assets, fetch 5-year hourly weather per asset, and write under data/weather/node_{pnode_id}
    data_dir = Path(__file__).resolve().parents[1]
    assets_csv = data_dir / "storage_assets.csv"
    out_root = data_dir / "weather"
    out_root.mkdir(parents=True, exist_ok=True)

    df_assets = pd.read_csv(assets_csv)

    # Determine column names robustly
    pnode_candidates = [
        "Inferred Pnode ID",
        "Inferred PnodeID",
        "Inferred Pnode",
        "Inferred_Pnode_ID",
    ]
    lat_candidates = ["Latitude", "latitude", "LATITUDE"]
    lon_candidates = ["Longitude", "longitude", "LONGITUDE"]

    pnode_col = next((c for c in pnode_candidates if c in df_assets.columns), None)
    lat_col = next((c for c in lat_candidates if c in df_assets.columns), None)
    lon_col = next((c for c in lon_candidates if c in df_assets.columns), None)
    if not pnode_col or not lat_col or not lon_col:
        raise KeyError(
            f"Missing required columns in {assets_csv}. "
            f"Found columns: {list(df_assets.columns)}"
        )

    # Clean and iterate rows
    sel = df_assets[[pnode_col, lat_col, lon_col]].copy()
    sel = sel.dropna(subset=[pnode_col, lat_col, lon_col])

    # Process each asset (row). To avoid duplicate downloads for the same pnode, collapse to unique combos.
    unique_rows = sel.drop_duplicates(subset=[pnode_col])

    def report_missing_hours(df: pd.DataFrame) -> int:
        t = pd.to_datetime(df["time"]).sort_values()
        # Ensure unique times before comparison
        t_unique = pd.DatetimeIndex(t.unique())
        expected = pd.date_range(
            t_unique.min(), t_unique.max(), freq="H", tz=t_unique.tz
        )
        missing = expected.difference(t_unique)
        return len(missing)

    for _, row in unique_rows.iterrows():
        # Skip non-integer PNODEs and avoid re-downloading if a CSV already exists
        try:
            pnode_id = int(row[pnode_col])
        except Exception:
            continue

        node_dir = out_root / f"node_{pnode_id}"
        if node_dir.exists():
            existing_files = list(node_dir.glob("weather_*.csv"))
            if existing_files:
                print(
                    f"  Skipping pnode {pnode_id}: found existing file {existing_files[0].name}"
                )
                continue
        try:
            pnode_id = int(row[pnode_col])
        except Exception:
            # Skip non-integer PNODEs
            continue
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        print(f"Fetching weather for pnode {pnode_id} at (lat,lon)=({lat},{lon})...")
        try:
            weather_df = get_weather_last_5_years(lat, lon)
        except Exception as e:
            print(f"  Failed to fetch pnode {pnode_id}: {e}")
            continue

        # Report missing hourly observations (if any)
        try:
            missing_count = report_missing_hours(weather_df)
        except Exception as e:
            print(f"  Warning: could not compute missing hours for {pnode_id}: {e}")
            missing_count = -1

        node_dir = out_root / f"node_{pnode_id}"
        node_dir.mkdir(parents=True, exist_ok=True)

        start_date = pd.to_datetime(weather_df["time"]).min().date()
        end_date = pd.to_datetime(weather_df["time"]).max().date()
        out_file = node_dir / f"weather_{start_date}_to_{end_date}.csv"
        weather_df.to_csv(out_file, index=False)

        if missing_count == 0:
            print(f"  Saved {out_file} (no missing hours)")
        elif missing_count > 0:
            print(f"  Saved {out_file} (missing hours: {missing_count})")
        else:
            print(f"  Saved {out_file} (missing-hour check unavailable)")

        time.sleep(15)  # be nice to the API
