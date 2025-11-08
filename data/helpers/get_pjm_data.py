import csv
import gc
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Timer

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

# Load API keys from .env or environment
load_dotenv()  # if .env file is present
api_keys_raw = os.getenv("PJM_API_KEYS")
if not api_keys_raw:
    raise RuntimeError("PJM_API_KEYS not set in environment")
api_keys = [key.strip() for key in api_keys_raw.split(",") if key.strip()]
if not api_keys:
    raise RuntimeError("No API keys provided in PJM_API_KEYS")


print("Loaded PJM_API_KEYS: " + ", ".join(k for k in api_keys))

# Constants for PJM API
PJM_API_ROOT = "https://api.pjm.com/api/v1"
BATCH_SIZE = (
    50000  # maximum rows per request (API limit):contentReference[oaicite:15]{index=15}
)
REQUESTS_PER_KEY_PER_MIN = (
    6  # rate limit per key (requests/min):contentReference[oaicite:16]{index=16}
)
# Per PJM metadata, feeds are archived after ~186 days; archived queries reject the 'fields' parameter.
ARCHIVE_CUTOFF_DAYS = 186

# Set up a pool of API keys to cycle through, allowing 6 uses per key per minute:contentReference[oaicite:17]{index=17}
available_keys = deque()
for key in api_keys:
    for _ in range(REQUESTS_PER_KEY_PER_MIN):
        available_keys.append(key)


def get_api_key():
    """Get an API key from the pool, recycling it after 60 seconds to enforce rate limits:contentReference[oaicite:18]{index=18}."""
    # Wait until a key is available
    while not available_keys:
        time.sleep(1)
    key = available_keys.popleft()
    # Return this key to the pool after 60 seconds (rate limiting)
    Timer(60.0, lambda k=key: available_keys.append(k)).start()
    return key


def fetch_one_day(date, endpoint, field_list):
    """Fetch all data for a given day and endpoint, returning a list of records (each as dict)."""
    # Construct the date range string: YYYY-MM-DDT00:00:00 to YYYY-MM-DDT23:59:00 (UTC)
    start_dt = datetime(date.year, date.month, date.day, 0, 0, 0)
    end_dt = start_dt + timedelta(days=1) - timedelta(minutes=1)
    range_param = f"{start_dt:%Y-%m-%dT%H:%M:%S} to {end_dt:%Y-%m-%dT%H:%M:%S}"
    params = {"download": "false", "datetime_beginning_utc": range_param}
    # Archived data (older than ~6 months) rejects the 'fields' parameter. For recent data, including
    # fields narrows payload and speeds up responses. We'll include fields only for non-archived dates
    # and add a runtime fallback if the API still rejects it.
    include_fields = (
        datetime.now(timezone.utc).date() - date
    ).days <= ARCHIVE_CUTOFF_DAYS
    if include_fields:
        params["fields"] = ",".join(field_list)

    all_records = []  # will collect all pages for this day
    start_row = 1
    while True:
        params["startRow"] = str(start_row)
        params["rowCount"] = str(BATCH_SIZE)
        print(
            f"[{endpoint}] Fetching rows {start_row} to {start_row + BATCH_SIZE - 1} for {date} ..."
        )
        # Build the request URL
        url = f"{PJM_API_ROOT}/{endpoint}"
        # Get an API key and make the request (with a 30s timeout)
        key = get_api_key()
        try:
            resp = requests.get(
                url,
                headers={
                    "Ocp-Apim-Subscription-Key": key,
                    "Accept": "application/json",
                },
                params=params,
                timeout=30,
            )
        except requests.RequestException as e:
            # Network or connection error, wait and retry
            print(
                f"[{endpoint}] Request error for {date}: {e}. Retrying in 30 seconds..."
            )
            time.sleep(30)
            continue
        if resp.status_code == 429:
            # Too Many Requests - wait and retry after 30s:contentReference[oaicite:21]{index=21}
            print(f"[{endpoint}] Rate limit hit for {date}, waiting 30 seconds...")
            time.sleep(1)
            continue
        if resp.status_code != 200:
            # If archived data rejects 'fields', remove and retry once
            if resp.status_code == 401:
                print(
                    f"[{endpoint}] Unauthorized (401). Check that your PJM API key is valid and enabled for Data Miner 2. "
                    f"Key used: {key}"
                )
                print(f"Response body: {resp.text}")
                www = resp.headers.get("WWW-Authenticate")
                if www:
                    print(f"WWW-Authenticate: {www}")
            elif resp.status_code == 403:
                print(
                    f"[{endpoint}] Forbidden (403). The key {key} may not have access to this feed, or IP restrictions are in place."
                )
            print(f"[{endpoint}] Error fetching data for {date}: {resp.text}")
            raise RuntimeError(
                f"[{endpoint}] Unexpected status code {resp.status_code} for {date}"
            )
        # Parse JSON response
        data = resp.json()
        items = data.get("items", [])
        total_rows = data.get("totalRows", 0)
        # Add this page of items to our results
        all_records.extend(items)
        # If we've retrieved all rows, exit loop
        if start_row + BATCH_SIZE > total_rows:
            break
        # Otherwise, get next page
        start_row += BATCH_SIZE
    return all_records


def _load_inferred_sets():
    """Load inferred pnodes from data/inferred_pnodes.csv into two sets (ids, names)."""
    data_dir = Path(__file__).resolve().parents[1]
    inferred_csv = data_dir / "inferred_pnodes.csv"
    ids, names = set(), set()
    if not inferred_csv.exists():
        print(f"WARNING: {inferred_csv} not found. No filtering will be applied.")
        return ids, names
    with inferred_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("Inferred Pnode") or "").strip()
            pid = (row.get("Inferred Pnode ID") or "").strip()
            if name and name.lower() != "nan" and name.lower() != "null":
                names.add(name)
            if pid and pid.lower() != "nan" and pid.lower() != "null":
                ids.add(pid)
    print(f"Loaded inferred pnodes: {len(ids)} IDs, {len(names)} names")
    return ids, names


def _to_id_str(v):
    if v is None:
        return None
    try:
        # Ensure integers stay integers when stringified
        if isinstance(v, float):
            if v.is_integer():
                return str(int(v))
        return str(v)
    except Exception:
        return str(v)


def _open_csv_with_header(path: str, header: list[str]):
    """Open CSV for appending; write header only if file is empty/new."""
    p = Path(path)
    exists = p.exists()
    f = open(path, "a" if exists and p.stat().st_size > 0 else "w", newline="")
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(header)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    return f, writer


def _resume_start_date(csv_path: str, default_start):
    """If CSV exists, resume from the day after the last date written; else default_start.

    Assumes first column is 'datetime_beginning_utc'. If parsing fails, falls back to default_start.
    """
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return default_start
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return default_start
            try:
                idx = header.index("datetime_beginning_utc")
            except ValueError:
                idx = 0
            last_date = None
            for row in reader:
                if len(row) <= idx:
                    continue
                ts = row[idx]
                if not ts:
                    continue
                # Normalize ISO, drop timezone letter if present
                tsv = ts.replace("Z", "")
                try:
                    d = datetime.fromisoformat(tsv).date()
                except Exception:
                    continue
                last_date = d
            if last_date is None:
                return default_start
            return max(default_start, last_date + timedelta(days=1))
    except Exception:
        return default_start


# Define endpoints and fields for real-time and day-ahead LMPs (from integration-svc):contentReference[oaicite:22]{index=22}:contentReference[oaicite:23]{index=23}
rt_endpoint = "rt_fivemin_hrl_lmps"  # 5-minute real-time LMPs (estimated):contentReference[oaicite:24]{index=24}
da_endpoint = (
    "da_hrl_lmps"  # hourly day-ahead LMPs:contentReference[oaicite:25]{index=25}
)

# Request vs output fields: request includes pnode_name for filtering; output excludes
# pnode_name, voltage, equipment, type, zone per user request.
rt_request_fields = [
    "datetime_beginning_utc",
    "pnode_id",
    "pnode_name",
    "voltage",
    "equipment",
    "type",
    "zone",
    "system_energy_price_rt",
    "total_lmp_rt",
    "congestion_price_rt",
    "marginal_loss_price_rt",
]
rt_output_fields = [
    "datetime_beginning_utc",
    "pnode_id",
    "system_energy_price_rt",
    "total_lmp_rt",
    "congestion_price_rt",
    "marginal_loss_price_rt",
]
da_request_fields = [
    "datetime_beginning_utc",
    "pnode_id",
    "pnode_name",
    "type",
    "zone",
    "system_energy_price_da",
    "total_lmp_da",
    "congestion_price_da",
    "marginal_loss_price_da",
]
da_output_fields = [
    "datetime_beginning_utc",
    "pnode_id",
    "system_energy_price_da",
    "total_lmp_da",
    "congestion_price_da",
    "marginal_loss_price_da",
]

# Determine date range: last 5 years up to yesterday (to have full days)
end_date = datetime.now(timezone.utc).date() - timedelta(days=1)
start_date = end_date - timedelta(days=5 * 365)  # ~5 years ago (approximate)
print(f"Fetching data from {start_date} to {end_date} ...")
infer_ids, infer_names = _load_inferred_sets()


def _process_endpoint_parallel(
    out_dir: Path,
    header: list[str],
    endpoint: str,
    request_fields: list[str],
    output_fields: list[str],
    infer_ids: set[str],
    infer_names: set[str],
    start_date,
    end_date,
    max_workers: int = 3,
):
    """Fetch multiple days concurrently and write sequentially (chronological)."""
    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    def _month_file_for_date(day):
        return out_dir / f"{endpoint}_{day:%Y-%m}.csv"

    def _resume_from_dir(default_start):
        # Determine the last written date across existing monthly files
        if not out_dir.exists():
            return default_start
        last = None
        for p in sorted(out_dir.glob(f"{endpoint}_*.csv")):
            try:
                with open(p, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header_row = next(reader, None)
                    if not header_row:
                        continue
                    try:
                        idx = header_row.index("datetime_beginning_utc")
                    except ValueError:
                        idx = 0
                    for row in reader:
                        if len(row) <= idx:
                            continue
                        ts = row[idx]
                        if not ts:
                            continue
                        tsv = ts.replace("Z", "")
                        try:
                            d = datetime.fromisoformat(tsv).date()
                        except Exception:
                            continue
                        if last is None or d > last:
                            last = d
            except Exception:
                continue
        if last is None:
            return default_start
        return max(default_start, last + timedelta(days=1))

    # Build date list from resume point in monthly directory
    resume_date = _resume_from_dir(start_date)
    dates = []
    d = resume_date
    while d <= end_date:
        dates.append(d)
        d = d + timedelta(days=1)
    if not dates:
        print(f"[{endpoint}] Up to date; nothing to fetch.")
        return

    # Submit fetches with concurrency
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for day in dates:
            futures[day] = ex.submit(fetch_one_day, day, endpoint, request_fields)

        # Write per-day in chronological order to the appropriate monthly file
        for day in dates:
            try:
                # Remove the Future so its internal result can be freed after use
                future = futures.pop(day)
                records = future.result()
            except Exception as e:
                print(f"[{endpoint}] Error fetching data for {day}: {e}")
                continue
            wrote = 0
            month_file = _month_file_for_date(day)
            with _open_csv_with_header(str(month_file), header)[0] as out_csv:
                writer = csv.writer(out_csv)
                for rec in records:
                    pid = _to_id_str(rec.get("pnode_id"))
                    pname = (rec.get("pnode_name") or "").strip()
                    if (pid and pid in infer_ids) or (pname and pname in infer_names):
                        row = [rec.get(col, "") for col in output_fields]
                        writer.writerow(row)
                        wrote += 1
                out_csv.flush()
                try:
                    os.fsync(out_csv.fileno())
                except OSError:
                    pass
            # Drop references and occasionally run GC
            records = None
            future = None
            gc.collect()

            print(
                f"[{endpoint}] {day}: wrote {wrote} rows (filtered) -> {month_file.name}"
            )
            # If this day is the last day of its month (next day is a new month or past end), upload
            next_day = day + timedelta(days=1)
            is_month_complete = next_day.month != day.month or next_day > end_date
            if is_month_complete:
                bucket_arn = os.getenv(
                    "PJM_S3_BUCKET_ARN", "arn:aws:s3:::pjmlmpsthesis"
                )
                # Extract bucket name from ARN (last token after :::)
                bucket_name = bucket_arn.split(":::")[-1]
                key = f"{endpoint}/{month_file.name}"
                try:
                    s3 = boto3.client("s3")
                    s3.upload_file(str(month_file), bucket_name, key)
                    print(
                        f"[S3] Uploaded {month_file.name} to s3://{bucket_name}/{key}"
                    )
                except (BotoCoreError, ClientError) as e:
                    print(f"[S3] Failed to upload {month_file.name}: {e}")


# Fetch Real-Time LMPs (5-minute) and Day-Ahead (hourly) using parallel fetching
data_root = Path(__file__).resolve().parents[1]
monthly_dir = data_root / "pjm_lmps"
rt_dir = monthly_dir  # same folder, different filename prefix per endpoint
da_dir = monthly_dir

# Allow customizing worker count via env var PJM_WORKERS
try:
    MAX_WORKERS = int(os.getenv("PJM_WORKERS", "3"))
    if MAX_WORKERS < 1:
        MAX_WORKERS = 1
except Exception:
    MAX_WORKERS = 3

print(f"Using up to {MAX_WORKERS} concurrent day fetch workers per endpoint.")

_process_endpoint_parallel(
    rt_dir,
    rt_output_fields,
    rt_endpoint,
    rt_request_fields,
    rt_output_fields,
    infer_ids,
    infer_names,
    start_date,
    end_date,
    MAX_WORKERS,
)
_process_endpoint_parallel(
    da_dir,
    da_output_fields,
    da_endpoint,
    da_request_fields,
    da_output_fields,
    infer_ids,
    infer_names,
    start_date,
    end_date,
    MAX_WORKERS,
)

print(f"Data fetching complete. Outputs written monthly under {monthly_dir}")
