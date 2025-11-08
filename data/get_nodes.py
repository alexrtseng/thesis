from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    data_dir = Path(__file__).resolve().parent
    src_csv = data_dir / "storage_assets.csv"
    out_csv = data_dir / "inferred_pnodes.csv"

    if not src_csv.exists():
        print(f"ERROR: Source CSV not found: {src_csv}", file=sys.stderr)
        return 1

    # Try pandas first for a convenient DataFrame, with a safe fallback to csv.DictReader.
    try:
        import pandas as pd  # type: ignore

        # Read all as strings to preserve large numeric IDs without scientific notation
        df = pd.read_csv(
            src_csv,
            dtype=str,
            na_values=["", "null", "NULL", "NaN", "N/A"],
            keep_default_na=True,
            engine="python",
        )

        required_cols = ["Inferred Pnode", "Inferred Pnode ID"]
        for col in required_cols:
            if col not in df.columns:
                print(
                    f"ERROR: Column '{col}' not found in {src_csv.name}.\n"
                    f"Available columns: {list(df.columns)}",
                    file=sys.stderr,
                )
                return 2

        sub = df[required_cols].copy()
        # Strip whitespace
        for col in required_cols:
            sub[col] = sub[col].astype(str).str.strip()

        # Drop NAs and empties
        sub = sub.dropna(how="any")
        sub = sub[(sub["Inferred Pnode"] != "") & (sub["Inferred Pnode ID"] != "")]

        # Deduplicate and sort for stable output
        sub = sub.drop_duplicates().sort_values(required_cols)

        # Write output
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_csv, index=False)

        print(
            f"Read {len(df)} rows from {src_csv.name}. Saved {len(sub)} unique inferred pnodes to {out_csv.name}."
        )
        return 0

    except ModuleNotFoundError:
        # Fallback to stdlib CSV if pandas isn't installed
        import csv

        unique_pairs = set()
        with src_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            # Validate columns
            fieldnames = reader.fieldnames or []
            if (
                "Inferred Pnode" not in fieldnames
                or "Inferred Pnode ID" not in fieldnames
            ):
                print(
                    "ERROR: Required columns not found in CSV. Found columns: "
                    + ", ".join(fieldnames),
                    file=sys.stderr,
                )
                return 2

            total = 0
            for row in reader:
                total += 1
                name = (row.get("Inferred Pnode") or "").strip()
                pid = (row.get("Inferred Pnode ID") or "").strip()
                if name and pid and name.lower() != "null" and pid.lower() != "null":
                    unique_pairs.add((name, pid))

        # Write output CSV
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Inferred Pnode", "Inferred Pnode ID"])
            for name, pid in sorted(unique_pairs):
                writer.writerow([name, pid])

        print(
            f"Read {total} rows from {src_csv.name}. Saved {len(unique_pairs)} unique inferred pnodes to {out_csv.name}."
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
