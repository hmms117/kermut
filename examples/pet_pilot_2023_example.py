"""Example utilities for the PET Pilot 2023 dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Mapping

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tournament"
VENDOR = DATA_ROOT / "vendor"


def head(path: Path, limit: int = 5) -> List[Mapping[str, str]]:
    """Return the first ``limit`` records from a CSV file."""

    rows: List[Mapping[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for _, row in zip(range(limit), reader):
            rows.append(row)
    return rows


def count_rows(path: Path) -> int:
    """Return the number of data rows in a CSV file."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # drop header
        return sum(1 for _ in reader)


def main() -> None:
    """Showcase loading the in vitro submission table."""

    if not VENDOR.exists():
        raise SystemExit(
            "PET Pilot 2023 submodule is missing. Run `git submodule update --init` first."
        )

    invitro_path = VENDOR / "in_vitro" / "input" / "pet_invitro_submissions.csv"
    if not invitro_path.exists():
        raise SystemExit(f"Could not find expected CSV at {invitro_path}")

    total_rows = count_rows(invitro_path)
    preview = head(invitro_path)

    print(f"Loaded {total_rows} rows from {invitro_path.relative_to(ROOT)}")
    if preview:
        fieldnames = preview[0].keys()
        print(f"Columns: {', '.join(fieldnames)}")
        print("First rows:")
        for row in preview:
            print({key: row[key] for key in fieldnames})
    else:
        print("File appears to be empty.")


if __name__ == "__main__":
    main()
