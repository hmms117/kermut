"""Quick inspection helper for the AAV Fit4Function dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Mapping

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "aav_fit4function"
VENDOR = DATA_ROOT / "vendor"


def head(path: Path, limit: int = 5) -> List[Mapping[str, str]]:
    """Return the first ``limit`` records from a CSV file."""

    rows: List[Mapping[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for _, row in zip(range(limit), reader):
            rows.append(row)
    return rows


def main() -> None:
    """Print the first few rows of the screening summary table."""

    if not VENDOR.exists():
        raise SystemExit(
            "Fit4Function submodule is missing. Run `git submodule update --init` first."
        )

    screen_path = VENDOR / "data" / "fit4function_library_screens.csv"
    if not screen_path.exists():
        raise SystemExit(f"Expected screening CSV at {screen_path}")

    preview = head(screen_path)
    print(f"Previewing {screen_path.relative_to(ROOT)}")
    if preview:
        fieldnames = preview[0].keys()
        print(f"Columns: {', '.join(fieldnames)}")
        for row in preview:
            print({key: row[key] for key in fieldnames})
    else:
        print("File appears to be empty.")


if __name__ == "__main__":
    main()
