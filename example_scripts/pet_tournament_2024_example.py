"""Placeholder example for the PET 2024 dataset."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "tournament2024"
VENDOR = DATA_ROOT / "vendor"


def main() -> None:
    """Report on the availability of the 2024 vendor data."""

    if VENDOR.exists() and any(VENDOR.iterdir()):
        print("PET 2024 vendor data is present. Extend this script to inspect tables.")
    else:
        print(
            "PET 2024 vendor submodule is not initialized."
            "\nFollow the README instructions in data/tournament2024 to add the repository once access is granted."
        )


if __name__ == "__main__":
    main()
