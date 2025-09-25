"""Input/output helpers for the Typer CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def load_assays(path: Path) -> pd.DataFrame:
    """Load an assay table from TSV or CSV format."""

    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def write_candidates(rows: Iterable[Dict[str, object]], destination: Path) -> None:
    """Write candidate information to ``destination`` as CSV."""

    rows = list(rows)
    if not rows:
        destination.write_text("sequence\n", encoding="utf-8")
        return

    # Build deterministic column order: sequence + sorted metadata columns
    fieldnames: List[str] = ["sequence"]
    extras = sorted({key for row in rows for key in row.keys() if key != "sequence"})
    fieldnames.extend(extras)

    destination.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df[fieldnames]
    df.to_csv(destination, index=False)


__all__ = ["load_assays", "write_candidates"]
