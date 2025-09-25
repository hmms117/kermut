"""Comprehensive end-to-end example for ``bopo_multiobj_pipeline``.

This script demonstrates how to stage the next experimental batch from a rich
synthetic assay capturing two competing objectives:

* ``activity_score`` is maximised – higher values indicate improved activity.
* ``aggregation_penalty`` is minimised – lower values imply better developability.

The accompanying TSV (``example_scripts/data/comprehensive_double_mutant_assay.tsv``)
contains 500 double mutants generated from the 1CRN crambin structure together
with anchor single mutants repeated across the first five batches. Both targets are
included for transparency and combined into a single ``objective`` column via
z-score normalisation (activity z-score minus aggregation z-score) so the
pipeline can consume a scalar reward while still exposing the raw measurements.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from pathlib import Path
from typing import Iterable, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bopo_multiobj_pipeline import main as run_pipeline

DATASET_PATH = Path(__file__).resolve().parent / "data" / "comprehensive_double_mutant_assay.tsv"


def summarise_dataset(path: Path) -> str:
    """Return a short textual summary of the TSV content."""

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        records = list(reader)
    if not records:
        raise ValueError("Dataset is empty; expected comprehensive assay records")
    reference_sequence = records[0]["reference_sequence"]
    double_mutants = sum(1 for row in records if mutation_count(reference_sequence, row["sequence"]) == 2)
    single_mutants = sum(1 for row in records if mutation_count(reference_sequence, row["sequence"]) == 1)
    batches = sorted({int(row["batch"]) for row in records})
    return (
        f"Loaded {len(records)} records covering batches {batches}. "
        f"Found {double_mutants} double mutants and {single_mutants} anchor single mutants."
    )


def mutation_count(reference: str, sequence: str) -> int:
    return sum(1 for ref, alt in zip(reference, sequence) if ref != alt)


def run_comprehensive_example(output_dir: Path, num_select: int = 12) -> Path:
    """Run the pipeline on the comprehensive assay and return the batch path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "pairs.tsv"
    candidates_path = output_dir / "candidates.tsv"
    args: List[str] = [
        "--input",
        str(DATASET_PATH),
        "--output-dir",
        str(output_dir),
        "--objective-column",
        "objective",
        "--num-select",
        str(num_select),
        "--max-mutations",
        "2",
        "--candidate-pool-size",
        "80",
        "--diversity-weight",
        "0.6",
        "--min-hamming-distance",
        "2",
        "--epochs",
        "1",
        "--export-pairs",
        str(pairs_path),
        "--export-candidates",
        str(candidates_path),
    ]
    run_pipeline(args)
    return output_dir / "next_batch.tsv"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the comprehensive BOPO example")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Where to write pipeline artefacts (defaults to a temporary directory)",
    )
    parser.add_argument("--num-select", type=int, default=12, help="Number of variants to promote in the next batch")
    args = parser.parse_args(list(argv) if argv is not None else None)

    summary = summarise_dataset(DATASET_PATH)
    print(summary)

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        batch_path = run_comprehensive_example(output_dir, num_select=args.num_select)
        print(f"Next batch selections written to {batch_path}")
        print(f"Full candidate pool exported to {output_dir / 'candidates.tsv'}")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            batch_path = run_comprehensive_example(output_dir, num_select=args.num_select)
            print(f"Next batch selections written to {batch_path}")
            print("Temporary directory cleaned up at exit")


if __name__ == "__main__":
    main()
