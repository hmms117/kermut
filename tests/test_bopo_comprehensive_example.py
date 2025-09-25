from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.bopo_cli_comprehensive_example import (
    DATASET_PATH,
    mutation_count,
    run_comprehensive_example,
    summarise_dataset,
)


def test_dataset_contains_expected_double_mutants() -> None:
    summary = summarise_dataset(DATASET_PATH)
    assert "500 double mutants" in summary
    with open(DATASET_PATH, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    reference = rows[0]["reference_sequence"]
    double_mutant_rows = [row for row in rows if mutation_count(reference, row["sequence"]) == 2]
    assert len(double_mutant_rows) == 500
    # Ensure both target columns are present and non-trivial.
    activities = {float(row["activity_score"]) for row in rows}
    penalties = {float(row["aggregation_penalty"]) for row in rows}
    assert len(activities) > 50
    assert len(penalties) > 50


def test_comprehensive_example_pipeline(tmp_path: Path) -> None:
    next_batch_path = run_comprehensive_example(tmp_path, num_select=10)
    assert next_batch_path.exists()
    lines = next_batch_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split("\t")[:4] == ["rank", "sequence", "predicted_score", "min_hamming_distance"]
    assert len(lines) == 11  # header + 10 selections
    candidate_path = tmp_path / "candidates.tsv"
    pairs_path = tmp_path / "pairs.tsv"
    assert candidate_path.exists()
    assert pairs_path.exists()
    # Ensure minimum Hamming distance constraint was respected.
    sequences = [line.split("\t")[1] for line in lines[1:]]
    for idx in range(len(sequences)):
        for jdx in range(idx + 1, len(sequences)):
            seq_a = sequences[idx]
            seq_b = sequences[jdx]
            distance = sum(1 for a, b in zip(seq_a, seq_b) if a != b)
            assert distance >= 2
