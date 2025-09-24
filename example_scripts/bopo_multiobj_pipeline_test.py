"""Smoke test for the BOPO multi-objective pipeline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bopo_multiobj_pipeline import main


def run_pipeline_test() -> None:
    """Runs the pipeline on a tiny assay with changing batch conditions."""

    assay_tsv = dedent(
        """
        batch\tcondition\tsequence\tobjective\treference_sequence
        1\tph7\tACDE\t0.10\tACDE
        1\tph7\tACDF\t0.32\tACDE
        1\tph7\tACGE\t0.28\tACDE
        2\tph7\tACDE\t0.12\tACDE
        2\tph7\tACDF\t0.31\tACDE
        2\tph7\tACTE\t0.20\tACDE
        3\tph9\tACDE\t0.20\tACDE
        3\tph9\tACDF\t0.27\tACDE
        3\tph9\tACGG\t0.41\tACDE
        4\tph6\tACDE\t0.05\tACDE
        4\tph6\tACFE\t0.34\tACDE
        4\tph6\tACGE\t0.38\tACDE
        5\tph6\tACDE\t0.06\tACDE
        5\tph6\tACTE\t0.24\tACDE
        5\tph6\tACGF\t0.45\tACDE
        """
    ).strip()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        assay_path = tmp_path / "toy_assay.tsv"
        assay_path.write_text(assay_tsv, encoding="utf-8")

        output_dir = tmp_path / "outputs"
        pairs_path = tmp_path / "pairs.tsv"
        candidates_path = tmp_path / "candidates.tsv"

        main(
            [
                "--input",
                str(assay_path),
                "--output-dir",
                str(output_dir),
                "--export-pairs",
                str(pairs_path),
                "--export-candidates",
                str(candidates_path),
                "--num-select",
                "3",
                "--max-mutations",
                "2",
                "--candidate-pool-size",
                "10",
                "--diversity-weight",
                "0.5",
                "--min-hamming-distance",
                "1",
            ]
        )

        next_batch_path = output_dir / "next_batch.tsv"
        assert next_batch_path.exists(), "next batch selections were not written"
        next_lines = next_batch_path.read_text(encoding="utf-8").strip().splitlines()
        assert next_lines[0].split("\t") == [
            "rank",
            "sequence",
            "predicted_score",
            "min_hamming_distance",
            "composite_score",
            "mutations",
            "approx_gain",
        ]
        assert len(next_lines) == 4, "expected exactly three selections"
        selected_sequences = [line.split("\t")[1] for line in next_lines[1:]]
        assert set(selected_sequences) == {"ACFF", "ACFG", "ACTF"}

        assert pairs_path.exists(), "preference pairs export missing"
        pair_lines = pairs_path.read_text(encoding="utf-8").strip().splitlines()
        assert pair_lines[0].split("\t") == [
            "seq_a",
            "seq_b",
            "label",
            "batch_a",
            "batch_b",
            "source",
            "anchor",
            "delta_a",
            "delta_b",
        ]
        assert any("cross_batch" in line for line in pair_lines[1:]), "missing cross-batch anchors"

        assert candidates_path.exists(), "candidate export missing"
        candidate_lines = candidates_path.read_text(encoding="utf-8").strip().splitlines()
        assert candidate_lines[0].split("\t") == [
            "sequence",
            "predicted_score",
            "distance",
            "mutations",
            "approx_gain",
        ]
        candidate_sequences = {line.split("\t")[0] for line in candidate_lines[1:]}
        for sequence in selected_sequences:
            assert sequence in candidate_sequences

    print("BOPO multi-objective pipeline test succeeded.")


if __name__ == "__main__":
    run_pipeline_test()
