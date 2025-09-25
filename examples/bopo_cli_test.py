"""Smoke test for the BOPO multi-objective pipeline."""

from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bopo_cli import main


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
        runner_path = tmp_path / "fake_kermut_runner.py"
        runner_code = dedent(
            """
            import argparse
            import csv
            from pathlib import Path

            POSTERIOR = {
                "ACDE": 0.10,
                "ACDF": 1.00,
                "ACFE": 0.50,
                "ACGE": 0.20,
                "ACTE": -0.30,
                "ACDG": 0.55,
            }

            def derive_mutations(sequence: str, wild_type: str) -> str:
                tokens = []
                for idx, (ref, mut) in enumerate(zip(wild_type, sequence), start=1):
                    if ref != mut:
                        tokens.append(f"{ref}{idx}{mut}")
                return ";".join(tokens) if tokens else "WT"

            def main() -> None:
                parser = argparse.ArgumentParser()
                parser.add_argument("input")
                parser.add_argument("--wild-type", required=True)
                parser.add_argument("--output-dir", required=True)
                parser.add_argument("--target-col", required=True)
                parser.add_argument("--sequence-col", required=True)
                parser.add_argument("--mutation-col")
                args, _ = parser.parse_known_args()

                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / "training_posterior.csv"

                with open(args.input, "r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)

                existing = {row[args.sequence_col] for row in rows}
                fieldnames = [args.sequence_col, args.target_col, args.mutation_col or "mutations", "posterior_mean", "posterior_std"]
                with open(output_path, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in rows:
                        sequence = row[args.sequence_col]
                        mutations = row.get(args.mutation_col or "mutations", "") or derive_mutations(sequence, args.wild_type)
                        posterior = POSTERIOR.get(sequence, float(row[args.target_col]))
                        writer.writerow(
                            {
                                args.sequence_col: sequence,
                                args.target_col: row[args.target_col],
                                args.mutation_col or "mutations": mutations,
                                "posterior_mean": posterior,
                                "posterior_std": 0.05,
                            }
                        )
                    for sequence, posterior in POSTERIOR.items():
                        if sequence in existing:
                            continue
                        writer.writerow(
                            {
                                args.sequence_col: sequence,
                                args.target_col: f"{posterior:.2f}",
                                args.mutation_col or "mutations": derive_mutations(sequence, args.wild_type),
                                "posterior_mean": posterior,
                                "posterior_std": 0.05,
                            }
                        )

            if __name__ == "__main__":
                main()
            """
        ).strip()
        runner_path.write_text(runner_code, encoding="utf-8")

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
                "--kermut-runner",
                sys.executable,
                str(runner_path),
                "--kermut-run-output-dir",
                "kermut_run",
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
        with candidates_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = reader.fieldnames
            assert fieldnames == [
                "sequence",
                "predicted_score",
                "distance",
                "mutations",
                "approx_gain",
            ]
            approx_gains = {row["sequence"]: float(row["approx_gain"]) for row in reader}
        for sequence in selected_sequences:
            assert sequence in approx_gains
        assert math.isclose(approx_gains["ACDG"], 0.45, rel_tol=1e-6)
        assert math.isclose(approx_gains["ACFF"], 0.65, rel_tol=1e-6)
        assert math.isclose(approx_gains["ACFG"], 0.425, rel_tol=1e-6)

    print("BOPO multi-objective pipeline test succeeded.")


if __name__ == "__main__":
    run_pipeline_test()
