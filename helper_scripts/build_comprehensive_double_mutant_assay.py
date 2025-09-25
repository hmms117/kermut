#!/usr/bin/env python3
"""Generate a comprehensive assay TSV with dual objectives.

This script fetches the amino-acid sequence for a protein from the RCSB PDB,
constructs 500 double-mutant variants together with anchor single mutants and
exports a TSV that can be consumed by ``bopo_cli.py``.

The generated TSV contains two experimental targets:

* ``activity_score`` – synthetic activity measurements where larger values are
  better.
* ``aggregation_penalty`` – synthetic liability/aggregation proxy where smaller
  values are preferred.

The script scales both targets and combines them into a single ``objective``
column (activity z-score minus aggregation z-score) so the pipeline can
optimise a single scalar while keeping the original measurements available in
the TSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, List, Sequence, Tuple
from urllib.error import URLError
from urllib.request import urlopen

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydropathy index.
KYTE_DOOLITTLE = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

# Relative residue volumes (normalized from Zamyatnin).
RESIDUE_VOLUME = {
    "A": 67.0,
    "C": 86.0,
    "D": 91.0,
    "E": 109.0,
    "F": 135.0,
    "G": 48.0,
    "H": 118.0,
    "I": 124.0,
    "K": 135.0,
    "L": 124.0,
    "M": 124.0,
    "N": 96.0,
    "P": 90.0,
    "Q": 114.0,
    "R": 148.0,
    "S": 73.0,
    "T": 93.0,
    "V": 105.0,
    "W": 163.0,
    "Y": 141.0,
}

# Net charge at physiological pH.
RESIDUE_CHARGE = {
    "A": 0.0,
    "C": 0.0,
    "D": -1.0,
    "E": -1.0,
    "F": 0.0,
    "G": 0.0,
    "H": 0.1,
    "I": 0.0,
    "K": 1.0,
    "L": 0.0,
    "M": 0.0,
    "N": 0.0,
    "P": 0.0,
    "Q": 0.0,
    "R": 1.0,
    "S": 0.0,
    "T": 0.0,
    "V": 0.0,
    "W": 0.0,
    "Y": 0.0,
}


DEFAULT_REFERENCE_SEQUENCES = {
    # Chain A from the crambin structure (PDB: 1CRN).
    "1CRN": "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN",
}

ALT_RESIDUES = ["A", "V", "L", "I", "G", "S", "T", "D"]


@dataclass
class Variant:
    sequence: str
    mutations: List[Tuple[int, str]]  # (0-indexed position, new residue)
    base_activity: float
    base_penalty: float


def fetch_reference_sequence(pdb_id: str) -> str:
    """Download the canonical sequence for a PDB entry from RCSB."""

    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    try:
        with urlopen(url) as handle:  # pragma: no cover - network access
            fasta = handle.read().decode("utf-8")
        lines = [line.strip() for line in fasta.splitlines() if line and not line.startswith(">")]
        sequence = "".join(lines)
        if not sequence:
            raise RuntimeError(f"Failed to extract sequence for {pdb_id}")
        return sequence
    except URLError as exc:
        fallback = DEFAULT_REFERENCE_SEQUENCES.get(pdb_id.upper())
        if fallback:
            print(
                f"Warning: failed to fetch sequence for {pdb_id} ({exc}); "
                "using bundled fallback sequence."
            )
            return fallback
        raise


def deterministic_choice(options: Sequence[str], token: str) -> str:
    """Select an option deterministically based on a hash token."""

    filtered = [residue for residue in options if residue in ALT_RESIDUES]
    if filtered:
        options = filtered
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:2], "big") % len(options)
    return options[index]


def apply_mutations(reference: str, mutations: Sequence[Tuple[int, str]]) -> str:
    residues = list(reference)
    for position, new_residue in mutations:
        residues[position] = new_residue
    return "".join(residues)


def enumerate_double_mutants(reference: str, limit: int) -> List[Variant]:
    """Generate deterministic double mutants for the reference sequence."""

    variants: List[Variant] = []
    for pos_a, pos_b in itertools.combinations(range(len(reference)), 2):
        ref_a = reference[pos_a]
        ref_b = reference[pos_b]
        alt_a = deterministic_choice([aa for aa in AMINO_ACIDS if aa != ref_a], f"double-{pos_a}-{pos_b}-a")
        alt_b = deterministic_choice([aa for aa in AMINO_ACIDS if aa != ref_b], f"double-{pos_a}-{pos_b}-b")
        mutations = [(pos_a, alt_a), (pos_b, alt_b)]
        sequence = apply_mutations(reference, mutations)
        activity = compute_activity(reference, sequence)
        penalty = compute_aggregation_penalty(reference, sequence)
        variants.append(Variant(sequence=sequence, mutations=list(mutations), base_activity=activity, base_penalty=penalty))
        if len(variants) >= limit:
            break
    return variants


def enumerate_single_mutants(reference: str, count: int) -> List[Variant]:
    """Return a deterministic panel of single mutants for anchor measurements."""

    variants: List[Variant] = []
    for position in range(len(reference)):
        ref_residue = reference[position]
        alt = deterministic_choice([aa for aa in AMINO_ACIDS if aa != ref_residue], f"single-{position}")
        sequence = apply_mutations(reference, [(position, alt)])
        activity = compute_activity(reference, sequence)
        penalty = compute_aggregation_penalty(reference, sequence)
        variants.append(
            Variant(sequence=sequence, mutations=[(position, alt)], base_activity=activity, base_penalty=penalty)
        )
        if len(variants) >= count:
            break
    return variants


def mutation_labels(reference: str, mutations: Sequence[Tuple[int, str]]) -> List[str]:
    labels: List[str] = []
    for position, new_residue in mutations:
        ref_residue = reference[position]
        labels.append(f"{ref_residue}{position + 1}{new_residue}")
    return labels


def compute_activity(reference: str, sequence: str) -> float:
    """Synthetic activity landscape favouring hydrophobic enrichment."""

    delta_hydro = 0.0
    delta_volume = 0.0
    delta_charge = 0.0
    mutated_positions: List[int] = []
    for index, (ref_residue, new_residue) in enumerate(zip(reference, sequence)):
        if ref_residue == new_residue:
            continue
        mutated_positions.append(index)
        delta_hydro += KYTE_DOOLITTLE[new_residue] - KYTE_DOOLITTLE[ref_residue]
        delta_volume += RESIDUE_VOLUME[new_residue] - RESIDUE_VOLUME[ref_residue]
        delta_charge += RESIDUE_CHARGE[new_residue] - RESIDUE_CHARGE[ref_residue]
    proximity_bonus = 0.0
    if len(mutated_positions) >= 2:
        i, j = mutated_positions
        separation = abs(i - j) + 1
        proximity_bonus = 1.0 / separation
    noise = hashed_noise(sequence, "activity")
    activity = 0.85 + 0.12 * delta_hydro - 0.05 * (delta_volume / 50.0) + 0.08 * abs(delta_charge) + 0.06 * proximity_bonus
    return activity + 0.05 * noise


def compute_aggregation_penalty(reference: str, sequence: str) -> float:
    """Synthetic liability landscape penalising hydrophobic and bulky residues."""

    hydrophobic_gain = 0.0
    volume_change = 0.0
    charge_abs = 0.0
    for ref_residue, new_residue in zip(reference, sequence):
        if ref_residue == new_residue:
            continue
        hydrophobic_gain += max(0.0, KYTE_DOOLITTLE[new_residue] - KYTE_DOOLITTLE[ref_residue])
        volume_change += abs(RESIDUE_VOLUME[new_residue] - RESIDUE_VOLUME[ref_residue]) / 50.0
        charge_abs += abs(RESIDUE_CHARGE[new_residue])
    noise = hashed_noise(sequence, "penalty")
    penalty = 0.35 + 0.18 * hydrophobic_gain + 0.07 * volume_change + 0.05 * charge_abs
    return penalty + 0.04 * noise


def hashed_noise(sequence: str, salt: str) -> float:
    digest = hashlib.sha256((sequence + salt).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def build_records(
    reference: str,
    doubles: Sequence[Variant],
    singles: Sequence[Variant],
    batches: int,
    doubles_per_batch: int,
) -> List[dict]:
    batch_conditions = [
        "pH6.5_low_salt",
        "pH7.0_high_salt",
        "pH7.5_low_salt",
        "pH8.0_high_buffer",
        "pH6.0_high_salt",
        "pH7.2_low_salt",
        "pH8.3_high_buffer",
        "pH6.8_moderate_salt",
        "pH7.8_low_salt",
        "pH6.3_high_salt",
    ]
    activity_offsets = [0.05, -0.02, 0.03, -0.01, 0.04, -0.03, 0.02, -0.04, 0.01, 0.035]
    penalty_offsets = [0.02, 0.04, -0.01, 0.03, -0.02, 0.015, 0.025, -0.03, 0.01, -0.015]
    records: List[dict] = []
    double_iter = iter(doubles)
    anchor_batches = min(5, batches)
    for batch_index in range(batches):
        batch_id = batch_index + 1
        condition = batch_conditions[batch_index % len(batch_conditions)]
        activity_shift = activity_offsets[batch_index % len(activity_offsets)]
        penalty_shift = penalty_offsets[batch_index % len(penalty_offsets)]
        if batch_index < anchor_batches:
            # Reference measurements for anchor batches only.
            ref_activity = compute_activity(reference, reference) + activity_shift + 0.02 * hashed_noise(
                f"ref-{batch_id}", "activity"
            )
            ref_penalty = compute_aggregation_penalty(reference, reference) + penalty_shift + 0.02 * hashed_noise(
                f"ref-{batch_id}", "penalty"
            )
            records.append(
                {
                    "batch": batch_id,
                    "condition": condition,
                    "sequence": reference,
                    "activity_score": ref_activity,
                    "aggregation_penalty": ref_penalty,
                    "reference_sequence": reference,
                    "mutations": "WT",
                }
            )
            # Anchor single mutants appearing in the anchor batches.
            for variant in singles:
                activity = variant.base_activity + activity_shift + 0.01 * hashed_noise(
                    f"single-{variant.sequence}-{batch_id}", "activity"
                )
                penalty = variant.base_penalty + penalty_shift + 0.01 * hashed_noise(
                    f"single-{variant.sequence}-{batch_id}", "penalty"
                )
                records.append(
                    {
                        "batch": batch_id,
                        "condition": condition,
                        "sequence": variant.sequence,
                        "activity_score": activity,
                        "aggregation_penalty": penalty,
                        "reference_sequence": reference,
                        "mutations": ";".join(mutation_labels(reference, variant.mutations)),
                    }
                )
        # Batch-specific double mutants.
        for _ in range(doubles_per_batch):
            variant = next(double_iter)
            activity = variant.base_activity + activity_shift + 0.015 * hashed_noise(
                f"double-{variant.sequence}-{batch_id}", "activity"
            )
            penalty = variant.base_penalty + penalty_shift + 0.015 * hashed_noise(
                f"double-{variant.sequence}-{batch_id}", "penalty"
            )
            records.append(
                {
                    "batch": batch_id,
                    "condition": condition,
                    "sequence": variant.sequence,
                    "activity_score": activity,
                    "aggregation_penalty": penalty,
                    "reference_sequence": reference,
                    "mutations": ";".join(mutation_labels(reference, variant.mutations)),
                }
            )
    return records


def add_objective(records: List[dict]) -> None:
    activities = [float(record["activity_score"]) for record in records]
    penalties = [float(record["aggregation_penalty"]) for record in records]
    activity_mean = mean(activities)
    penalty_mean = mean(penalties)
    activity_std = pstdev(activities)
    penalty_std = pstdev(penalties)
    for record, activity, penalty in zip(records, activities, penalties):
        activity_z = (activity - activity_mean) / activity_std if activity_std else 0.0
        penalty_z = (penalty - penalty_mean) / penalty_std if penalty_std else 0.0
        record["objective"] = activity_z - penalty_z


def export_tsv(records: Iterable[dict], path: Path) -> None:
    headers = [
        "batch",
        "condition",
        "sequence",
        "activity_score",
        "aggregation_penalty",
        "objective",
        "reference_sequence",
        "mutations",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, delimiter="\t")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct a comprehensive assay TSV with dual objectives")
    parser.add_argument("--pdb-id", default="1CRN", help="PDB identifier to fetch the reference sequence from (default 1CRN)")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "examples" / "data" / "comprehensive_double_mutant_assay.tsv"),
        help="Destination TSV path",
    )
    parser.add_argument(
        "--single-per-batch",
        type=int,
        default=5,
        help="Number of anchor single mutants to include in every batch",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=50,
        help="Number of assay batches to simulate",
    )
    parser.add_argument(
        "--double-per-batch",
        type=int,
        default=10,
        help="Number of double mutants to generate per batch (total = batches * double_per_batch)",
    )
    parser.add_argument(
        "--reference-sequence",
        default="",
        help="Optional reference sequence override; skips fetching if provided",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_sequence:
        reference = args.reference_sequence.strip().upper()
    else:
        reference = fetch_reference_sequence(args.pdb_id)
    singles = enumerate_single_mutants(reference, args.single_per_batch)
    doubles = enumerate_double_mutants(reference, args.batches * args.double_per_batch)
    records = build_records(reference, doubles, singles, args.batches, args.double_per_batch)
    add_objective(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_tsv(records, output_path)
    print(f"Wrote {len(records)} assay records to {output_path}")


if __name__ == "__main__":
    main()
