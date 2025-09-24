"""Command line pipeline for multi-objective batch BOPO selections.

This module implements a simplified BOPO-style pipeline that can be executed as a
standalone script.  The pipeline ingests TSV assay files, cleans the objective
values, discovers anchors (control sequences that appear across multiple
batches), trains a lightweight preference model, enumerates new DMS-driven
candidates and performs Pareto-aware selection balancing predicted preference
score with Hamming-distance diversity.

The implementation is dependency-light and relies solely on Python's standard
library to keep the script portable.  It is therefore intentionally simple, but
covers the full data-processing flow required to stage the next experimental
batch.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class AssayRecord:
    """A single measurement for a sequence under a specific batch/condition."""

    batch: int
    condition: str
    sequence: str
    objective: float
    source_file: str


@dataclass
class PreferencePair:
    """Represents a directional preference between two sequences."""

    seq_a: str
    seq_b: str
    label: int
    batch_a: int
    batch_b: int
    source: str
    anchor: Optional[str] = None
    delta_a: Optional[float] = None
    delta_b: Optional[float] = None


@dataclass
class CandidateProposal:
    """Container for a candidate sequence prior to preference scoring."""

    sequence: str
    mutations: List[str]
    approx_gain: float


@dataclass
class CandidateScore:
    """Stores the predicted score and diversity metadata for a candidate."""

    sequence: str
    score: float
    distance: int
    mutations: List[str]
    approx_gain: float

@dataclass
class SelectedCandidate(CandidateScore):
    """Extends :class:`CandidateScore` with ranking metadata."""

    rank: int = 0
    composite_score: float = 0.0


class MutationFeaturizer:
    """Binary featurization of sequences based on reference mutations."""

    def __init__(self, reference_sequence: str) -> None:
        self.reference_sequence = reference_sequence
        self._feature_index: Dict[str, int] = {}

    def register_sequence(self, sequence: str) -> None:
        for idx, (ref_aa, seq_aa) in enumerate(
            zip(self.reference_sequence, sequence), start=1
        ):
            if ref_aa != seq_aa:
                mutation = f"{ref_aa}{idx}{seq_aa}"
                if mutation not in self._feature_index:
                    self._feature_index[mutation] = len(self._feature_index)

    def feature_names(self) -> List[str]:
        return [mutation for mutation, _ in sorted(self._feature_index.items(), key=lambda item: item[1])]

    def vector(self, sequence: str) -> List[float]:
        vector = [0.0] * len(self._feature_index)
        for idx, (ref_aa, seq_aa) in enumerate(
            zip(self.reference_sequence, sequence), start=1
        ):
            if ref_aa != seq_aa:
                mutation = f"{ref_aa}{idx}{seq_aa}"
                feature_index = self._feature_index.get(mutation)
                if feature_index is not None:
                    vector[feature_index] = 1.0
        return vector


class PreferenceNet:
    """Simple logistic preference model trained on pairwise differences."""

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        l2: float = 0.01,
        epochs: int = 200,
        tolerance: float = 1e-6,
    ) -> None:
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.tolerance = tolerance
        self.weights: List[float] = [0.0] * n_features

    def train(self, training_pairs: Sequence[Tuple[List[float], int]]) -> None:
        if not training_pairs:
            return
        weights = self.weights[:]
        for _ in range(self.epochs):
            gradient = [0.0] * self.n_features
            max_step = 0.0
            for diff_vector, label in training_pairs:
                dot = _dot(weights, diff_vector)
                prob = _sigmoid(dot)
                error = prob - float(label)
                for idx, value in enumerate(diff_vector):
                    if value != 0.0:
                        gradient[idx] += error * value
            for idx in range(self.n_features):
                gradient[idx] = gradient[idx] / float(len(training_pairs)) + self.l2 * weights[idx]
                step = self.learning_rate * gradient[idx]
                weights[idx] -= step
                max_step = max(max_step, abs(step))
            if max_step < self.tolerance:
                break
        self.weights = weights

    def score(self, features: Sequence[float]) -> float:
        return _dot(self.weights, features)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BOPO-style multi-objective pipeline")
    parser.add_argument("--input", "-i", nargs="+", help="TSV files with assay measurements", required=True)
    parser.add_argument("--output-dir", "-o", required=True, help="Directory for pipeline outputs")
    parser.add_argument("--objective-column", default="objective")
    parser.add_argument("--batch-column", default="batch")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--condition-column", default="condition")
    parser.add_argument("--reference-column", default="reference_sequence")
    parser.add_argument("--reference-sequence", default="", help="Override reference sequence")
    parser.add_argument("--num-select", type=int, default=10, help="Number of sequences to select")
    parser.add_argument("--max-mutations", type=int, default=2, help="Maximum mutations per enumerated candidate")
    parser.add_argument("--candidate-pool-size", type=int, default=50, help="Maximum number of candidate sequences to score")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate for the preference net")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs for the preference net")
    parser.add_argument("--l2", type=float, default=0.01, help="L2 regularization strength")
    parser.add_argument("--diversity-weight", type=float, default=0.2, help="Weight for diversity during selection")
    parser.add_argument(
        "--min-hamming-distance",
        type=int,
        default=1,
        help="Minimum allowable Hamming distance between selected sequences",
    )
    parser.add_argument("--export-pairs", default="", help="Optional TSV path to export generated preference pairs")
    parser.add_argument("--export-candidates", default="", help="Optional TSV path for the full candidate pool")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed controlling deterministic orderings")
    parser.add_argument(
        "--kermut-singles",
        default="",
        help="Optional CSV/TSV with Kermut posterior means for single mutants",
    )
    parser.add_argument(
        "--kermut-sequence-column",
        default="sequence",
        help="Column in the Kermut file containing variant sequences",
    )
    parser.add_argument(
        "--kermut-score-column",
        default="posterior_mean",
        help="Column in the Kermut file containing predicted variant effects",
    )
    parser.add_argument(
        "--kermut-delimiter",
        default=",",
        help="Delimiter used in the Kermut single mutant file (default comma)",
    )
    parser.add_argument(
        "--kermut-runner",
        nargs="+",
        default=[],
        help=(
            "Command used to generate Kermut single-mutant posteriors from the assay TSV. "
            "Provide, for example, `python -m kermut.cmdline.one_liner run`."
        ),
    )
    parser.add_argument(
        "--kermut-run-output-dir",
        default="kermut_outputs",
        help="Sub-directory (relative to --output-dir) where runner artefacts are written",
    )
    parser.add_argument(
        "--kermut-run-sequence-column",
        default="sequence",
        help="Sequence column name supplied to the runner command",
    )
    parser.add_argument(
        "--kermut-run-target-column",
        default="objective",
        help="Target/fitness column name supplied to the runner command",
    )
    parser.add_argument(
        "--kermut-run-mutation-column",
        default="mutations",
        help="Mutation annotation column name supplied to the runner command",
    )
    parser.add_argument(
        "--kermut-run-extra-args",
        nargs="*",
        default=None,
        help="Additional arguments appended to the runner command",
    )
    parser.add_argument(
        "--kermut-run-training-file",
        default="training_posterior.csv",
        help="File produced by the runner containing posterior statistics",
    )
    return parser.parse_args(argv)


def read_assay_records(
    paths: Sequence[str],
    sequence_column: str,
    batch_column: str,
    condition_column: str,
    objective_column: str,
    reference_column: str,
    reference_override: str,
) -> Tuple[List[AssayRecord], str]:
    reference_sequence = reference_override
    records: List[AssayRecord] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                sequence = (row.get(sequence_column) or "").strip()
                objective = _to_float(row.get(objective_column))
                batch = _to_int(row.get(batch_column))
                condition = (row.get(condition_column) or "").strip()
                reference_value = (row.get(reference_column) or "").strip()
                if not sequence or objective is None or batch is None:
                    continue
                if not reference_sequence:
                    reference_sequence = reference_value
                if reference_sequence and len(sequence) != len(reference_sequence):
                    raise ValueError(
                        "Encountered sequence with length mismatch to reference; ensure input is aligned"
                    )
                records.append(
                    AssayRecord(
                        batch=batch,
                        condition=condition,
                        sequence=sequence,
                        objective=objective,
                        source_file=os.path.basename(path),
                    )
                )
    if not records:
        raise ValueError("No valid assay rows were loaded from the provided TSV files")
    if not reference_sequence:
        raise ValueError(
            "Unable to determine the reference sequence; provide --reference-sequence or reference column"
        )
    return records, reference_sequence


def aggregate_records(records: Sequence[AssayRecord], reference_sequence: str) -> List[Dict[str, object]]:
    aggregated: Dict[Tuple[int, str], Dict[str, object]] = {}
    for record in records:
        key = (record.batch, record.sequence)
        entry = aggregated.get(key)
        if entry is None:
            entry = {
                "batch": record.batch,
                "sequence": record.sequence,
                "condition": record.condition,
                "objectives": [record.objective],
                "sources": [record.source_file],
            }
            aggregated[key] = entry
        else:
            entry["objectives"].append(record.objective)
            entry["sources"].append(record.source_file)
    result: List[Dict[str, object]] = []
    for entry in aggregated.values():
        objectives = entry["objectives"]  # type: ignore[assignment]
        averaged_objective = sum(objectives) / float(len(objectives))
        sequence = entry["sequence"]  # type: ignore[assignment]
        result.append(
            {
                "batch": entry["batch"],
                "sequence": sequence,
                "condition": entry["condition"],
                "objective": averaged_objective,
                "count": len(objectives),
                "mutations": sequence_mutations(reference_sequence, sequence),
            }
        )
    return sorted(result, key=lambda item: (int(item["batch"]), item["sequence"]))


def detect_anchors(records: Sequence[Dict[str, object]]) -> Dict[str, List[int]]:
    occurrences: Dict[str, List[int]] = {}
    for record in records:
        sequence = record["sequence"]  # type: ignore[assignment]
        batch = int(record["batch"])
        batches = occurrences.setdefault(sequence, [])
        if batch not in batches:
            batches.append(batch)
    anchors = {sequence: sorted(batches) for sequence, batches in occurrences.items() if len(batches) > 1}
    return anchors


def group_by_batch(records: Sequence[Dict[str, object]]) -> Dict[int, List[Dict[str, object]]]:
    batches: Dict[int, List[Dict[str, object]]] = {}
    for record in records:
        batch = int(record["batch"])
        batches.setdefault(batch, []).append(record)
    return batches


def generate_preference_pairs(
    batches: Dict[int, List[Dict[str, object]]],
    anchors: Dict[str, List[int]],
) -> List[PreferencePair]:
    pairs: List[PreferencePair] = []
    # Within-batch comparisons
    for batch, records in batches.items():
        sorted_records = sorted(records, key=lambda item: float(item["objective"]))
        for idx in range(len(sorted_records) - 1, -1, -1):
            better = sorted_records[idx]
            for jdx in range(idx):
                worse = sorted_records[jdx]
                if better["sequence"] == worse["sequence"]:
                    continue
                pairs.append(
                    PreferencePair(
                        seq_a=str(better["sequence"]),
                        seq_b=str(worse["sequence"]),
                        label=1,
                        batch_a=int(better["batch"]),
                        batch_b=int(worse["batch"]),
                        source="within_batch",
                        delta_a=float(better["objective"]),
                        delta_b=float(worse["objective"]),
                    )
                )
    # Cross-batch anchors
    anchor_records: Dict[str, Dict[int, Dict[str, object]]] = {}
    for sequence, batches_with_anchor in anchors.items():
        anchor_records[sequence] = {}
        for batch in batches_with_anchor:
            for record in batches[batch]:
                if record["sequence"] == sequence:
                    anchor_records[sequence][batch] = record
                    break
    seen_pairs: set[Tuple[str, str, int, int]] = set()
    for anchor_sequence, batch_records in anchor_records.items():
        batch_ids = sorted(batch_records.keys())
        for first_index in range(len(batch_ids)):
            batch_a = batch_ids[first_index]
            anchor_a = batch_records[batch_a]
            for second_index in range(first_index + 1, len(batch_ids)):
                batch_b = batch_ids[second_index]
                anchor_b = batch_records[batch_b]
                baseline_a = float(anchor_a["objective"])
                baseline_b = float(anchor_b["objective"])
                for candidate_a in batches[batch_a]:
                    if candidate_a["sequence"] == anchor_sequence:
                        continue
                    delta_a = float(candidate_a["objective"]) - baseline_a
                    for candidate_b in batches[batch_b]:
                        if candidate_b["sequence"] == anchor_sequence:
                            continue
                        delta_b = float(candidate_b["objective"]) - baseline_b
                        if abs(delta_a - delta_b) < 1e-12:
                            continue
                        if delta_a > delta_b:
                            seq_a = str(candidate_a["sequence"])
                            seq_b = str(candidate_b["sequence"])
                            first_batch = batch_a
                            second_batch = batch_b
                            da = delta_a
                            db = delta_b
                        else:
                            seq_a = str(candidate_b["sequence"])
                            seq_b = str(candidate_a["sequence"])
                            first_batch = batch_b
                            second_batch = batch_a
                            da = delta_b
                            db = delta_a
                        key = (seq_a, seq_b, first_batch, second_batch)
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        pairs.append(
                            PreferencePair(
                                seq_a=seq_a,
                                seq_b=seq_b,
                                label=1,
                                batch_a=first_batch,
                                batch_b=second_batch,
                                source="cross_batch",
                                anchor=anchor_sequence,
                                delta_a=da,
                                delta_b=db,
                            )
                        )
    return pairs


def build_training_data(
    pairs: Sequence[PreferencePair],
    featurizer: MutationFeaturizer,
) -> List[Tuple[List[float], int]]:
    training_data: List[Tuple[List[float], int]] = []
    feature_cache: Dict[str, List[float]] = {}
    for pair in pairs:
        vec_a = feature_cache.get(pair.seq_a)
        if vec_a is None:
            vec_a = featurizer.vector(pair.seq_a)
            feature_cache[pair.seq_a] = vec_a
        vec_b = feature_cache.get(pair.seq_b)
        if vec_b is None:
            vec_b = featurizer.vector(pair.seq_b)
            feature_cache[pair.seq_b] = vec_b
        diff_ab = [a - b for a, b in zip(vec_a, vec_b)]
        training_data.append((diff_ab, 1))
        diff_ba = [b - a for a, b in zip(vec_a, vec_b)]
        training_data.append((diff_ba, 0))
    return training_data


def compute_reference_baseline(
    records: Sequence[Dict[str, object]], reference_sequence: str
) -> Optional[float]:
    reference_objectives = [float(record["objective"]) for record in records if record["sequence"] == reference_sequence]
    if reference_objectives:
        return mean(reference_objectives)
    return None


def compute_mutation_scores(
    records: Sequence[Dict[str, object]],
    reference_sequence: str,
    baseline: Optional[float] = None,
) -> Dict[str, float]:
    if baseline is None:
        baseline = compute_reference_baseline(records, reference_sequence) or 0.0
    score_accumulator: Dict[str, float] = {}
    count_accumulator: Dict[str, int] = {}
    for record in records:
        sequence = record["sequence"]
        if sequence == reference_sequence:
            continue
        delta = float(record["objective"]) - baseline
        for mutation in record["mutations"]:  # type: ignore[assignment]
            score_accumulator[mutation] = score_accumulator.get(mutation, 0.0) + delta
            count_accumulator[mutation] = count_accumulator.get(mutation, 0) + 1
    mutation_scores: Dict[str, float] = {}
    for mutation, total in score_accumulator.items():
        count = count_accumulator[mutation]
        if count:
            mutation_scores[mutation] = total / float(count)
    return mutation_scores


def load_kermut_single_mutation_scores(
    path: str,
    sequence_column: str,
    score_column: str,
    reference_sequence: str,
    delimiter: str,
    fallback_baseline: Optional[float],
) -> Dict[str, float]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Kermut single mutant file not found: {path}")
    records: List[Tuple[str, float]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("Kermut single mutant file is missing a header row")
        for row in reader:
            sequence = row.get(sequence_column)
            score = _to_float(row.get(score_column))
            if sequence is None or score is None:
                continue
            sequence = sequence.strip()
            if not sequence:
                continue
            records.append((sequence, score))
    if not records:
        return {}
    baseline = fallback_baseline
    for sequence, score in records:
        if sequence == reference_sequence:
            baseline = score
            break
    if baseline is None:
        baseline = 0.0
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for sequence, score in records:
        if sequence == reference_sequence or len(sequence) != len(reference_sequence):
            continue
        mutations = sequence_mutations(reference_sequence, sequence)
        if len(mutations) != 1:
            continue
        mutation = mutations[0]
        totals[mutation] = totals.get(mutation, 0.0) + (score - baseline)
        counts[mutation] = counts.get(mutation, 0) + 1
    mutation_scores: Dict[str, float] = {}
    for mutation, total in totals.items():
        count = counts[mutation]
        if count:
            mutation_scores[mutation] = total / float(count)
    return mutation_scores


def prepare_kermut_runner_input(
    records: Sequence[Dict[str, object]],
    reference_sequence: str,
    sequence_column: str,
    target_column: str,
    mutation_column: Optional[str],
    path: Path,
    baseline: Optional[float],
) -> None:
    rows: List[Tuple[str, float, Optional[str]]] = []
    seen_sequences: set[str] = set()
    for record in records:
        sequence = str(record["sequence"])
        if sequence in seen_sequences:
            continue
        objective = float(record["objective"])
        mutations: List[str] = record.get("mutations", [])  # type: ignore[assignment]
        if sequence == reference_sequence:
            rows.append((sequence, objective, "WT" if mutation_column else None))
        elif len(mutations) == 1:
            rows.append((sequence, objective, mutations[0] if mutation_column else None))
        else:
            continue
        seen_sequences.add(sequence)
    if not rows:
        raise ValueError("No single mutants were found for the Kermut runner input")
    if baseline is not None and not any(sequence == reference_sequence for sequence, _, _ in rows):
        rows.append((reference_sequence, baseline, "WT" if mutation_column else None))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        headers = [sequence_column, target_column]
        if mutation_column:
            headers.append(mutation_column)
        writer.writerow(headers)
        for sequence, objective, mutation in rows:
            row: List[object] = [sequence, f"{objective:.12f}"]
            if mutation_column:
                row.append(mutation or "")
            writer.writerow(row)


def run_kermut_runner(
    records: Sequence[Dict[str, object]],
    reference_sequence: str,
    command: Sequence[str],
    output_dir: Path,
    sequence_column: str,
    target_column: str,
    mutation_column: Optional[str],
    extra_args: Optional[Sequence[str]],
    training_filename: str,
    fallback_baseline: Optional[float],
) -> Dict[str, float]:
    if not command:
        raise ValueError("Kermut runner command was not provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / "kermut_runner_input.csv"
        prepare_kermut_runner_input(
            records,
            reference_sequence,
            sequence_column,
            target_column,
            mutation_column,
            tmp_input,
            fallback_baseline,
        )
        runner_cmd = list(command)
        runner_cmd.append(str(tmp_input))
        runner_cmd.extend(["--wild-type", reference_sequence])
        runner_cmd.extend(["--output-dir", str(output_dir)])
        runner_cmd.extend(["--target-col", target_column])
        runner_cmd.extend(["--sequence-col", sequence_column])
        if mutation_column:
            runner_cmd.extend(["--mutation-col", mutation_column])
        if extra_args:
            runner_cmd.extend(extra_args)
        try:
            subprocess.run(runner_cmd, check=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Failed to launch Kermut runner command '{command[0]}'"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Kermut runner exited with a non-zero status"
            ) from exc
        training_path = output_dir / training_filename
        if not training_path.exists():
            raise FileNotFoundError(
                f"Kermut runner did not produce expected file: {training_path}"
            )
        mutation_scores: Dict[str, List[float]] = {}
        baseline = fallback_baseline
        with open(training_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(
                    f"Kermut runner output {training_path} is missing a header row"
                )
            for row in reader:
                sequence = (row.get(sequence_column) or "").strip()
                if not sequence:
                    continue
                posterior_value = _to_float(row.get("posterior_mean"))
                if posterior_value is None:
                    continue
                if sequence == reference_sequence:
                    baseline = posterior_value
                    continue
                mutation_label = (row.get(mutation_column) or "").strip() if mutation_column else ""
                mutation: Optional[str] = None
                if mutation_label and mutation_label.upper() not in {"WT", "-"}:
                    tokens = [token for token in mutation_label.replace(",", ";").split(";") if token]
                    if len(tokens) == 1:
                        mutation = tokens[0]
                if mutation is None:
                    derived = sequence_mutations(reference_sequence, sequence)
                    if len(derived) == 1:
                        mutation = derived[0]
                if mutation is None:
                    continue
                mutation_scores.setdefault(mutation, []).append(posterior_value)
        baseline_value = baseline if baseline is not None else fallback_baseline
        if baseline_value is None:
            baseline_value = 0.0
        averaged: Dict[str, float] = {}
        for mutation, values in mutation_scores.items():
            if values:
                averaged[mutation] = sum(values) / float(len(values)) - baseline_value
        return averaged


def enumerate_candidates(
    reference_sequence: str,
    mutation_scores: Dict[str, float],
    max_mutations: int,
    pool_size: int,
    existing_sequences: Iterable[str],
) -> List[CandidateProposal]:
    sorted_mutations = sorted(
        mutation_scores.items(), key=lambda item: (item[1], item[0]), reverse=True
    )
    existing_set = set(existing_sequences)
    proposals: Dict[str, CandidateProposal] = {}

    def backtrack(start: int, current: List[Tuple[str, float]], used_positions: List[int]) -> None:
        if current:
            mutations = [mutation for mutation, _ in current]
            sequence = apply_mutations(reference_sequence, mutations)
            if sequence not in existing_set and sequence not in proposals:
                approx = sum(score for _, score in current) / float(len(current))
                proposals[sequence] = CandidateProposal(sequence=sequence, mutations=mutations[:], approx_gain=approx)
                if len(proposals) >= pool_size:
                    return
        if len(current) >= max_mutations:
            return
        for index in range(start, len(sorted_mutations)):
            mutation, score = sorted_mutations[index]
            position = parse_mutation(mutation)[1]
            if position in used_positions:
                continue
            current.append((mutation, score))
            used_positions.append(position)
            backtrack(index + 1, current, used_positions)
            if len(proposals) >= pool_size:
                return
            current.pop()
            used_positions.pop()

    backtrack(0, [], [])
    return list(proposals.values())


def apply_mutations(reference_sequence: str, mutations: Sequence[str]) -> str:
    sequence_list = list(reference_sequence)
    for mutation in mutations:
        ref, position, alt = parse_mutation(mutation)
        sequence_list[position - 1] = alt
    return "".join(sequence_list)


def parse_mutation(mutation: str) -> Tuple[str, int, str]:
    if len(mutation) < 3:
        raise ValueError(f"Invalid mutation token: {mutation}")
    ref = mutation[0]
    alt = mutation[-1]
    position_str = mutation[1:-1]
    position = int(position_str)
    return ref, position, alt


def sequence_mutations(reference_sequence: str, sequence: str) -> List[str]:
    return [
        f"{ref}{idx}{alt}"
        for idx, (ref, alt) in enumerate(zip(reference_sequence, sequence), start=1)
        if ref != alt
    ]


def hamming_distance(seq_a: str, seq_b: str) -> int:
    if len(seq_a) != len(seq_b):
        raise ValueError("Cannot compute Hamming distance for sequences of different length")
    return sum(1 for char_a, char_b in zip(seq_a, seq_b) if char_a != char_b)


def compute_pareto_front(candidates: Sequence[CandidateScore]) -> List[CandidateScore]:
    front: List[CandidateScore] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if candidate is other:
                continue
            if (other.score >= candidate.score and other.distance >= candidate.distance) and (
                other.score > candidate.score or other.distance > candidate.distance
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    front.sort(key=lambda item: (-item.score, -item.distance, item.sequence))
    return front


def select_next_batch(
    candidates: Dict[str, CandidateScore],
    num_select: int,
    diversity_weight: float,
    min_hamming_distance: int,
    reference_length: int,
) -> List[SelectedCandidate]:
    selected: List[SelectedCandidate] = []
    chosen_sequences: List[str] = []
    remaining_sequences = set(candidates.keys())
    while len(selected) < num_select and remaining_sequences:
        candidate_scores: List[CandidateScore] = []
        for sequence in sorted(remaining_sequences):
            candidate = candidates[sequence]
            if chosen_sequences:
                distances = [hamming_distance(sequence, other) for other in chosen_sequences]
                min_distance = min(distances)
            else:
                min_distance = reference_length
            if min_distance < min_hamming_distance:
                continue
            candidate_scores.append(
                CandidateScore(
                    sequence=sequence,
                    score=candidate.score,
                    distance=min_distance,
                    mutations=candidate.mutations,
                    approx_gain=candidate.approx_gain,
                )
            )
        if not candidate_scores:
            break
        front = compute_pareto_front(candidate_scores)
        best = max(
            front,
            key=lambda item: (
                item.score + diversity_weight * float(item.distance),
                item.distance,
                item.sequence,
            ),
        )
        composite = best.score + diversity_weight * float(best.distance)
        selected.append(
            SelectedCandidate(
                sequence=best.sequence,
                score=best.score,
                distance=best.distance,
                mutations=best.mutations,
                approx_gain=best.approx_gain,
                rank=len(selected) + 1,
                composite_score=composite,
            )
        )
        chosen_sequences.append(best.sequence)
        remaining_sequences.remove(best.sequence)
    return selected


def export_pairs(path: Path, pairs: Sequence[PreferencePair]) -> None:
    headers = [
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
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        for pair in pairs:
            writer.writerow(
                [
                    pair.seq_a,
                    pair.seq_b,
                    pair.label,
                    pair.batch_a,
                    pair.batch_b,
                    pair.source,
                    pair.anchor or "",
                    "" if pair.delta_a is None else f"{pair.delta_a:.6f}",
                    "" if pair.delta_b is None else f"{pair.delta_b:.6f}",
                ]
            )


def export_candidates(path: Path, candidates: Sequence[CandidateScore]) -> None:
    headers = ["sequence", "predicted_score", "distance", "mutations", "approx_gain"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.sequence,
                    f"{candidate.score:.6f}",
                    candidate.distance,
                    ";".join(candidate.mutations),
                    f"{candidate.approx_gain:.6f}",
                ]
            )


def export_selection(path: Path, selections: Sequence[SelectedCandidate]) -> None:
    headers = [
        "rank",
        "sequence",
        "predicted_score",
        "min_hamming_distance",
        "composite_score",
        "mutations",
        "approx_gain",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(headers)
        for candidate in selections:
            writer.writerow(
                [
                    candidate.rank,
                    candidate.sequence,
                    f"{candidate.score:.6f}",
                    candidate.distance,
                    f"{candidate.composite_score:.6f}",
                    ";".join(candidate.mutations),
                    f"{candidate.approx_gain:.6f}",
                ]
            )


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    records, reference_sequence = read_assay_records(
        args.input,
        args.sequence_column,
        args.batch_column,
        args.condition_column,
        args.objective_column,
        args.reference_column,
        args.reference_sequence,
    )
    aggregated_records = aggregate_records(records, reference_sequence)
    batches = group_by_batch(aggregated_records)
    anchors = detect_anchors(aggregated_records)
    preference_pairs = generate_preference_pairs(batches, anchors)
    featurizer = MutationFeaturizer(reference_sequence)
    for record in aggregated_records:
        featurizer.register_sequence(record["sequence"])
    training_data = build_training_data(preference_pairs, featurizer)
    preference_net = PreferenceNet(
        n_features=len(featurizer.feature_names()),
        learning_rate=args.learning_rate,
        l2=args.l2,
        epochs=args.epochs,
    )
    preference_net.train(training_data)
    baseline = compute_reference_baseline(aggregated_records, reference_sequence)
    output_dir = Path(args.output_dir)
    mutation_scores: Dict[str, float] = {}
    if args.kermut_runner:
        try:
            mutation_scores = run_kermut_runner(
                records=aggregated_records,
                reference_sequence=reference_sequence,
                command=args.kermut_runner,
                output_dir=output_dir / args.kermut_run_output_dir,
                sequence_column=args.kermut_run_sequence_column,
                target_column=args.kermut_run_target_column,
                mutation_column=args.kermut_run_mutation_column or None,
                extra_args=args.kermut_run_extra_args or [],
                training_filename=args.kermut_run_training_file,
                fallback_baseline=baseline,
            )
        except Exception as exc:
            raise RuntimeError(f"Kermut runner failed: {exc}") from exc
    if not mutation_scores and args.kermut_singles:
        mutation_scores = load_kermut_single_mutation_scores(
            path=args.kermut_singles,
            sequence_column=args.kermut_sequence_column,
            score_column=args.kermut_score_column,
            reference_sequence=reference_sequence,
            delimiter=args.kermut_delimiter,
            fallback_baseline=baseline,
        )
    if not mutation_scores:
        mutation_scores = compute_mutation_scores(aggregated_records, reference_sequence, baseline)
    candidate_proposals = enumerate_candidates(
        reference_sequence,
        mutation_scores,
        args.max_mutations,
        args.candidate_pool_size,
        [record["sequence"] for record in aggregated_records],
    )
    if not candidate_proposals:
        candidate_proposals = [
            CandidateProposal(
                sequence=record["sequence"],
                mutations=record["mutations"],
                approx_gain=float(record["objective"]),
            )
            for record in sorted(aggregated_records, key=lambda item: float(item["objective"]), reverse=True)
        ]
    candidate_scores: Dict[str, CandidateScore] = {}
    for proposal in candidate_proposals:
        features = featurizer.vector(proposal.sequence)
        score = preference_net.score(features)
        candidate_scores[proposal.sequence] = CandidateScore(
            sequence=proposal.sequence,
            score=score,
            distance=0,
            mutations=proposal.mutations,
            approx_gain=proposal.approx_gain,
        )
    reference_length = len(reference_sequence)
    selections = select_next_batch(
        candidate_scores,
        args.num_select,
        args.diversity_weight,
        args.min_hamming_distance,
        reference_length,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    export_selection(output_dir / "next_batch.tsv", selections)
    if args.export_pairs:
        export_pairs(Path(args.export_pairs), preference_pairs)
    if args.export_candidates:
        # Export the full pool with diversity computed relative to selections for transparency
        enriched_candidates: List[CandidateScore] = []
        for sequence, candidate in candidate_scores.items():
            if selections:
                min_distance = min(
                    hamming_distance(sequence, selected.sequence) for selected in selections
                )
            else:
                min_distance = reference_length
            enriched_candidates.append(
                CandidateScore(
                    sequence=sequence,
                    score=candidate.score,
                    distance=min_distance,
                    mutations=candidate.mutations,
                    approx_gain=candidate.approx_gain,
                )
            )
        export_candidates(Path(args.export_candidates), enriched_candidates)


if __name__ == "__main__":
    main()
