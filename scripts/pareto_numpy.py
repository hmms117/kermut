"""Pareto utilities shared by the BAGEL and BOPO Typer scripts."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

SequenceScore = Tuple[str, np.ndarray]


def _is_dominated(candidate: np.ndarray, challenger: np.ndarray) -> bool:
    """Return ``True`` if *candidate* is dominated by *challenger* in max-space."""

    return np.all(challenger >= candidate) and np.any(challenger > candidate)


def pareto_front(items: Sequence[SequenceScore]) -> List[SequenceScore]:
    """Compute the non-dominated set.

    Each item is a pair ``(sequence, scores)`` where ``scores`` is already in
    "maximize" orientation.  Duplicate sequences are merged by taking the
    element-wise maximum of their score vectors to keep the best observed
    projection for each objective.
    """

    aggregated: dict[str, np.ndarray] = {}
    for seq, scores in items:
        if seq in aggregated:
            aggregated[seq] = np.maximum(aggregated[seq], scores)
        else:
            aggregated[seq] = np.array(scores, dtype=float, copy=True)

    sequences = list(aggregated.keys())
    vectors = [aggregated[seq] for seq in sequences]

    front: List[SequenceScore] = []
    for idx, vector in enumerate(vectors):
        dominated = False
        for jdx, other in enumerate(vectors):
            if idx == jdx:
                continue
            if _is_dominated(vector, other):
                dominated = True
                break
        if not dominated:
            front.append((sequences[idx], vector))
    return front


def hamming(a: str, b: str) -> int:
    """Return the Hamming distance between two aligned sequences."""

    if len(a) != len(b):
        raise ValueError("Sequences must have the same length for Hamming distance.")
    return sum(char_a != char_b for char_a, char_b in zip(a, b))


def select_diverse(
    front: Sequence[SequenceScore],
    *,
    k: int,
    min_hamming: int = 2,
) -> List[SequenceScore]:
    """Greedy selection favouring Hamming-diverse sequences."""

    if k <= 0:
        return []

    ordered = sorted(front, key=lambda item: -float(np.sum(item[1])))
    chosen: List[SequenceScore] = []
    for seq, vector in ordered:
        if len(chosen) >= k:
            break
        if all(hamming(seq, existing_seq) >= min_hamming for existing_seq, _ in chosen):
            chosen.append((seq, vector))
    return chosen


__all__ = ["pareto_front", "select_diverse", "hamming"]
