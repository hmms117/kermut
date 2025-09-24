"""Pareto front utilities shared across strategy implementations."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return ``True`` if point ``a`` dominates point ``b`` in maximisation sense."""

    return np.all(a >= b) and np.any(a > b)


def non_dominated_sorting(points: np.ndarray) -> List[List[int]]:
    """Fast non-dominated sorting for a set of points."""

    if points.ndim != 2:
        raise ValueError("Points array must be two-dimensional.")

    n_points = points.shape[0]
    domination_counts = np.zeros(n_points, dtype=int)
    dominated: List[List[int]] = [[] for _ in range(n_points)]
    fronts: List[List[int]] = [[]]

    for p in range(n_points):
        for q in range(n_points):
            if p == q:
                continue
            if dominates(points[p], points[q]):
                dominated[p].append(q)
            elif dominates(points[q], points[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            fronts[0].append(p)

    front_idx = 0
    while fronts[front_idx]:
        next_front: List[int] = []
        for p in fronts[front_idx]:
            for q in dominated[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        front_idx += 1
        fronts.append(next_front)
    if not fronts[-1]:
        fronts.pop()
    return fronts


def _max_min_diversity(features: np.ndarray, k: int) -> List[int]:
    """Greedy max-min diversity selection."""

    if features.ndim != 2:
        raise ValueError("Features must be two-dimensional.")
    n = features.shape[0]
    if k >= n:
        return list(range(n))

    norms = np.linalg.norm(features, axis=1)
    first = int(np.argmax(norms))
    selected = [first]
    remaining = set(range(n))
    remaining.remove(first)

    while len(selected) < k and remaining:
        best_idx = None
        best_distance = -np.inf
        for idx in list(remaining):
            dist = min(
                float(np.linalg.norm(features[idx] - features[sel])) for sel in selected
            )
            if dist > best_distance:
                best_distance = dist
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected


def select_diverse(
    front: pd.DataFrame,
    k: int,
    *,
    feature_columns: Sequence[str] | None = None,
    embeddings: np.ndarray | None = None,
) -> pd.DataFrame:
    """Select a diverse subset from the provided Pareto front."""

    if k <= 0:
        raise ValueError("k must be positive.")
    if len(front) == 0:
        return front.copy()

    if embeddings is not None:
        if embeddings.shape[0] != len(front):
            raise ValueError("Embeddings must match the number of rows in the front.")
        features = embeddings
    else:
        if not feature_columns:
            raise ValueError(
                "feature_columns must be provided when embeddings are not supplied."
            )
        for column in feature_columns:
            if column not in front.columns:
                raise KeyError(f"Feature column '{column}' missing from Pareto front data frame.")
        features = front.loc[:, feature_columns].to_numpy()

    indices = _max_min_diversity(features, k)
    return front.iloc[indices]
