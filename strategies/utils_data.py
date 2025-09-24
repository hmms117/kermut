"""Utilities to load multi-objective assay datasets and prepare them for optimisation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


class Direction(str, Enum):
    """Optimisation direction for a particular assay objective."""

    MAX = "max"
    MIN = "min"

    @classmethod
    def from_string(cls, value: str) -> "Direction":
        """Normalise user-provided direction strings."""

        value = value.lower().strip()
        if value in {"max", "maximize", "maximise", "+"}:
            return cls.MAX
        if value in {"min", "minimize", "minimise", "-"}:
            return cls.MIN
        raise ValueError(f"Unsupported direction '{value}'. Use 'max' or 'min'.")


@dataclass(frozen=True)
class ObjectiveConfig:
    """Configuration for a single assay objective."""

    name: str
    direction: Direction
    weight: float = 1.0

    @property
    def signed_weight(self) -> float:
        """Return the weight multiplied by the optimisation sign."""

        return self.weight if self.direction is Direction.MAX else -self.weight


@dataclass
class PreparedDataset:
    """Container storing a multi-objective dataset and derived metadata."""

    raw: pd.DataFrame
    table: pd.DataFrame
    configs: Sequence[ObjectiveConfig]
    anchor_offsets: Mapping[int, pd.Series]
    sequence_column: str
    batch_column: str = "batch"
    condition_column: str = "condition"

    def objective_columns(self) -> List[str]:
        """Return the raw objective column names."""

        return [cfg.name for cfg in self.configs]

    def signed_columns(self) -> List[str]:
        """Return the column names storing direction-adjusted objectives."""

        return [f"{cfg.name}_signed" for cfg in self.configs]

    def aligned_columns(self) -> List[str]:
        """Return the column names storing anchor-aligned objectives."""

        return [f"{cfg.name}_aligned" for cfg in self.configs]

    @property
    def energy_column(self) -> str:
        """Name of the BAGEL-style energy column."""

        return "bagel_energy"

    @property
    def score_column(self) -> str:
        """Name of the aggregated weighted score column."""

        return "weighted_score"

    def to_objective_matrix(self) -> np.ndarray:
        """Return a numpy array of the signed objectives for Pareto sorting."""

        return self.table[self.signed_columns()].to_numpy()

    def summary(self, indices: Sequence[int]) -> pd.DataFrame:
        """Return a tidy summary table for the requested row indices."""

        base_columns = [self.sequence_column, self.batch_column, self.condition_column]
        base_columns = [col for col in base_columns if col in self.table.columns]
        extra_columns = (
            list(dict.fromkeys(base_columns + self.objective_columns()))
            + self.aligned_columns()
            + self.signed_columns()
            + [self.score_column, self.energy_column]
        )
        intersecting = [col for col in extra_columns if col in self.table.columns]
        return self.table.loc[list(indices), intersecting].reset_index(drop=True)


def load_data(
    path: Path,
    *,
    batch_column: str = "batch",
    condition_column: str = "condition",
) -> pd.DataFrame:
    """Load an assay table, automatically detecting the delimiter."""

    if not Path(path).exists():
        raise FileNotFoundError(f"Assay file not found at {path}")

    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)

    for column in (batch_column, condition_column):
        if column not in df.columns:
            raise KeyError(
                f"Required column '{column}' missing from assay file {path.name}."
            )

    rename: Dict[str, str] = {}
    if batch_column != "batch":
        rename[batch_column] = "batch"
    if condition_column != "condition":
        rename[condition_column] = "condition"
    if rename:
        df = df.rename(columns=rename)

    if df["batch"].isna().any():
        raise ValueError("Batch column contains missing values.")

    df["batch"] = df["batch"].astype(int)
    df["condition"] = df["condition"].astype(str)
    return df


def infer_objective_configs(
    objectives: Sequence[str],
    directions: Sequence[str],
    weights: Optional[Sequence[float]] = None,
) -> List[ObjectiveConfig]:
    """Build a list of :class:`ObjectiveConfig` from CLI arguments."""

    if not objectives:
        raise ValueError("At least one objective must be provided.")

    if len(directions) != len(objectives):
        raise ValueError("Number of directions must match number of objectives.")

    if weights is None:
        weights = [1.0] * len(objectives)
    elif len(weights) != len(objectives):
        raise ValueError("Number of weights must match number of objectives.")

    if not all(weight >= 0 for weight in weights):
        raise ValueError("Objective weights must be non-negative.")

    configs: List[ObjectiveConfig] = []
    for name, direction, weight in zip(objectives, directions, weights):
        configs.append(
            ObjectiveConfig(name=name, direction=Direction.from_string(direction), weight=weight)
        )
    return configs


def compute_anchor_offsets(
    df: pd.DataFrame,
    objectives: Sequence[str],
    *,
    batch_column: str = "batch",
    condition_column: str = "condition",
) -> Dict[int, pd.Series]:
    """Estimate per-batch offsets using anchor conditions observed across rounds."""

    if batch_column not in df.columns or condition_column not in df.columns:
        raise KeyError("Data frame must contain batch and condition columns for alignment.")

    batches = sorted(df[batch_column].unique())
    if not batches:
        raise ValueError("The dataset does not contain any batches.")

    base_batch = batches[0]
    offsets: Dict[int, pd.Series] = {}
    base = (
        df[df[batch_column] == base_batch]
        .groupby(condition_column)[list(objectives)]
        .mean(numeric_only=True)
    )

    zero = pd.Series(0.0, index=objectives, dtype=float)
    offsets[base_batch] = zero

    for batch in batches[1:]:
        current = (
            df[df[batch_column] == batch]
            .groupby(condition_column)[list(objectives)]
            .mean(numeric_only=True)
        )
        overlap = current.index.intersection(base.index)
        if len(overlap) == 0:
            offsets[batch] = zero.copy()
            continue
        diff = current.loc[overlap, objectives] - base.loc[overlap, objectives]
        offsets[batch] = diff.mean(axis=0)
    return offsets


def apply_anchor_offsets(
    df: pd.DataFrame,
    offsets: Mapping[int, pd.Series],
    objectives: Sequence[str],
    *,
    batch_column: str = "batch",
) -> pd.DataFrame:
    """Return a new data frame where objectives are aligned using anchor offsets."""

    adjusted = df.copy()
    aligned_columns = {obj: f"{obj}_aligned" for obj in objectives}

    for obj in objectives:
        adjusted[aligned_columns[obj]] = adjusted[obj]

    for batch, offset in offsets.items():
        mask = adjusted[batch_column] == batch
        if mask.any():
            adjusted.loc[mask, list(aligned_columns.values())] = (
                adjusted.loc[mask, objectives]
                - np.asarray([offset[obj] for obj in objectives])
            )
    return adjusted


def add_directional_scores(
    df: pd.DataFrame,
    configs: Sequence[ObjectiveConfig],
) -> pd.DataFrame:
    """Attach direction-adjusted and weighted scores required by the strategies."""

    scored = df.copy()
    score = np.zeros(len(scored), dtype=float)
    for config in configs:
        aligned_col = f"{config.name}_aligned"
        if aligned_col not in scored.columns:
            raise KeyError(
                f"Aligned column '{aligned_col}' missing. Did you call apply_anchor_offsets?"
            )
        signed_col = f"{config.name}_signed"
        sign = 1.0 if config.direction is Direction.MAX else -1.0
        scored[signed_col] = scored[aligned_col] * sign
        score = score + scored[signed_col] * config.weight
    scored["weighted_score"] = score
    scored["bagel_energy"] = -score
    return scored


def infer_sequence_column(df: pd.DataFrame, preferred: str = "sequence") -> str:
    """Best-effort detection of the sequence column in the assay table."""

    candidates = [preferred, "mutated_sequence", "sequence_str", "protein_sequence"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(
        "Unable to locate a sequence column. Provide --sequence-column in the CLI."
    )


def prepare_multiobjective_dataset(
    path: Path,
    objectives: Sequence[str],
    directions: Sequence[str],
    weights: Optional[Sequence[float]] = None,
    *,
    batch_column: str = "batch",
    condition_column: str = "condition",
    sequence_column: Optional[str] = None,
) -> PreparedDataset:
    """Load a dataset, align batches, and attach scoring metadata."""

    raw = load_data(path, batch_column=batch_column, condition_column=condition_column)
    configs = infer_objective_configs(objectives, directions, weights)

    for config in configs:
        if config.name not in raw.columns:
            raise KeyError(f"Objective column '{config.name}' not present in the assay table.")

    seq_col = sequence_column or infer_sequence_column(raw)

    offsets = compute_anchor_offsets(
        raw,
        [cfg.name for cfg in configs],
        batch_column="batch",
        condition_column="condition",
    )
    aligned = apply_anchor_offsets(raw, offsets, [cfg.name for cfg in configs])
    scored = add_directional_scores(aligned, configs)

    return PreparedDataset(
        raw=raw,
        table=scored,
        configs=configs,
        anchor_offsets=offsets,
        sequence_column=seq_col,
        batch_column="batch",
        condition_column="condition",
    )


def prep_kermut(
    dataset: PreparedDataset,
    checkpoint: Optional[Path] = None,
    *,
    fine_tune: bool = True,
    extra_config: Optional[Mapping[str, object]] = None,
) -> object:
    """Placeholder for integrating Kermut fine-tuning within the strategies."""

    warnings.warn(
        "prep_kermut is a stub. Plug in your project-specific fine-tuning routine.",
        stacklevel=2,
    )
    return {
        "checkpoint": Path(checkpoint) if checkpoint else None,
        "fine_tune": fine_tune,
        "config": extra_config or {},
        "n_sequences": len(dataset.table),
    }
