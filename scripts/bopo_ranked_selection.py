"""Rank existing assay variants with a BOPO-inspired acquisition function.

In contrast to ``bopo_policy_learning`` (which trains a generative mutation
policy) this interface computes acquisition values for the multi-objective assay
table directly, producing a Pareto-diverse batch of existing variants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import typer

from scripts.datasets import PreparedDataset, prepare_multiobjective_dataset, prep_kermut
from scripts.pareto_pandas import non_dominated_sorting, select_diverse

app = typer.Typer(help="BOPO-inspired Bayesian optimisation front-end for Kermut assays.")


class BOPOPlanner:
    """Heuristic Bayesian-style planner leveraging anchor-aware normalisation."""

    def __init__(
        self,
        dataset: PreparedDataset,
        *,
        model: object | None = None,
        exploration_weight: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if exploration_weight < 0:
            raise ValueError("exploration_weight must be non-negative.")
        self.dataset = dataset
        self.model = model
        self.exploration_weight = exploration_weight
        self.rng = np.random.default_rng(seed)

    def _condition_uncertainty(self) -> np.ndarray:
        table = self.dataset.table
        counts = (
            table.groupby(self.dataset.condition_column)[self.dataset.sequence_column]
            .transform("count")
            .to_numpy()
        )
        counts = counts.astype(float)
        counts[counts <= 0] = 1.0
        uncertainty = 1.0 / np.sqrt(counts)
        return uncertainty

    def _batch_uncertainty(self) -> np.ndarray:
        table = self.dataset.table
        counts = (
            table.groupby(self.dataset.batch_column)[self.dataset.sequence_column]
            .transform("count")
            .to_numpy()
        )
        counts = counts.astype(float)
        counts[counts <= 0] = 1.0
        return 1.0 / np.sqrt(counts)

    def _novelty_bonus(self) -> np.ndarray:
        table = self.dataset.table
        anchor_bonus = np.zeros(len(table), dtype=float)
        # Reward sequences in batches with large anchor corrections (changed conditions).
        for batch, offset in self.dataset.anchor_offsets.items():
            magnitude = float(np.linalg.norm(offset.to_numpy()))
            if magnitude == 0:
                continue
            mask = table[self.dataset.batch_column] == batch
            anchor_bonus[mask.to_numpy()] = magnitude
        if anchor_bonus.max() > 0:
            anchor_bonus = anchor_bonus / anchor_bonus.max()
        noise = self.rng.normal(scale=0.05, size=len(table))
        return anchor_bonus + noise

    def acquisition(self) -> pd.Series:
        table = self.dataset.table
        exploration = self._condition_uncertainty() + self._batch_uncertainty() + self._novelty_bonus()
        if np.allclose(exploration.std(), 0):
            exploration = np.zeros_like(exploration)
        else:
            exploration = (exploration - exploration.mean()) / (exploration.std() + 1e-8)
        acquisition = table[self.dataset.score_column].to_numpy() + self.exploration_weight * exploration
        return pd.Series(acquisition, index=table.index, name="acquisition")

    def suggest(self, batch_size: int) -> pd.DataFrame:
        table = self.dataset.table.copy()
        table["_dataset_index"] = table.index
        table["acquisition"] = self.acquisition()

        fronts = non_dominated_sorting(table[self.dataset.signed_columns()].to_numpy())
        selected_frames: list[pd.DataFrame] = []
        remaining = batch_size
        for front_indices in fronts:
            if remaining <= 0:
                break
            front_df = table.iloc[front_indices].copy()
            front_df = front_df.sort_values("acquisition", ascending=False)
            if len(front_df) <= remaining:
                selected_frames.append(front_df)
                remaining -= len(front_df)
                continue
            selected = select_diverse(
                front_df.head(remaining * 3),
                remaining,
                feature_columns=self.dataset.signed_columns(),
            ).sort_values("acquisition", ascending=False)
            selected_frames.append(selected)
            remaining -= len(selected)
        if remaining > 0:
            leftovers = (
                table.drop(pd.concat(selected_frames).index)
                .sort_values("acquisition", ascending=False)
                .head(remaining)
            )
            if not leftovers.empty:
                selected_frames.append(leftovers)
        selected = pd.concat(selected_frames, ignore_index=True)
        if "_dataset_index" not in selected.columns:
            selected["_dataset_index"] = selected.index
        return selected


@app.command()
def suggest(
    assay_file: Path = typer.Argument(..., help="Path to the TSV/CSV assay table."),
    *,
    objectives: Sequence[str] = typer.Option(
        ..., "--objectives", help="Objective column names, e.g. activity stability."
    ),
    directions: Sequence[str] = typer.Option(
        ..., "--directions", help="Optimisation direction for each objective (max/min)."
    ),
    weights: Optional[Sequence[float]] = typer.Option(
        None, "--weights", help="Relative weight for each objective (defaults to 1.0)."
    ),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", help="Optional Kermut checkpoint to warm-start from."
    ),
    batch_size: int = typer.Option(96, help="Number of variants to return."),
    exploration_weight: float = typer.Option(
        0.5,
        help="Trade-off between exploitation (posterior mean) and exploration bonuses.",
    ),
    sequence_column: Optional[str] = typer.Option(
        None, "--sequence-column", help="Column containing amino-acid sequences."
    ),
    batch_column: str = typer.Option("batch", help="Column describing assay batch numbers."),
    condition_column: str = typer.Option(
        "condition", help="Column describing shared experimental conditions."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Optional path to save the suggested variants as TSV."
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed controlling tie-breaks."),
) -> None:
    """Run a BOPO-inspired acquisition routine on an assay table."""

    typer.secho("📂 Loading assay measurements", fg=typer.colors.CYAN)
    dataset = prepare_multiobjective_dataset(
        assay_file,
        objectives=objectives,
        directions=directions,
        weights=weights,
        batch_column=batch_column,
        condition_column=condition_column,
        sequence_column=sequence_column,
    )

    typer.secho("🧠 Preparing Kermut backbone", fg=typer.colors.CYAN)
    model = prep_kermut(dataset, checkpoint)

    typer.secho("🤖 Running BOPO acquisition", fg=typer.colors.CYAN)
    planner = BOPOPlanner(
        dataset,
        model=model,
        exploration_weight=exploration_weight,
        seed=seed,
    )
    suggestions = planner.suggest(batch_size)

    if "_dataset_index" not in suggestions.columns:
        raise RuntimeError("Internal error: suggestions missing '_dataset_index' column.")

    typer.secho("📊 Collating summary table", fg=typer.colors.CYAN)
    dataset_indices = suggestions["_dataset_index"].astype(int).tolist()
    summary = dataset.summary(dataset_indices)
    summary.insert(0, "dataset_index", dataset_indices)
    summary["acquisition"] = suggestions["acquisition"].to_numpy()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, sep="\t", index=False)
        typer.secho(f"✅ Saved {len(summary)} variants to {output}", fg=typer.colors.GREEN)
    else:
        typer.echo(summary.to_csv(sep="\t", index=False))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()
