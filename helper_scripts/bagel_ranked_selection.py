"""Rank existing assay measurements with a BAGEL-style energy model.

This interface works on recorded assay rows and therefore complements
``bagel_monte_carlo`` which searches sequence space using stochastic mutation
proposals.  Both entry points consume the same multi-objective tabular format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import typer

from helper_scripts.datasets import PreparedDataset, prepare_multiobjective_dataset, prep_kermut
from helper_scripts.pareto_pandas import non_dominated_sorting, select_diverse

app = typer.Typer(help="BAGEL Monte Carlo exploration tailored for multi-objective assays.")


class BAGELPlanner:
    """Simple Monte Carlo sampler that ranks existing variants using BAGEL energy."""

    def __init__(
        self,
        dataset: PreparedDataset,
        *,
        model: object | None = None,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be strictly positive.")
        self.dataset = dataset
        self.model = model
        self.temperature = float(temperature)
        self.rng = np.random.default_rng(seed)

    def _boltzmann_weights(self, energies: np.ndarray) -> np.ndarray:
        scaled = -energies / self.temperature
        scaled -= scaled.max()
        weights = np.exp(scaled)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return np.full_like(weights, 1.0 / len(weights))
        return weights / total

    def _candidate_pool(self, batch_size: int, pool_multiplier: int = 25) -> pd.DataFrame:
        table = self.dataset.table
        energies = table[self.dataset.energy_column].to_numpy()
        weights = self._boltzmann_weights(energies)
        pool_size = min(len(table), max(batch_size * pool_multiplier, batch_size))
        if pool_size == len(table):
            indices = np.arange(len(table))
        else:
            indices = self.rng.choice(len(table), size=pool_size, replace=False, p=weights)
        pool = table.iloc[indices].copy()
        pool["_dataset_index"] = pool.index
        return pool

    def _select_from_fronts(self, candidate_pool: pd.DataFrame, batch_size: int) -> pd.DataFrame:
        if len(candidate_pool) <= batch_size:
            return candidate_pool.reset_index(drop=True)

        fronts = non_dominated_sorting(
            candidate_pool[self.dataset.signed_columns()].to_numpy()
        )
        selected_frames: list[pd.DataFrame] = []
        remaining = batch_size
        for front_indices in fronts:
            if remaining <= 0:
                break
            front_df = candidate_pool.iloc[front_indices].copy()
            if len(front_df) <= remaining:
                selected_frames.append(front_df)
                remaining -= len(front_df)
                continue
            selected = select_diverse(
                front_df,
                remaining,
                feature_columns=self.dataset.signed_columns(),
            )
            selected_frames.append(selected)
            remaining -= len(selected)
        if remaining > 0:
            # Fallback to top-scoring sequences if Pareto fronts are exhausted.
            leftovers = (
                candidate_pool.drop(pd.concat(selected_frames).index)
                .sort_values(self.dataset.score_column, ascending=False)
                .head(remaining)
            )
            if not leftovers.empty:
                selected_frames.append(leftovers)
        return pd.concat(selected_frames, ignore_index=True)

    def suggest(self, batch_size: int) -> pd.DataFrame:
        """Return BAGEL-ranked suggestions with Pareto diversity."""

        pool = self._candidate_pool(batch_size)
        selected = self._select_from_fronts(pool, batch_size).copy()
        if "_dataset_index" not in selected.columns:
            selected["_dataset_index"] = selected.index
        return selected.reset_index(drop=True)


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
    sequence_column: Optional[str] = typer.Option(
        None, "--sequence-column", help="Column containing amino-acid sequences."
    ),
    batch_column: str = typer.Option("batch", help="Column describing assay batch numbers."),
    condition_column: str = typer.Option(
        "condition", help="Column describing shared experimental conditions."
    ),
    temperature: float = typer.Option(
        1.0, help="Boltzmann temperature controlling energy diversity in the proposal."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", help="Optional path to save the suggested variants as TSV."
    ),
    seed: Optional[int] = typer.Option(None, help="Random seed for the Monte Carlo sampler."),
) -> None:
    """Rank candidates using BAGEL energy and select a diverse Pareto subset."""

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

    typer.secho("🎲 Running BAGEL-style Monte Carlo", fg=typer.colors.CYAN)
    planner = BAGELPlanner(dataset, model=model, temperature=temperature, seed=seed)
    suggestions = planner.suggest(batch_size)

    typer.secho("📊 Collating summary table", fg=typer.colors.CYAN)
    if "_dataset_index" not in suggestions.columns:
        raise RuntimeError("Internal error: suggestions missing '_dataset_index' column.")

    dataset_indices = suggestions["_dataset_index"].astype(int).tolist()
    summary = dataset.summary(dataset_indices)
    summary.insert(0, "dataset_index", dataset_indices)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, sep="\t", index=False)
        typer.secho(f"✅ Saved {len(summary)} variants to {output}", fg=typer.colors.GREEN)
    else:
        typer.echo(summary.to_csv(sep="\t", index=False))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    app()
