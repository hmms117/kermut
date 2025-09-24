"""Monte Carlo BAGEL-style exploration driven by a generative proposal policy.

This script performs stochastic search over sequence space by repeatedly
proposing neighbours of the current best design.  It complements
``bagel_ranked_selection`` which instead re-ranks an existing assay table.
Both entry points operate on the shared multi-objective tabular format where
each objective is provided as a separate column.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import typer

from scripts.io import load_assays, write_candidates
from scripts.model_adapter import KermutAdapter
from scripts.pareto_numpy import pareto_front, select_diverse
from scripts.sequences import count_muts, propose_neighbors

app = typer.Typer(help="BAGEL-inspired Monte Carlo exploration on top of Kermut.")


def _normalise_weights(weights: Sequence[float]) -> List[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Objective weights must sum to a positive value.")
    return [float(weight) / total for weight in weights]


def _scalar_energy(
    mu_by_obj: Dict[str, float],
    *,
    objectives: Sequence[str],
    directions: Sequence[str],
    weights: Sequence[float],
    uncertainty: Optional[Dict[str, float]] = None,
    uncertainty_beta: float = 0.0,
) -> float:
    """Return a scalarised energy where lower values are better."""

    norm_weights = _normalise_weights(weights)
    energy = 0.0
    for idx, objective in enumerate(objectives):
        direction = directions[idx].lower()
        sign = -1.0 if direction.startswith("max") else 1.0
        energy += norm_weights[idx] * sign * float(mu_by_obj[objective])
        if uncertainty_beta and uncertainty is not None:
            energy += norm_weights[idx] * uncertainty_beta * float(
                uncertainty.get(objective, 0.0)
            )
    return energy


def _as_maximise_vector(
    mu_by_obj: Dict[str, float],
    *,
    objectives: Sequence[str],
    directions: Sequence[str],
) -> np.ndarray:
    values: List[float] = []
    for idx, objective in enumerate(objectives):
        value = float(mu_by_obj[objective])
        if directions[idx].lower().startswith("min"):
            value = -value
        values.append(value)
    return np.array(values, dtype=float)


@app.command()
def suggest(
    *,
    data: Path = typer.Option(..., help="TSV/CSV with columns: sequence, objectives"),
    ckpt: Path = typer.Option(..., help="Path to the Kermut checkpoint"),
    objectives: List[str] = typer.Option(..., help="Objective column names"),
    directions: List[str] = typer.Option(..., help="Direction per objective: max|min"),
    weights: List[float] = typer.Option(..., help="Priority weight per objective"),
    batch_size: int = typer.Option(96, help="Number of variants to return"),
    iters: int = typer.Option(25_000, help="Monte Carlo iterations"),
    min_muts: int = typer.Option(2, help="Minimum mutations per variant"),
    max_muts: int = typer.Option(11, help="Maximum mutations per variant"),
    beam_width: int = typer.Option(200, help="Pre-filter size for neighbour proposals"),
    site_pool: Optional[str] = typer.Option(None, help="Comma-separated 0-based sites"),
    pareto: bool = typer.Option(True, help="Return Pareto front instead of scalar top-K"),
    uncertainty_beta: float = typer.Option(
        0.0, help=">=0 penalises high-σ; <0 encourages exploration"
    ),
    base_seq: Optional[str] = typer.Option(None, help="Wild-type sequence override"),
    out_csv: Path = typer.Option(Path("bagel_next_batch.csv")),
    device: str = typer.Option("cuda"),
    seed: int = typer.Option(1234, help="Random seed for reproducibility"),
) -> None:
    """Suggest the next experimental batch via energy-based Monte Carlo search."""

    if not (len(objectives) == len(directions) == len(weights)):
        raise typer.BadParameter("Objectives, directions and weights must have equal length.")

    df = load_assays(data)
    if "sequence" not in df.columns:
        raise typer.BadParameter("Input data must contain a 'sequence' column.")

    wild_type = base_seq or df.iloc[0]["sequence"]
    if not isinstance(wild_type, str):
        raise typer.BadParameter("Unable to infer wild-type sequence from the data.")

    mutable_sites = [int(token) for token in site_pool.split(",")] if site_pool else None

    model = KermutAdapter(str(ckpt), device=device)
    model.fit(df)

    rng = np.random.default_rng(seed)
    current = wild_type
    best_sequence = current
    best_energy = float("inf")
    pareto_pool: List[Tuple[str, np.ndarray]] = []
    score_cache: Dict[str, Dict[str, float]] = {}

    temperature_initial = 1.0
    temperature_final = 0.01

    for step in range(max(1, iters)):
        neighbours = list(
            propose_neighbors(
                current,
                min_muts=min_muts,
                max_muts=max_muts,
                site_pool=mutable_sites,
                rng=rng,
                max_candidates=beam_width,
            )
        )
        sequences = [current] + neighbours
        mu_list, sigma_list = model.predict_many(sequences, objectives)

        energies = []
        maximise_vectors = []
        for seq, mu, sigma in zip(sequences, mu_list, sigma_list):
            score_cache[seq] = mu
            energies.append(
                _scalar_energy(
                    mu,
                    objectives=objectives,
                    directions=directions,
                    weights=weights,
                    uncertainty=sigma,
                    uncertainty_beta=uncertainty_beta,
                )
            )
            maximise_vectors.append(
                _as_maximise_vector(mu, objectives=objectives, directions=directions)
            )

        candidate_index = int(np.argmin(energies[1:], axis=0)) + 1
        candidate_seq = sequences[candidate_index]
        candidate_energy = energies[candidate_index]
        candidate_vector = maximise_vectors[candidate_index]

        t_ratio = float(step) / float(max(1, iters - 1))
        temperature = temperature_initial * (temperature_final / temperature_initial) ** t_ratio
        energy_delta = candidate_energy - energies[0]
        accept = energy_delta <= 0 or rng.random() < math.exp(
            -energy_delta / max(temperature, 1e-8)
        )

        if accept:
            current = candidate_seq
            if candidate_energy < best_energy:
                best_energy = candidate_energy
                best_sequence = candidate_seq

        pareto_pool.append((candidate_seq, candidate_vector))
        if best_sequence in score_cache:
            pareto_pool.append(
                (
                    best_sequence,
                    _as_maximise_vector(
                        score_cache[best_sequence],
                        objectives=objectives,
                        directions=directions,
                    ),
                )
            )

    if pareto:
        front = pareto_front(pareto_pool)
        selected = select_diverse(front, k=batch_size, min_hamming=2)
        selected_sequences = [seq for seq, _ in selected]
    else:
        scores: Dict[str, Tuple[np.ndarray, float]] = {}
        norm_weights = np.array(_normalise_weights(weights), dtype=float)
        for seq, vector in pareto_pool:
            if seq not in scores:
                scores[seq] = (vector, float(np.dot(norm_weights, vector)))
        ranked = sorted(scores.items(), key=lambda item: -item[1][1])
        selected_sequences = [seq for seq, _ in ranked[:batch_size]]

    mu_selected, sigma_selected = model.predict_many(selected_sequences, objectives)
    records = []
    for seq, mu, sigma in zip(selected_sequences, mu_selected, sigma_selected):
        record: Dict[str, float | str | int] = {
            "sequence": seq,
            "n_mut": count_muts(wild_type, seq),
        }
        for objective in objectives:
            record[f"{objective}_pred"] = float(mu[objective])
            record[f"{objective}_std"] = float(sigma[objective])
        records.append(record)

    write_candidates(records, out_csv)
    typer.secho(f"Wrote {len(records)} candidates → {out_csv}", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover - Typer entry point
    app()
