"""BOPO-style preference-optimised proposal learning for Kermut."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import typer

from kermut_ext.io_utils import load_assays, write_candidates
from kermut_ext.kermut_adapter import KermutAdapter
from kermut_ext.pareto import pareto_front, select_diverse

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"

app = typer.Typer(help="BOPO-inspired proposal learning on top of Kermut.")


class IndependentMutationPolicy(nn.Module):
    """Factorised categorical policy over per-site amino-acid substitutions."""

    def __init__(
        self,
        length: int,
        *,
        aa_vocab: str = AA_VOCAB,
        site_pool: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.length = length
        self.aa_vocab = aa_vocab
        self.site_pool = list(site_pool) if site_pool is not None else list(range(length))
        self.num_aa = len(self.aa_vocab)
        self.logits = nn.Parameter(torch.zeros(length, self.num_aa))

    def sample_batch(
        self,
        wild_type: str,
        *,
        size: int,
        min_muts: int,
        max_muts: int,
        rng: np.random.Generator,
    ) -> Tuple[List[str], np.ndarray]:
        sequences: List[str] = []
        log_probs: List[float] = []
        for _ in range(size):
            n_mut = int(rng.integers(min_muts, max_muts + 1))
            positions = rng.choice(self.site_pool, size=n_mut, replace=False)
            seq = list(wild_type)
            log_prob = 0.0
            for pos in positions:
                probs = torch.softmax(self.logits[pos], dim=-1)
                idx = torch.distributions.Categorical(probs=probs).sample().item()
                aa = self.aa_vocab[idx]
                if aa == seq[pos]:
                    idx = (idx + 1) % self.num_aa
                    aa = self.aa_vocab[idx]
                log_prob += float(torch.log(probs[idx] + 1e-12))
                seq[pos] = aa
            sequences.append("".join(seq))
            log_probs.append(log_prob)
        return sequences, np.asarray(log_probs, dtype=float)

    def avg_log_prob(self, sequence: str, wild_type: str) -> torch.Tensor:
        muts = [(idx, aa) for idx, aa in enumerate(sequence) if aa != wild_type[idx]]
        if not muts:
            return torch.tensor(0.0, device=self.logits.device)
        log_probs = []
        for pos, aa in muts:
            idx = self.aa_vocab.index(aa)
            log_prob = torch.log_softmax(self.logits[pos], dim=-1)[idx]
            log_probs.append(log_prob)
        stacked = torch.stack(log_probs)
        return stacked.mean()


def _scalar_objective(
    mu_by_obj: Dict[str, float],
    *,
    objectives: Sequence[str],
    directions: Sequence[str],
    weights: Sequence[float],
) -> float:
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        raise ValueError("Objective weights must sum to a positive value.")
    weights = weights / weight_sum
    total = 0.0
    for idx, objective in enumerate(objectives):
        value = float(mu_by_obj[objective])
        if directions[idx].lower().startswith("max"):
            value = -value
        total += weights[idx] * value
    return total


@app.command()
def suggest(
    *,
    data: Path = typer.Option(..., help="TSV/CSV with columns: sequence, objectives"),
    ckpt: Path = typer.Option(..., help="Path to the Kermut checkpoint"),
    objectives: List[str] = typer.Option(..., help="Objective column names"),
    directions: List[str] = typer.Option(..., help="Direction per objective: max|min"),
    weights: List[float] = typer.Option(..., help="Priority weight per objective"),
    epochs: int = typer.Option(30, help="Number of BOPO optimisation epochs"),
    B: int = typer.Option(256, help="Hybrid rollout solutions per epoch"),
    K: int = typer.Option(16, help="Filtered candidates for preference pairs"),
    batch_size: int = typer.Option(96, help="Number of variants to return"),
    min_muts: int = typer.Option(2, help="Minimum mutations per variant"),
    max_muts: int = typer.Option(11, help="Maximum mutations per variant"),
    site_pool: Optional[str] = typer.Option(None, help="Comma-separated 0-based mutable sites"),
    lr: float = typer.Option(5e-2, help="Learning rate for the policy optimiser"),
    seed: int = typer.Option(123, help="Random seed"),
    device: str = typer.Option("cuda"),
    base_seq: Optional[str] = typer.Option(None, help="Wild-type sequence override"),
    out_csv: Path = typer.Option(Path("bopo_next_batch.csv")),
) -> None:
    """Learn a proposal policy via BOPO and sample the next Pareto batch."""

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

    policy = IndependentMutationPolicy(len(wild_type), site_pool=mutable_sites).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)

    for _ in range(epochs):
        policy.train()
        batch, _ = policy.sample_batch(
            wild_type,
            size=B,
            min_muts=min_muts,
            max_muts=max_muts,
            rng=rng,
        )
        mu_list, _ = model.predict_many(batch, objectives)
        scalar_values = np.asarray(
            [
                _scalar_objective(
                    mu,
                    objectives=objectives,
                    directions=directions,
                    weights=weights,
                )
                for mu in mu_list
            ],
            dtype=float,
        )
        order = np.argsort(scalar_values)
        filtered_indices = np.linspace(0, len(batch) - 1, num=min(K, len(batch)), dtype=int)
        filtered_sequences = [batch[order[idx]] for idx in filtered_indices]
        filtered_scores = [scalar_values[order[idx]] for idx in filtered_indices]

        if len(filtered_sequences) < 2:
            continue

        best_sequence = filtered_sequences[0]
        best_score = filtered_scores[0]
        loss = torch.tensor(0.0, device=device)
        optimiser.zero_grad()
        for sequence, score in zip(filtered_sequences[1:], filtered_scores[1:]):
            scale = float(score / (best_score + 1e-12))
            best_logprob = policy.avg_log_prob(best_sequence, wild_type)
            seq_logprob = policy.avg_log_prob(sequence, wild_type)
            diff = best_logprob - seq_logprob
            loss = loss - torch.log(torch.sigmoid(scale * diff) + 1e-12)
        loss = loss / max(1, len(filtered_sequences) - 1)
        loss.backward()
        optimiser.step()

    policy.eval()
    sample_size = max(5_000, batch_size * 200)
    samples, _ = policy.sample_batch(
        wild_type,
        size=sample_size,
        min_muts=min_muts,
        max_muts=max_muts,
        rng=rng,
    )
    mu_list, _ = model.predict_many(samples, objectives)

    pareto_items: List[Tuple[str, np.ndarray]] = []
    for seq, mu in zip(samples, mu_list):
        vector = []
        for idx, objective in enumerate(objectives):
            value = float(mu[objective])
            if directions[idx].lower().startswith("min"):
                value = -value
            vector.append(value)
        pareto_items.append((seq, np.asarray(vector, dtype=float)))

    front = pareto_front(pareto_items)
    selected = select_diverse(front, k=batch_size, min_hamming=2)
    selected_sequences = [seq for seq, _ in selected]

    mu_selected, sigma_selected = model.predict_many(selected_sequences, objectives)
    rows: List[Dict[str, float | str]] = []
    for seq, mu, sigma in zip(selected_sequences, mu_selected, sigma_selected):
        row: Dict[str, float | str] = {"sequence": seq}
        for objective in objectives:
            row[f"{objective}_pred"] = float(mu[objective])
            row[f"{objective}_std"] = float(sigma[objective])
        rows.append(row)

    write_candidates(rows, out_csv)
    typer.secho(f"Wrote {len(rows)} candidates → {out_csv}", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover - Typer entry point
    app()
