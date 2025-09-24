"""Sequence helpers for the Typer CLIs."""

from __future__ import annotations
"""Sequence utilities shared by the planning scripts."""

from typing import Iterable, Iterator, Optional

import numpy as np

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def count_muts(reference: str, sequence: str) -> int:
    """Return the number of positions that differ from the reference sequence."""

    if len(reference) != len(sequence):
        raise ValueError("Sequences must have identical length when counting mutations.")
    return sum(ref != seq for ref, seq in zip(reference, sequence))


def propose_neighbors(
    current: str,
    *,
    min_muts: int,
    max_muts: int,
    site_pool: Optional[Iterable[int]] = None,
    rng: np.random.Generator,
    max_candidates: int = 200,
) -> Iterator[str]:
    """Yield neighbouring sequences with bounded Hamming distance."""

    if min_muts > max_muts:
        raise ValueError("min_muts must be less than or equal to max_muts")
    if min_muts < 0:
        raise ValueError("min_muts must be non-negative")

    length = len(current)
    sites = list(site_pool) if site_pool is not None else list(range(length))
    if not sites:
        raise ValueError("site_pool must contain at least one position")

    max_mutations = min(max_muts, len(sites))
    if min_muts > max_mutations:
        raise ValueError("Not enough mutable sites for the requested min_muts")

    seen: set[str] = set()
    attempts = 0
    max_attempts = max(1, max_candidates) * 20

    while len(seen) < max_candidates and attempts < max_attempts:
        attempts += 1
        n_mut = int(rng.integers(min_muts, max_mutations + 1))

        if n_mut == 0:
            continue

        positions = rng.choice(sites, size=n_mut, replace=False)
        mutated = list(current)

        for raw_idx in np.atleast_1d(positions):
            idx = int(raw_idx)
            original = mutated[idx]
            choices = [aa for aa in AA_ALPHABET if aa != original]
            mutated[idx] = str(rng.choice(choices))

        proposal = "".join(mutated)
        if proposal == current or proposal in seen:
            continue

        distance = count_muts(current, proposal)
        if min_muts <= distance <= max_muts:
            seen.add(proposal)
            yield proposal


__all__ = ["AA_ALPHABET", "count_muts", "propose_neighbors"]
