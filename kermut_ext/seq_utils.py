"""Sequence helpers for the Typer CLIs."""

from __future__ import annotations

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
    """Yield neighbouring sequences via single-position perturbations."""

    if min_muts > max_muts:
        raise ValueError("min_muts must be less than or equal to max_muts")

    length = len(current)
    sites = list(site_pool) if site_pool is not None else list(range(length))
    seen: set[str] = set()

    while len(seen) < max_candidates:
        idx = int(rng.choice(sites))
        mutated = list(current)
        aa = AA_ALPHABET[rng.integers(0, len(AA_ALPHABET))]
        if aa == mutated[idx]:
            continue
        mutated[idx] = aa
        proposal = "".join(mutated)
        n_mut = count_muts(current, proposal)
        if min_muts <= n_mut <= max_muts and proposal not in seen:
            seen.add(proposal)
            yield proposal


__all__ = ["AA_ALPHABET", "count_muts", "propose_neighbors"]
