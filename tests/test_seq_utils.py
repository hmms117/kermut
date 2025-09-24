import itertools

import pytest

np = pytest.importorskip("numpy")

from scripts.sequences import count_muts, propose_neighbors


def test_propose_neighbors_respects_bounds() -> None:
    rng = np.random.default_rng(123)
    current = "ACDEFG"
    neighbours = list(
        itertools.islice(
            propose_neighbors(
                current,
                min_muts=2,
                max_muts=4,
                rng=rng,
                max_candidates=25,
            ),
            25,
        )
    )
    assert neighbours, "Expected at least one neighbour"
    assert len(set(neighbours)) == len(neighbours)
    for neighbour in neighbours:
        distance = count_muts(current, neighbour)
        assert 2 <= distance <= 4


def test_propose_neighbors_validates_pool_size() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        list(
            propose_neighbors(
                "AAAA",
                min_muts=3,
                max_muts=5,
                site_pool=[0, 1],
                rng=rng,
                max_candidates=5,
            )
        )
