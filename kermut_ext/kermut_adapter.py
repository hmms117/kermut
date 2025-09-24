"""Adapter around the core Kermut model used by the Typer CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd


@dataclass
class Prediction:
    """Container holding per-objective posterior statistics."""

    mean: Dict[str, float]
    std: Dict[str, float]


class KermutAdapter:
    """Thin abstraction that exposes a uniform multi-objective API.

    The Typer CLIs added in this repository only rely on two methods:
    :meth:`fit`, which should retrain or refresh a Kermut model using all
    available assay data, and :meth:`predict_many`, which should return
    posterior means and uncertainties for multiple sequences at once.

    The default implementation here is intentionally lightweight.  It keeps
    method signatures stable and raises :class:`NotImplementedError` so that
    downstream users are forced to plug in the project-specific training and
    inference code.  This avoids accidentally running a partially configured
    workflow while still providing a clear surface for integration.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda") -> None:
        self.ckpt_path = ckpt_path
        self.device = device

    # ------------------------------------------------------------------
    # Public API expected by the Typer scripts
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "KermutAdapter":
        """Refresh the underlying Kermut model using the provided data.

        Parameters
        ----------
        df:
            Data frame containing at least a ``sequence`` column along with the
            objective columns passed via ``--objectives`` on the command line.
        """

        raise NotImplementedError(
            "Replace KermutAdapter.fit with your project-specific training logic."
        )

    def predict_many(
        self, sequences: Sequence[str], objectives: Sequence[str]
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        """Return posterior mean and standard deviation for each sequence.

        Returns two lists of dictionaries matching ``objectives``: the first
        contains the posterior means and the second the associated standard
        deviations.
        """

        raise NotImplementedError(
            "Replace KermutAdapter.predict_many with your inference implementation."
        )


__all__ = ["KermutAdapter", "Prediction"]
