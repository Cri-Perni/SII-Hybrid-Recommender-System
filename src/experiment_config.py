"""Configurazione riproducibile per gli esperimenti nested del recommender."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    """Parametri immutabili condivisi da tuning, valutazione e report."""

    outer_splits: int = 5
    inner_splits: int = 3
    random_state: int = 42
    alpha_grid: Tuple[float, ...] = tuple(round(step / 10, 1) for step in range(11))
    theta_grid: Tuple[float, ...] = (3.0, 4.0)
    ranking_k: Tuple[int, ...] = (5, 10)
    ranking_relevance_threshold: float = 4.0
    ranking_max_users: int = 500
    cold_item_max_train_ratings: int = 2
    n_factors: int = 50

    def __post_init__(self) -> None:
        if self.outer_splits < 2 or self.inner_splits < 2:
            raise ValueError("outer_splits e inner_splits devono essere almeno 2.")
        if not self.alpha_grid or any(alpha < 0.0 or alpha > 1.0 for alpha in self.alpha_grid):
            raise ValueError("alpha_grid deve contenere valori nell'intervallo [0, 1].")
        if not self.theta_grid or any(theta < 1.0 or theta > 5.0 for theta in self.theta_grid):
            raise ValueError("theta_grid deve contenere soglie di rating valide.")
        if not self.ranking_k or any(k <= 0 for k in self.ranking_k):
            raise ValueError("ranking_k deve contenere valori positivi.")
        if self.ranking_max_users <= 0:
            raise ValueError("ranking_max_users deve essere positivo.")
        if self.cold_item_max_train_ratings < 0:
            raise ValueError("cold_item_max_train_ratings non può essere negativo.")

    def to_dict(self) -> dict:
        """Restituisce una rappresentazione JSON-serializzabile e versionabile."""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, tuple):
                data[key] = list(value)
        return data
