"""Modello ibrido con una sola semantica di fusione, stabile rispetto al batch."""

from __future__ import annotations

import numpy as np
import pandas as pd


class HybridRecommender:
    """Fonde direttamente predizioni rating CF e CB sulla scala comune [1, 5].

    La stessa predizione è usata per RMSE/MAE e per ordinare gli item Top-N.
    Nessuna normalizzazione min-max dipendente dal batch di test è applicata.
    """

    def __init__(self, cf_model, cb_model, alpha: float = 0.5, r_min: float = 1.0, r_max: float = 5.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha deve appartenere all'intervallo [0, 1].")
        if r_min >= r_max:
            raise ValueError("r_min deve essere minore di r_max.")
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = float(alpha)
        self.r_min = float(r_min)
        self.r_max = float(r_max)

    def fit(self, train_df: pd.DataFrame):
        """Convenienza per uso standalone; non chiamare dopo fit dei componenti."""
        self.cf_model.fit(train_df)
        self.cb_model.fit(train_df)
        return self

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        cf_ratings = self.cf_model.predict_batch(test_df)
        cb_ratings = self.cb_model.predict_batch(test_df)
        predictions = self.alpha * cf_ratings + (1.0 - self.alpha) * cb_ratings
        return np.clip(predictions, self.r_min, self.r_max)

    def predict_score_normalized(self, test_df: pd.DataFrame) -> np.ndarray:
        """Trasformazione fissa delle stesse predizioni rating in [0, 1]."""
        ratings = self.predict_batch(test_df)
        return np.clip((ratings - self.r_min) / (self.r_max - self.r_min), 0.0, 1.0)
