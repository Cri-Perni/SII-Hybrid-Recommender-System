import pandas as pd
import numpy as np

class HybridRecommender:
    """
    Weighted Hybrid Recommender combining Collaborative Filtering (S_CF) and Content-Based (S_CB).
    S_Hybrid = alpha * S_CF + (1 - alpha) * S_CB
    R_hat = r_min + S_Hybrid * (r_max - r_min)
    """
    def __init__(self, cf_model, cb_model, alpha: float = 0.5, r_min: float = 1.0, r_max: float = 5.0):
        self.cf_model = cf_model
        self.cb_model = cb_model
        self.alpha = alpha
        self.r_min = r_min
        self.r_max = r_max

    def fit(self, train_df: pd.DataFrame):
        self.cf_model.fit(train_df)
        self.cb_model.fit(train_df)
        return self

    def predict_score_normalized(self, test_df: pd.DataFrame) -> np.ndarray:
        s_cf = self.cf_model.predict_score_normalized(test_df, r_min=self.r_min, r_max=self.r_max)
        s_cb = self.cb_model.predict_score_normalized(test_df)
        s_hybrid = self.alpha * s_cf + (1.0 - self.alpha) * s_cb
        return np.clip(s_hybrid, 0.0, 1.0)

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        s_hybrid = self.predict_score_normalized(test_df)
        r_hat = self.r_min + s_hybrid * (self.r_max - self.r_min)
        return np.clip(r_hat, self.r_min, self.r_max)
