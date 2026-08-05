import pandas as pd
import numpy as np

class GlobalMeanBaseline:
    """
    Baseline model that always predicts the global mean of all ratings in the training set.
    """
    def __init__(self):
        self.global_mean = 3.0

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df['rating'].mean())
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        return self.global_mean

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(test_df), self.global_mean, dtype=float)


class UserMeanBaseline:
    """
    Baseline model that predicts the mean rating given by the user.
    Falls back to global mean if user was not seen in training.
    """
    def __init__(self):
        self.user_means = {}
        self.global_mean = 3.0

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df['rating'].mean())
        user_grouped = train_df.groupby('user_id')['rating'].mean()
        self.user_means = user_grouped.to_dict()
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        return self.user_means.get(user_id, self.global_mean)

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = test_df['user_id'].map(self.user_means).fillna(self.global_mean).values
        return preds.astype(float)


class ItemMeanBaseline:
    """
    Baseline model that predicts the mean rating received by the item.
    Falls back to global mean if item was not seen in training.
    """
    def __init__(self):
        self.item_means = {}
        self.global_mean = 3.0

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df['rating'].mean())
        item_grouped = train_df.groupby('item_id')['rating'].mean()
        self.item_means = item_grouped.to_dict()
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        return self.item_means.get(item_id, self.global_mean)

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = test_df['item_id'].map(self.item_means).fillna(self.global_mean).values
        return preds.astype(float)
