import pandas as pd
import numpy as np

class BiasedMatrixFactorization:
    """
    Fallback NumPy implementation of Biased Matrix Factorization (FunkSVD with biases).
    R_ui ≈ μ + b_u + b_i + P_u · Q_i
    """
    def __init__(self, n_factors=50, lr=0.005, reg=0.02, n_epochs=20, random_state=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        
        self.global_mean = 3.0
        self.user_biases = {}
        self.item_biases = {}
        self.user_factors = {}
        self.item_factors = {}
        
    def fit(self, train_df: pd.DataFrame):
        np.random.seed(self.random_state)
        self.global_mean = float(train_df['rating'].mean())
        
        users = train_df['user_id'].unique()
        items = train_df['item_id'].unique()
        
        # Initialize biases and factors
        self.user_biases = {u: 0.0 for u in users}
        self.item_biases = {i: 0.0 for i in items}
        self.user_factors = {u: np.random.normal(0, 0.1, self.n_factors) for u in users}
        self.item_factors = {i: np.random.normal(0, 0.1, self.n_factors) for i in items}
        
        records = train_df[['user_id', 'item_id', 'rating']].to_dict('records')
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(records)
            for rec in records:
                u = rec['user_id']
                i = rec['item_id']
                r = rec['rating']
                
                bu = self.user_biases[u]
                bi = self.item_biases[i]
                pu = self.user_factors[u]
                qi = self.item_factors[i]
                
                pred = self.global_mean + bu + bi + np.dot(pu, qi)
                err = r - pred
                
                # Update biases
                self.user_biases[u] += self.lr * (err - self.reg * bu)
                self.item_biases[i] += self.lr * (err - self.reg * bi)
                
                # Update latent factors
                self.user_factors[u] += self.lr * (err * qi - self.reg * pu)
                self.item_factors[i] += self.lr * (err * pu - self.reg * qi)
                
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        bu = self.user_biases.get(user_id, 0.0)
        bi = self.item_biases.get(item_id, 0.0)
        pu = self.user_factors.get(user_id, None)
        qi = self.item_factors.get(item_id, None)
        
        if pu is None or qi is None:
            pred = self.global_mean + bu + bi
        else:
            pred = self.global_mean + bu + bi + np.dot(pu, qi)
            
        return float(np.clip(pred, 1.0, 5.0))


class CollaborativeFilteringModel:
    """
    Collaborative Filtering wrapper using scikit-surprise SVD if available,
    falling back to NumPy Biased Matrix Factorization.
    """
    def __init__(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.random_state = random_state
        self.surprise_svd = None
        self.fallback_svd = None
        self.use_surprise = False
        
        try:
            from surprise import SVD
            self.surprise_svd = SVD(
                n_factors=self.n_factors,
                n_epochs=self.n_epochs,
                lr_all=self.lr_all,
                reg_all=self.reg_all,
                random_state=self.random_state
            )
            self.use_surprise = True
        except ImportError:
            self.use_surprise = False

    def fit(self, train_df: pd.DataFrame):
        if self.use_surprise:
            from surprise import Dataset, Reader
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'rating']], reader)
            trainset = data.build_full_trainset()
            self.surprise_svd.fit(trainset)
        else:
            self.fallback_svd = BiasedMatrixFactorization(
                n_factors=self.n_factors,
                lr=self.lr_all,
                reg=self.reg_all,
                n_epochs=self.n_epochs,
                random_state=self.random_state
            )
            self.fallback_svd.fit(train_df)
        return self

    def predict(self, user_id: int, item_id: int) -> float:
        if self.use_surprise:
            pred = self.surprise_svd.predict(user_id, item_id).est
        else:
            pred = self.fallback_svd.predict(user_id, item_id)
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.use_surprise:
            preds = [self.surprise_svd.predict(u, i).est for u, i in zip(test_df['user_id'], test_df['item_id'])]
        else:
            preds = [self.fallback_svd.predict(u, i) for u, i in zip(test_df['user_id'], test_df['item_id'])]
        return np.clip(np.array(preds, dtype=float), 1.0, 5.0)

    def predict_score_normalized(self, test_df: pd.DataFrame, r_min=1.0, r_max=5.0) -> np.ndarray:
        """
        Returns prediction normalized in [0, 1].
        S_CF(u, i) = (r_hat - r_min) / (r_max - r_min)
        """
        raw_preds = self.predict_batch(test_df)
        scores = (raw_preds - r_min) / (r_max - r_min)
        return np.clip(scores, 0.0, 1.0)
