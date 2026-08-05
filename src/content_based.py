import pandas as pd
import numpy as np

class ContentBasedRecommender:
    """
    Content-Based Recommender using cosine similarity between user profiles and item feature vectors.
    Calibrates predictions around user mean rating for improved rating estimation.
    """
    def __init__(self, item_feature_matrix: np.ndarray, item_id_to_idx: dict, rating_threshold: float = 3.0):
        self.item_feature_matrix = item_feature_matrix
        self.item_id_to_idx = item_id_to_idx
        self.rating_threshold = rating_threshold
        self.feature_dim = item_feature_matrix.shape[1]
        
        self.user_profiles = {}          # user_id -> vector (D,)
        self.user_means = {}             # user_id -> mean rating
        self.user_sim_means = {}         # user_id -> mean similarity
        self.user_sim_stds = {}          # user_id -> std similarity
        self.global_mean_rating = 3.523
        self.global_user_profile = np.mean(item_feature_matrix, axis=0)
        
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def fit(self, train_df: pd.DataFrame):
        self.user_profiles = {}
        self.user_means = {}
        self.user_sim_means = {}
        self.user_sim_stds = {}
        self.global_mean_rating = float(train_df['rating'].mean())
        
        grouped = train_df.groupby('user_id')
        
        for user_id, group in grouped:
            self.user_means[user_id] = float(group['rating'].mean())
            
            # Filter positive ratings
            pos_ratings = group[group['rating'] >= self.rating_threshold]
            if pos_ratings.empty:
                pos_ratings = group
                
            weighted_vectors = []
            weights = []
            
            for _, row in pos_ratings.iterrows():
                item_id = int(row['item_id'])
                r = float(row['rating'])
                if item_id in self.item_id_to_idx:
                    idx = self.item_id_to_idx[item_id]
                    v_i = self.item_feature_matrix[idx]
                    weighted_vectors.append(v_i * r)
                    weights.append(r)
                    
            if weighted_vectors and sum(weights) > 0:
                p_u = np.sum(weighted_vectors, axis=0) / sum(weights)
            else:
                p_u = self.global_user_profile.copy()
                
            self.user_profiles[user_id] = p_u

            # Compute mean and std of similarities for this user across rated items
            sims = []
            for _, row in group.iterrows():
                item_id = int(row['item_id'])
                if item_id in self.item_id_to_idx:
                    idx = self.item_id_to_idx[item_id]
                    v_i = self.item_feature_matrix[idx]
                    sims.append(self._cosine_similarity(p_u, v_i))
            
            if sims:
                self.user_sim_means[user_id] = float(np.mean(sims))
                self.user_sim_stds[user_id] = float(np.std(sims)) if np.std(sims) > 1e-5 else 1.0
            else:
                self.user_sim_means[user_id] = 0.5
                self.user_sim_stds[user_id] = 1.0

        return self

    def predict_score_normalized(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Calculates S_CB(u, i) in range [0, 1] for each pair in test_df.
        """
        scores = []
        for u, i in zip(test_df['user_id'], test_df['item_id']):
            p_u = self.user_profiles.get(u, self.global_user_profile)
            if i in self.item_id_to_idx:
                idx = self.item_id_to_idx[i]
                v_i = self.item_feature_matrix[idx]
                sim = self._cosine_similarity(p_u, v_i)
            else:
                sim = 0.0
            scores.append(sim)
            
        raw_scores = np.array(scores, dtype=float)
        # Normalize scores to [0, 1]
        s_min, s_max = raw_scores.min(), raw_scores.max()
        if s_max > s_min:
            norm_scores = (raw_scores - s_min) / (s_max - s_min)
        else:
            norm_scores = raw_scores
        return np.clip(norm_scores, 0.0, 1.0)

    def predict_batch(self, test_df: pd.DataFrame, r_min: float = 1.0, r_max: float = 5.0) -> np.ndarray:
        """
        Calculates predicted ratings r_hat = user_mean + (sim - user_sim_mean) * scale.
        """
        preds = []
        for u, i in zip(test_df['user_id'], test_df['item_id']):
            u_mean = self.user_means.get(u, self.global_mean_rating)
            p_u = self.user_profiles.get(u, self.global_user_profile)
            sim_mean = self.user_sim_means.get(u, 0.5)
            sim_std = self.user_sim_stds.get(u, 1.0)
            
            if i in self.item_id_to_idx:
                idx = self.item_id_to_idx[i]
                v_i = self.item_feature_matrix[idx]
                sim = self._cosine_similarity(p_u, v_i)
            else:
                sim = sim_mean
                
            # Calibrated prediction around user mean
            delta = (sim - sim_mean) / (sim_std + 1e-5)
            pred = u_mean + delta * 0.75
            preds.append(pred)
            
        return np.clip(np.array(preds, dtype=float), r_min, r_max)
