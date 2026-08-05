import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon, ttest_rel

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def precision_at_k(recommended_items: list, relevant_items: set, k: int) -> float:
    """Computes Precision@K for a single user."""
    top_k = recommended_items[:k]
    if not top_k:
        return 0.0
    hits = len(set(top_k) & relevant_items)
    return hits / k


def ndcg_at_k(recommended_items: list, relevant_items: set, k: int) -> float:
    """Computes NDCG@K for a single user."""
    top_k = recommended_items[:k]
    if not top_k or not relevant_items:
        return 0.0
    
    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in relevant_items:
            dcg += 1.0 / np.log2(idx + 2)
            
    ideal_hits = min(k, len(relevant_items))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(ideal_hits))
    
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_top_n(model, train_df: pd.DataFrame, test_df: pd.DataFrame, all_item_ids: list, k_list=[5, 10], relevance_threshold=4.0):
    """
    Evaluates Precision@K and NDCG@K for a given model across test users.
    Samples 500 representative test users if N > 500 for instant evaluation.
    """
    test_rel = test_df[test_df['rating'] >= relevance_threshold]
    relevant_by_user = test_rel.groupby('user_id')['item_id'].apply(set).to_dict()
    train_items_by_user = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    precision_results = {k: [] for k in k_list}
    ndcg_results = {k: [] for k in k_list}
    
    all_set = set(all_item_ids)
    max_k = max(k_list)
    
    user_keys = list(relevant_by_user.keys())
    if len(user_keys) > 500:
        rng = np.random.RandomState(42)
        user_keys = list(rng.choice(user_keys, size=500, replace=False))
    
    for u in user_keys:
        rel_items = relevant_by_user[u]
        if not rel_items:
            continue
            
        seen_items = train_items_by_user.get(u, set())
        unseen_items = list(all_set - seen_items)
        if not unseen_items:
            continue
            
        candidate_df = pd.DataFrame({
            'user_id': np.full(len(unseen_items), u),
            'item_id': unseen_items
        })
        
        preds = model.predict_batch(candidate_df)
        
        # Fast top-K selection using NumPy argpartition
        if len(preds) > max_k:
            top_k_indices = np.argpartition(-preds, max_k)[:max_k]
            top_k_sorted = top_k_indices[np.argsort(-preds[top_k_indices])]
        else:
            top_k_sorted = np.argsort(-preds)
            
        recommended_items = [unseen_items[idx] for idx in top_k_sorted]
        
        for k in k_list:
            precision_results[k].append(precision_at_k(recommended_items, rel_items, k))
            ndcg_results[k].append(ndcg_at_k(recommended_items, rel_items, k))
            
    summary = {}
    for k in k_list:
        summary[f"Precision@{k}"] = float(np.mean(precision_results[k])) if precision_results[k] else 0.0
        summary[f"NDCG@{k}"] = float(np.mean(ndcg_results[k])) if ndcg_results[k] else 0.0
        
    return summary


def run_5fold_cross_validation(ratings_df: pd.DataFrame, model_factory_fn, random_state=42):
    """
    Runs 5-Fold Cross Validation on ratings_df using model_factory_fn(fold_idx).
    Returns list of dicts with fold metrics: rmse, mae.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
        train_df = ratings_df.iloc[train_idx].copy()
        test_df = ratings_df.iloc[test_idx].copy()
        
        model = model_factory_fn(fold_idx)
        model.fit(train_df)
        
        preds = model.predict_batch(test_df)
        y_true = test_df['rating'].values
        
        rmse = compute_rmse(y_true, preds)
        mae = compute_mae(y_true, preds)
        
        results.append({'fold': fold_idx, 'rmse': rmse, 'mae': mae})
        
    return results


def perform_statistical_test(model_a_rmses: list, model_b_rmses: list, model_a_name="Model A", model_b_name="Model B"):
    """
    Performs paired t-test and Wilcoxon signed-rank test on fold RMSE scores.
    """
    a = np.array(model_a_rmses)
    b = np.array(model_b_rmses)
    
    t_stat, t_p = ttest_rel(a, b)
    
    try:
        w_stat, w_p = wilcoxon(a, b)
    except Exception:
        w_stat, w_p = 0.0, 1.0
        
    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "t_statistic": float(t_stat),
        "t_p_value": float(t_p),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p_value": float(w_p),
        "statistically_significant_p05": bool(t_p < 0.05)
    }
