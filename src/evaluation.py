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
    Evaluates Precision@K and NDCG@K for a given model across users in test_df.
    Only considers users with at least 1 relevant item in test_df (rating >= relevance_threshold).
    """
    test_rel = test_df[test_df['rating'] >= relevance_threshold]
    relevant_by_user = test_rel.groupby('user_id')['item_id'].apply(set).to_dict()
    
    train_items_by_user = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    precision_results = {k: [] for k in k_list}
    ndcg_results = {k: [] for k in k_list}
    
    for u, rel_items in relevant_by_user.items():
        if not rel_items:
            continue
            
        seen_items = train_items_by_user.get(u, set())
        unseen_items = [item_id for item_id in all_item_ids if item_id not in seen_items]
        
        if not unseen_items:
            continue
            
        candidate_df = pd.DataFrame({
            'user_id': [u] * len(unseen_items),
            'item_id': unseen_items
        })
        
        preds = model.predict_batch(candidate_df)
        candidate_df['pred_rating'] = preds
        
        # Sort top K
        sorted_candidates = candidate_df.sort_values('pred_rating', ascending=False)['item_id'].tolist()
        
        for k in k_list:
            precision_results[k].append(precision_at_k(sorted_candidates, rel_items, k))
            ndcg_results[k].append(ndcg_at_k(sorted_candidates, rel_items, k))
            
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
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
        train_df = ratings_df.iloc[train_idx].copy()
        test_df = ratings_df.iloc[test_idx].copy()
        
        model = model_factory_fn(fold_idx)
        model.fit(train_df)
        
        preds = model.predict_batch(test_df)
        y_true = test_df['rating'].values
        
        rmse = compute_rmse(y_true, preds)
        mae = compute_mae(y_true, preds)
        
        fold_metrics.append({
            'fold': fold_idx,
            'rmse': rmse,
            'mae': mae
        })
        
    return fold_metrics


def perform_statistical_test(scores_a: list, scores_b: list, name_a="Model A", name_b="Model B"):
    """
    Performs Wilcoxon signed-rank test and paired t-test between scores_a and scores_b.
    Returns test statistics and p-values.
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    # Paired t-test
    t_stat, p_t = ttest_rel(scores_a, scores_b)
    
    # Wilcoxon signed-rank test
    try:
        w_stat, p_w = wilcoxon(scores_a, scores_b)
    except Exception:
        w_stat, p_w = np.nan, np.nan
        
    return {
        "comparison": f"{name_a} vs {name_b}",
        "mean_a": float(np.mean(scores_a)),
        "std_a": float(np.std(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "std_b": float(np.std(scores_b)),
        "t_statistic": float(t_stat),
        "t_p_value": float(p_t),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_p_value": float(p_w),
        "statistically_significant_p05": bool(p_t < 0.05)
    }
