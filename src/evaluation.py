"""Metriche, ranking e utilità statistiche per esperimenti riproducibili."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import KFold


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def precision_at_k(recommended_items: Sequence[int], relevant_items: set[int], k: int) -> float:
    """Computes Precision@K for one user."""
    top_k = recommended_items[:k]
    return len(set(top_k).intersection(relevant_items)) / k if top_k else 0.0


def ndcg_at_k(recommended_items: Sequence[int], relevant_items: set[int], k: int) -> float:
    """Computes binary-relevance NDCG@K for one user."""
    top_k = recommended_items[:k]
    if not top_k or not relevant_items:
        return 0.0
    dcg = sum(1.0 / np.log2(position + 2) for position, item in enumerate(top_k) if item in relevant_items)
    ideal_hits = min(k, len(relevant_items))
    idcg = sum(1.0 / np.log2(position + 2) for position in range(ideal_hits))
    return float(dcg / idcg) if idcg else 0.0


def evaluate_top_n_detailed(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_item_ids: Iterable[int],
    k_list: Sequence[int] = (5, 10),
    relevance_threshold: float = 4.0,
    max_users: int = 500,
    sample_seed: int = 42,
) -> dict:
    """Valuta ranking esclusivamente su utenti/item del test fold.

    Per ogni utente eleggibile, i candidati sono tutti gli item del catalogo
    non osservati nel training fold. Il campionamento di utenti, se necessario,
    è deterministico e viene restituito assieme alle metriche.
    """
    if not k_list or any(k <= 0 for k in k_list):
        raise ValueError("k_list deve contenere soli valori positivi.")
    if max_users <= 0:
        raise ValueError("max_users deve essere positivo.")

    relevant_by_user = (
        test_df.loc[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )
    train_items_by_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    user_keys = sorted(relevant_by_user)
    eligible_users = len(user_keys)
    if eligible_users > max_users:
        rng = np.random.RandomState(sample_seed)
        user_keys = sorted(rng.choice(user_keys, size=max_users, replace=False).tolist())

    catalogue = sorted(set(all_item_ids))
    max_k = max(k_list)
    precision_values = {k: [] for k in k_list}
    ndcg_values = {k: [] for k in k_list}
    candidate_counts = []

    for user_id in user_keys:
        seen_items = train_items_by_user.get(user_id, set())
        candidates = [item_id for item_id in catalogue if item_id not in seen_items]
        if not candidates:
            continue
        candidate_df = pd.DataFrame({
            "user_id": np.full(len(candidates), user_id),
            "item_id": candidates,
        })
        predictions = model.predict_batch(candidate_df)
        # Ordine secondario sul item_id: ranking riproducibile anche a parità di score.
        ranked_indices = np.lexsort((np.asarray(candidates), -np.asarray(predictions)))[:max_k]
        recommended_items = [candidates[index] for index in ranked_indices]
        relevant_items = relevant_by_user[user_id]
        candidate_counts.append(len(candidates))
        for k in k_list:
            precision_values[k].append(precision_at_k(recommended_items, relevant_items, k))
            ndcg_values[k].append(ndcg_at_k(recommended_items, relevant_items, k))

    metrics = {}
    for k in k_list:
        metrics[f"Precision@{k}"] = float(np.mean(precision_values[k])) if precision_values[k] else 0.0
        metrics[f"NDCG@{k}"] = float(np.mean(ndcg_values[k])) if ndcg_values[k] else 0.0
    return {
        "metrics": metrics,
        "support": {
            "eligible_users": eligible_users,
            "evaluated_users": len(candidate_counts),
            "sampled_users": len(user_keys),
            "max_users": max_users,
            "sample_seed": sample_seed,
            "mean_candidate_items": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        },
    }


def evaluate_top_n(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_item_ids: Iterable[int],
    k_list: Sequence[int] = (5, 10),
    relevance_threshold: float = 4.0,
) -> dict:
    """Compatibilità con la precedente API: restituisce solo le metriche."""
    return evaluate_top_n_detailed(
        model, train_df, test_df, all_item_ids, k_list, relevance_threshold
    )["metrics"]


def run_5fold_cross_validation(ratings_df: pd.DataFrame, model_factory_fn, random_state: int = 42) -> list[dict]:
    """Compatibilità per notebook: CV semplice, non adatta al tuning finale."""
    splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(ratings_df)):
        train_df = ratings_df.iloc[train_idx]
        test_df = ratings_df.iloc[test_idx]
        model = model_factory_fn(fold_idx).fit(train_df)
        predictions = model.predict_batch(test_df)
        results.append({
            "fold": fold_idx,
            "rmse": compute_rmse(test_df["rating"].to_numpy(), predictions),
            "mae": compute_mae(test_df["rating"].to_numpy(), predictions),
        })
    return results


def summarize_metric_values(values: Iterable[float]) -> dict:
    """Media, deviazione campionaria e IC t al 95% per valori per-fold."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {"mean": None, "std": None, "n": 0, "ci95": [None, None]}
    mean = float(np.mean(array))
    if array.size == 1:
        return {"mean": mean, "std": 0.0, "n": 1, "ci95": [mean, mean]}
    std = float(np.std(array, ddof=1))
    half_width = float(student_t.ppf(0.975, df=array.size - 1) * std / np.sqrt(array.size))
    return {"mean": mean, "std": std, "n": int(array.size), "ci95": [mean - half_width, mean + half_width]}


def perform_statistical_test(
    model_a_rmses: Sequence[float],
    model_b_rmses: Sequence[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> dict:
    """Test paired su outer-fold allineati, con effetto e IC della differenza."""
    a = np.asarray(model_a_rmses, dtype=float)
    b = np.asarray(model_b_rmses, dtype=float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("I due vettori paired devono avere stessa cardinalità e almeno due valori.")
    differences = a - b
    t_statistic, t_p_value = ttest_rel(a, b)
    try:
        wilcoxon_statistic, wilcoxon_p_value = wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        wilcoxon_statistic, wilcoxon_p_value = 0.0, 1.0
    difference_summary = summarize_metric_values(differences)
    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "paired_folds": int(a.size),
        "mean_rmse_difference_a_minus_b": difference_summary["mean"],
        "difference_ci95": difference_summary["ci95"],
        "t_statistic": float(t_statistic),
        "t_p_value": float(t_p_value),
        "t_test_significant_p05": bool(t_p_value < 0.05),
        "wilcoxon_statistic": float(wilcoxon_statistic),
        "wilcoxon_p_value": float(wilcoxon_p_value),
        "wilcoxon_significant_p05": bool(wilcoxon_p_value < 0.05),
        "interpretation": "Negative differences favor model_a because lower RMSE is better.",
    }
