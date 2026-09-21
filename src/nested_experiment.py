"""Nested cross-validation leakage-free per il recommender ibrido.

Il modulo usa esclusivamente i training fold interni per selezionare alpha e
la soglia CB. Ogni outer test fold viene consultato una sola volta, dopo il
refit finale sul corrispondente outer training fold.
"""

from __future__ import annotations

import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .baselines import GlobalMeanBaseline, ItemMeanBaseline, UserMeanBaseline
from .collaborative import CollaborativeFilteringModel
from .content_based import ContentBasedRecommender
from .evaluation import (
    compute_mae,
    compute_rmse,
    evaluate_top_n_detailed,
    perform_statistical_test,
    summarize_metric_values,
)
from .experiment_config import ExperimentConfig
from .hybrid import HybridRecommender


MODEL_NAMES = (
    "global_mean",
    "user_mean",
    "item_mean",
    "cf_only",
    "cb_only",
    "hybrid_selected",
)


def _model_seed(config: ExperimentConfig, outer_fold: int, inner_fold: int = 0) -> int:
    return config.random_state + 10_000 * (outer_fold + 1) + inner_fold


def _make_cf(config: ExperimentConfig, seed: int) -> CollaborativeFilteringModel:
    return CollaborativeFilteringModel(n_factors=config.n_factors, random_state=seed)


def _make_cb(feature_matrix: np.ndarray, item_id_to_idx: Mapping[int, int], theta: float) -> ContentBasedRecommender:
    return ContentBasedRecommender(feature_matrix, item_id_to_idx, rating_threshold=theta)


def _blend_ratings(cf_predictions: np.ndarray, cb_predictions: np.ndarray, alpha: float) -> np.ndarray:
    """La sola definizione dell'ibrido: media pesata di rating in [1, 5]."""
    return np.clip(alpha * cf_predictions + (1.0 - alpha) * cb_predictions, 1.0, 5.0)


def _rank_configs(summary: Mapping[Tuple[float, float], Mapping[str, object]]) -> Tuple[float, float]:
    """Minimizza RMSE; a parità usa alpha e theta più piccoli in modo deterministico."""
    return min(
        summary,
        key=lambda params: (
            float(summary[params]["mean_rmse"]),
            float(params[0]),
            float(params[1]),
        ),
    )


def _inner_search(
    outer_train: pd.DataFrame,
    feature_matrix: np.ndarray,
    item_id_to_idx: Mapping[int, int],
    config: ExperimentConfig,
    outer_fold: int,
) -> dict:
    """Seleziona alpha e theta solo tramite validation fold interni."""
    splitter = KFold(
        n_splits=config.inner_splits,
        shuffle=True,
        random_state=_model_seed(config, outer_fold, 1),
    )
    scores: Dict[Tuple[float, float], list[float]] = {
        (alpha, theta): [] for theta in config.theta_grid for alpha in config.alpha_grid
    }

    for inner_fold, (train_idx, validation_idx) in enumerate(splitter.split(outer_train)):
        inner_train = outer_train.iloc[train_idx]
        inner_validation = outer_train.iloc[validation_idx]
        y_validation = inner_validation["rating"].to_numpy(dtype=float)

        cf = _make_cf(config, _model_seed(config, outer_fold, inner_fold + 10)).fit(inner_train)
        cf_predictions = cf.predict_batch(inner_validation)

        for theta in config.theta_grid:
            cb = _make_cb(feature_matrix, item_id_to_idx, theta).fit(inner_train)
            cb_predictions = cb.predict_batch(inner_validation)
            for alpha in config.alpha_grid:
                scores[(alpha, theta)].append(
                    compute_rmse(y_validation, _blend_ratings(cf_predictions, cb_predictions, alpha))
                )

    summary = {
        params: {
            "alpha": float(params[0]),
            "theta": float(params[1]),
            "fold_rmse": [float(value) for value in values],
            "mean_rmse": float(np.mean(values)),
            "std_rmse": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
        for params, values in scores.items()
    }
    selected_alpha, selected_theta = _rank_configs(summary)

    # La baseline CB e il controllo alpha=0.5 sono scelti anch'essi solo internamente.
    cb_candidates = {(0.0, theta): summary[(0.0, theta)] for theta in config.theta_grid}
    _, selected_cb_theta = _rank_configs(cb_candidates)
    alpha_half = min(config.alpha_grid, key=lambda alpha: abs(alpha - 0.5))
    half_candidates = {(alpha_half, theta): summary[(alpha_half, theta)] for theta in config.theta_grid}
    _, selected_half_theta = _rank_configs(half_candidates)

    return {
        "seed": _model_seed(config, outer_fold, 1),
        "configurations": [summary[(alpha, theta)] for theta in config.theta_grid for alpha in config.alpha_grid],
        "selected": {"alpha": float(selected_alpha), "theta": float(selected_theta)},
        "selected_cb_theta": float(selected_cb_theta),
        "selected_alpha_half_theta": float(selected_half_theta),
    }


def _pointwise_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict:
    return {"rmse": compute_rmse(y_true, predictions), "mae": compute_mae(y_true, predictions)}


def _aggregate_pointwise(outer_folds: Iterable[dict]) -> dict:
    result = {}
    for model_name in MODEL_NAMES:
        result[model_name] = {
            "rmse": summarize_metric_values([fold["pointwise"][model_name]["rmse"] for fold in outer_folds]),
            "mae": summarize_metric_values([fold["pointwise"][model_name]["mae"] for fold in outer_folds]),
        }
    return result


def _aggregate_top_n(outer_folds: Iterable[dict], config: ExperimentConfig) -> dict:
    result = {}
    models = ("cf_only", "cb_only", "hybrid_selected")
    for model_name in models:
        model_folds = [fold["top_n"][model_name] for fold in outer_folds]
        metrics = {}
        for k in config.ranking_k:
            for metric_name in (f"Precision@{k}", f"NDCG@{k}"):
                metrics[metric_name] = summarize_metric_values(
                    [fold_result["metrics"][metric_name] for fold_result in model_folds]
                )
        result[model_name] = {
            "metrics": metrics,
            "support": {
                "eligible_users_total": int(sum(item["support"]["eligible_users"] for item in model_folds)),
                "evaluated_users_total": int(sum(item["support"]["evaluated_users"] for item in model_folds)),
                "mean_candidate_items": summarize_metric_values(
                    [item["support"]["mean_candidate_items"] for item in model_folds]
                ),
            },
        }
    return result


def _aggregate_cold_start(outer_folds: Iterable[dict]) -> dict:
    model_names = ("cf_only", "cb_only", "user_mean", "hybrid_selected", "hybrid_alpha_0_5")
    result = {}
    for model_name in model_names:
        valid = [fold["cold_start"]["models"][model_name] for fold in outer_folds if fold["cold_start"]["support"]["test_ratings"]]
        result[model_name] = {
            "rmse": summarize_metric_values([item["rmse"] for item in valid]),
            "mae": summarize_metric_values([item["mae"] for item in valid]),
        }
    result["support"] = {
        "test_ratings_total": int(sum(fold["cold_start"]["support"]["test_ratings"] for fold in outer_folds)),
        "cold_items_total": int(sum(fold["cold_start"]["support"]["cold_items"] for fold in outer_folds)),
        "zero_train_test_items_total": int(sum(fold["cold_start"]["support"]["zero_train_test_items"] for fold in outer_folds)),
        "valid_outer_folds": int(sum(bool(fold["cold_start"]["support"]["test_ratings"]) for fold in outer_folds)),
    }
    return result


def execution_environment() -> dict:
    packages = ("numpy", "pandas", "scipy", "scikit-learn")
    package_versions = {}
    for package in packages:
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions,
    }


def run_nested_experiment(
    ratings_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    item_id_to_idx: Mapping[int, int],
    all_item_ids: Iterable[int],
    config: ExperimentConfig,
) -> dict:
    """Esegue nested 5×3 CV con test esterni isolati.

    La funzione non legge né scrive file: il chiamante è responsabile della
    persistenza degli artefatti, rendendo il calcolo testabile su fixture.
    """
    required_columns = {"user_id", "item_id", "rating"}
    missing = required_columns.difference(ratings_df.columns)
    if missing:
        raise ValueError(f"ratings_df non contiene le colonne richieste: {sorted(missing)}")
    if len(ratings_df) < config.outer_splits:
        raise ValueError("Il dataset non ha abbastanza righe per gli outer fold richiesti.")

    started_at = time.perf_counter()
    outer_splitter = KFold(n_splits=config.outer_splits, shuffle=True, random_state=config.random_state)
    outer_folds = []
    all_item_ids = sorted(set(all_item_ids))

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splitter.split(ratings_df)):
        outer_train = ratings_df.iloc[train_idx].copy()
        outer_test = ratings_df.iloc[test_idx].copy()
        tuning = _inner_search(outer_train, feature_matrix, item_id_to_idx, config, outer_fold)
        selected_alpha = tuning["selected"]["alpha"]
        selected_theta = tuning["selected"]["theta"]

        seed = _model_seed(config, outer_fold)
        cf_model = _make_cf(config, seed).fit(outer_train)
        cb_selected = _make_cb(feature_matrix, item_id_to_idx, selected_theta).fit(outer_train)
        cb_only = _make_cb(feature_matrix, item_id_to_idx, tuning["selected_cb_theta"]).fit(outer_train)
        cb_half = _make_cb(feature_matrix, item_id_to_idx, tuning["selected_alpha_half_theta"]).fit(outer_train)
        hybrid_selected = HybridRecommender(cf_model, cb_selected, alpha=selected_alpha)
        hybrid_alpha_half = HybridRecommender(cf_model, cb_half, alpha=0.5)

        y_test = outer_test["rating"].to_numpy(dtype=float)
        models = {
            "global_mean": GlobalMeanBaseline().fit(outer_train),
            "user_mean": UserMeanBaseline().fit(outer_train),
            "item_mean": ItemMeanBaseline().fit(outer_train),
            "cf_only": cf_model,
            "cb_only": cb_only,
            "hybrid_selected": hybrid_selected,
        }
        pointwise = {name: _pointwise_metrics(y_test, model.predict_batch(outer_test)) for name, model in models.items()}

        ranking_seed = config.random_state + 50_000 + outer_fold
        top_n = {
            "cf_only": evaluate_top_n_detailed(
                cf_model, outer_train, outer_test, all_item_ids, config.ranking_k,
                config.ranking_relevance_threshold, config.ranking_max_users, ranking_seed,
            ),
            "cb_only": evaluate_top_n_detailed(
                cb_only, outer_train, outer_test, all_item_ids, config.ranking_k,
                config.ranking_relevance_threshold, config.ranking_max_users, ranking_seed,
            ),
            "hybrid_selected": evaluate_top_n_detailed(
                hybrid_selected, outer_train, outer_test, all_item_ids, config.ranking_k,
                config.ranking_relevance_threshold, config.ranking_max_users, ranking_seed,
            ),
        }

        item_counts = outer_train["item_id"].value_counts()
        # Include anche gli item con zero interazioni nel training outer: sono cold per definizione.
        cold_items = {
            item_id for item_id in outer_test["item_id"].unique()
            if item_counts.get(item_id, 0) <= config.cold_item_max_train_ratings
        }
        cold_test = outer_test[outer_test["item_id"].isin(cold_items)]
        cold_support = {
            "test_ratings": int(len(cold_test)),
            "cold_items": int(len(cold_items)),
            "zero_train_test_items": int(sum(item_counts.get(item_id, 0) == 0 for item_id in cold_items)),
        }
        if len(cold_test):
            y_cold = cold_test["rating"].to_numpy(dtype=float)
            cold_models = {
                "cf_only": cf_model,
                "cb_only": cb_only,
                "user_mean": models["user_mean"],
                "hybrid_selected": hybrid_selected,
                "hybrid_alpha_0_5": hybrid_alpha_half,
            }
            cold_metrics = {name: _pointwise_metrics(y_cold, model.predict_batch(cold_test)) for name, model in cold_models.items()}
        else:
            cold_metrics = {name: {"rmse": None, "mae": None} for name in ("cf_only", "cb_only", "user_mean", "hybrid_selected", "hybrid_alpha_0_5")}

        outer_folds.append({
            "outer_fold": outer_fold,
            "outer_seed": seed,
            "ranking_seed": ranking_seed,
            "train_rows": int(len(outer_train)),
            "test_rows": int(len(outer_test)),
            "tuning": tuning,
            "pointwise": pointwise,
            "top_n": top_n,
            "cold_start": {"support": cold_support, "models": cold_metrics},
        })

    pointwise = _aggregate_pointwise(outer_folds)
    hybrid_rmses = [fold["pointwise"]["hybrid_selected"]["rmse"] for fold in outer_folds]
    cf_rmses = [fold["pointwise"]["cf_only"]["rmse"] for fold in outer_folds]
    cb_rmses = [fold["pointwise"]["cb_only"]["rmse"] for fold in outer_folds]
    selected_parameters = [fold["tuning"]["selected"] for fold in outer_folds]

    return {
        "protocol": {
            "name": "nested_kfold_rating_cv",
            "version": "2.0",
            "outer_test_policy": "Each outer test fold is used only after inner tuning and outer-train refit.",
            "hybrid_prediction": "clip(alpha * CF_rating + (1 - alpha) * CB_rating, 1, 5)",
            "ranking_candidate_policy": "catalogue items excluding items observed by the user in the outer training fold",
            "ranking_relevance_policy": f"outer-test rating >= {config.ranking_relevance_threshold}",
            "cold_item_policy": f"item has <= {config.cold_item_max_train_ratings} ratings in the outer training fold",
        },
        "config": config.to_dict(),
        "environment": execution_environment(),
        "outer_folds": outer_folds,
        "selected_parameters": selected_parameters,
        "pointwise": pointwise,
        "top_n": _aggregate_top_n(outer_folds, config),
        "cold_start": _aggregate_cold_start(outer_folds),
        "statistical_tests": {
            "hybrid_selected_vs_cf_only": perform_statistical_test(hybrid_rmses, cf_rmses, "Hybrid selected", "CF-only"),
            "hybrid_selected_vs_cb_only": perform_statistical_test(hybrid_rmses, cb_rmses, "Hybrid selected", "CB-only"),
        },
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }
