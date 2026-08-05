import os
import sys
import json
import numpy as np
import pandas as pd

# Enforce UTF-8 encoding on standard output for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.data_loader import (
    download_and_extract_movielens, load_ratings, load_items,
    build_item_feature_matrix, compute_sparsity, GENRE_NAMES
)
from src.baselines import GlobalMeanBaseline, UserMeanBaseline, ItemMeanBaseline
from src.collaborative import CollaborativeFilteringModel
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender
from src.evaluation import (
    compute_rmse, compute_mae, evaluate_top_n,
    run_5fold_cross_validation, perform_statistical_test
)
from src.utils import (
    plot_rating_distribution, plot_long_tail, plot_genre_distribution,
    plot_alpha_sensitivity, plot_cold_start_comparison,
    plot_comparative_alpha_sensitivity, plot_comparative_cold_start,
    plot_comparative_rating_distribution, plot_comparative_long_tail
)

def run_dataset_pipeline(dataset_name="ml-100k"):
    label = "MovieLens 100k" if dataset_name == "ml-100k" else "MovieLens 1M"
    print("\n" + "=" * 70)
    print(f"  EXPERIMENTAL PIPELINE FOR {label.upper()} (FULL DATASET - NO SAMPLING)")
    print("=" * 70)

    # ---------------------------------------------------------
    # FASE 1: Setup & Data Loading
    # ---------------------------------------------------------
    print(f"\n[{label} - Fase 1] Downloading & Loading Dataset...")
    ml_dir = download_and_extract_movielens(dataset_name=dataset_name, data_dir="data")
    ratings_df = load_ratings(ml_dir, dataset_name=dataset_name)
    items_df = load_items(ml_dir, dataset_name=dataset_name)
    
    output_dir = f"report/figures/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("report/figures/comparison", exist_ok=True)
    os.makedirs("report", exist_ok=True)

    # ---------------------------------------------------------
    # FASE 2: Exploratory Data Analysis (EDA)
    # ---------------------------------------------------------
    print(f"\n[{label} - Fase 2] Exploratory Data Analysis (EDA)...")
    eda_stats = compute_sparsity(ratings_df, items_df)
    print(f"  - Users: {eda_stats['num_users']}")
    print(f"  - Items: {eda_stats['num_items']}")
    print(f"  - Ratings: {eda_stats['num_ratings']}")
    print(f"  - Matrix Sparsity: {eda_stats['sparsity'] * 100:.2f}%")

    print("  - Generating EDA figures...")
    plot_rating_distribution(ratings_df, save_path=os.path.join(output_dir, "rating_distribution.png"), dataset_label=label)
    plot_long_tail(ratings_df, save_path=os.path.join(output_dir, "long_tail.png"), dataset_label=label)
    plot_genre_distribution(items_df, GENRE_NAMES, save_path=os.path.join(output_dir, "genre_distribution.png"), dataset_label=label)

    # Feature matrix construction
    feature_matrix, item_id_to_idx, idx_to_item_id = build_item_feature_matrix(items_df, include_year=True)
    all_item_ids = list(items_df['item_id'].values)

    # ---------------------------------------------------------
    # FASE 3: Baseline Models (5-Fold CV)
    # ---------------------------------------------------------
    print(f"\n[{label} - Fase 3] Evaluating Baseline Models (5-Fold Cross Validation)...")
    
    res_global = run_5fold_cross_validation(ratings_df, lambda f: GlobalMeanBaseline())
    res_user = run_5fold_cross_validation(ratings_df, lambda f: UserMeanBaseline())
    res_item = run_5fold_cross_validation(ratings_df, lambda f: ItemMeanBaseline())

    baseline_summary = {
        "Global Mean": {"rmse": float(np.mean([f['rmse'] for f in res_global])), "std_rmse": float(np.std([f['rmse'] for f in res_global])),
                        "mae": float(np.mean([f['mae'] for f in res_global])), "std_mae": float(np.std([f['mae'] for f in res_global]))},
        "User Mean":   {"rmse": float(np.mean([f['rmse'] for f in res_user])), "std_rmse": float(np.std([f['rmse'] for f in res_user])),
                        "mae": float(np.mean([f['mae'] for f in res_user])), "std_mae": float(np.std([f['mae'] for f in res_user]))},
        "Item Mean":   {"rmse": float(np.mean([f['rmse'] for f in res_item])), "std_rmse": float(np.std([f['rmse'] for f in res_item])),
                        "mae": float(np.mean([f['mae'] for f in res_item])), "std_mae": float(np.std([f['mae'] for f in res_item]))},
    }

    for name, metrics in baseline_summary.items():
        print(f"  - {name:12s} | RMSE: {metrics['rmse']:.4f} +/- {metrics['std_rmse']:.4f} | MAE: {metrics['mae']:.4f} +/- {metrics['std_mae']:.4f}")

    # ---------------------------------------------------------
    # FASE 4: Sensitivity Analysis on alpha (0.0 to 1.0) & 5-Fold CV
    # ---------------------------------------------------------
    print(f"\n[{label} - Fase 4] Sensitivity Analysis on Hybrid Weight alpha in [0.0, 1.0]...")
    alphas = np.linspace(0.0, 1.0, 11)
    alpha_results = {round(a, 1): [] for a in alphas}

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(ratings_df)):
        train_df = ratings_df.iloc[train_idx].copy()
        test_df = ratings_df.iloc[test_idx].copy()

        cf = CollaborativeFilteringModel(n_factors=50, n_epochs=20, random_state=42 + fold_idx)
        cb = ContentBasedRecommender(feature_matrix, item_id_to_idx, rating_threshold=3.0)
        
        cf.fit(train_df)
        cb.fit(train_df)

        # Precompute individual model predictions on test split for rapid vectorized alpha evaluation
        preds_cf = cf.predict_batch(test_df)
        preds_cb = cb.predict_batch(test_df)
        y_true = test_df['rating'].values

        for a in alphas:
            a_round = round(a, 1)
            preds_hybrid = np.clip(a_round * preds_cf + (1.0 - a_round) * preds_cb, 1.0, 5.0)
            
            rmse = compute_rmse(y_true, preds_hybrid)
            mae = compute_mae(y_true, preds_hybrid)
            
            alpha_results[a_round].append({'rmse': rmse, 'mae': mae})

    mean_rmses = [float(np.mean([x['rmse'] for x in alpha_results[round(a, 1)]])) for a in alphas]
    std_rmses = [float(np.std([x['rmse'] for x in alpha_results[round(a, 1)]])) for a in alphas]
    mean_maes = [float(np.mean([x['mae'] for x in alpha_results[round(a, 1)]])) for a in alphas]
    std_maes = [float(np.std([x['mae'] for x in alpha_results[round(a, 1)]])) for a in alphas]

    best_alpha_idx = np.argmin(mean_rmses)
    best_alpha = round(alphas[best_alpha_idx], 1)
    print(f"  - Optimal alpha*: {best_alpha} (RMSE = {mean_rmses[best_alpha_idx]:.4f} +/- {std_rmses[best_alpha_idx]:.4f})")

    plot_alpha_sensitivity(alphas, np.array(mean_rmses), np.array(std_rmses),
                           np.array(mean_maes), np.array(std_maes),
                           save_path=os.path.join(output_dir, "alpha_sensitivity.png"),
                           dataset_label=label)

    # ---------------------------------------------------------
    # FASE 5: Content-Based Profile Threshold Experiment (theta = 3 vs 4)
    # ---------------------------------------------------------
    print(f"\n[{label} - Fase 5] Content-Based Profile Threshold Experiment (theta = 3.0 vs theta = 4.0)...")
    res_cb_th3 = run_5fold_cross_validation(ratings_df, lambda f: ContentBasedRecommender(feature_matrix, item_id_to_idx, rating_threshold=3.0))
    res_cb_th4 = run_5fold_cross_validation(ratings_df, lambda f: ContentBasedRecommender(feature_matrix, item_id_to_idx, rating_threshold=4.0))

    cb_th3_rmse = float(np.mean([f['rmse'] for f in res_cb_th3]))
    cb_th4_rmse = float(np.mean([f['rmse'] for f in res_cb_th4]))

    print(f"  - CB (theta = 3.0) RMSE: {cb_th3_rmse:.4f}")
    print(f"  - CB (theta = 4.0) RMSE: {cb_th4_rmse:.4f}")

    # ---------------------------------------------------------
    # FASE 6: Top-N Ranking Evaluation (Precision@K, NDCG@K)
    # ---------------------------------------------------------
    print(f"\n[{label} - Fase 6] Evaluating Top-N Ranking Metrics (Precision@K, NDCG@K)...")
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    cf_model = CollaborativeFilteringModel(n_factors=50, n_epochs=20, random_state=42).fit(train_df)
    cb_model = ContentBasedRecommender(feature_matrix, item_id_to_idx, rating_threshold=3.0).fit(train_df)
    hybrid_best = HybridRecommender(cf_model, cb_model, alpha=best_alpha).fit(train_df)

    top_n_cf = evaluate_top_n(cf_model, train_df, test_df, all_item_ids, k_list=[5, 10])
    top_n_cb = evaluate_top_n(cb_model, train_df, test_df, all_item_ids, k_list=[5, 10])
    top_n_hybrid = evaluate_top_n(hybrid_best, train_df, test_df, all_item_ids, k_list=[5, 10])

    print("  - Top-N Ranking Results:")
    print(f"    * CF-only (alpha=1.0) : {top_n_cf}")
    print(f"    * CB-only (alpha=0.0) : {top_n_cb}")
    print(f"    * Hybrid  (alpha={best_alpha:.1f}) : {top_n_hybrid}")

    # ---------------------------------------------------------
    # FASE 7: Cold Start Experiment
    # ---------------------------------------------------------
    print(f"\n[{label} - Fase 7] Cold Start Experiment (< 3 ratings per item)...")
    item_counts = train_df['item_id'].value_counts()
    cold_items = set(item_counts[item_counts < 3].index)
    
    test_cold = test_df[test_df['item_id'].isin(cold_items)].copy()
    print(f"  - Evaluated on {len(test_cold)} cold-start item test ratings...")

    cold_results_df = pd.DataFrame()
    if len(test_cold) > 0:
        y_true_cold = test_cold['rating'].values
        
        p_cf = cf_model.predict_batch(test_cold)
        p_cb = cb_model.predict_batch(test_cold)
        p_hyb_opt = hybrid_best.predict_batch(test_cold)
        p_hyb_50 = HybridRecommender(cf_model, cb_model, alpha=0.5).predict_batch(test_cold)
        p_user_mean = UserMeanBaseline().fit(train_df).predict_batch(test_cold)
        
        cold_results_df = pd.DataFrame([
            {"Model": "CF-only (alpha=1.0)", "RMSE": compute_rmse(y_true_cold, p_cf), "MAE": compute_mae(y_true_cold, p_cf)},
            {"Model": "CB-only (alpha=0.0)", "RMSE": compute_rmse(y_true_cold, p_cb), "MAE": compute_mae(y_true_cold, p_cb)},
            {"Model": f"Hybrid (alpha={best_alpha})", "RMSE": compute_rmse(y_true_cold, p_hyb_opt), "MAE": compute_mae(y_true_cold, p_hyb_opt)},
            {"Model": "Hybrid (alpha=0.5)", "RMSE": compute_rmse(y_true_cold, p_hyb_50), "MAE": compute_mae(y_true_cold, p_hyb_50)},
            {"Model": "User Mean Baseline", "RMSE": compute_rmse(y_true_cold, p_user_mean), "MAE": compute_mae(y_true_cold, p_user_mean)},
        ])
        print(cold_results_df.to_string(index=False))
        plot_cold_start_comparison(cold_results_df, save_path=os.path.join(output_dir, "cold_start_comparison.png"), dataset_label=label)

    # ---------------------------------------------------------
    # FASE 8: Statistical Significance Tests
    # ---------------------------------------------------------
    print(f"\n[{label} - Fase 8] Statistical Significance Tests (Hybrid vs CF-only & CB-only)...")
    hybrid_fold_rmses = [x['rmse'] for x in alpha_results[best_alpha]]
    cf_fold_rmses = [x['rmse'] for x in alpha_results[1.0]]
    cb_fold_rmses = [x['rmse'] for x in alpha_results[0.0]]

    stat_hyb_vs_cf = perform_statistical_test(hybrid_fold_rmses, cf_fold_rmses, "Hybrid (optimal alpha)", "CF-only (alpha=1.0)")
    stat_hyb_vs_cb = perform_statistical_test(hybrid_fold_rmses, cb_fold_rmses, "Hybrid (optimal alpha)", "CB-only (alpha=0.0)")

    print(f"  - Hybrid vs CF-only | t-stat: {stat_hyb_vs_cf['t_statistic']:.4f}, p-val: {stat_hyb_vs_cf['t_p_value']:.4f} (Significant: {stat_hyb_vs_cf['statistically_significant_p05']})")
    print(f"  - Hybrid vs CB-only | t-stat: {stat_hyb_vs_cb['t_statistic']:.4f}, p-val: {stat_hyb_vs_cb['t_p_value']:.4f} (Significant: {stat_hyb_vs_cb['statistically_significant_p05']})")

    dataset_summary = {
        "eda": eda_stats,
        "baselines": baseline_summary,
        "best_alpha": best_alpha,
        "alpha_sensitivity": {
            "alphas": [float(a) for a in alphas],
            "mean_rmse": [float(x) for x in mean_rmses],
            "std_rmse": [float(x) for x in std_rmses],
            "mean_mae": [float(x) for x in mean_maes],
            "std_mae": [float(x) for x in std_maes],
        },
        "cb_threshold_experiment": {
            "theta_3_rmse": cb_th3_rmse,
            "theta_4_rmse": cb_th4_rmse
        },
        "top_n_ranking": {
            "cf_only": top_n_cf,
            "cb_only": top_n_cb,
            "hybrid": top_n_hybrid
        },
        "cold_start": cold_results_df.to_dict('records') if not cold_results_df.empty else [],
        "statistical_tests": {
            "hybrid_vs_cf": stat_hyb_vs_cf,
            "hybrid_vs_cb": stat_hyb_vs_cb
        }
    }

    return dataset_summary, cold_results_df, alphas, np.array(mean_rmses)


def main():
    print("=" * 70)
    print("  SII HYBRID RECOMMENDER SYSTEM - MULTI-DATASET COMPARATIVE PIPELINE")
    print("  Datasets: MovieLens 100k & MovieLens 1M (FULL DATASETS - NO SAMPLING)")
    print("=" * 70)

    # Run MovieLens 100k
    summary_100k, cold_100k, alphas, rmses_100k = run_dataset_pipeline("ml-100k")

    # Run MovieLens 1M
    summary_1m, cold_1m, alphas, rmses_1m = run_dataset_pipeline("ml-1m")

    # ---------------------------------------------------------
    # Comparative Figures & Results Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  GENERATING SIDE-BY-SIDE COMPARATIVE PLOTS & SUMMARY")
    print("=" * 70)

    plot_comparative_alpha_sensitivity(alphas, rmses_100k, rmses_1m,
                                       save_path="report/figures/comparison/alpha_sensitivity_comparison.png")

    if not cold_100k.empty and not cold_1m.empty:
        plot_comparative_cold_start(cold_100k, cold_1m,
                                    save_path="report/figures/comparison/cold_start_comparison.png")

    # Generate EDA comparative figures
    ml_100k_dir = os.path.join("data", "ml-100k")
    ml_1m_dir = os.path.join("data", "ml-1m")
    ratings_100k = load_ratings(ml_100k_dir, "ml-100k")
    ratings_1m = load_ratings(ml_1m_dir, "ml-1m")

    plot_comparative_rating_distribution(ratings_100k, ratings_1m,
                                           save_path="report/figures/comparison/rating_distribution_comparison.png")
    plot_comparative_long_tail(ratings_100k, ratings_1m,
                               save_path="report/figures/comparison/long_tail_comparison.png")

    summary_data = {
        "ml-100k": summary_100k,
        "ml-1m": summary_1m,
        "comparison_highlights": {
            "ml100k_users": summary_100k['eda']['num_users'],
            "ml1m_users": summary_1m['eda']['num_users'],
            "ml100k_ratings": summary_100k['eda']['num_ratings'],
            "ml1m_ratings": summary_1m['eda']['num_ratings'],
            "ml100k_sparsity_pct": round(summary_100k['eda']['sparsity'] * 100, 2),
            "ml1m_sparsity_pct": round(summary_1m['eda']['sparsity'] * 100, 2),
            "ml100k_cf_rmse": summary_100k['baselines']['Global Mean']['rmse'],
            "ml100k_best_alpha": summary_100k['best_alpha'],
            "ml1m_best_alpha": summary_1m['best_alpha'],
            "ml100k_hybrid_min_rmse": float(np.min(rmses_100k)),
            "ml1m_hybrid_min_rmse": float(np.min(rmses_1m))
        }
    }

    with open("report/experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTAL PIPELINES COMPLETED SUCCESSFULLY!")
    print("  - ML-100k Figures saved to: report/figures/ml-100k/")
    print("  - ML-1M Figures saved to:   report/figures/ml-1m/")
    print("  - Comparison Figures saved to: report/figures/comparison/")
    print("  - Aggregated JSON saved to: report/experiment_results.json")
    print("=" * 70)

if __name__ == "__main__":
    main()
