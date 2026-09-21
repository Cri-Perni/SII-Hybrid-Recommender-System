import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
import pandas as pd

# Enforce UTF-8 encoding on standard output for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.data_loader import download_and_extract_movielens, load_ratings, load_items, build_item_feature_matrix, GENRE_NAMES
from src.collaborative import CollaborativeFilteringModel
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender

def main():
    print("=" * 70)
    print("  GENERATING QUALITATIVE CASE STUDIES & OVERLAP ANALYSIS")
    print("=" * 70)

    ml_dir = download_and_extract_movielens("ml-100k", data_dir="data")
    ratings_df = load_ratings(ml_dir, "ml-100k")
    items_df = load_items(ml_dir, "ml-100k")

    feature_matrix, item_id_to_idx, idx_to_item_id = build_item_feature_matrix(items_df, include_year=True)
    all_item_ids = list(items_df['item_id'].values)
    item_title_map = items_df.set_index('item_id')['title'].to_dict()
    item_genre_map = {}
    for _, row in items_df.iterrows():
        genres = [g for g in GENRE_NAMES if g in row and row[g] == 1]
        item_genre_map[row['item_id']] = ", ".join(genres) if genres else "Unknown"

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

    cf_model = CollaborativeFilteringModel(n_factors=50, n_epochs=20, random_state=42).fit(train_df)
    cb_model = ContentBasedRecommender(feature_matrix, item_id_to_idx, rating_threshold=3.0).fit(train_df)
    # Analisi qualitativa esplorativa: i modelli sono già addestrati sul train split.
    # HybridRecommender usa la stessa fusione diretta dei rating della pipeline nested.
    hybrid_best = HybridRecommender(cf_model, cb_model, alpha=0.9)

    train_items_by_user = train_df.groupby('user_id')['item_id'].apply(set).to_dict()
    all_set = set(all_item_ids)

    # 1. Jaccard Overlap Analysis between CF and CBF Top-10 lists
    sample_users = list(train_items_by_user.keys())[:100]
    jaccard_sims = []
    
    for u in sample_users:
        seen = train_items_by_user.get(u, set())
        unseen = list(all_set - seen)
        if not unseen:
            continue
        c_df = pd.DataFrame({'user_id': np.full(len(unseen), u), 'item_id': unseen})
        
        p_cf = cf_model.predict_batch(c_df)
        p_cb = cb_model.predict_batch(c_df)
        
        top10_cf = [unseen[i] for i in np.argsort(-p_cf)[:10]]
        top10_cb = [unseen[i] for i in np.argsort(-p_cb)[:10]]
        
        intersection = len(set(top10_cf) & set(top10_cb))
        union = len(set(top10_cf) | set(top10_cb))
        jaccard_sims.append(intersection / union if union > 0 else 0.0)

    avg_jaccard = float(np.mean(jaccard_sims))
    print(f"\n[Differentiation Metric] Average Top-10 Jaccard Overlap (CF vs CBF): {avg_jaccard * 100:.2f}%")
    print(f"-> L'overlap del {avg_jaccard * 100:.1f}% descrive la diversità delle liste; non costituisce da solo una misura di qualità.")

    # 2. Select representative users for case studies
    user_counts = train_df['user_id'].value_counts()
    cold_users = user_counts[user_counts <= 5].index.tolist()
    cold_u = cold_users[0] if cold_users else sample_users[-1]

    target_users = [1, 5, cold_u]
    case_studies = []

    for u in target_users:
        seen = train_items_by_user.get(u, set())
        unseen = list(all_set - seen)
        c_df = pd.DataFrame({'user_id': np.full(len(unseen), u), 'item_id': unseen})
        
        p_cf = cf_model.predict_batch(c_df)
        p_cb = cb_model.predict_batch(c_df)
        p_hyb = hybrid_best.predict_batch(c_df)
        
        top5_cf_ids = [unseen[i] for i in np.argsort(-p_cf)[:5]]
        top5_cb_ids = [unseen[i] for i in np.argsort(-p_cb)[:5]]
        top5_hyb_ids = [unseen[i] for i in np.argsort(-p_hyb)[:5]]

        user_history = train_df[train_df['user_id'] == u].sort_values('rating', ascending=False).head(5)
        history_items = [
            {"title": item_title_map.get(row['item_id'], "Unknown"),
             "genres": item_genre_map.get(row['item_id'], "Unknown"),
             "rating": float(row['rating'])}
            for _, row in user_history.iterrows()
        ]

        case_studies.append({
            "user_id": int(u),
            "num_ratings": int(len(seen)),
            "history": history_items,
            "top5_cf": [{"title": item_title_map.get(i, "Unknown"), "genres": item_genre_map.get(i, "Unknown")} for i in top5_cf_ids],
            "top5_cb": [{"title": item_title_map.get(i, "Unknown"), "genres": item_genre_map.get(i, "Unknown")} for i in top5_cb_ids],
            "top5_hybrid": [{"title": item_title_map.get(i, "Unknown"), "genres": item_genre_map.get(i, "Unknown")} for i in top5_hyb_ids],
        })

    result_data = {
        "avg_jaccard_overlap_pct": round(avg_jaccard * 100, 2),
        "case_studies": case_studies
    }

    os.makedirs("report", exist_ok=True)
    with open("report/case_studies.json", "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print("\nCase Studies generated successfully and saved to report/case_studies.json")

if __name__ == "__main__":
    main()
