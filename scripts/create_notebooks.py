import json
import os

def make_cell(cell_type, source):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }
    if cell_type == "code":
        cell["execution_count"] = None
    return cell

def create_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filename}")

def main():
    # 01_EDA.ipynb
    eda_cells = [
        make_cell("markdown", "# Analisi Esplorativa dei Dati (EDA) - MovieLens 100k\n**Corso:** Sistemi Intelligenti per Internet (SII)"),
        make_cell("code", """import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import download_and_extract_movielens, load_ratings, load_items, compute_sparsity, GENRE_NAMES
from src.utils import plot_rating_distribution, plot_long_tail, plot_genre_distribution
"""),
        make_cell("markdown", "## 1. Caricamento Dataset e Calcolo Metrise Esplorative"),
        make_cell("code", """ml_dir = download_and_extract_movielens(data_dir="../data")
ratings_df = load_ratings(ml_dir)
items_df = load_items(ml_dir)

stats = compute_sparsity(ratings_df, items_df)
print("Statistiche Dataset:")
print(f" - Numero Utenti: {stats['num_users']}")
print(f" - Numero Film: {stats['num_items']}")
print(f" - Numero Valutazioni Totali: {stats['num_ratings']}")
print(f" - Sparsità Matrice: {stats['sparsity']*100:.2f}%")
"""),
        make_cell("markdown", "## 2. Visualizzazione delle Distribuzioni"),
        make_cell("code", """# Distribuzione dei voti (1-5)
plot_rating_distribution(ratings_df)

# Long Tail Analysis
plot_long_tail(ratings_df)

# Distribuzione dei Generi Cinematografici
plot_genre_distribution(items_df, GENRE_NAMES)
""")
    ]
    create_notebook(eda_cells, "notebooks/01_EDA.ipynb")

    # 02_models.ipynb
    models_cells = [
        make_cell("markdown", "# Modelli di Raccomandazione: Baselines, CF, CB, Hybrid\n**Corso:** Sistemi Intelligenti per Internet (SII)"),
        make_cell("code", """import sys
sys.path.append("..")

from sklearn.model_selection import train_test_split
from src.data_loader import download_and_extract_movielens, load_ratings, load_items, build_item_feature_matrix
from src.baselines import GlobalMeanBaseline, UserMeanBaseline, ItemMeanBaseline
from src.collaborative import CollaborativeFilteringModel
from src.content_based import ContentBasedRecommender
from src.hybrid import HybridRecommender
from src.evaluation import compute_rmse, compute_mae
"""),
        make_cell("markdown", "## 1. Split Train / Test (80/20) e Training Modelli"),
        make_cell("code", """ml_dir = download_and_extract_movielens(data_dir="../data")
ratings_df = load_ratings(ml_dir)
items_df = load_items(ml_dir)

feature_matrix, item_id_to_idx, _ = build_item_feature_matrix(items_df, include_year=True)
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Fit models
g_mean = GlobalMeanBaseline().fit(train_df)
u_mean = UserMeanBaseline().fit(train_df)
i_mean = ItemMeanBaseline().fit(train_df)

cf = CollaborativeFilteringModel(n_factors=50, n_epochs=20, random_state=42).fit(train_df)
cb = ContentBasedRecommender(feature_matrix, item_id_to_idx, rating_threshold=3.0).fit(train_df)
hybrid = HybridRecommender(cf, cb, alpha=0.5).fit(train_df)

print("Modelli addestrati con successo.")
"""),
        make_cell("markdown", "## 2. Valutazione Singola Predizione"),
        make_cell("code", """y_true = test_df['rating'].values

models = {
    "Global Mean": g_mean.predict_batch(test_df),
    "User Mean": u_mean.predict_batch(test_df),
    "Item Mean": i_mean.predict_batch(test_df),
    "CF-only (SVD)": cf.predict_batch(test_df),
    "CB-only (Cosine)": cb.predict_batch(test_df),
    "Hybrid (alpha=0.5)": hybrid.predict_batch(test_df)
}

for name, preds in models.items():
    rmse = compute_rmse(y_true, preds)
    mae = compute_mae(y_true, preds)
    print(f"{name:20s} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
""")
    ]
    create_notebook(models_cells, "notebooks/02_models.ipynb")

    # 03_experiments.ipynb
    exp_cells = [
        make_cell("markdown", "# Esperimenti Avanzati e Valutazione Sperimentale\n**Corso:** Sistemi Intelligenti per Internet (SII)"),
        make_cell("code", """import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from src.data_loader import download_and_extract_movielens, load_ratings, load_items, build_item_feature_matrix
from src.evaluation import run_5fold_cross_validation, perform_statistical_test
from src.utils import plot_alpha_sensitivity, plot_cold_start_comparison
"""),
        make_cell("markdown", "## 1. 5-Fold Cross-Validation e Sensitivity Analysis su Alpha"),
        make_cell("code", """# Per eseguire tutti gli esperimenti e generare la reportistica completa:
# Eseguire il file python run_pipeline.py dal terminale.
!python ../run_pipeline.py
""")
    ]
    create_notebook(exp_cells, "notebooks/03_experiments.ipynb")

if __name__ == "__main__":
    main()
