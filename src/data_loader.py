import os
import re
import urllib.request
import zipfile
import pandas as pd
import numpy as np

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def download_and_extract_movielens(dataset_name="ml-100k", data_dir="data"):
    """
    Downloads and extracts MovieLens dataset ("ml-100k" or "ml-1m") if not present.
    Returns path to extracted folder.
    """
    os.makedirs(data_dir, exist_ok=True)
    target_dir = os.path.join(data_dir, dataset_name)
    
    if dataset_name == "ml-100k":
        if os.path.exists(os.path.join(target_dir, "u.data")) and os.path.exists(os.path.join(target_dir, "u.item")):
            print(f"[DataLoader] MovieLens 100k dataset found in {target_dir}")
            return target_dir
        url = MOVIELENS_100K_URL
    elif dataset_name == "ml-1m":
        if os.path.exists(os.path.join(target_dir, "ratings.dat")) and os.path.exists(os.path.join(target_dir, "movies.dat")):
            print(f"[DataLoader] MovieLens 1M dataset found in {target_dir}")
            return target_dir
        url = MOVIELENS_1M_URL
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}. Expected 'ml-100k' or 'ml-1m'.")

    zip_path = os.path.join(data_dir, f"{dataset_name}.zip")
    print(f"[DataLoader] Downloading {dataset_name} from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print(f"[DataLoader] Extracting {dataset_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print(f"[DataLoader] Extracted dataset to {target_dir}")
    return target_dir


def load_ratings(ml_dir="data/ml-100k", dataset_name="ml-100k"):
    """
    Loads ratings into a pandas DataFrame.
    Columns: user_id, item_id, rating, timestamp
    """
    if dataset_name == "ml-100k":
        u_data_path = os.path.join(ml_dir, "u.data")
        ratings = pd.read_csv(
            u_data_path,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1"
        )
    elif dataset_name == "ml-1m":
        r_data_path = os.path.join(ml_dir, "ratings.dat")
        ratings = pd.read_csv(
            r_data_path,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1"
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
        
    return ratings


def load_items(ml_dir="data/ml-100k", dataset_name="ml-100k"):
    """
    Loads item metadata into a pandas DataFrame.
    Columns: item_id, title, release_year, and 19 genre binary columns.
    """
    if dataset_name == "ml-100k":
        u_item_path = os.path.join(ml_dir, "u.item")
        cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_NAMES
        items = pd.read_csv(
            u_item_path,
            sep="|",
            names=cols,
            encoding="latin-1"
        )
        items['release_year'] = pd.to_datetime(items['release_date'], format='%d-%b-%Y', errors='coerce').dt.year
    elif dataset_name == "ml-1m":
        m_item_path = os.path.join(ml_dir, "movies.dat")
        items = pd.read_csv(
            m_item_path,
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            encoding="latin-1"
        )
        
        # Parse release year from Title e.g. "Toy Story (1995)"
        extracted_years = items['title'].str.extract(r'\((\d{4})\)$')[0]
        items['release_year'] = pd.to_numeric(extracted_years, errors='coerce')
        
        # Create binary genre columns matching GENRE_NAMES
        for g in GENRE_NAMES:
            items[g] = items['genres'].apply(lambda x: 1 if isinstance(x, str) and g in x.split('|') else 0)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
        
    # Fill missing years with median year
    median_year = items['release_year'].median()
    items['release_year'] = items['release_year'].fillna(median_year if not pd.isna(median_year) else 1995).astype(int)
    
    return items


def build_item_feature_matrix(items_df, include_year=True):
    """
    Builds item content feature matrix (M x D).
    D = 19 (genres) if include_year=False, or D = 20 (genres + minmax release_year) if include_year=True.
    Returns:
        feature_matrix: np.ndarray of shape (num_items, D)
        item_id_to_idx: dict mapping item_id -> row index in matrix
        idx_to_item_id: dict mapping row index -> item_id
    """
    items_sorted = items_df.sort_values('item_id').reset_index(drop=True)
    item_ids = items_sorted['item_id'].values
    
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    idx_to_item_id = {idx: item_id for idx, item_id in enumerate(item_ids)}
    
    genre_features = items_sorted[GENRE_NAMES].values.astype(float)
    
    if include_year:
        years = items_sorted['release_year'].values.astype(float)
        y_min, y_max = years.min(), years.max()
        years_norm = (years - y_min) / (y_max - y_min + 1e-8)
        feature_matrix = np.hstack([genre_features, years_norm.reshape(-1, 1)])
    else:
        feature_matrix = genre_features
        
    return feature_matrix, item_id_to_idx, idx_to_item_id


def compute_sparsity(ratings_df, items_df):
    """
    Computes user-item matrix sparsity.
    Sparsity = 1 - (Num Ratings / (Num Users * Num Items))
    """
    num_users = ratings_df['user_id'].nunique()
    num_items = items_df['item_id'].nunique()
    num_ratings = len(ratings_df)
    
    sparsity = 1.0 - (num_ratings / (num_users * num_items))
    return {
        "num_users": int(num_users),
        "num_items": int(num_items),
        "num_ratings": int(num_ratings),
        "sparsity": float(sparsity)
    }
