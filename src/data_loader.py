import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

MOVIELENS_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

def download_and_extract_movielens(data_dir="data"):
    """
    Downloads MovieLens 100k dataset if not already present in data_dir.
    Returns path to ml-100k folder.
    """
    os.makedirs(data_dir, exist_ok=True)
    target_dir = os.path.join(data_dir, "ml-100k")
    
    if os.path.exists(os.path.join(target_dir, "u.data")) and os.path.exists(os.path.join(target_dir, "u.item")):
        print(f"[DataLoader] Dataset found in {target_dir}")
        return target_dir

    zip_path = os.path.join(data_dir, "ml-100k.zip")
    print(f"[DataLoader] Downloading MovieLens 100k from {MOVIELENS_100K_URL}...")
    urllib.request.urlretrieve(MOVIELENS_100K_URL, zip_path)
    
    print("[DataLoader] Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print(f"[DataLoader] Extracted dataset to {target_dir}")
    return target_dir


def load_ratings(ml_dir="data/ml-100k"):
    """
    Loads u.data ratings into a pandas DataFrame.
    Columns: user_id, item_id, rating, timestamp
    """
    u_data_path = os.path.join(ml_dir, "u.data")
    ratings = pd.read_csv(
        u_data_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        encoding="latin-1"
    )
    return ratings


def load_items(ml_dir="data/ml-100k"):
    """
    Loads u.item movie metadata into a pandas DataFrame.
    Columns: item_id, title, release_date, video_release_date, imdb_url, 19 genre binary columns.
    """
    u_item_path = os.path.join(ml_dir, "u.item")
    cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRE_NAMES
    items = pd.read_csv(
        u_item_path,
        sep="|",
        names=cols,
        encoding="latin-1"
    )
    
    # Extract release year from release_date (e.g. '01-Jan-1995' -> 1995)
    items['release_year'] = pd.to_datetime(items['release_date'], format='%d-%b-%Y', errors='coerce').dt.year
    # Fill missing years with median year
    median_year = items['release_year'].median()
    items['release_year'] = items['release_year'].fillna(median_year).astype(int)
    
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
        "num_users": num_users,
        "num_items": num_items,
        "num_ratings": num_ratings,
        "sparsity": sparsity
    }
