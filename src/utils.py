import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent aesthetic plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})

def plot_architecture_diagram(save_path: str = "report/figures/architecture.png"):
    """
    Generates a high-resolution graphical architecture diagram for the Hybrid Recommender System.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Background color
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    box_props_input = dict(boxstyle="round,pad=0.5", facecolor="#34495e", edgecolor="#2c3e50", linewidth=2)
    box_props_cf = dict(boxstyle="round,pad=0.5", facecolor="#2980b9", edgecolor="#1b4f72", linewidth=2)
    box_props_cb = dict(boxstyle="round,pad=0.5", facecolor="#27ae60", edgecolor="#196f3d", linewidth=2)
    box_props_fusion = dict(boxstyle="round,pad=0.5", facecolor="#8e44ad", edgecolor="#512e5f", linewidth=2)
    box_props_output = dict(boxstyle="round,pad=0.5", facecolor="#d35400", edgecolor="#873600", linewidth=2)
    
    # Coordinates (x, y)
    # 1. Input Box
    ax.text(0.5, 0.90, "Input Dataset\n(MovieLens 100k / MovieLens 1M)\nUtente u, Item i", 
            ha="center", va="center", color="white", fontsize=11, fontweight="bold", bbox=box_props_input)
    
    # 2. CF Box (Left)
    ax.text(0.25, 0.62, "Collaborative Filtering\n(Biased Matrix Factorization - SVD)\nScore S_CF(u, i) in [0, 1]", 
            ha="center", va="center", color="white", fontsize=10, fontweight="bold", bbox=box_props_cf)
    
    # 3. CB Box (Right)
    ax.text(0.75, 0.62, "Content-Based Filtering\n(Profilo Utente theta + Cosine Sim)\nScore S_CB(u, i) in [0, 1]", 
            ha="center", va="center", color="white", fontsize=10, fontweight="bold", bbox=box_props_cb)
    
    # 4. Fusion Box
    ax.text(0.5, 0.35, "Modulo di Fusione Pesata (Weighted Fusion)\nS_Hybrid(u, i) = alpha * S_CF + (1 - alpha) * S_CB\nalpha in [0.0, 1.0]", 
            ha="center", va="center", color="white", fontsize=11, fontweight="bold", bbox=box_props_fusion)
    
    # 5. Output Box
    ax.text(0.5, 0.10, "Predizione Rating Finale R_hat(u, i)\nDenormalizzata nel range originario [1.0, 5.0]", 
            ha="center", va="center", color="white", fontsize=11, fontweight="bold", bbox=box_props_output)
    
    # Arrows
    arrow_props = dict(arrowstyle="->", color="#2c3e50", lw=2.5, mutation_scale=15)
    
    # Input to CF & CB
    ax.annotate("", xy=(0.25, 0.72), xytext=(0.42, 0.84), arrowprops=arrow_props)
    ax.annotate("", xy=(0.75, 0.72), xytext=(0.58, 0.84), arrowprops=arrow_props)
    
    # CF & CB to Fusion
    ax.annotate("", xy=(0.42, 0.43), xytext=(0.25, 0.52), arrowprops=arrow_props)
    ax.annotate("", xy=(0.58, 0.43), xytext=(0.75, 0.52), arrowprops=arrow_props)
    
    # Fusion to Output
    ax.annotate("", xy=(0.5, 0.17), xytext=(0.5, 0.27), arrowprops=arrow_props)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_rating_distribution(ratings_df: pd.DataFrame, save_path: str = None, dataset_label: str = "MovieLens"):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=ratings_df, x='rating', hue='rating', palette="viridis", legend=False, ax=ax)
    ax.set_title(f"Distribuzione delle Valutazioni ({dataset_label})", fontsize=13, fontweight='bold')
    ax.set_xlabel("Rating", fontsize=11)
    ax.set_ylabel("Conteggio", fontsize=11)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height):,}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_long_tail(ratings_df: pd.DataFrame, save_path: str = None, dataset_label: str = "MovieLens"):
    item_counts = ratings_df['item_id'].value_counts().values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(item_counts, color='#2b5c8f', linewidth=2)
    ax.set_title(f"Popolarità dei Film - Long Tail Analysis ({dataset_label})", fontsize=13, fontweight='bold')
    ax.set_xlabel("Indice Film (ordinato per popolarità)", fontsize=11)
    ax.set_ylabel("Numero di Valutazioni", fontsize=11)
    ax.fill_between(range(len(item_counts)), item_counts, color='#2b5c8f', alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_genre_distribution(items_df: pd.DataFrame, genre_names: list, save_path: str = None, dataset_label: str = "MovieLens"):
    genre_counts = items_df[genre_names].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = pd.DataFrame({'genre': genre_counts.index, 'count': genre_counts.values})
    sns.barplot(data=df_plot, x='count', y='genre', hue='genre', palette="mako", legend=False, ax=ax)
    ax.set_title(f"Distribuzione dei Generi Cinematografici ({dataset_label})", fontsize=13, fontweight='bold')
    ax.set_xlabel("Numero di Film", fontsize=11)
    ax.set_ylabel("Genere", fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_alpha_sensitivity(alphas: np.ndarray, mean_rmses: np.ndarray, std_rmses: np.ndarray,
                           mean_maes: np.ndarray, std_maes: np.ndarray, save_path: str = None, dataset_label: str = "MovieLens"):
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = '#1f77b4'
    ax1.set_xlabel('Parametro di Fusione Pesata (alpha)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('RMSE (Root Mean Squared Error)', color=color1, fontsize=11, fontweight='bold')
    line1 = ax1.plot(alphas, mean_rmses, color=color1, marker='o', linewidth=2, label='RMSE')
    ax1.fill_between(alphas, mean_rmses - std_rmses, mean_rmses + std_rmses, color=color1, alpha=0.15)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(alphas)

    # Highlight optimal alpha
    best_idx = np.argmin(mean_rmses)
    best_alpha = alphas[best_idx]
    best_rmse = mean_rmses[best_idx]
    ax1.annotate(f'alpha* = {best_alpha:.1f}\nRMSE = {best_rmse:.4f}',
                 xy=(best_alpha, best_rmse), xytext=(best_alpha, best_rmse + 0.02),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, fontweight='bold', color='red', ha='center')

    color2 = '#ff7f0e'
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE (Mean Absolute Error)', color=color2, fontsize=11, fontweight='bold')
    line2 = ax2.plot(alphas, mean_maes, color=color2, marker='s', linestyle='--', linewidth=2, label='MAE')
    ax2.fill_between(alphas, mean_maes - std_maes, mean_maes + std_maes, color=color2, alpha=0.15)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f"Sensitivity Analysis: Errore vs Peso alpha - {dataset_label} (5-Fold CV)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_cold_start_comparison(results_df: pd.DataFrame, save_path: str = None, dataset_label: str = "MovieLens"):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, results_df['RMSE'], width, label='RMSE', color='#3498db')
    rects2 = ax.bar(x + width/2, results_df['MAE'], width, label='MAE', color='#e74c3c')

    ax.set_ylabel('Errore', fontsize=11, fontweight='bold')
    ax.set_title(f'Scenario Cold Start (< 3 Rating per Item) - {dataset_label}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=15, ha='right', fontsize=10)
    ax.legend()
    ax.set_ylim(0, max(results_df['RMSE'].max(), results_df['MAE'].max()) * 1.2)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', xytext=(0, 3),
                        textcoords='offset points', fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_comparative_alpha_sensitivity(alphas: np.ndarray, rmses_100k: np.ndarray, rmses_1m: np.ndarray, save_path: str = None):
    """
    Plots comparative RMSE sensitivity across alpha for MovieLens 100k vs MovieLens 1M side-by-side.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.plot(alphas, rmses_100k, color='#e74c3c', marker='o', linewidth=2.5, label='MovieLens 100k')
    ax.plot(alphas, rmses_1m, color='#2ecc71', marker='s', linewidth=2.5, linestyle='--', label='MovieLens 1M')
    
    ax.set_title("Confronto Sensitivity Analysis: RMSE vs Alpha (MovieLens 100k vs 1M)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Parametro di Fusione Pesata (alpha)", fontsize=11, fontweight='bold')
    ax.set_ylabel("RMSE (Root Mean Squared Error)", fontsize=11, fontweight='bold')
    ax.set_xticks(alphas)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_comparative_cold_start(cold_100k: pd.DataFrame, cold_1m: pd.DataFrame, save_path: str = None):
    """
    Plots comparative Cold Start RMSE for MovieLens 100k vs MovieLens 1M.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    models = cold_100k['Model'].values
    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, cold_100k['RMSE'], width, label='MovieLens 100k', color='#34495e')
    ax.bar(x + width/2, cold_1m['RMSE'], width, label='MovieLens 1M', color='#e67e22')

    ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
    ax.set_title('Confronto Performance Cold Start (RMSE) - MovieLens 100k vs 1M', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.3f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', xytext=(0, 3),
                        textcoords='offset points', fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_comparative_rating_distribution(ratings_100k: pd.DataFrame, ratings_1m: pd.DataFrame, save_path: str = None):
    """
    Plots side-by-side percentage rating distribution comparison for MovieLens 100k vs MovieLens 1M.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    
    p_100k = ratings_100k['rating'].value_counts(normalize=True).sort_index() * 100
    p_1m = ratings_1m['rating'].value_counts(normalize=True).sort_index() * 100
    
    ratings = np.arange(1, 6)
    width = 0.35
    
    ax.bar(ratings - width/2, [p_100k.get(r, 0) for r in ratings], width, label='MovieLens 100k (%)', color='#2b5c8f')
    ax.bar(ratings + width/2, [p_1m.get(r, 0) for r in ratings], width, label='MovieLens 1M (%)', color='#27ae60')
    
    ax.set_xlabel('Rating (Stelle)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Percentuale (%)', fontsize=11, fontweight='bold')
    ax.set_title('Confronto Distribuzione Rating (%) - MovieLens 100k vs 1M', fontsize=13, fontweight='bold')
    ax.set_xticks(ratings)
    ax.legend(fontsize=11)
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}%',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', xytext=(0, 3),
                        textcoords='offset points', fontsize=9)
                        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_comparative_long_tail(ratings_100k: pd.DataFrame, ratings_1m: pd.DataFrame, save_path: str = None):
    """
    Plots comparative Long Tail popularity curves for MovieLens 100k vs MovieLens 1M.
    """
    counts_100k = ratings_100k['item_id'].value_counts().values
    counts_1m = ratings_1m['item_id'].value_counts().values
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    ax.plot(counts_100k, color='#e74c3c', linewidth=2, label='MovieLens 100k (1.682 Film)')
    ax.plot(counts_1m, color='#2980b9', linewidth=2, linestyle='--', label='MovieLens 1M (3.883 Film)')
    
    ax.set_title("Confronto Popolarità Film - Long Tail Analysis (ML-100k vs ML-1M)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Indice Film (ordinato per popolarità)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Numero di Valutazioni", fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)
