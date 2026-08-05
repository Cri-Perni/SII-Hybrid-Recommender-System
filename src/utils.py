import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent aesthetic plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.family': 'sans-serif'})

def plot_rating_distribution(ratings_df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=ratings_df, x='rating', hue='rating', palette="viridis", legend=False, ax=ax)
    ax.set_title("Distribuzione delle Valutazioni (Rating 1-5)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Rating", fontsize=11)
    ax.set_ylabel("Conteggio", fontsize=11)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_long_tail(ratings_df: pd.DataFrame, save_path: str = None):
    item_counts = ratings_df['item_id'].value_counts().values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(item_counts, color='#2b5c8f', linewidth=2)
    ax.set_title("Popolarità dei Film - Long Tail Analysis", fontsize=13, fontweight='bold')
    ax.set_xlabel("Indice Film (ordinato per popolarità)", fontsize=11)
    ax.set_ylabel("Numero di Valutazioni", fontsize=11)
    ax.fill_between(range(len(item_counts)), item_counts, color='#2b5c8f', alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_genre_distribution(items_df: pd.DataFrame, genre_names: list, save_path: str = None):
    genre_counts = items_df[genre_names].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    df_plot = pd.DataFrame({'genre': genre_counts.index, 'count': genre_counts.values})
    sns.barplot(data=df_plot, x='count', y='genre', hue='genre', palette="mako", legend=False, ax=ax)
    ax.set_title("Distribuzione dei Generi Cinematografici (u.item)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Numero di Film", fontsize=11)
    ax.set_ylabel("Genere", fontsize=11)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_alpha_sensitivity(alphas: np.ndarray, mean_rmses: np.ndarray, std_rmses: np.ndarray,
                           mean_maes: np.ndarray, std_maes: np.ndarray, save_path: str = None):
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

    plt.title("Sensitivity Analysis: Errore vs Peso di Fusione alpha (5-Fold CV)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_cold_start_comparison(results_df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, results_df['RMSE'], width, label='RMSE', color='#3498db')
    rects2 = ax.bar(x + width/2, results_df['MAE'], width, label='MAE', color='#e74c3c')

    ax.set_ylabel('Errore', fontsize=11, fontweight='bold')
    ax.set_title('Confronto Performance in Scenario Cold Start (< 3 Rating per Item)', fontsize=13, fontweight='bold')
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
