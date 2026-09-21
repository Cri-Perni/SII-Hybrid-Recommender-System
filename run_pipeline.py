"""Pipeline riproducibile per la valutazione nested 5×3 del recommender ibrido."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import (
    GENRE_NAMES,
    build_item_feature_matrix,
    compute_sparsity,
    download_and_extract_movielens,
    load_items,
    load_ratings,
)
from src.experiment_config import ExperimentConfig
from src.nested_experiment import run_nested_experiment
from src.utils import (
    plot_genre_distribution,
    plot_long_tail,
    plot_nested_pointwise,
    plot_rating_distribution,
    plot_selected_parameters,
)

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"


def _format(summary: dict, decimals: int = 4) -> str:
    mean = summary["mean"]
    std = summary["std"]
    return "n/a" if mean is None else f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def _dataset_label(dataset_name: str) -> str:
    return "MovieLens 100k" if dataset_name == "ml-100k" else "MovieLens 1M"


def run_dataset_pipeline(dataset_name: str, config: ExperimentConfig, generate_eda: bool) -> dict:
    label = _dataset_label(dataset_name)
    print(f"\n{'=' * 72}\n{label}: nested {config.outer_splits}×{config.inner_splits} CV\n{'=' * 72}")
    data_dir = ROOT / "data"
    ml_dir = download_and_extract_movielens(dataset_name=dataset_name, data_dir=str(data_dir))
    ratings_df = load_ratings(ml_dir, dataset_name=dataset_name)
    items_df = load_items(ml_dir, dataset_name=dataset_name)
    feature_matrix, item_id_to_idx, _ = build_item_feature_matrix(items_df, include_year=True)
    figure_dir = FIGURES_DIR / dataset_name
    figure_dir.mkdir(parents=True, exist_ok=True)

    if generate_eda:
        plot_rating_distribution(ratings_df, str(figure_dir / "rating_distribution.png"), label)
        plot_long_tail(ratings_df, str(figure_dir / "long_tail.png"), label)
        plot_genre_distribution(items_df, GENRE_NAMES, str(figure_dir / "genre_distribution.png"), label)

    nested = run_nested_experiment(
        ratings_df=ratings_df,
        feature_matrix=feature_matrix,
        item_id_to_idx=item_id_to_idx,
        all_item_ids=items_df["item_id"].tolist(),
        config=config,
    )
    plot_nested_pointwise(nested["pointwise"], str(figure_dir / "nested_pointwise_rmse.png"), label)
    plot_selected_parameters(nested["selected_parameters"], str(figure_dir / "nested_selected_parameters.png"), label)

    hybrid = nested["pointwise"]["hybrid_selected"]["rmse"]
    print(f"  Hybrid nested outer RMSE: {_format(hybrid)}")
    print(f"  Completato in {nested['elapsed_seconds']:.1f} s")
    return {"eda": compute_sparsity(ratings_df, items_df), "nested_5x3": nested}


def write_markdown_summary(results: dict, destination: Path) -> None:
    lines = [
        "# Risultati sperimentali riproducibili",
        "",
        "Questo file è generato automaticamente da `run_pipeline.py`. Le metriche finali sono medie sui cinque outer test fold di una nested 5×3 CV; i parametri `alpha` e `theta` sono scelti esclusivamente nei fold interni.",
        "",
    ]
    for dataset_name, dataset_result in results["datasets"].items():
        nested = dataset_result["nested_5x3"]
        lines.extend([
            f"## {_dataset_label(dataset_name)}",
            "",
            f"- Rating: {dataset_result['eda']['num_ratings']:,}; utenti: {dataset_result['eda']['num_users']:,}; film: {dataset_result['eda']['num_items']:,}.",
            f"- Durata: {nested['elapsed_seconds']:.1f} s; ranking: massimo {nested['config']['ranking_max_users']} utenti idonei per outer fold.",
            "",
            "### Accuratezza pointwise",
            "",
            "| Modello | RMSE (mean ± std) | MAE (mean ± std) |",
            "|---|---:|---:|",
        ])
        for name, metrics in nested["pointwise"].items():
            lines.append(f"| {name} | {_format(metrics['rmse'])} | {_format(metrics['mae'])} |")
        lines.extend(["", "### Ranking Top-N", "", "| Modello | Precision@5 | NDCG@5 | Precision@10 | NDCG@10 | Utenti valutati |", "|---|---:|---:|---:|---:|---:|"])
        for name, ranking in nested["top_n"].items():
            metrics, support = ranking["metrics"], ranking["support"]
            lines.append(
                f"| {name} | {_format(metrics['Precision@5'])} | {_format(metrics['NDCG@5'])} | "
                f"{_format(metrics['Precision@10'])} | {_format(metrics['NDCG@10'])} | {support['evaluated_users_total']} |"
            )
        stat = nested["statistical_tests"]["hybrid_selected_vs_cf_only"]
        lines.extend([
            "",
            "### Confronto statistico Hybrid vs CF",
            "",
            f"- Differenza RMSE Hybrid − CF: {stat['mean_rmse_difference_a_minus_b']:.6f}; IC 95% [{stat['difference_ci95'][0]:.6f}, {stat['difference_ci95'][1]:.6f}].",
            f"- paired t-test: t={stat['t_statistic']:.4f}, p={stat['t_p_value']:.6g}; Wilcoxon: W={stat['wilcoxon_statistic']:.4f}, p={stat['wilcoxon_p_value']:.6g}.",
            "",
        ])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(results: dict, destination: Path) -> None:
    config = results["config"]
    lines = [
        "# Relazione di Progetto: Sistema di Raccomandazione Ibrido per SII",
        "## Valutazione riproducibile mediante Nested Cross-Validation",
        "",
        "**Corso:** Sistemi Intelligenti per Internet (SII)  ",
        "**Anno Accademico:** 2025/2026  ",
        "**Studente:** Cristian Perniconi — Matricola: 566835",
        "",
        "---",
        "",
        "## Abstract",
        "",
        "Il progetto confronta Collaborative Filtering (CF), Content-Based Filtering (CB) e una fusione ibrida pesata sui dataset MovieLens 100k e MovieLens 1M. Per evitare che il test influisca sulla scelta degli iperparametri, ogni risultato finale è stimato con una nested cross-validation: cinque fold esterni costituiscono test indipendenti e, per ciascuno, tre fold interni selezionano `alpha` e la soglia del profilo CB `theta`. Il modello ibrido usa una sola definizione sia per RMSE/MAE sia per il ranking: `clip(alpha·r_CF + (1-alpha)·r_CB, 1, 5)`. Le tabelle sono generate dal file `report/experiment_results.json`; pertanto ogni valore riportato è riconducibile ai singoli fold e alla configurazione che l’ha prodotto.",
        "",
        "## 1. Protocollo sperimentale",
        "",
        f"- **Split:** nested K-Fold con {config['outer_splits']} fold esterni e {config['inner_splits']} fold interni, con seed `{config['random_state']}`.",
        f"- **Tuning interno:** `alpha ∈ {config['alpha_grid']}` e `theta ∈ {config['theta_grid']}`; il criterio di selezione è l’RMSE medio sui validation fold interni.",
        "- **Valutazione finale:** dopo il tuning, CF e CB vengono riaddestrati sull’intero training outer e misurati una sola volta sul test outer.",
        f"- **Top-N:** candidati = catalogo meno item osservati dall’utente nel training outer; rilevante = rating nel test outer ≥ {config['ranking_relevance_threshold']}; massimo {config['ranking_max_users']} utenti idonei per fold, campionati deterministicamente quando necessario.",
        f"- **Cold item:** item con al più {config['cold_item_max_train_ratings']} rating nel training outer; il filtro non consulta mai le frequenze del test.",
        "- **Statistiche:** paired t-test e Wilcoxon sono applicati ai cinque RMSE dei test outer appaiati. Con cinque osservazioni il Wilcoxon bilaterale ha potenza limitata e viene interpretato con cautela.",
        "",
        "## 2. Modelli",
        "",
        "Il CF usa TruncatedSVD sulla matrice sparsa dei residui con bias marginali. Il CB costruisce un profilo utente dai film positivi, usando 19 indicatori di genere armonizzati e l’anno normalizzato.",
        "",
        "### 2.1 Significato e ruolo della soglia $\\theta$ nel Content-Based",
        "",
        "La soglia $\\theta$ stabilisce quali rating dell’utente vengono considerati positivi per costruire il profilo contenutistico. Formalmente, il profilo viene calcolato sui soli film appartenenti all’insieme $R_u^+(\\theta)=\\{i:r_{u,i}\\geq\\theta\\}$. Nel progetto sono confrontate due configurazioni: $\\theta=3.0$, che include i film valutati 3, 4 o 5, e $\\theta=4.0$, che include soltanto i film valutati 4 o 5.",
        "",
        "$$\\mathbf{p}_u(\\theta)=\\frac{\\sum_{i\\in R_u^+(\\theta)}r_{u,i}\\mathbf{v}_i}{\\sum_{i\\in R_u^+(\\theta)}r_{u,i}}.$$",
        "",
        "La scelta di $\\theta$ modifica direttamente la composizione del profilo: con $\\theta=3.0$ il profilo contiene più film ma anche preferenze moderate; con $\\theta=4.0$ il profilo è più selettivo e rappresenta soprattutto i gusti forti dell’utente. Di conseguenza cambiano la cosine similarity, il rating CB e la correzione che il CB può applicare alla predizione CF. Se nessun rating supera la soglia, l’implementazione usa lo storico disponibile dell’utente come fallback.",
        "",
        "Nel modello ibrido, $\\theta$ non è un peso indipendente da $\\alpha$: determina il contenuto del segnale CB che viene poi pesato da $(1-\\alpha)$. Con $\\alpha=0.9$, la scelta di $\\theta$ influenza il 10% della predizione ibrida. Un profilo più selettivo può migliorare la coerenza semantica della correzione, ma può anche ridurne la copertura; per questo $\\theta$ viene selezionato nella CV interna insieme ad $\\alpha$, senza usare l’outer test.",
        "",
        "La nested CV ha selezionato $\\theta=4.0$ in tutti i cinque outer fold di entrambi i dataset. Questo risultato indica che, con le feature disponibili (generi e anno), includere soltanto i film valutati almeno 4 produce una correzione CB più utile rispetto a includere anche i rating pari a 3.",
        "",
        "Il modello primario combina direttamente predizioni di rating già limitate alla scala MovieLens:",
        "",
        "$$\\hat{r}^{\\mathrm{Hybrid}}_{u,i}=\\operatorname{clip}\\left(\\alpha\\hat{r}^{\\mathrm{CF}}_{u,i}+(1-\\alpha)\\hat{r}^{\\mathrm{CB}}_{u,i},\\,1,\\,5\\right).$$",
        "",
        "Questa scelta elimina la normalizzazione min-max dipendente dal batch: una predizione e il suo ordinamento non cambiano se il batch di candidati viene riordinato o suddiviso.",
        "",
        "## 3. Risultati",
        "",
    ]
    for dataset_name, dataset_result in results["datasets"].items():
        nested = dataset_result["nested_5x3"]
        eda = dataset_result["eda"]
        lines.extend([
            f"### 3.{1 if dataset_name == 'ml-100k' else 2} {_dataset_label(dataset_name)}",
            "",
            f"Il dataset contiene {eda['num_ratings']:,} rating, {eda['num_users']:,} utenti e {eda['num_items']:,} film; la sparsità della matrice è {eda['sparsity'] * 100:.2f}%.",
            "",
            "| Modello | RMSE (mean ± std) | MAE (mean ± std) |",
            "|---|---:|---:|",
        ])
        for name, metrics in nested["pointwise"].items():
            lines.append(f"| {name} | {_format(metrics['rmse'])} | {_format(metrics['mae'])} |")
        lines.extend([
            "",
            f"![RMSE nested](figures/{dataset_name}/nested_pointwise_rmse.png)",
            "",
            f"![Parametri scelti internamente](figures/{dataset_name}/nested_selected_parameters.png)",
            "",
            "#### Ranking Top-N",
            "",
            "| Modello | Precision@5 | NDCG@5 | Precision@10 | NDCG@10 | Supporto utenti |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for name, ranking in nested["top_n"].items():
            metrics, support = ranking["metrics"], ranking["support"]
            lines.append(
                f"| {name} | {_format(metrics['Precision@5'])} | {_format(metrics['NDCG@5'])} | "
                f"{_format(metrics['Precision@10'])} | {_format(metrics['NDCG@10'])} | {support['evaluated_users_total']} |"
            )
        cold = nested["cold_start"]
        lines.extend(["", "#### Cold-item", "", f"Supporto complessivo: {cold['support']['test_ratings_total']} rating test; {cold['support']['cold_items_total']} occorrenze di item cold conteggiate nei fold, di cui {cold['support'].get('zero_train_test_items_total', 0)} con zero interazioni nel training (fold validi: {cold['support']['valid_outer_folds']}).", "", "| Modello | Cold RMSE (mean ± std) | Cold MAE (mean ± std) |", "|---|---:|---:|"])
        for name, metrics in cold.items():
            if name != "support":
                lines.append(f"| {name} | {_format(metrics['rmse'])} | {_format(metrics['mae'])} |")
        stat = nested["statistical_tests"]["hybrid_selected_vs_cf_only"]
        lines.extend([
            "",
            "#### Significatività e rilevanza pratica",
            "",
            f"Per Hybrid selezionato contro CF-only, la differenza media RMSE (Hybrid − CF) è {stat['mean_rmse_difference_a_minus_b']:.6f}, con IC 95% [{stat['difference_ci95'][0]:.6f}, {stat['difference_ci95'][1]:.6f}]. Il paired t-test restituisce `t={stat['t_statistic']:.4f}`, `p={stat['t_p_value']:.6g}`; il Wilcoxon bilaterale restituisce `W={stat['wilcoxon_statistic']:.4f}`, `p={stat['wilcoxon_p_value']:.6g}`. Un’eventuale significatività deve essere letta insieme alla dimensione dell’effetto, non come prova autonoma di un grande beneficio pratico.",
            "",
        ])
    dataset_summaries = []
    for dataset_name, dataset_result in results["datasets"].items():
        nested = dataset_result["nested_5x3"]
        pointwise = nested["pointwise"]
        ranking = nested["top_n"]
        cold = nested["cold_start"]
        cf_rmse = pointwise["cf_only"]["rmse"]["mean"]
        cb_rmse = pointwise["cb_only"]["rmse"]["mean"]
        hybrid_rmse = pointwise["hybrid_selected"]["rmse"]["mean"]
        cf_p5 = ranking["cf_only"]["metrics"]["Precision@5"]["mean"]
        cb_p5 = ranking["cb_only"]["metrics"]["Precision@5"]["mean"]
        hybrid_p5 = ranking["hybrid_selected"]["metrics"]["Precision@5"]["mean"]
        cf_ndcg5 = ranking["cf_only"]["metrics"]["NDCG@5"]["mean"]
        hybrid_ndcg5 = ranking["hybrid_selected"]["metrics"]["NDCG@5"]["mean"]
        selected_parameters = nested["selected_parameters"]
        selected_alpha_values = [parameters["alpha"] for parameters in selected_parameters]
        selected_theta_values = [parameters["theta"] for parameters in selected_parameters]
        dataset_summaries.append({
            "label": _dataset_label(dataset_name),
            "cf_rmse": cf_rmse,
            "cb_rmse": cb_rmse,
            "hybrid_rmse": hybrid_rmse,
            "cf_vs_cb_rmse_gap": cb_rmse - cf_rmse,
            "hybrid_vs_cf_rmse_delta": hybrid_rmse - cf_rmse,
            "hybrid_vs_cf_rmse_pct": 100 * (hybrid_rmse - cf_rmse) / cf_rmse,
            "cf_p5": cf_p5,
            "cb_p5": cb_p5,
            "hybrid_p5": hybrid_p5,
            "hybrid_vs_cf_p5_pct": 100 * (hybrid_p5 - cf_p5) / cf_p5 if cf_p5 else 0.0,
            "cf_ndcg5": cf_ndcg5,
            "hybrid_ndcg5": hybrid_ndcg5,
            "hybrid_vs_cf_ndcg5_pct": 100 * (hybrid_ndcg5 - cf_ndcg5) / cf_ndcg5 if cf_ndcg5 else 0.0,
            "cold_cf": cold["cf_only"]["rmse"]["mean"],
            "cold_hybrid": cold["hybrid_selected"]["rmse"]["mean"],
            "cold_half": cold["hybrid_alpha_0_5"]["rmse"]["mean"],
            "cold_user": cold["user_mean"]["rmse"]["mean"],
            "selected_alpha_values": selected_alpha_values,
            "selected_theta_values": selected_theta_values,
        })

    lines.extend([
        "## 4. Discussione: contributo distinto di CF e CB",
        "",
        "La nested CV separa il contributo dei due rami perché misura CF-only, CB-only e Hybrid sugli stessi outer test fold. Il risultato non va letto come una semplice gara fra modelli: CF e CB forniscono segnali diversi, con utilità che dipende da scala del dataset, metrica e disponibilità di interazioni.",
        "",
        "### 4.1 Contributo del Collaborative Filtering",
        "",
        "Il CF è il **motore predittivo principale**. I bias utente/item e i fattori latenti appresi dalla TruncatedSVD sfruttano le correlazioni collettive tra valutazioni: per questo il CF ricostruisce meglio il rating individuale rispetto ai soli metadati descrittivi. Il suo vantaggio aumenta quando ogni item e utente dispone di più interazioni, perché i fattori latenti sono stimati con evidenza più stabile.",
        "",
    ])
    for summary in dataset_summaries:
        lines.append(
            f"- **{summary['label']}:** CF-only ottiene RMSE `{summary['cf_rmse']:.4f}`, contro `{summary['cb_rmse']:.4f}` di CB-only (vantaggio assoluto CF di `{summary['cf_vs_cb_rmse_gap']:.4f}`). Questo quantifica che generi e anno non sostituiscono il segnale collaborativo per la predizione puntuale."
        )
    lines.extend([
        "",
        "Il CF, tuttavia, non possiede una nozione semantica esplicita di contenuto: due film possono essere vicini nello spazio latente anche se non condividono generi, e un item con poche interazioni ha fattori meno affidabili. Questi limiti sono lo spazio in cui il CB può essere complementare.",
        "",
        "### 4.2 Contributo del Content-Based Filtering",
        "",
        "Il CB costruisce un profilo dai film positivamente valutati e confronta tale profilo con i vettori item basati su generi e anno. Il suo apporto è quindi **semantico e locale**: favorisce item coerenti con preferenze esplicite anche quando il segnale collaborativo è debole. Non è però competitivo come predittore di rating autonomo, perché 20 feature poco ricche non rappresentano registi, attori, temi, tag o caratteristiche narrative.",
        "",
    ])
    for summary in dataset_summaries:
        lines.append(
            f"- **{summary['label']}:** CB-only ottiene Precision@5 `{summary['cb_p5']:.4f}`, rispetto a `{summary['cf_p5']:.4f}` del CF. Il confronto mostra che il CB può ordinare item rilevanti in alcuni contesti, pur avendo un RMSE molto più elevato; ranking e predizione del rating sono quindi obiettivi distinti."
        )
    lines.extend([
        "",
        "### 4.3 Contributo di CF e CB *dentro* la predizione ibrida",
        "",
        "Nel modello ibrido CF e CB non contribuiscono allo stesso modo: la predizione usa direttamente i due rating sulla medesima scala MovieLens,",
        "",
        "$$\\hat{r}^{\\mathrm{Hybrid}}_{u,i}=\\alpha\\hat{r}^{\\mathrm{CF}}_{u,i}+(1-\\alpha)\\hat{r}^{\\mathrm{CB}}_{u,i}.$$",
        "",
        "Riscrivendo la formula rispetto alla previsione CF si ottiene `r_Hybrid − r_CF = (1−alpha) · (r_CB − r_CF)`. Il CF fornisce quindi la previsione di base; il CB sposta tale previsione solo in proporzione al suo peso e al disaccordo fra i due modelli. Se CF e CB concordano, il CB non modifica il risultato; se il CB assegna un rating maggiore/minore, applica una correzione positiva/negativa.",
        "",
    ])
    for summary in dataset_summaries:
        alpha_values = summary["selected_alpha_values"]
        theta_values = summary["selected_theta_values"]
        alpha_counts = ", ".join(
            f"{alpha:.1f} ({alpha_values.count(alpha)}/{len(alpha_values)} fold)"
            for alpha in sorted(set(alpha_values))
        )
        theta_counts = ", ".join(
            f"{theta:.1f} ({theta_values.count(theta)}/{len(theta_values)} fold)"
            for theta in sorted(set(theta_values))
        )
        cb_weight_mean = 1.0 - sum(alpha_values) / len(alpha_values)
        lines.append(
            f"- **{summary['label']}:** la CV interna ha selezionato `alpha={alpha_counts}` e `theta={theta_counts}`. In pratica il CF pesa in media `{1.0 - cb_weight_mean:.0%}` e il CB `{cb_weight_mean:.0%}`; `theta=4.0` indica che il profilo CB usa soltanto film valutati almeno 4 dall’utente."
        )
    lines.extend([
        "",
        "Nel caso osservato, `alpha=0.9` in tutti i fold significa che l’ibrido non fa una media paritaria: è un **CF dominante con correzione CB del 10%**. In formule, `r_Hybrid = r_CF + 0.1 · (r_CB − r_CF)`. Il CB non può quindi ribaltare da solo una predizione CF molto diversa, ma può modificare l’ordine di item con score CF vicini; questo è precisamente il meccanismo di *tie-breaking* che può incidere sul Top-N più che sull’RMSE globale.",
        "",
        "Il ruolo di `theta=4.0` è altrettanto importante: la correzione CB viene calcolata da un profilo costruito solo con i titoli che l’utente ha apprezzato molto. Il 10% CB non rappresenta dunque un generico segnale di genere, ma una piccola spinta verso film simili alle preferenze forti dell’utente.",
        "",
        "#### Effetto misurato della correzione CB",
        "",
    ])
    for summary in dataset_summaries:
        direction = "migliora" if summary["hybrid_vs_cf_p5_pct"] >= 0 else "peggiora"
        lines.append(
            f"- **{summary['label']}:** rispetto alla base CF, la correzione CB del peso selezionato porta l’RMSE da `{summary['cf_rmse']:.4f}` a `{summary['hybrid_rmse']:.4f}` ({summary['hybrid_vs_cf_rmse_delta']:+.4f}; {summary['hybrid_vs_cf_rmse_pct']:+.2f}%). Nel ranking, {direction} Precision@5 da `{summary['cf_p5']:.4f}` a `{summary['hybrid_p5']:.4f}` ({summary['hybrid_vs_cf_p5_pct']:+.1f}%) e NDCG@5 da `{summary['cf_ndcg5']:.4f}` a `{summary['hybrid_ndcg5']:.4f}` ({summary['hybrid_vs_cf_ndcg5_pct']:+.1f}%)."
        )
    lines.extend([
        "",
        "Il contributo CB è quindi **condizionale**. Su ML-1M, il 10% di correzione semantica migliora il ranking perché il CF produce una base latente già solida e il CB aiuta a discriminare candidati vicini. Su ML-100k, la stessa correzione migliora leggermente RMSE ma peggiora il Top-N: la preferenza di genere non si allinea sempre con gli item rilevanti nel test. Il risultato mostra che il CB contribuisce come regolatore/tie-breaker, non come sostituto del CF; un peso scelto per RMSE non è necessariamente il peso ottimale per ranking.",
        "",
        "### 4.4 Cold-item e decisione operativa",
        "",
        "Nel cold-item, un peso CB più alto può attenuare l’incertezza del CF, ma i metadati disponibili non sono sufficienti a superare una stima robusta della media personale dell’utente. I risultati riportano sia l’ibrido selezionato per RMSE globale sia `alpha=0.5` come controllo esplorativo, senza presentare quest’ultimo come configurazione ottimizzata.",
        "",
    ])
    for summary in dataset_summaries:
        lines.append(
            f"- **{summary['label']}:** Cold RMSE CF=`{summary['cold_cf']:.4f}`, Hybrid selezionato=`{summary['cold_hybrid']:.4f}`, Hybrid `alpha=0.5`=`{summary['cold_half']:.4f}`, User Mean=`{summary['cold_user']:.4f}`. L’ibrido migliora il CF, ma User Mean resta il riferimento più stabile nello scenario estremo."
        )
    lines.extend([
        "",
        "**Sintesi:** usare il CF come base per l’accuratezza dei rating; usare il CB come complemento per aumentare la coerenza semantica, specialmente nel ranking di cataloghi più ricchi e per attenuare il cold-item. Per un sistema operativo, una possibile evoluzione è un peso adattivo: più CF per item popolari, più CB per item con poca evidenza, e selezione di `alpha` separata per RMSE e per ranking quando l’obiettivo primario è Top-N.",
        "",
        "### 4.5 Limiti metodologici",
        "",
        "La nested CV rimuove l’ottimismo dovuto alla scelta di `alpha` e `theta` sugli stessi fold di test. Rimangono limiti espliciti: lo split è casuale per interazioni e misura soprattutto la predizione di rating mancanti, non un vero scenario temporale; le feature CB sono limitate a generi e anno; il ranking ML-1M usa un campione deterministico di utenti per contenere i costi. Il cold-item valuta la scarsità di item, non il cold-user, e una baseline User Mean può restare competitiva perché non dipende dalla stima dell’item.",
        "",
        "## 5. Riproducibilità",
        "",
        "Eseguire `python run_pipeline.py` dopo l’installazione di `requirements.txt`. La pipeline rigenera il manifest `report/experiment_manifest.json`, i risultati versionati `report/experiment_results.json`, il riepilogo `report/nested_results.md`, figure e questa relazione. Il JSON conserva seed, ambiente, configurazioni scelte e metriche per fold.",
        "",
        "## Bibliografia",
        "",
        "1. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.",
        "2. Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix factorization techniques for recommender systems*. Computer, 42(8), 30–37.",
        "3. GroupLens Research. *MovieLens Datasets*. https://grouplens.org/datasets/movielens/",
    ])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("ml-100k", "ml-1m"), default=("ml-100k", "ml-1m"))
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--ranking-max-users", type=int, default=500)
    parser.add_argument("--skip-eda", action="store_true", help="Non rigenerare i grafici EDA.")
    parser.add_argument(
        "--render-existing-results",
        action="store_true",
        help="Rigenera riepilogo Markdown e relazione dal JSON esistente, senza addestrare modelli.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.render_existing_results:
        results_path = REPORT_DIR / "experiment_results.json"
        if not results_path.exists():
            raise FileNotFoundError("Manca report/experiment_results.json: eseguire prima la pipeline completa.")
        results = json.loads(results_path.read_text(encoding="utf-8"))
        write_markdown_summary(results, REPORT_DIR / "nested_results.md")
        write_report(results, REPORT_DIR / "Relazione_Progetto_SII_Recommender_Ibrido.md")
        print(f"Artefatti testuali rigenerati da: {results_path}")
        return

    config = ExperimentConfig(
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        ranking_max_users=args.ranking_max_users,
    )
    manifest = {
        "schema_version": "2.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "datasets": list(args.datasets),
        "command": "python run_pipeline.py",
    }
    (REPORT_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dataset_results = {
        dataset_name: run_dataset_pipeline(dataset_name, config, generate_eda=not args.skip_eda)
        for dataset_name in args.datasets
    }
    results = {
        "schema_version": "2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "datasets": dataset_results,
    }
    (REPORT_DIR / "experiment_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown_summary(results, REPORT_DIR / "nested_results.md")
    write_report(results, REPORT_DIR / "Relazione_Progetto_SII_Recommender_Ibrido.md")
    print(f"\nRisultati salvati in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
