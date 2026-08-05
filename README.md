# SII Hybrid Recommender System (Multi-Dataset Benchmark)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Progettazione, sviluppo e valutazione sperimentale di un **Sistema di Raccomandazione Ibrido** (Collaborative Filtering + Content-Based Filtering) applicato in forma integrale (nessun sottocampionamento) sui benchmark **MovieLens 100k** e **MovieLens 1M** per il corso di *Sistemi Intelligenti per Internet (SII)*.

---

## 📌 Architettura del Progetto

Il sistema combina due approcci complementari mediante fusione pesata:
1. **Collaborative Filtering (CF):** Matrix Factorization vettorizzata (SciPy CSR + TruncatedSVD) per apprendere lo spazio latente utente-item.
2. **Content-Based Filtering (CB):** Profilazione utente con filtro di soglia $\theta \in \{3.0, 4.0\}$ e Cosine Similarity vettorizzata sui generi cinematografici e anno di uscita.
3. **Fusione Ibrida Pesata:** $S_{\text{Hybrid}}(u, i) = \alpha \cdot S_{\text{CF}}(u, i) + (1 - \alpha) \cdot S_{\text{CB}}(u, i)$.

---

## 📁 Struttura del Repository

```text
├── data/                     # Data directory (ML-100k e ML-1M scaricati automaticamente)
│   ├── ml-100k/
│   └── ml-1m/
├── notebooks/                # Jupyter Notebooks di analisi ed esperimenti
│   ├── 01_EDA.ipynb          # Analisi Esplorativa dei Dati
│   ├── 02_models.ipynb       # Training e predizioni singoli modelli
│   └── 03_experiments.ipynb  # Pipeline sperimentale completa
├── src/                      # Moduli Python del sistema
│   ├── data_loader.py        # Caricamento dataset (ML-100k & ML-1M) e feature matrix
│   ├── baselines.py          # Baseline naive (Global, User, Item Mean)
│   ├── collaborative.py      # Collaborative Filtering vettorizzato (CSR + TruncatedSVD)
│   ├── content_based.py      # Content-Based vettorizzato (Profilo Utente + Cosine Sim)
│   ├── hybrid.py             # Modello Ibrido con fusione pesata alpha
│   ├── evaluation.py         # Metriche predittive (RMSE, MAE), Top-N (Precision, NDCG), CV, Test Statistici
│   └── utils.py              # Visualizzazioni grafiche e plot comparativi side-by-side
├── report/                   # Relazione finale e grafici ad alta risoluzione
│   ├── figures/              # Figures generate per dataset e comparative
│   │   ├── ml-100k/
│   │   ├── ml-1m/
│   │   └── comparison/
│   ├── experiment_results.json
│   └── Relazione_Progetto_SII_Recommender_Ibrido.md
├── run_pipeline.py           # Master script multi-dataset per eseguire tutti gli esperimenti
├── requirements.txt          # Dipendenze del progetto
└── README.md
```

---

## 🚀 Guida all'Uso

### 1. Installazione Dipendenze
```bash
python -m venv .venv
# Su Windows:
.venv\Scripts\activate
# Su Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Esecuzione Esperimenti Multi-Dataset Completi
Per scaricare sia MovieLens 100k che MovieLens 1M (1.000.209 rating), eseguire 5-Fold Cross Validation su entrambi i dataset senza campionamento, calcolare la Sensitivity Analysis su $\alpha$, test Top-N, Cold Start e test di significatività statistica:

```bash
python run_pipeline.py
```

I grafici comparativi side-by-side verranno automaticamente salvati in `report/figures/comparison/` e la relazione finale in `report/Relazione_Progetto_SII_Recommender_Ibrido.md`.

---

## 📊 Risultati Principali a Confronto

| Dataset | Volume Rating | Sparsità | Baselines User Mean RMSE | Optimal $\alpha^*$ | Hybrid RMSE (5-Fold CV) | Top-N Precision@5 | Top-N NDCG@5 |
|---|---|---|---|---|---|---|---|
| **MovieLens 100k** | 100.000 | 93.70% | 1.0419 | 0.9 | **0.9445 ± 0.0022** | 0.0057 | 0.0071 |
| **MovieLens 1M** | 1.000.209 | 95.74% | 1.0355 | 0.9 | **0.8971 ± 0.0024** | **0.0235** (+39%) | **0.0269** (+30%) |

### Takeaways Chiave:
1. **Riduzione dell'Errore con la Scala:** Con 1 milione di valutazioni (ML-1M), la Matrix Factorization (SVD) riduce l'RMSE da `0.9445` a **`0.8971`**, beneficiando della maggiore densità d'interazione per utente.
2. **Qualità del Ranking nei Top-N:** L'ibridazione ($\alpha^*=0.9$) su MovieLens 1M migliora la Precision@5 del **+39%** e l'NDCG@5 del **+30%** rispetto al filtro collaborativo puro.
3. **Mitigazione Cold Start:** Nei film con meno di 3 valutazioni, la fusione pesata ($\alpha=0.5$) riduce l'RMSE rispetto al CF puro.
