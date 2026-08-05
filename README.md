# SII Hybrid Recommender System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Progettazione, sviluppo e valutazione sperimentale di un **Sistema di Raccomandazione Ibrido** (Collaborative Filtering + Content-Based Filtering) applicato al dataset **MovieLens 100k** per il corso di *Sistemi Intelligenti per Internet (SII)*.

---

## 📌 Architettura del Progetto

Il sistema combina due approcci complementari mediante fusione pesata:
1. **Collaborative Filtering (CF):** Modellato tramite Matrix Factorization (SVD / Biased MF) per apprendere pattern di preferenza latenti.
2. **Content-Based Filtering (CB):** Basato su profilazione utente e Cosine Similarity applicata ai generi e all'anno di uscita dei film.
3. **Fusione Ibrida Pesata:** $S_{\text{Hybrid}}(u, i) = \alpha \cdot S_{\text{CF}}(u, i) + (1 - \alpha) \cdot S_{\text{CB}}(u, i)$.

---

## 📁 Struttura del Repository

```text
├── data/                     # Dataset MovieLens 100k (download automatico)
├── notebooks/                # Jupyter Notebooks di analisi ed esperimenti
│   ├── 01_EDA.ipynb          # Analisi Esplorativa dei Dati
│   ├── 02_models.ipynb       # Training e predizioni singoli modelli
│   └── 03_experiments.ipynb  # Pipeline sperimentale completa
├── src/                      # Moduli Python del sistema
│   ├── data_loader.py        # Caricamento dataset e costruzione feature matrix
│   ├── baselines.py          # Baseline naive (Global, User, Item Mean)
│   ├── collaborative.py      # Modello Collaborative Filtering (SVD / Biased MF)
│   ├── content_based.py      # Modello Content-Based (Profilo Utente + Cosine Sim)
│   ├── hybrid.py             # Modello Ibrido con fusione pesata alpha
│   ├── evaluation.py         # Metriche predittive (RMSE, MAE), Top-N (Precision, NDCG), CV, Test Statistici
│   └── utils.py              # Visualizzazioni grafiche e utility
├── report/                   # Relazione finale e grafici ad alta risoluzione
│   ├── figures/              # Grafici generati automaticamente
│   ├── experiment_results.json
│   └── Relazione_Progetto_SII_Recommender_Ibrido.md
├── run_pipeline.py           # Master script di esecuzione esperimenti
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

### 2. Esecuzione Esperimenti Completi
Per scaricare il dataset, eseguire la 5-Fold Cross-Validation, la Sensitivity Analysis su $\alpha$, i test di ranking e Cold Start, e generare tutti i grafici nella cartella `report/figures/`:

```bash
python run_pipeline.py
```

### 3. Notebooks Interattivi
Puoi esplorare i singoli componenti aprendo i Jupyter Notebooks presenti nella cartella `notebooks/`:
- `01_EDA.ipynb`: Analisi esplorativa dei voti e dei generi.
- `02_models.ipynb`: Confronto preliminare dei modelli.
- `03_experiments.ipynb`: Esecuzione controllata degli esperimenti.

---

## 📊 Risultati Principali

- **RMSE Modello Ibrido ($\alpha^*=0.8$):** `0.9241 ± 0.0028` (rispetto a `0.9385` del solo CF e `1.0582` del solo CB).
- **Miglioramento su Cold Start:** L'approccio ibrido riduce drasticamente l'errore rispetto al filtro collaborativo quando gli item hanno pochissime valutazioni.
- **Significatività Statistica:** Test di Wilcoxon $p < 0.01$ conferma il valore aggiunto dell'ibridazione.
