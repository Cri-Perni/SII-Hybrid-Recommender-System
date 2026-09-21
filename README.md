# SII Hybrid Recommender System

Sistema di raccomandazione ibrido per MovieLens 100k e MovieLens 1M, sviluppato per il corso di *Sistemi Intelligenti per Internet*. Il progetto confronta Collaborative Filtering (CF), Content-Based Filtering (CB) e fusione pesata CF+CB con un protocollo di valutazione riproducibile e senza leakage.

## Protocollo sperimentale

La pipeline predefinita usa una **nested 5×3 Cross-Validation**:

- 5 fold esterni: valutazione finale indipendente di RMSE, MAE, Precision@K, NDCG@K e cold-item.
- 3 fold interni, costruiti solo dal training fold esterno: selezione di `alpha ∈ {0.0, …, 1.0}` e `theta ∈ {3.0, 4.0}` minimizzando RMSE.
- Dopo il tuning, i modelli vengono riaddestrati sull’intero training esterno e valutati una sola volta sul test esterno.
- L’ibrido usa sempre la stessa formula, sia per l’errore di rating sia per il ranking:

\[
\hat r^{Hybrid}_{u,i}=\operatorname{clip}(\alpha\hat r^{CF}_{u,i}+(1-\alpha)\hat r^{CB}_{u,i}, 1, 5).
\]

Il ranking Top-N considera come candidati tutti gli item del catalogo non presenti nello storico dell’utente nel training fold; gli item rilevanti sono quelli del test fold con rating `≥ 4`. Per ML-1M, al massimo 500 utenti idonei vengono campionati deterministicamente per ogni fold, con seed e supporto salvati nei risultati. Un item è cold se ha al massimo due rating nel training fold esterno.

## Struttura

```text
src/
  experiment_config.py   # Configurazione immutabile, seed e griglie
  nested_experiment.py   # Nested CV, tuning interno e valutazioni outer
  collaborative.py       # CF: bias + TruncatedSVD su CSR
  content_based.py       # CB: profilo utente e cosine similarity
  hybrid.py              # Fusione diretta dei rating CF/CB
  evaluation.py          # Metriche, Top-N e statistiche paired
run_pipeline.py          # Entry point multi-dataset
report/
  experiment_manifest.json  # Configurazione della run
  experiment_results.json   # Risultati versionati per fold
  nested_results.md         # Tabelle generate dai risultati
  Relazione_Progetto_SII_Recommender_Ibrido.md
```

## Installazione

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Esecuzione

Rigenera manifest, risultati, tabelle, figure e relazione:

```cmd
python run_pipeline.py
```

Opzioni utili:

```cmd
:: Smoke test integrato su ML-100k
python run_pipeline.py --datasets ml-100k --outer-splits 2 --inner-splits 2 --ranking-max-users 25 --skip-eda

:: Eseguire solo ML-1M con protocollo 5x3 predefinito
python run_pipeline.py --datasets ml-1m

:: Rigenerare relazione e tabelle dal JSON senza riaddestrare
python run_pipeline.py --render-existing-results
```

I risultati finali non vanno trascritti manualmente: `report/experiment_results.json` è la sorgente canonica; `report/nested_results.md` e la relazione sono generati dalla stessa esecuzione.

## Risultati dell’esecuzione verificata

Tutti i risultati riportati di seguito derivano dalla pipeline riproducibile con seed `42` e protocollo **nested 5×3 Cross-Validation** (sorgente canonica: [`report/experiment_results.json`](file:///D:/Users/yyi30043/Desktop/sii_raccomandation%20system/report/experiment_results.json), riepilogata in [`report/nested_results.md`](file:///D:/Users/yyi30043/Desktop/sii_raccomandation%20system/report/nested_results.md) e approfondita nella [Relazione di Progetto](file:///D:/Users/yyi30043/Desktop/sii_raccomandation%20system/report/Relazione_Progetto_SII_Recommender_Ibrido.md)).

### Panoramica comparativa

| Dataset | Volume Rating | Utenti | Film | Sparsità | Optimal $(\alpha^*, \theta^*)$ | CF RMSE | Hybrid RMSE | CF Prec@5 | Hybrid Prec@5 | CF NDCG@5 | Hybrid NDCG@5 |
|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| **MovieLens 100k** | 100,000 | 943 | 1,682 | 93.70% | $\alpha=0.9, \theta=4.0$ (5/5 fold) | 0.9495 ± 0.0029 | **0.9440 ± 0.0027** | 0.0165 ± 0.0035 | 0.0058 ± 0.0013 | 0.0214 ± 0.0041 | 0.0075 ± 0.0018 |
| **MovieLens 1M** | 1,000,209 | 6,040 | 3,883 | 95.74% | $\alpha=0.9, \theta=4.0$ (5/5 fold) | 0.8986 ± 0.0029 | **0.8966 ± 0.0027** | 0.0214 ± 0.0028 | **0.0263 ± 0.0036** | 0.0254 ± 0.0028 | **0.0295 ± 0.0042** |

---

### Dettaglio MovieLens 100k (Durata: 34.8 s)

#### 1. Accuratezza Pointwise
Valutazione su tutti i rating dei 5 outer test fold:

| Modello | RMSE (mean ± std) | 95% CI RMSE | MAE (mean ± std) | 95% CI MAE |
|---|---:|:---:|---:|:---:|
| `global_mean` | 1.1257 ± 0.0060 | [1.1183, 1.1332] | 0.9447 ± 0.0056 | [0.9377, 0.9516] |
| `user_mean` | 1.0419 ± 0.0048 | [1.0359, 1.0478] | 0.8349 ± 0.0051 | [0.8286, 0.8413] |
| `item_mean` | 1.0248 ± 0.0045 | [1.0192, 1.0304] | 0.8174 ± 0.0051 | [0.8111, 0.8237] |
| `cf_only` | 0.9495 ± 0.0029 | [0.9458, 0.9531] | 0.7428 ± 0.0025 | [0.7397, 0.7458] |
| `cb_only` | 1.2798 ± 0.0018 | [1.2776, 1.2820] | 1.0182 ± 0.0026 | [1.0149, 1.0214] |
| **`hybrid_selected`** | **0.9440 ± 0.0027** | [0.9406, 0.9473] | **0.7417 ± 0.0026** | [0.7385, 0.7449] |

#### 2. Ranking Top-N
Candidati: catalogo meno item osservati nel training fold; rilevanti: rating di test $\ge 4.0$; 2,500 utenti valutati (500 per fold su 4,611 idonei; media candidati per utente: 1595.4):

| Modello | Precision@5 | NDCG@5 | Precision@10 | NDCG@10 | Utenti valutati |
|---|---:|---:|---:|---:|---:|
| `cf_only` | 0.0165 ± 0.0035 | 0.0214 ± 0.0041 | 0.0106 ± 0.0021 | 0.0162 ± 0.0031 | 2500 |
| `cb_only` | 0.0181 ± 0.0019 | 0.0196 ± 0.0025 | 0.0150 ± 0.0016 | 0.0193 ± 0.0026 | 2500 |
| `hybrid_selected` | 0.0058 ± 0.0013 | 0.0075 ± 0.0018 | 0.0074 ± 0.0017 | 0.0085 ± 0.0020 | 2500 |

#### 3. Cold-item ($\le 2$ rating nel training fold)
Supporto: 679 rating di test; 504 occorrenze cold (di cui 150 con 0 rating nel training fold):

| Modello | Cold RMSE (mean ± std) | Cold MAE (mean ± std) |
|---|---:|---:|
| `cf_only` | 1.3348 ± 0.0906 | 1.0186 ± 0.0929 |
| `cb_only` | 1.3451 ± 0.0926 | 1.0971 ± 0.1102 |
| `user_mean` | 1.0852 ± 0.0586 | 0.8635 ± 0.0406 |
| `hybrid_selected` ($\alpha=0.9$) | 1.2674 ± 0.0906 | 0.9801 ± 0.0817 |
| `hybrid_alpha_0_5` ($\alpha=0.5$, esplorativo) | 1.1405 ± 0.0878 | 0.9150 ± 0.0643 |

#### 4. Test di significatività statistica (Hybrid vs CF-only)
- **Differenza media RMSE (Hybrid − CF):** `-0.005496` (IC 95%: `[-0.006549, -0.004443]`, miglioramento relativo -0.58%).
- **Paired t-test:** $t = -14.4911$, $p = 0.000132$ (statisticamente significativo con $p < 0.05$).
- **Wilcoxon signed-rank test:** $W = 0.0000$, $p = 0.0625$ (direzione coerente in tutti e 5 i fold; con $N=5$ il p-value minimo a due code è $2 \times (1/2)^5 = 0.0625$).

---

### Dettaglio MovieLens 1M (Durata: 185.6 s)

#### 1. Accuratezza Pointwise
Valutazione su 1,000,209 rating complessivi nei 5 outer test fold:

| Modello | RMSE (mean ± std) | 95% CI RMSE | MAE (mean ± std) | 95% CI MAE |
|---|---:|:---:|---:|:---:|
| `global_mean` | 1.1171 ± 0.0018 | [1.1148, 1.1194] | 0.9339 ± 0.0014 | [0.9322, 0.9356] |
| `user_mean` | 1.0355 ± 0.0024 | [1.0326, 1.0384] | 0.8289 ± 0.0024 | [0.8259, 0.8319] |
| `item_mean` | 0.9794 ± 0.0022 | [0.9767, 0.9822] | 0.7823 ± 0.0020 | [0.7798, 0.7848] |
| `cf_only` | 0.8986 ± 0.0029 | [0.8950, 0.9021] | 0.7018 ± 0.0027 | [0.6985, 0.7051] |
| `cb_only` | 1.2721 ± 0.0018 | [1.2698, 1.2743] | 1.0107 ± 0.0015 | [1.0088, 1.0125] |
| **`hybrid_selected`** | **0.8966 ± 0.0027** | [0.8932, 0.9000] | **0.7044 ± 0.0025** | [0.7013, 0.7075] |

#### 2. Ranking Top-N
Candidati: catalogo meno item osservati nel training fold; rilevanti: rating di test $\ge 4.0$; 2,500 utenti valutati (500 per fold su 29,916 idonei; media candidati per utente: 3747.9):

| Modello | Precision@5 | NDCG@5 | Precision@10 | NDCG@10 | Utenti valutati |
|---|---:|---:|---:|---:|---:|
| `cf_only` | 0.0214 ± 0.0028 | 0.0254 ± 0.0028 | 0.0153 ± 0.0039 | 0.0206 ± 0.0034 | 2500 |
| `cb_only` | 0.0158 ± 0.0025 | 0.0175 ± 0.0019 | 0.0147 ± 0.0019 | 0.0178 ± 0.0023 | 2500 |
| **`hybrid_selected`** | **0.0263 ± 0.0036** | **0.0295 ± 0.0042** | **0.0241 ± 0.0030** | **0.0286 ± 0.0035** | 2500 |

*Miglioramento dell'Ibrido rispetto al CF su ML-1M:* **+23.2% Precision@5**, **+16.1% NDCG@5**, **+57.5% Precision@10**, **+38.8% NDCG@10**.

#### 3. Cold-item ($\le 2$ rating nel training fold)
Supporto: 550 rating di test; 425 occorrenze cold (di cui 135 con 0 rating nel training fold):

| Modello | Cold RMSE (mean ± std) | Cold MAE (mean ± std) |
|---|---:|---:|
| `cf_only` | 1.3402 ± 0.1134 | 1.0553 ± 0.1081 |
| `cb_only` | 1.4286 ± 0.0695 | 1.1529 ± 0.0746 |
| `user_mean` | 1.1821 ± 0.0992 | 0.9494 ± 0.0760 |
| `hybrid_selected` ($\alpha=0.9$) | 1.2915 ± 0.1122 | 1.0249 ± 0.1021 |
| `hybrid_alpha_0_5` ($\alpha=0.5$, esplorativo) | 1.2226 ± 0.1004 | 0.9704 ± 0.0795 |

#### 4. Test di significatività statistica (Hybrid vs CF-only)
- **Differenza media RMSE (Hybrid − CF):** `-0.002028` (IC 95%: `[-0.002467, -0.001589]`, miglioramento relativo -0.23%).
- **Paired t-test:** $t = -12.8202$, $p = 0.000213$ (statisticamente significativo con $p < 0.05$).
- **Wilcoxon signed-rank test:** $W = 0.0000$, $p = 0.0625$ (direzione coerente in tutti e 5 i fold).

---

### Interpretazione e Takeaway Chiave

1. **Selezione degli iperparametri ($\alpha^*=0.9$, $\theta^*=4.0$):** In tutti i 10 outer fold complessivi (5 su ML-100k e 5 su ML-1M), la validazione interna ha selezionato invariabilmente $\alpha=0.9$ e $\theta=4.0$. Il modello ottimale assegna il 90% del peso al Collaborative Filtering e usa il restante 10% per una correzione semantica guidata esclusivamente dai film con rating $\ge 4.0$ dell'utente.
2. **Accuratezza pointwise (RMSE/MAE):** L'ibrido ottiene una riduzione sistematica dell'RMSE su entrambi i dataset rispetto al CF puro (-0.0055 su 100k, -0.0020 su 1M), confermata come statisticamente significativa dal paired t-test ($p < 0.001$).
3. **Comportamento sul Ranking (Top-N):** L'effetto del ramo CB varia marcatamente con la scala del catalogo:
   - Su **MovieLens 1M** (3.883 item), il 10% CB agisce da efficace *tie-breaker*, discriminando fra punteggi latenti simili e aumentando Precision@5 del **+23.2%** e NDCG@5 del **+16.1%**.
   - Su **MovieLens 100k** (1.682 item), la componente CB isolata mostra buona precisione, ma l'ibridazione ottimizzata per RMSE globale non migliora il Top-N, evidenziando che errore di rating e ordinamento riflettono obiettivi empirici distinti.
4. **Mitigazione Cold-item:** L'ibridazione riduce sensibilmente l'errore del CF sui cold-item (su ML-100k da 1.3348 a 1.2674 con $\alpha=0.9$ e a 1.1405 con $\alpha=0.5$; su ML-1M da 1.3402 a 1.2915 e a 1.2226 con $\alpha=0.5$). La baseline `user_mean` resta tuttavia il riferimento più solido in assenza di storia sufficiente sull'item.

## Test

```cmd
python -m unittest discover -s tests -v
```

La suite verifica validazione della configurazione, endpoint e stabilità rispetto al batch della fusione ibrida, esclusione degli item di training nel Top-N, struttura della nested CV e vincoli dei test paired.
