# Piano di Progetto: Sistema di Raccomandazione Ibrido per SII

**Corso:** Sistemi Intelligenti per Internet (SII)  
**Modalità:** Progetto Individuale  
**Titolo Progetto:** Progettazione, Sviluppo e Valutazione Sperimentale di un Sistema di Raccomandazione Ibrido (Filtro Collaborativo + Content-Based) su Dataset MovieLens 100k  

---

## 1. Obiettivi del Progetto
- Sviluppare un **Recommender System Ibrido** che combini la fattorizzazione di matrice / filtro collaborativo (Collaborative Filtering - $S_{CF}$) con un approccio basato sul contenuto dei film ($S_{CB}$).
- Analizzare sperimentalmente come varia l'accuratezza (RMSE, MAE) al variare del parametro di fusione pesata $\alpha \in [0, 1]$.
- Dimostrare la capacità dell'approccio ibrido nel mitigare le problematiche classiche dei Recommender Systems: **Sparsità della Matrice** e **Cold Start** (film/utenti con poche valutazioni).
- Garantire **solida rigore metodologico** nella validazione (Train/Test Split, Cross-Validation) e produrre una documentazione chiara con codice riproducibile su GitHub.

---

## 2. Stack Tecnologico e Librerie
- **Linguaggio:** Python 3.10+
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Machine Learning & Cosine Similarity:** `scikit-learn`
- **Collaborative Filtering:** `scikit-surprise` (o algoritmo SVD / KNN custom)
- **Visualizzazione Grafica:** `matplotlib`, `seaborn`
- **Dataset:** MovieLens 100k (`u.data` per le valutazioni, `u.item` per metadati/generi)

---

## 3. Roadmap Dettagliata (Step-by-Step)

### Fase 1: Approvazione Docenti e Setup Iniziale
1. **Contatto con i Docenti:** Inviare l'email di proposta del progetto per ottenere la formale approvazione (obbligatoria).
2. **Setup Repository GitHub:**
   - Creare una repo pubblica (es. `SII-Hybrid-Recommender-System`).
   - Aggiungere `.gitignore` per Python e file `README.md` iniziale.
   - Organizzazione cartelle:
     ```text
     ├── data/                 # Dataset MovieLens 100k
     ├── notebooks/            # Jupyter Notebooks di sviluppo ed esperimenti
     ├── src/                  # Moduli Python trasversali (opzionale)
     ├── report/               # Relazione finale in PDF/LaTeX
     ├── README.md
     └── requirements.txt
     ```

### Fase 2: Analisi Esplorativa dei Dati (EDA)
1. **Caricamento Dataset:** Leggere `u.data` (userId, itemId, rating, timestamp) e `u.item` (itemId, title, release_date, generi).
2. **Calcolo Metriche Esplorative:**
   - Numero totale di utenti ($N$) e oggetti ($M$).
   - Percentuale di **Sparsità della Matrice**:
     $$\text{Sparsity} = 1 - \frac{\text{Num. Rating Totali}}{N \times M}$$
   - Distribuzione dei voti (istogramma rating da 1 a 5).
   - Distribuzione della popolarità dei film (Long Tail Analysis).
   - Distribuzione dei generi dei film.

### Fase 3: Implementazione dei Modelli

#### A. Modulo Collaborative Filtering ($S_{CF}$)
1. Utilizzare `Surprise` con algoritmo **SVD** (Singular Value Decomposition) o **KNNWithMeans**.
2. Eseguire l'addestramento sui rating dell'utente.
3. Scalare/normalizzare le predizioni $\hat{r}_{u,i}$ nell'intervallo $[0, 1]$:
   $$S_{CF}(u, i) = \frac{\hat{r}_{u,i} - r_{\min}}{r_{\max} - r_{\min}}$$

#### B. Modulo Content-Based ($S_{CB}$)
1. Estrarre il vettore binario/TF-IDF dei generi dei film da `u.item`.
2. Per ciascun utente $u$, costruire il **Profilo Utente** $P_u$ calcolando la media pesata dei vettori dei film che l'utente ha valutato positivamente ($\ge 3$ o $\ge 4$ stelle).
3. Calcolare la **Cosine Similarity** tra il profilo dell'utente $P_u$ e il vettore del genere del film $i$:
   $$S_{CB}(u, i) = \text{CosineSimilarity}(P_u, V_i)$$

#### C. Modulo di Fusione Ibrida ($S_{\text{Hybrid}}$)
1. Calcolare il punteggio finale combinato:
   $$S_{\text{Hybrid}}(u, i) = \alpha \cdot S_{CF}(u, i) + (1 - \alpha) \cdot S_{CB}(u, i)$$
2. Convertire lo score di nuovo nella scala dei rating $[1, 5]$ per le valutazioni di errore standard:
   $$\hat{R}_{u,i} = r_{\min} + S_{\text{Hybrid}}(u, i) \cdot (r_{\max} - r_{\min})$$

### Fase 4: Esperimenti e Valutazione Sperimentale
1. **Train / Test Split:** Suddivisione 80% Train, 20% Test (oppure 5-Fold Cross-Validation).
2. **Sensitivity Analysis su $\alpha$:**
   - Far variare $\alpha$ da $0.0$ a $1.0$ con passo $0.1$.
   - Calcolare **RMSE** (Root Mean Squared Error) e **MAE** (Mean Absolute Error) per ciascun valore.
   - Individuare il valore ottimo di $\alpha$.
3. **Esperimento Cold Start (Punto di forza metodologico):**
   - Creare un test set sintetico rimuovendo la maggior parte delle valutazioni per un sottoinsieme di film (es. tenendo solo $<3$ rating).
   - Confrontare l'errore del modello solo Collaborative ($lpha = 1.0$) con il modello Ibrido ($lpha = 0.5$).
   - Dimostrare che il modello ibrido mantiene performance superiori quando le valutazioni scarseggiano.

### Fase 5: Stesura del Report Finale e Consegna
1. Redigere un report chiaro, strutturato e professionale.
2. Inserire tabelle riassuntive e grafici (andamento RMSE vs $\alpha$, confronto Cold Start).
3. Verificare che tutti i requisiti del corso siano rispettati (nomi, matricola, link GitHub, librerie usate, bibliografia).

---

## 4. Modello Email di Proposta ai Docenti

```text
Oggetto: Richiesta approvazione idea progettuale SII - [Nome Cognome] [Matricola]

Gentile Professore/essa,

desidero sottoporre alla Sua attenzione la proposta per lo svolgimento del progetto individuale per il corso di Sistemi Intelligenti per Internet (SII).

Titolo: Progettazione e valutazione sperimentale di un Sistema di Raccomandazione Ibrido (Collaborative Filtering + Content-Based) su dataset MovieLens 100k.

Sintesi del lavoro:
L'obiettivo del progetto è realizzare una pipeline completa in Python che fonda un modello di Filtro Collaborativo (SVD) e un modulo Content-Based basato sui generi dei film (Cosine Similarity). Verrà analizzata l'accuratezza (RMSE/MAE) al variare del peso di fusione pesata alpha e si condurrà un esperimento specifico per valutare l'efficacia dell'approccio ibrido nel mitigare il problema del Cold Start per oggetti con poche valutazioni.

Linguaggio e tool: Python (pandas, scikit-learn, scikit-surprise, matplotlib).
Il codice completo, ben documentato e riproducibile, verrà caricato su GitHub e affiancato da un report finale dettagliato.

Resto a disposizione per eventuali modifiche o suggerimenti.

Cordiali saluti,
[Nome Cognome]
[Matricola]
[Corso di Laurea]
```

---

## 5. Struttura Suggerita per il Report Finale (PDF)

1. **Intestazione:** Titolo, Nome, Cognome, Matricola, Corso di Studi, Link Repository GitHub.
2. **1. Introduzione ed Obiettivi:** Presentazione del problema della raccomandazione, motivazione della scelta dell'approccio ibrido.
3. **2. Dataset e Analisi Esplorativa (EDA):** Descrizione di MovieLens 100k, matrice di sparsità, distribuzione dei voti e dei generi.
4. **3. Architettura e Metodologia:** Formule e spiegazione di $S_{CF}$, $S_{CB}$ e della funzione di fusione pesata con parametro $\alpha$.
5. **4. Valutazione Sperimentale e Risultati:**
   - Setup di test e metriche utilizzate (RMSE, MAE).
   - Grafico dell'andamento delle metriche al variare di $\alpha$.
   - Risultati dell'esperimento sul problema del Cold Start.
6. **5. Discussione e Conclusioni:** Considerazioni critiche sui trade-off riscontrati e possibili sviluppi futuri.
7. **Bibliografia e Sitografia:** Riferimenti a paper/documentazione utilizzata.

---

## 6. Checklist Finale prima della Consegna
- [ ] Email inviata e progetto confermato dai docenti.
- [ ] Codice funzionante, pulito e commentato su GitHub.
- [ ] File `requirements.txt` aggiornato nella repo.
- [ ] Grafici generati e integrati nel report.
- [ ] Link GitHub verificato e funzionante nel report.
- [ ] Report impaginato correttamente in formato PDF.
