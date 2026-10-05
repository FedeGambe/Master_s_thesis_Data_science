# Revisione del progetto di tesi — BEV adoption

> Branch: `revisione-progetto` · Data revisione: 5 ottobre 2026
> Oggetto: intera repository (`Analisi_Statistiche/`, `Apprendimento_Automatico_analisi_predittive/`, `Dataset/`, CI)

---

## 0. Stato dei lavori (aggiornamento)

Decisioni prese: script R mantenuti e affiancati da una versione Python; logica in `src/` con notebook di presentazione eseguibili su Colab; lingua italiana; pagina HTML dei risultati da pubblicare.

| Fase | Stato | Dove |
|---|---|---|
| 0. Pulizia e nuova struttura | ✅ (eliminazioni da confermare, vedi sotto) | `dati/`, `legacy/` |
| 1. Fondamenta (`src/tesi_bev`, percorsi relativi) | ✅ | `src/tesi_bev/` |
| 2. Correzioni ML (L1–L5, B1, B2) | ✅ | `scripts/05_machine_learning.py` |
| 3. Statistica (B7, IC, effetti marginali) | ✅ | `scripts/02_statistica.py` |
| 4. Reti neurali con validazione separata | ✅ | `scripts/06_reti_neurali.py` |
| 5. Dashboard (B3–B6) | ✅ (deploy online da fare) | `app/dashboard.py` |
| 6. Miglioramenti (XGBoost/LightGBM, ablation, SHAP, McNemar) | ✅ | `risultati/machine_learning/` |
| 7. Comunicazione (README, pagina risultati) | ✅ (pubblicazione da fare) | `README.md`, `docs/risultati.html` |
| R: script ripuliti e verificati contro Python | ✅ | `R/`, `risultati/r/` |

### Problemi emersi durante l'implementazione

| # | Dove | Problema | Stato |
|---|---|---|---|
| B8 | `Analisi_Esplorative_MCA.R` | `status.socioeconomico` e `distanza.sostenibilità` prendevano i punteggi dell'MCA **sulla mobilità** invece di quella socio-demografica | Corretto |
| B9 | `Analisi_Esplorative_CLUSTER.R` | Il clustering gerarchico usa `data_ric`, mai definito; nello script PCA restano righe dell'esercizio `Hitters`/`Salary` | Corretto/rimosso |
| B10 | notebook statistico, cap. 5.3 | Odds ratio letti come variazioni di probabilità ("OR 1,22 → +22% di probabilità"); coefficienti estratti per posizione (`tabella_sign.index[5]`) | Corretto: effetti marginali e accesso per nome |
| B11 | notebook statistico, cap. 7.1 | `data_kmode_r` caricava l'URL di `dataset_dummy` invece di `data_kmode_R.csv` | Superato dalla nuova pipeline |
| D1 | `dataset_partenza.csv` vs `dataset_originale.csv` | **Il genere è etichettato al contrario** nei due file (Maschio 8.019 in uno, Femmina 8.026 nell'altro); `partenza` ha 7 righe in più e un valore anomalo (2,2·10¹⁰ viaggi) | Da verificare con la fonte; si usa `dataset_originale` |
| D2 | `Età` usata nel cluster | Non è un'età vera ma il limite inferiore della classe d'età | Ricostruita da `dataset_originale` |

### Risultati che cambiano l'interpretazione

- **VIF**: con l'intercetta tutti i VIF sono < 3; l'esclusione di due variabili non era necessaria.
- **H2.1 non è supportata**: chi aveva un'auto ICE ha odds di BEV 1,5 volte quelle di chi aveva una PHEV.
- **Cluster**: i 4 cluster coincidono con il tipo di auto precedente (ARI 0,997). Senza quelle dummy la silhouette scende a 0,21.
- **Indici di mobilità**: alpha di Cronbach 0,33, bassa coerenza interna.
- **Modelli**: i primi 6 hanno ROC-AUC 0,720–0,725 con IC sovrapposti (McNemar n.s.); il "GB ridotto" perde il vantaggio.
- **Replica R ↔ Python**: PCA e MCA identiche, FAMD quasi identica, K-Means R con WSS 34.629,87 come nella tesi.

### Da fare (richiede una tua decisione)

- Eliminare i file spostati in `legacy/` che non servono più (`Import_file_from_github.py`, `read`, i 34 modelli `.pkl` in `legacy/.../Salvataggio_Modelli/`) e il workflow `.github/workflows/r.yml`, che fallisce sempre: l'eliminazione non è stata eseguita automaticamente.
- Al merge su `main`, aggiornare `RAMO_GITHUB` in `src/tesi_bev/config.py` e `RAMO` in `scripts/genera_notebook.py` (badge Colab).
- Pubblicare `docs/risultati.html` e, se vuoi, la dashboard (es. Hugging Face Spaces).

---

## 1. Sintesi

Il progetto è **completo nei contenuti** (EDA, regressione logistica con 4 ipotesi, FAMD/MCA/PCA, clustering, 13 modelli ML, reti neurali, dashboard) ma soffre di tre tipi di problemi:

| Area | Gravità | In breve |
|---|---|---|
| **Correttezza metodologica** | 🔴 Alta | Data leakage in più punti (scaler fittato sul test set, feature selection su tutto il dataset, early stopping sul test set). Le metriche riportate sono quindi leggermente ottimistiche e non del tutto confrontabili. |
| **Bug nel codice** | 🔴 Alta | Tabella finale dei modelli con colonne disallineate; dashboard che inverte il genere e non scala gli input; report MLP calcolato sulle predizioni SVM. |
| **Organizzazione e riproducibilità** | 🟠 Media | Notebook monolitici (5.366 righe) esportati da Colab, percorsi Google Drive / `~/Desktop`, dataset duplicati, README con link rotti, CI R sempre in errore. |

**Buona notizia:** il dataset è pulito (10.688 righe, 0 NaN, 0 duplicati) e il target è quasi bilanciato (54% BEV / 46% non‑BEV), quindi correggere i problemi è un lavoro di *riorganizzazione*, non di rifacimento.

---

## 2. Stato attuale

```
Master_s_thesis_Data_science/
├── README.md                          ← link a notebook inesistenti
├── .github/workflows/r.yml            ← rcmdcheck su un non‑pacchetto → fallisce sempre
├── Dataset/
│   ├── dataset_originale.csv          (10.688 × 18)
│   ├── dataset_dummy.csv
│   ├── Dataset_partenza - Dataset_partenza.csv
│   ├── data_con_cluster.csv  ┐
│   ├── data_con_cluster2.csv ├ 3 file quasi identici (2,6 MB ciascuno)
│   ├── data_con_clusterorg.csv┘
│   ├── data_kmode_R.csv
│   ├── Import_file_from_github.py     ← scarica dati di ALTRI progetti (Bitcoin, Vehicle Price)
│   └── read                           ← file vuoto
├── Analisi_Statistiche/
│   ├── *.ipynb + *.py (3.183 righe, export Colab)
│   └── 4 script R  (leggono ~/Desktop/EDA/*.csv non presenti nel repo)
└── Apprendimento_Automatico_analisi_predittive/
    ├── *.ipynb + *.py (5.366 righe, export Colab con `!pip install`)
    ├── Dashboard_di_Predizione.ipynb + .py
    └── Salvataggio_Modelli/  (34 file .pkl/.joblib/.keras versionati in git)
```

### Risultati attualmente riportati

**Screening iniziale (10‑fold CV su train, accuracy):**

| Modello | Acc. media | Dev. std |
|---|---|---|
| Gradient Boosting | **0,682** | 0,011 |
| AdaBoost | 0,671 | 0,014 |
| Random Forest | 0,669 | 0,011 |
| SVM | 0,653 | 0,016 |
| MLP | 0,647 | 0,012 |
| Naive Bayes | 0,644 | 0,013 |
| Logistica | 0,628 | 0,017 |
| Ridge / LDA | 0,625 | 0,016 |
| SGD | 0,610 | 0,016 |
| QDA | 0,596 | 0,019 |
| KNN | 0,592 | 0,011 |
| Decision Tree | 0,587 | 0,017 |

**Modelli ottimizzati (test set):**

| Modello | Accuracy | F1 | Recall | Precision | AUC |
|---|---|---|---|---|---|
| Gradient Boosting (ridotto) | **0,674** | **0,718** | 0,773 | **0,671** | **0,738** |
| Gradient Boosting | 0,660 | 0,702 | 0,748 | 0,661 | 0,711 |
| Voting (soft) | 0,659 | 0,694 | 0,733 | 0,658 | 0,711 |
| AdaBoost | 0,654 | 0,697 | 0,743 | 0,656 | 0,698 |
| ANN (regolarizzata) | 0,663 | — | 0,770 | 0,660 | 0,708 |
| MLP | 0,651 | 0,699 | 0,760 | 0,647 | 0,695 |
| Random Forest | 0,648 | 0,693 | 0,743 | 0,649 | 0,712 |
| SVM | 0,633 | 0,700 | **0,800** | 0,622 | 0,683 |
| Regressione logistica | 0,621 | 0,636 | 0,621 | 0,652 | 0,665 |

> ⚠️ Questi numeri vanno **ricalcolati** dopo le correzioni della sezione 3: le differenze tra modelli (1–3 punti) sono dello stesso ordine di grandezza della variabilità tra fold, e il "GB ridotto" è valutato su uno split diverso dagli altri.

---

## 3. Problemi critici (da correggere per primi)

### 3.1 Data leakage

| # | Dove | Problema | Correzione |
|---|---|---|---|
| L1 | `apprendimento_automatico(ml_e_dl).py:150`, `dashboard_di_predizione.py:420` | `X_test = sc.fit_transform(X_test)` — lo scaler viene ri‑fittato sul test set, che quindi ha una scala diversa dal train. | `X_test = sc.transform(X_test)`, o meglio un `Pipeline`. |
| L2 | `…ml_e_dl).py:196‑240`, `:4721‑4730` | `SelectKBest`, `PolynomialFeatures` e la scelta di *k* sono fatti su **tutto** `X, y`: il test set influenza la selezione delle feature. | Mettere scaler → poly → selector dentro una `Pipeline` e ottimizzare *k* con `GridSearchCV` solo sul train. |
| L3 | `…ml_e_dl).py:4626, 4673, 4829, 4987, 5138` | Reti neurali: `validation_data=(X_test, y_test)` + `EarlyStopping(restore_best_weights=True)` → il test set sceglie l'epoca migliore. | Usare `validation_split=0.2` sul train (o un set di validazione dedicato) e toccare il test **una sola volta** alla fine. |
| L4 | `…ml_e_dl).py:4372‑4373` | GB ridotto: scaler fittato su tutto `X_reduced`, split con `random_state=42` (gli altri usano `343`). Le variabili ridotte derivano dalle importanze di un modello già valutato sul test. | Stesso split per tutti i modelli; selezione delle feature dentro la CV. |
| L5 | `…ml_e_dl).py:658` (e analoghi) | Si prende lo stimatore del **fold con lo score più alto** e lo si usa sul test → selezione ottimistica e modello addestrato su meno dati. | Dopo la CV, ri‑addestrare il `best_estimator_` su tutto il train (è ciò che `GridSearchCV(refit=True)` già fa). |

### 3.2 Bug nel codice

| # | Dove | Problema | Effetto |
|---|---|---|---|
| B1 | `…ml_e_dl).py:4115‑4126` | Le liste che costruiscono `best_model_df` hanno **ordini diversi** rispetto a `models` (es. `f1scores_todf` ha SVM in 4ª posizione dove c'è "Voting Classifier"). Inoltre l'accuracy del Voting è quella del modello *soft ridotto*, le altre metriche quelle dell'*hard completo*. | La tabella/grafico "Metriche a confronto" della tesi attribuisce valori ai modelli sbagliati (es. "Recall Class 1 = 0,8004 → Voting", che in realtà è la SVM). |
| B2 | `…ml_e_dl).py:2830` | `class_repost_mlp = classification_report(y_test, y_pred_svm)` | Il report dell'MLP mostra i numeri della SVM. |
| B3 | `dashboard_di_predizione.py:90, 242` | Nella dashboard `Maschio=0, Femmina=1`; nel training `Maschio=1, Femmina=0`. | Le predizioni usano il genere invertito. |
| B4 | `dashboard_di_predizione.py:79‑82` | `scaled_data` viene calcolato ma poi `poly.transform(data)` usa i dati **non scalati**. | Input fuori distribuzione → predizioni inaffidabili. |
| B5 | `dashboard_di_predizione.py:185` | Mostra "Accuratezza predizione" = accuracy globale del modello sul test. | Messaggio fuorviante: non è l'affidabilità della singola predizione. |
| B6 | `dashboard_di_predizione.py:96, 251` | Etichetta `'54-64'` invece di `'55-64'`. | Refuso. |
| B7 | `analisi_statistiche_….py:1340‑1345` | VIF e Logit per McFadden calcolati **senza costante**; il "modello nullo" è con tutti i coefficienti a 0, non quello con sola intercetta. | R² di McFadden e VIF non standard. Usare `sm.add_constant` e `result.prsquared`. |

### 3.3 Scelte metodologiche da motivare o rivedere

- **Metrica di ottimizzazione**: tutte le grid search usano `scoring='accuracy'`. Con un target 54/46 è accettabile, ma per una tesi conviene ottimizzare **ROC‑AUC** o **F1** e riportare anche Brier score / calibrazione.
- **Variabile "Auto precedente BEV"**: è un predittore fortissimo e quasi tautologico (chi aveva un BEV ne ricompra uno). Utile mostrare un modello *con* e *senza* per capire quanto il resto delle variabili spiega davvero.
- **Sezione bilanciamento (ADASYN/SMOTE)**: con classi 54/46 non serve; meglio rimuoverla o spiegare perché non viene usata.
- **Significatività delle differenze**: tra il miglior modello e il 5° ci sono ~2 punti di accuracy. Servono intervalli di confidenza (bootstrap sul test) o un test (McNemar / corrected resampled t‑test) prima di dichiarare un vincitore.
- **Regressione logistica**: affiancare agli odds ratio gli **intervalli di confidenza** e gli *average marginal effects* (`result.get_margeff()`), più leggibili per le ipotesi H1–H4.
- **Clustering**: i seed cambiano tra esecuzioni (`42`, `367889`, nessuno); manca una metrica di validazione interna riportata in modo sistematico (silhouette, costo k‑modes, stabilità via bootstrap).

---

## 4. Organizzazione e riproducibilità

| Problema | Proposta |
|---|---|
| Script `.py` sono export Colab con `!pip install`, `drive.mount`, codice "commentato con stringhe" (`"""…"""`) per caricare/salvare. | Codice riutilizzabile in un pacchetto `src/`, notebook snelli che importano da lì. Flag `RETRAIN = False` al posto dei blocchi commentati. |
| Percorsi `/content/drive/MyDrive/…` e `~/Desktop/EDA/…`; gli R leggono file (`data_eda.csv`, `data_per_cluster.csv`, `data_eda_2.csv`) **non presenti** nel repo. | Percorsi relativi da un unico `config` (`data/raw`, `data/processed`). Generare i file intermedi con uno script e versionarli o renderli riproducibili. |
| Dati caricati da URL GitHub fissati a un vecchio commit (`750651…`). | Lettura locale; per Colab, una sola funzione `load_data()` che sceglie locale/remoto. |
| `data_con_cluster*.csv` triplicati, `read` vuoto, `Import_file_from_github.py` relativo ad altri progetti. | Eliminare/accorpare; tenere un `data/README.md` con *data dictionary* (significato, unità, codifiche). |
| 34 modelli `.pkl`/`.keras` in git; versioni di scikit‑learn non fissate → i pickle possono non caricarsi più. | `requirements.txt`/`environment.yml` con versioni; modelli in `models/` (o Git LFS / release GitHub); salvare **un'unica `Pipeline`** per la dashboard. |
| Preprocessing ripetuto 3 volte (ML, dashboard, ANN) con piccole differenze. | Una funzione `prepare_features()` in `src/` usata ovunque. |
| Blocco per modello copiato 7 volte (~500 righe ciascuno) con variabili `_gb1`, `_ad`, `_rf`… | Un dizionario `{nome: (stimatore, griglia)}` + una funzione `evaluate(model)` che restituisce una riga di metriche → elimina alla radice il bug B1. |
| README principale e README delle cartelle duplicati; link a `Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb` che non esiste. | README unico con abstract, risultati chiave, struttura, istruzioni di esecuzione. |
| `.github/workflows/r.yml` esegue `rcmdcheck` ma il repo non è un pacchetto R → fallisce sempre. | Sostituire con una CI Python (lint + `pytest` + esecuzione notebook con `nbmake`) oppure rimuovere. |
| Nessuna licenza né citazione. | Aggiungere `LICENSE` e `CITATION.cff` (titolo tesi, ateneo, anno). |

---

## 5. Struttura proposta

```
Master_s_thesis_Data_science/
├── README.md                 # abstract, risultati, come eseguire
├── LICENSE · CITATION.cff
├── requirements.txt · renv.lock (R)
├── data/
│   ├── raw/                  # dataset_originale.csv, Dataset_partenza.csv
│   ├── processed/            # dataset_dummy.csv, data_kmode.csv, data_con_cluster.csv
│   └── README.md             # data dictionary
├── src/bev/
│   ├── config.py             # percorsi, seed, costanti
│   ├── data.py               # load_data(), mapping, prepare_features()
│   ├── models.py             # registry modelli + griglie
│   ├── evaluate.py           # metriche, CI bootstrap, tabelle, grafici
│   └── plots.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_logistic_hypotheses.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_neural_networks.ipynb
├── R/
│   ├── famd.R · mca.R · pca.R · cluster.R   # con here::here() e percorsi relativi
├── models/                   # pipeline finale (+ metadata versioni)
├── app/dashboard.py          # Dash, carica UNA pipeline
├── reports/
│   ├── figures/
│   └── results.md            # tabelle generate automaticamente
├── docs/                     # questa revisione (md + html)
└── tests/                    # test su prepare_features e pipeline
```

---

## 6. Miglioramenti possibili (oltre alle correzioni)

**Modellistica**
1. Aggiungere **XGBoost / LightGBM / CatBoost** (già importati ma non usati) — tipicamente i più forti su dati tabellari.
2. Ottimizzazione con **Optuna** o `HalvingGridSearchCV` invece di griglie da 60+ minuti.
3. **Nested cross‑validation** per una stima onesta della performance del processo di tuning.
4. **Calibrazione** delle probabilità (`CalibratedClassifierCV`) e scelta della soglia in base all'obiettivo (es. massimizzare recall dei futuri acquirenti BEV).
5. **Ablation** "con/senza auto precedente BEV".

**Interpretabilità**
6. SHAP su un modello addestrato con la pipeline corretta, con *dependence plot* per reddito, VMT e sensibilità ambientale — collega direttamente il ML alle ipotesi H1–H4.
7. Confronto esplicito: le variabili significative nella logistica coincidono con le più importanti per SHAP?

**Presentazione**
8. Tabelle dei risultati **generate dal codice** (`reports/results.md`), non copiate a mano.
9. Dashboard: input validati, probabilità con intervallo/indicazione dell'incertezza, spiegazione SHAP della singola predizione; deploy su Hugging Face Spaces o Render.
10. Una pagina riassuntiva (GitHub Pages) con abstract, grafici chiave e link a notebook/Colab.

---

## 7. Roadmap consigliata

| Fase | Attività | Output | Stima |
|---|---|---|---|
| **0. Pulizia** | Eliminare duplicati/file estranei, `requirements.txt`, nuova struttura cartelle, sistemare README e CI | Repo ordinato, link funzionanti | 0,5 g |
| **1. Fondamenta** | `src/bev/data.py` + `config.py`; script R con percorsi relativi e generazione dei CSV intermedi | Dati riproducibili da zero | 1 g |
| **2. Correzioni ML** | Pipeline senza leakage, split unico, refit corretto, tabella risultati automatica (fix L1–L5, B1, B2) | Nuove metriche affidabili | 1–2 g |
| **3. Statistica** | Logit con costante, IC, effetti marginali, seed fissi nel clustering (fix B7) | Ipotesi H1–H4 rivalidate | 0,5 g |
| **4. Reti neurali** | Validation set separato, confronto equo con GB | Sezione DL corretta | 0,5 g |
| **5. Dashboard** | Una pipeline salvata, fix B3–B6, deploy | Demo online | 0,5–1 g |
| **6. Miglioramenti** | XGBoost/LightGBM, calibrazione, ablation, SHAP | Capitolo "estensioni" | 1–2 g |
| **7. Comunicazione** | README finale, pagina HTML dei risultati, CITATION | Portfolio pronto | 0,5 g |

**Ordine suggerito:** 0 → 1 → 2 → 3/4 in parallelo → 5 → 6 → 7. Ogni fase in un commit (o PR) separato su questo branch.

---

## 8. Decisioni da prendere

1. **Notebook vs script**: tenere i notebook come "racconto" importando da `src/` (consigliato) o passare a script puri?
2. **Colab**: deve continuare a funzionare su Colab (badge nel README) oppure basta l'esecuzione locale?
3. **Modelli salvati**: rimuoverli dal repo e rigenerarli, oppure spostarli in una Release GitHub / Git LFS?
4. **Lingua**: codice e README in italiano (come ora) o in inglese per renderlo un progetto da portfolio?
5. **Risultati della tesi originale**: conservarli in `reports/tesi_originale/` per confronto "prima/dopo le correzioni"?
