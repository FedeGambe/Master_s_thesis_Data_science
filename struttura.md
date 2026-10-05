# Struttura del progetto

Questo documento descrive in dettaglio l'organizzazione del repository: cosa contiene ogni cartella, quale script produce ogni file e in che ordine vanno eseguite le fasi. Per i risultati dell'analisi si veda [`docs/RISULTATI.md`](docs/RISULTATI.md); per l'installazione e i comandi rapidi il [`README.md`](README.md).

## Principi di organizzazione

- **La logica sta in un solo posto.** Tutte le trasformazioni dei dati, i modelli e le metriche sono funzioni del pacchetto `src/tesi_bev`. Script, notebook e dashboard le importano e non duplicano codice: la stessa codifica dei dati è usata in addestramento, nei test e nella dashboard.
- **Gli script calcolano, i notebook mostrano.** Gli script in `scripts/` eseguono le analisi e salvano tabelle e modelli; i notebook leggono gli stessi risultati e li presentano con testo e grafici.
- **Ogni output è rigenerabile.** `dati/elaborati/`, `risultati/`, `modelli/`, `docs/` e `notebooks/` sono prodotti dal codice; `scripts/esegui_tutto.py` li ricrea tutti a partire dal solo `dati/originali/dataset_originale.csv`.
- **Percorsi relativi alla radice.** `src/tesi_bev/config.py` calcola i percorsi a partire dalla posizione del pacchetto, così il codice funziona allo stesso modo in locale, in CI e su Google Colab. Gli script R vanno lanciati dalla radice del repository.
- **Riproducibilità.** Seed unico `343` (lo stesso dello split della tesi), split train/test 80/20 stratificato e unico per tutti i modelli, versioni dei pacchetti fissate in `requirements.txt`.

## Albero delle cartelle

```
Master_s_thesis_Data_science/
├── README.md                    presentazione del progetto e comandi principali
├── struttura.md                 questo documento
├── pyproject.toml               metadati del pacchetto tesi_bev e configurazione di pytest
├── requirements.txt             dipendenze Python con versioni esatte
├── requirements-colab.txt       pacchetti da aggiungere all'ambiente standard di Colab
├── .gitignore
├── .github/workflows/           integrazione continua
├── dati/
│   ├── originali/               dati di partenza (versionati)
│   └── elaborati/               dataset derivati (non versionati)
├── src/tesi_bev/                pacchetto Python con tutta la logica
├── scripts/                     fasi dell'analisi, orchestrazione e generatori
├── R/                           analisi esplorative e cluster in R
├── notebooks/                   notebook di presentazione (generati)
├── app/                         dashboard Dash e sorgenti della dashboard HTML
├── modelli/                     modello finale salvato
├── risultati/                   tabelle e JSON prodotti dalle fasi
├── docs/                        pagina dei risultati, dashboard HTML, tesi e presentazione
├── tests/                       test automatici
└── legacy/                      materiale originale della tesi, conservato come archivio
```

## Flusso dei dati

```
dati/originali/dataset_originale.csv
        │
        ├─ 01_prepara_dati ──► dati/elaborati/{dataset_ml, dataset_logit, dataset_categoriale}.csv
        │                      risultati/dati/
        ├─ 02_statistica ────► risultati/statistica/
        ├─ 03_esplorativa ───► risultati/esplorativa/
        │                      dati/elaborati/dataset_cluster.csv  (indici compositi per il clustering)
        ├─ 04_cluster ───────► risultati/cluster/, dati/elaborati/etichette_cluster.csv
        ├─ 05_machine_learning ► risultati/machine_learning/, modelli/modello_finale.{joblib,json}
        ├─ 06_reti_neurali ──► risultati/reti_neurali/
        │
        ├─ R/esegui_tutto.R ─► risultati/r/ (+ figure/)      legge dati/elaborati/ (dopo 01 e 03)
        │
        ├─ genera_notebook ──► notebooks/*.ipynb              leggono risultati/ e src/tesi_bev
        ├─ 07_pagina_risultati ► docs/risultati.html, docs/RISULTATI.md   legge risultati/
        └─ 08_dashboard_html ► docs/dashboard.html            legge modelli/modello_finale.*
```

Le fasi 02, 03, 05 e 06 leggono direttamente il dataset originale tramite `tesi_bev.dati`; la 04 ha bisogno di `dataset_cluster.csv` creato dalla 03; la 07 e la 08 non ricalcolano nulla, ma impaginano ciò che le fasi precedenti hanno salvato.

---

## `dati/`

### `dati/originali/` (versionata)

| File | Contenuto |
|---|---|
| `dataset_originale.csv` | **La fonte di tutte le analisi.** 10.688 righe, 18 colonne, nessun valore mancante. Contiene le 16 variabili della tesi (socio-demografiche, abitative, sensibilità ambientale, auto precedente, mobilità e target `BEV dummy`) più `Auto attuale` (modello del veicolo) e `Tipologia di auto attuale` (BEV, PHEV, HEV, FCEV), da cui deriva il target. |
| `dataset_partenza.csv` | Versione già codificata usata nel lavoro originale (dummy e confronti con le categorie di riferimento, es. `Classe d'età: <25 vs 45-54`). Conservata per confronto, non è letta dal codice. |

Se il file locale non è presente (ad esempio un notebook aperto da solo su Colab), `config.URL_DATASET` punta alla copia su GitHub nel branch `main`.

### `dati/elaborati/` (ignorata da git)

Dataset intermedi rigenerabili, usati soprattutto dagli script R e dal clustering.

| File | Creato da | Usato da |
|---|---|---|
| `dataset_ml.csv` | `01_prepara_dati.py` | consultazione (stessa codifica di `dati.prepara_ml`) |
| `dataset_logit.csv` | `01_prepara_dati.py` | consultazione (variabili della regressione logistica) |
| `dataset_categoriale.csv` | `01_prepara_dati.py` | `R/01`–`R/03` (PCA, MCA, FAMD) |
| `dataset_cluster.csv` | `03_esplorativa.py` | `04_cluster.py`, `R/04_cluster.R` |
| `etichette_cluster.csv` | `04_cluster.py` | consultazione (etichette K-Mode e delle verifiche per ogni BEV) |

---

## `src/tesi_bev/` — il pacchetto

Installabile con `pip install -e .`; gli script lo rendono comunque importabile tramite `scripts/_percorso.py`, i notebook e i test tramite `sys.path` o la configurazione di pytest.

| Modulo | Responsabilità | Funzioni principali |
|---|---|---|
| `__init__.py` | descrizione del pacchetto e versione (`2.0.0`) | — |
| `config.py` | percorsi, URL del dataset, seed (`343`), dimensione del test (20%), numero di fold (10), nomi delle colonne, ordine delle categorie, categorie di riferimento della logistica, nomi brevi per i grafici | `nome_breve` |
| `dati.py` | caricamento e **unica fonte delle codifiche**: mappe ordinali (età, reddito, istruzione), dummy dell'auto precedente, variabili categoriali per MCA e K-Mode, filtro sui soli BEV | `carica_dati`, `codifica_ml`, `prepara_ml`, `prepara_logit`, `crea_variabili_categoriali`, `solo_bev` |
| `statistica.py` | descrittive, frequenze, test bivariati (con V di Cramér), quota di BEV per categoria, VIF, regressione logistica (statsmodels), odds ratio, effetti marginali, bontà di adattamento, verifica delle ipotesi H1–H4 | `test_bivariati`, `vif`, `logit`, `tabella_logit`, `effetti_marginali`, `verifica_ipotesi` |
| `esplorativa.py` | PCA (scikit-learn), MCA e FAMD (`prince`, stesse definizioni di FactoMineR), alpha di Cronbach, indici compositi da usare nel clustering | `pca`, `mca`, `mca_gruppi`, `famd`, `indici_compositi`, `alpha_cronbach` |
| `cluster.py` | K-Mode (segmentazione finale), K-Means, K-Prototype e gerarchico (verifiche), scelta di k (gomito, silhouette, costo), stabilità tra seed e bootstrap (ARI), profili dei cluster | `kmode`, `stabilita_kmode`, `kmeans`, `gerarchico`, `gomito_silhouette`, `profilo_cluster` |
| `modelli.py` | ogni modello è una `Pipeline` scaler + stimatore, così lo scaler si addestra solo sul training di ogni fold; split, CV stratificata, 15 modelli di screening, spazi di ricerca, `RandomizedSearchCV`, voting | `pipeline`, `dividi`, `modelli_screening`, `screening`, `ottimizza`, `voting` |
| `reti_neurali.py` | reti Keras (base, regolarizzata, avanzata della tesi con 189 feature polinomiali, LeakyReLU, L1+L2, BatchNorm, AdamW); l'early stopping usa una validazione estratta dal training, mai il test | `ARCHITETTURE`, `crea_rete`, `ReteNeurale` |
| `valutazione.py` | metriche (accuracy, bilanciata, precision, recall, F1, ROC-AUC, PR-AUC, Brier), IC bootstrap, test di McNemar, soglia di Youden, curve ROC e di calibrazione, permutation importance, SHAP | `metriche`, `tabella_risultati`, `bootstrap_ic`, `confronto_mcnemar`, `valori_shap` |
| `grafici.py` | grafici Plotly con stile e palette comuni (distribuzioni, forest plot, scree, mappe MCA, cluster, ROC, calibrazione, importanza, matrici di confusione, curve di addestramento) | una funzione per tipo di grafico |
| `archivio.py` | salvataggio e lettura uniforme in `risultati/<sezione>/` (CSV, JSON) e `modelli/` (joblib) | `salva_tabella`, `carica_tabella`, `salva_json`, `carica_json`, `salva_modello`, `carica_modello` |
| `esporta_web.py` | traduce la pipeline finale (StandardScaler + LightGBM) in JSON e scrive `docs/dashboard.html` combinando `app/modello_dashboard.html` e `app/modello_bev.js` | `dati_modello`, `scrivi_pagina` |

---

## `scripts/` — le fasi dell'analisi

Ogni script numerato è una fase indipendente con una funzione `main()`; si lancia dalla radice del repository.

| File | Cosa fa | Scrive in |
|---|---|---|
| `_percorso.py` | aggiunge `src/` a `sys.path`, così `tesi_bev` è importabile senza installarlo | — |
| `01_prepara_dati.py` | controlli di qualità (righe, mancanti, duplicati, quota BEV), frequenze delle variabili categoriali, dataset elaborati | `risultati/dati/`, `dati/elaborati/` |
| `02_statistica.py` | descrittive, test bivariati, quota di BEV per categoria, VIF, logit completo e ridotto, effetti marginali, verifica delle ipotesi | `risultati/statistica/` |
| `03_esplorativa.py` | PCA sulla mobilità, alpha di Cronbach, MCA tematiche (abitative, mobilità, socio-demografiche), FAMD, indici compositi | `risultati/esplorativa/`, `dati/elaborati/dataset_cluster.csv` |
| `04_cluster.py` | K-Mode con k = 4 (finale) e stabilità tra seed; verifiche con K-Means (con e senza auto precedente) e gerarchico; profili e distribuzioni per cluster | `risultati/cluster/`, `dati/elaborati/etichette_cluster.csv` |
| `05_machine_learning.py` | split unico, screening di 15 modelli in 10-fold CV, ottimizzazione, valutazione sul test (IC bootstrap, McNemar, calibrazione), ablation senza *auto precedente BEV*, SHAP, salvataggio del modello finale. Opzione `--veloce` | `risultati/machine_learning/`, `modelli/` |
| `06_reti_neurali.py` | addestra le tre architetture sullo stesso split del ML e le valuta sul test | `risultati/reti_neurali/` |
| `07_pagina_risultati.py` | raccoglie le tabelle di tutte le fasi e scrive la pagina e il documento dei risultati, usando il modello HTML `modello_pagina_risultati.html` | `docs/risultati.html`, `docs/RISULTATI.md` |
| `modello_pagina_risultati.html` | template (stile, struttura, grafici) della pagina dei risultati | — |
| `08_dashboard_html.py` | esporta il modello finale nella dashboard HTML autonoma | `docs/dashboard.html` |
| `genera_notebook.py` | costruisce i notebook in `notebooks/`; con `--esegui` li esegue e salva gli output | `notebooks/` |
| `esegui_tutto.py` | esegue in ordine le fasi 01–06, gli script R (se `Rscript` è disponibile), i notebook con output, la fase 07 e la 08. Opzioni `--veloce` e `--senza-r` | tutto quanto sopra |

---

## `R/` — analisi in R

Riproducono in R le analisi esplorative e il clustering, come nella tesi originale. Leggono `dati/elaborati/` (quindi richiedono prima le fasi 01 e 03) e scrivono in `risultati/r/`, con i grafici in `risultati/r/figure/`.

| File | Contenuto |
|---|---|
| `00_comune.R` | pacchetti, seed, percorsi e funzioni comuni (`leggi`, `salva`, `salva_grafico`, `autovalori`, `dati_bev`) |
| `01_pca.R` | PCA su tutte le variabili quantitative e sulle sole variabili di mobilità, alpha di Cronbach (`psych`) |
| `02_mca.R` | MCA tematiche; le prime due dimensioni diventano indici compositi |
| `03_famd.R` | FAMD sulle variabili miste |
| `04_cluster.R` | K-Mode (`klaR`), K-Means con scelta di k e gerarchico di Ward |
| `esegui_tutto.R` | esegue i quattro script in sequenza |

Pacchetti necessari: `dplyr`, `FactoMineR`, `factoextra`, `cluster`, `psych`, `ggplot2`, `klaR`. I segni delle componenti possono differire tra R e Python: gli assi sono definiti a meno del segno.

---

## `notebooks/` — presentazione

Generati da `scripts/genera_notebook.py`: per cambiarli si modifica lo script, non il `.ipynb`. Non contengono logica di analisi; la prima cella, su Colab, clona il repository e installa `requirements-colab.txt`.

| Notebook | Contenuto |
|---|---|
| `01_dati_e_analisi_descrittive.ipynb` | qualità dei dati, distribuzioni, test bivariati |
| `02_regressione_logistica.ipynb` | VIF, logit, effetti marginali, ipotesi H1–H4 |
| `03_analisi_esplorative.ipynb` | PCA, MCA, FAMD, indici compositi |
| `04_cluster.ipynb` | K-Mode e verifiche con K-Means e gerarchico |
| `05_machine_learning.ipynb` | screening, ottimizzazione, IC bootstrap, McNemar, SHAP |
| `06_reti_neurali.ipynb` | reti Keras con validazione separata |
| `07_dashboard.ipynb` | dashboard di predizione dentro il notebook |

---

## `app/` — dashboard

| File | Contenuto |
|---|---|
| `dashboard.py` | dashboard Dash (`python app/dashboard.py`, poi http://127.0.0.1:8050): probabilità che il veicolo di un cliente sia un BEV e contributi SHAP. Gli input passano per `dati.codifica_ml`, la stessa funzione dell'addestramento |
| `modello_bev.js` | logica della dashboard HTML in JavaScript: codifica degli input, standardizzazione, somma delle foglie degli alberi LightGBM e TreeSHAP esatto |
| `modello_dashboard.html` | template della dashboard HTML. Va modificato questo file, non `docs/dashboard.html`, che viene rigenerato |

## `modelli/`

| File | Contenuto |
|---|---|
| `modello_finale.joblib` | pipeline addestrata (StandardScaler + LightGBM) salvata dalla fase 05 |
| `modello_finale.json` | metadati: nome del modello, ordine delle feature, soglie (0,5 e Youden), metriche sul test, versioni dei pacchetti |

## `risultati/`

Una sottocartella per fase, scritta tramite `tesi_bev.archivio`. I file sono versionati per poter leggere i risultati senza rieseguire l'analisi.

| Cartella | Contenuto principale |
|---|---|
| `dati/` | `qualita.json`, frequenze delle variabili categoriali |
| `statistica/` | descrittive, test bivariati, quote di BEV per categoria, VIF, logit completo e ridotto, effetti marginali, bontà di adattamento, verifica delle ipotesi |
| `esplorativa/` | varianza e loadings della PCA, autovalori e categorie delle MCA, FAMD, alpha di Cronbach |
| `cluster/` | scelta di k, centroidi e distribuzioni del K-Mode, profili delle verifiche, `riepilogo.json` (dimensioni, ARI di stabilità e con l'auto precedente) |
| `machine_learning/` | split, screening in CV, ottimizzazione, risultati sul test, probabilità, curve ROC, calibrazione, McNemar, ablation, importanze (permutazione e SHAP), confronto tesi/revisione, metadati del modello finale |
| `reti_neurali/` | architetture, curve di addestramento, risultati e probabilità sul test |
| `r/` | output degli script R (PCA, MCA, FAMD, K-Means, gerarchico, K-Mode) e `figure/` con i grafici PNG |

## `docs/`

| File | Contenuto |
|---|---|
| `RISULTATI.md` | risultati completi, capitolo per capitolo (generato dalla fase 07) |
| `risultati.html` | stessi contenuti in una pagina autonoma con indice e grafici interattivi (generata dalla fase 07) |
| `dashboard.html` | dashboard di predizione in un unico file, si apre con un doppio clic senza Python né server (generata dalla fase 08) |
| `favicon.svg` | icona delle pagine |
| `Tesi_Gamberini_v_digitale.pdf` | testo completo della tesi |
| `MCI_178147.pdf` | presentazione di laurea, riferimento per lo stile della pagina dei risultati |

## `tests/`

Si eseguono con `pytest` dalla radice (la configurazione è in `pyproject.toml`).

| File | Cosa verifica |
|---|---|
| `test_dati.py` | integrità del dataset (10.688 × 18, nessun mancante), codifica ML (genere maschio = 1, dummy dell'auto precedente esclusive), coerenza della codifica di una singola riga, errore su valori sconosciuti, categorie di riferimento della logistica, assenza di NaN nelle variabili categoriali |
| `test_modelli.py` | split stratificato e riproducibile, scaler addestrato solo sul training, coerenza delle metriche, tabella dei risultati, McNemar su predizioni identiche |
| `test_dashboard_html.py` | la dashboard JavaScript restituisce le stesse probabilità e gli stessi SHAP del modello Python (saltato se mancano Node.js o il modello salvato) |

## `.github/workflows/`

| File | Contenuto |
|---|---|
| `test.yml` | a ogni push e pull request installa le dipendenze principali su Python 3.12 ed esegue `pytest` |
| `r.yml` | modello standard di GitHub per pacchetti R (`rcmdcheck`), attivo solo su `main`. Il repository non è un pacchetto R (manca il file `DESCRIPTION`), quindi questo controllo non è adatto agli script in `R/` |

## `legacy/` — materiale originale

Codice e file della tesi così come erano prima della revisione, conservati come archivio. Non sono usati dal nuovo codice.

| Cartella | Contenuto |
|---|---|
| `Analisi_Statistiche/` | notebook e script Python delle analisi statistiche, logistica e cluster; script R originali di PCA, MCA, FAMD e cluster; README originale |
| `Apprendimento_Automatico_analisi_predittive/` | notebook e script di machine e deep learning e della dashboard originale; `Salvataggio_Modelli/` con i modelli addestrati allora (`.pkl`, `.joblib`, `.keras`) |
| `dati/` | dataset intermedi (con le etichette dei cluster, dummy, K-Mode da R) e lo script di import da GitHub |
| `revisione/` | `REVISIONE_PROGETTO.md` e la versione HTML: problemi trovati nel codice originale (data leakage, codifiche, errori negli script) e stato delle correzioni |
