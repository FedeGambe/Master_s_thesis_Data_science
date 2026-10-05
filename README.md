# Analisi data-driven per l'adozione di veicoli elettrici

*Analisi statistiche e predittive sugli acquirenti di auto elettriche*

Tesi magistrale di **Federico Gamberini** (mat. 178147), relatore **Prof. Roberto Cavicchioli**. Laurea Magistrale in Management e Comunicazione d'Impresa (LM77), Università degli Studi di Modena e Reggio Emilia, Dipartimento di Comunicazione ed Economia, A.A. 2023/2024.

Il progetto analizza **10.688 possessori di veicoli a basse emissioni** in California per capire cosa distingue chi guida un'auto 100% elettrica (BEV) da chi ha scelto un ibrido plug-in, un full hybrid o l'idrogeno. Comprende analisi statistiche e una regressione logistica per le ipotesi di ricerca, analisi multivariate (PCA, MCA, FAMD), una segmentazione dei possessori di BEV e modelli predittivi di machine e deep learning, con una dashboard interattiva.

## Risultati principali

| | |
|---|---|
| Quota di BEV nel campione | 54,2% |
| Ipotesi confermate | H1 (reddito), H2.3 (BEV precedente), H3 (percorrenze, in parte), H4 (sensibilità ambientale) |
| Ipotesi non supportata | H2.1: chi aveva una PHEV ha odds di BEV 0,66 volte quelle di chi aveva un'auto ICE |
| Ipotesi esplorativa | H2.2: rispetto alla PHEV, l'auto ICE aumenta la probabilità di BEV (OR 1,51); HEV e GNC senza effetto significativo |
| Segmentazione | K-Mode (k = 4), come nella tesi: coppie con reddito alto e VMT basso, famiglie pendolari con 3 auto, over 65 con reddito medio, famiglie con reddito molto alto. I profili sono descrittivi, perché la partizione cambia con il seed (ARI 0,13); il K-Means, scartato nella tesi, coincide con l'auto precedente (ARI 0,997) |
| Reti neurali | la rete avanzata della tesi (189 feature polinomiali), validata senza usare il test set: accuracy 64,6%, ROC-AUC 0,672, sotto la rete più semplice (0,694) |
| Miglior modello predittivo | Voting/LightGBM, ROC-AUC 0,725 (IC 95% 0,705–0,746); i primi 6 modelli sono statisticamente equivalenti |
| Fattori più importanti (SHAP) | viaggi lunghi e VMT (in negativo), reddito, auto precedente BEV, sensibilità ambientale |

Tutti i risultati, capitolo per capitolo, sono in [`docs/RISULTATI.md`](docs/RISULTATI.md) e nella pagina interattiva [`docs/risultati.html`](docs/risultati.html), con indice e grafici (GitHub ne mostra solo il codice: va aperta nel browser dopo averla scaricata). Il testo completo della tesi e la presentazione di laurea sono in [`docs/Tesi_Gamberini_v_digitale.pdf`](docs/Tesi_Gamberini_v_digitale.pdf) e [`docs/MCI_178147.pdf`](docs/MCI_178147.pdf).

## Limiti e conclusioni

I limiti principali sono il contesto geografico (solo California), una capacità esplicativa contenuta (pseudo-R² 0,064, ROC-AUC 0,725) e l'assenza di variabili psicologiche e culturali. Reddito, miglia annue e viaggi lunghi sono i fattori più influenti; l'adozione del BEV appare come una transizione culturale oltre che tecnologica. Il testo completo è in [`docs/RISULTATI.md`](docs/RISULTATI.md).

## Notebook

I notebook presentano le analisi: la logica è nel pacchetto `src/tesi_bev`. Si aprono direttamente su Google Colab: la prima cella clona il repository e installa le dipendenze mancanti (`requirements-colab.txt`). I notebook sono generati da `scripts/genera_notebook.py`: per modificarli si cambia lo script, non il file `.ipynb`.

| Notebook | Contenuto | |
|---|---|---|
| [01 Dati e analisi descrittive](notebooks/01_dati_e_analisi_descrittive.ipynb) | qualità dei dati, distribuzioni, test bivariati | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedeGambe/Master_s_thesis_Data_science/blob/main/notebooks/01_dati_e_analisi_descrittive.ipynb) |
| [02 Regressione logistica](notebooks/02_regressione_logistica.ipynb) | VIF, logit, effetti marginali, ipotesi H1-H4 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedeGambe/Master_s_thesis_Data_science/blob/main/notebooks/02_regressione_logistica.ipynb) |
| [03 Analisi esplorative](notebooks/03_analisi_esplorative.ipynb) | PCA, MCA, FAMD, indici compositi (versione Python degli script R) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedeGambe/Master_s_thesis_Data_science/blob/main/notebooks/03_analisi_esplorative.ipynb) |
| [04 Cluster](notebooks/04_cluster.ipynb) | K-Mode (segmentazione finale), verificato con K-Means e gerarchico | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedeGambe/Master_s_thesis_Data_science/blob/main/notebooks/04_cluster.ipynb) |
| [05 Machine learning](notebooks/05_machine_learning.ipynb) | 15 modelli, ottimizzazione, IC bootstrap, McNemar, SHAP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedeGambe/Master_s_thesis_Data_science/blob/main/notebooks/05_machine_learning.ipynb) |
| [06 Reti neurali](notebooks/06_reti_neurali.ipynb) | reti Keras con validazione separata | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedeGambe/Master_s_thesis_Data_science/blob/main/notebooks/06_reti_neurali.ipynb) |
| [07 Dashboard](notebooks/07_dashboard.ipynb) | dashboard di predizione | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FedeGambe/Master_s_thesis_Data_science/blob/main/notebooks/07_dashboard.ipynb) |

## Struttura

```
├── dati/
│   ├── originali/         dataset di partenza (dataset_originale.csv è la fonte usata da tutte le analisi)
│   └── elaborati/         dataset derivati per R e per il clustering (non versionati, creati dalle fasi 01 e 03)
├── src/tesi_bev/          pacchetto Python: configurazione, dati, statistica, esplorativa, cluster,
│                          modelli, reti neurali, valutazione, grafici, salvataggio, esportazione web
├── scripts/               una fase dell'analisi per script (01 → 08), esegui_tutto.py, genera_notebook.py
├── R/                     PCA, MCA, FAMD e cluster in R (stesse analisi della versione Python)
├── notebooks/             notebook di presentazione, eseguibili su Colab
├── app/                   dashboard Dash (dashboard.py) e sorgenti della dashboard HTML (modello_bev.js, modello_dashboard.html)
├── modelli/               pipeline finale (StandardScaler + LightGBM) usata dalle dashboard
├── risultati/             tabelle CSV/JSON prodotte dagli script, una cartella per fase (risultati/r/ per R)
├── docs/                  pagina e documento dei risultati, dashboard.html, tesi e presentazione in PDF
├── tests/                 test automatici (pytest)
├── .github/workflows/     integrazione continua (test Python)
└── legacy/                codice, dataset intermedi, modelli e notebook originali della tesi
```

La descrizione dettagliata di ogni cartella e file, con il flusso dei dati tra le fasi, è in [`struttura.md`](struttura.md).

## Eseguire l'analisi in locale

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/esegui_tutto.py            # tutte le fasi, R, notebook, pagina e dashboard (circa 25 minuti)
                                          # --veloce: prova rapida del ML; --senza-r: salta gli script R
# oppure una fase alla volta:
python scripts/01_prepara_dati.py         # crea anche i CSV in dati/elaborati/ letti dagli script R
python scripts/02_statistica.py
python scripts/03_esplorativa.py
python scripts/04_cluster.py
python scripts/05_machine_learning.py
python scripts/06_reti_neurali.py
python scripts/07_pagina_risultati.py     # docs/risultati.html e docs/RISULTATI.md
python scripts/08_dashboard_html.py       # docs/dashboard.html (dashboard autonoma, si apre con doppio clic)

Rscript R/esegui_tutto.R                  # analisi R (dopo le fasi 01 e 03)
python app/dashboard.py                   # dashboard Dash su http://127.0.0.1:8050
pytest                                    # test
```

Python 3.10 o successivo; le versioni esatte con cui sono stati prodotti i risultati sono in `requirements.txt`. Per `tests/test_dashboard_html.py` serve anche Node.js (senza, il test viene saltato).

Pacchetti R richiesti: `dplyr`, `FactoMineR`, `factoextra`, `cluster`, `psych`, `ggplot2`, `klaR`.

## Dati

Fonte: *Sociodemographic data for BEV owning households in California* (UC Davis), derivato dalla ricerca *Understanding the Early Adopters of Fuel Cell Vehicles* di Scott Hardman. Dai 27.021 casi e 27 variabili originali la pulizia (OpenRefine e Python) ha portato a 10.688 casi e 16 variabili in 5 gruppi (il file ha 18 colonne: in più ci sono il modello e la tipologia dell'auto attuale, da cui deriva `BEV dummy`): socio-demografiche, abitative, sensibilità ambientale, classe veicolare, mobilità e percorrenza.

Il campione è composto da possessori californiani di veicoli a basse emissioni (BEV, PHEV, HEV, FCEV). La variabile dipendente `BEV dummy` vale 1 se il veicolo attuale è un BEV. Le variabili indipendenti riguardano caratteristiche socio-demografiche (genere, età, reddito, istruzione), abitative (casa di proprietà o indipendente, componenti e auto in famiglia), l'importanza attribuita alla riduzione delle emissioni, il tipo di auto precedente e la mobilità (viaggio più lungo, viaggi oltre 200 miglia, distanza casa-lavoro, miglia annue).
