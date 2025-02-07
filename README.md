# Progetto Tesi Magistrale in Data Science

Questa repository contiene il codice e i materiali relativi alla **Tesi Magistrale** incentrata sull'ambito della _Data Science_, svolta nell'ambito del corso di Laurea Magistrale in Management e Comunicazione d'Impresa (LM77) presso l'Università degli Studi di Modena e Reggio Emilia. Il lavoro si concentra sull'analisi statistica e predittiva per comprendere le caratteristiche degli utilizzatori di auto elettriche.

La repository è organizzata in due cartelle principali:

- **[Analisi_Statistiche](Analisi_Statistiche)**: contiene i materiali e il codice per le Analisi Statistiche ed Esplorative.
- **[Analisi_Predittive](Analisi_Predittive)**: contiene il codice per l'analisi predittiva tramite algoritmi di Machine e Deep Learning.

---

## **1. ANALISI STATISTICHE**

Le analisi statistiche sono contenute nell'apposita cartella e includono diversi passaggi per esplorare e comprendere i dati, tra cui analisi univariate e bivariate, analisi VIF, regressione logistica e analisi cluster. L'analisi esplorativa è stata eseguita principalmente in Python, mentre alcune tecniche più avanzate (FAMD, MCA, PCA) sono state eseguite in R.

### **Contenuti della cartella Analisi_Statistiche:**

#### 1. Raccolta ed Elaborazione dei Dati
Caricamento e preparazione dei dataset utilizzati in per le future analisi statistiche.

- [Cap. 1 Caricamento e preparazione dataset, Notebook: Python](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb)
- Link Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bPEGyp3IGkF0hbej1MXPqgE-6VONrB5h?usp=sharing)

#### 2. Analisi Univariata
Sono state condotte analisi univariate per ciascuna delle variabili. Queste analisi hanno permesso di esplorare la distribuzione, la centralità e la dispersione delle variabili.

- [Cap. 2 Analisi Univariata, Notebook: Python](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb)
- Link Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bPEGyp3IGkF0hbej1MXPqgE-6VONrB5h?usp=sharing)

#### 3. Analisi Bivariata
È stata eseguita un'analisi bivariata per esaminare le relazioni tra ciascuna delle variabili indipendenti e la variabile dipendente.

- [Cap. 3 Analisi Bivariata, Notebook: Python](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb)
- Link Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bPEGyp3IGkF0hbej1MXPqgE-6VONrB5h?usp=sharing)
  
#### 4. Formulazione delle Ipotesi di Ricerca
Dopo le analisi bivariate, sono state redatte le 4 Ipotesi di Ricerca focalizzate sul tema della logistica.

- [Cap. 4 Ipotesi di Ricerca, Notebook: Python](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb)
- Link Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bPEGyp3IGkF0hbej1MXPqgE-6VONrB5h?usp=sharing)

#### 5. Analisi VIF e Correlazione
Sono stati calcolati gli Indici di Inflazione della Varianza (VIF) per identificare potenziali problemi di multicollinearità tra le variabili indipendenti.

- [Cap. 5.1, 5.2 Analisi VIF e Correlazione, Notebook: Python](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb)
- Link Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bPEGyp3IGkF0hbej1MXPqgE-6VONrB5h?usp=sharing)

#### 6. Analisi di Regressione Logistica
L'analisi di regressione logistica è stata eseguita per testare le Ipotesi di Ricerca formulate, valutando l'influenza delle variabili indipendenti sulla probabilità della variabile dipendente.

- [Cap. 5.3, 5.4 Regressione Logistica, Notebook: Python](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb)
- Link Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bPEGyp3IGkF0hbej1MXPqgE-6VONrB5h?usp=sharing)

#### 7. Trasformazione delle Variabili
Le variabili quantitative sono state trasformate in variabili ordinali per la successiva analisi tramite **Multiple Correspondence Analysis (MCA)**.

- [Cap. 6 Trasformazione delle Variabili, Notebook: Python](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb)
- Link Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bPEGyp3IGkF0hbej1MXPqgE-6VONrB5h?usp=sharing)

#### 8. Analisi Esplorative
Sono state condotte le seguenti analisi esplorative utilizzando R:

- **FAMD (Factor Analysis of Mixed Data)**: Analisi di variabili sia quantitative che qualitative.
  - [File R: FAMD](Analisi_Statistiche/Analisi_Esplorative_FAMD.R)
  
- **MCA (Multiple Correspondence Analysis)**: Analisi delle relazioni tra variabili categoriche.
  - [File R: MCA](Analisi_Statistiche/Analisi_Esplorative_MCA.R)
  
- **PCA (Principal Component Analysis)**: Riduzione della dimensionalità delle variabili quantitative.
  - [File R: PCA](Analisi_Statistiche/Analisi_Esplorative_PCA.R)
  
- **Cluster Analysis**: Utilizzo di vari algoritmi di clustering (K-Means, K-Mode, K-Prototype, e Clustering Gerarchico) per raggruppare le osservazioni simili.
  - [File R: Cluster](Analisi_Statistiche/Analisi_Esplorative_CLUSTER.R)
  - [Cap. 7 Analisi Cluster, Notebook: Python](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb)
  - Link Colab: [![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bPEGyp3IGkF0hbej1MXPqgE-6VONrB5h?usp=sharing)

---

## **2. APPRENDIMENTO AUTOMATICO:** analisi predittive

La cartella **Apprendimento_Automatico_analisi_Predittive** contiene gli script per le analisi predittive, che includono la preparazione dei dati e l'applicazione di modelli di machine learning (ML) e deep learning.

### **Contenuti della cartella Apprendimento_Automatico_analisi_Predittive*e:**

#### 1. Raccolta ed Elaborazione dei Dati
Descrizione della preparazione dei dati per le analisi predittive.

- [Cap. 1 Caricamento e preparazione dataset, Notebook: Python](Apprendimento_Automatico_analisi_Predittive/Apprendimento_Automatico(ML_e_DL).ipynb)
- Link Colab: [![Apri con Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1HAzHywdEYMXUam888JsdAmGZRZmNiSH1/view?usp=sharing)

#### 2. Preparazione per le Analisi Predittive ML
Descrizione delle tecniche utilizzate per preparare i dati per l'analisi predittiva.

- [Cap. 2 Preparazione per le Analisi Predittive, Notebook: Python](Apprendimento_Automatico_analisi_Predittive/Apprendimento_Automatico(ML_e_DL).ipynb)
- Link Colab: [![Apri con Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1HAzHywdEYMXUam888JsdAmGZRZmNiSH1/view?usp=sharing)

#### 3. Analisi dei Modelli ML
Applicazione dei modelli di machine learning per predire i comportamenti degli utilizzatori di auto elettriche.

- [Cap. 3 Modelli ML, Notebook: Python](Analisi_Predittive)
- Link Colab: [![Apri con Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1HAzHywdEYMXUam888JsdAmGZRZmNiSH1/view?usp=sharing)

#### 4. Migliore Modello e Migliori Features
Analisi dei modelli migliori e selezione delle migliori features in base ai risultati.

- [Cap. 4 Migliore Modello e Migliori Features, Notebook: Python](Apprendimento_Automatico_analisi_Predittive/Apprendimento_Automatico(ML_e_DL).ipynb)
- Link Colab: [![Apri con Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1HAzHywdEYMXUam888JsdAmGZRZmNiSH1/view?usp=sharing)

#### 5. Artificial Neural Network
Creazione di modello più e meno complessi a base di reti neurali artificiali.

- [Cap. 5 Reti Neurali, Notebook: Python](Apprendimento_Automatico_analisi_Predittive/Apprendimento_Automatico(ML_e_DL).ipynb)
- Link Colab: [![Apri con Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1HAzHywdEYMXUam888JsdAmGZRZmNiSH1/view?usp=sharing)

#### 6. Dashboard di predizione
Le dashboard di predizioni offrono uno strumento interattivo per testare i modelli di previsione, consentendo di inserire nuovi dati relativi a potenziali consumatori. Una volta inseriti, i modelli analizzano le caratteristiche del cliente e restituiscono una predizione sulla probabilità che quel consumatore possa acquisire una vettura elettrica in futuro.

- [Dashboard di predizione: Python](Apprendimento_Automatico_analisi_Predittive/Dashboard_di_Predizione.ipynb)
- Link Colab: [![Apri con Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1hmreiqoanLHRvS9VwairauEKCfJY3der/view?usp=sharing)
