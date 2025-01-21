Questa repository contiene il codice e i materiali relativi alla tesi magistrale, con focus sulle analisi statistiche e sulle analisi predittive. La repository è organizzata in due cartelle principali:

- **analisi_statistiche**: contiene i materiali e il codice per le analisi statistiche esplorative.
- **analisi_predittive**: contiene il codice per l'analisi predittiva.

____________

**1. ANALISI STATISTICHE**

Le analisi statistiche sono contenute nell'appositca cartella. Comprendono analisi univariata delle variabili, analisi bivariate con la variabile dipendente, Analisi VIF, Analisi di regressione logistica e Analisi Cluster, queste analisi sono contenute nello stesso notebook/file/link colab. Le restanti analisi esplorative, FAMD, MCA, PCA e Cluster sono state condotte su R, di conseguenza sono presenti i file di R per ogniune delle 4 analisi. 
(Per l'analisi cluster, è stata condotta sia su R che su Python, sia per confrontare le librerie per l'algoritmo K-Mode, su Python inoltre sono stati testati anche gli algoritmo K-Means e K-Prototype e la tecnicna di cluster Gerarchica).

Contenuti della cartella:

**1. Analisi Univariata**: Inizialmente sono state condotte analisi univariate per ciascuna delle variabili. Queste analisi hanno permesso di esplorare la distribuzione, la centralità e la dispersione delle variabili, fornendo una panoramica generale dei dati. 
[CAP. 2. Analisi Univariata, Bivariata, Logistica e Cluster](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb), link colab: ![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)

**2. Analisi Bivariata:** Successivamente, è stata eseguita un'analisi bivariata per esaminare le relazioni tra ciascuna delle variabili indipendenti e la variabile dipendente. Questo passaggio ha permesso di identificare potenziali associazioni o pattern tra le variabili di interesse, contribuendo a formulare le ipotesi di Ricerca. 
[CAP. 3. Analisi Univariata, Bivariata, Logistica e Cluster](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb), link colab: ![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)

**3. Formulazione delle Ipotesi di Ricerca**: Dopo le analisi bivariate, sono state redatte le 4 Ipotesi di Ricerca focalizzate sul tema della logistica. Queste ipotesi sono state formulate per testare l'influenza delle variabili indipendenti sulla variabile dipendente e per esaminare le dinamiche specifiche del contesto logistico.
[CAP. 4. Analisi Univariata, Bivariata, Logistica e Cluster](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb), link colab: ![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)

**4. Analisi VIF e Correlazione**: In questa fase, sono stati calcolati gli Indici di Inflazione della Varianza (VIF) per identificare potenziali problemi di multi-collinearità tra le variabili indipendenti. Parallelamente, è stata condotta un'analisi di correlazione per valutare le relazioni lineari tra le variabili e per selezionare quelle da includere nei modelli di regressione.
[CAP. 5.1. Analisi Univariata, Bivariata, Logistica e Cluster](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb),![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)

**5. Analisi di Regressione Logistica**: Successivamente, è stata eseguita un'analisi di regressione logistica per testare le Ipotesi di Ricerca formulate. L'analisi ha permesso di valutare l'influenza delle variabili indipendenti sulle probabilità della variabile dipendente, utilizzando il modello logit. 
[CAP. 5.2., 5.3., 5.4. Analisi Univariata, Bivariata, Logistica e Cluster](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb),![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)

**6. Trasformazione delle Variabili**: Le variabili quantitative sono state trasformate in variabili ordinali per la successiva analisi tramite Multiple Correspondence Analysis (MCA). Le trasformazioni sono state realizzate all'interno del notebook Python, preparando così i dati per l'analisi esplorativa in R.
[CAP. 6. Analisi Univariata, Bivariata, Logistica e Cluster](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb), ![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)

**7. Analisi Esplorative**: Una volta preparati i dati, sono state condotte le seguenti analisi esplorative:
- FAMD (Factor Analysis of Mixed Data): utilizzata per analizzare variabili sia quantitative che qualitative, riducendo la dimensionalità e identificando pattern comuni.
  [File R: FAMD](Analisi_Statistiche/Analisi_Esplorative_FAMD.R)
- MCA (Multiple Correspondence Analysis): eseguita su R per esplorare le relazioni tra le variabili categoriche e ottenere una visualizzazione delle associazioni tra di esse.
  [File R: MCA](Analisi_Statistiche/Analisi_Esplorative_MCA.R)
- PCA (Principal Component Analysis): condotta per ridurre la dimensionalità delle variabili quantitative e identificare le componenti principali che spiegano la maggior parte della varianza.
  [File R: PCA](Analisi_Statistiche/Analisi_Esplorative_PCA.R)
- Cluster Analysis: sia su R che su Python sono stati utilizzati diversi algoritmi di clustering (K-Means, K-Mode, K-Prototype e Clustering Gerarchico) per raggruppare le osservazioni simili e identificare eventuali pattern strutturali nei dati.
  [File R: Cluster](Analisi_Statistiche/Analisi_Esplorative_CLUSTER.R), 
[CAP. 7. Analisi Univariata, Bivariata, Logistica e Cluster](Analisi_Statistiche/Analisi_univariata,_bivariata,_logistica_e_cluster.ipynb), ![Apri su Colab](https://colab.research.google.com/assets/colab-badge.svg)

____________

**2. ANALISI PREDITTIVE**
La cartella analisi_predittive contiene gli script per le analisi predittive, inclusi i modelli di machine e deep learning applicati sui dati.

IN PROGRESS...
