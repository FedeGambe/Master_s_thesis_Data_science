# Analisi data-driven per l'adozione di veicoli elettrici

*Analisi statistiche e predittive sugli acquirenti di auto elettriche*

Laureando: **Federico Gamberini** (mat. 178147) · Relatore: **Prof. Roberto Cavicchioli**  
Laurea Magistrale in Management e Comunicazione d'Impresa, Università degli Studi di Modena e Reggio Emilia, Dipartimento di Comunicazione ed Economia, A.A. 2023/2024

> Versione interattiva con grafici: [`docs/risultati.html`](risultati.html).

## Indice

- [Introduzione](#introduzione)
- [1. Analisi statistica](#1-analisi-statistica)
- [2. Analisi esplorativa](#2-analisi-esplorativa)
- [3. Cluster](#3-cluster)
- [4. Machine learning](#4-machine-learning)
- [5. Deep learning](#5-deep-learning)
- [Limiti e conclusioni](#limiti-e-conclusioni)

## Introduzione

**Contesto.** Transizione del settore automotive verso la riduzione delle emissioni, con normative ambientali e incentivi in aumento. **Obiettivo.** Individuare i fattori socio-demografici e le variabili chiave nella scelta di un veicolo elettrico. **Implicazioni.** Strategie di marketing per la mobilità elettrica e politiche pubbliche per la sostenibilità.

**Dataset.** *Sociodemographic data for BEV owning households in California* (UC Davis), derivato dalla ricerca *Understanding the Early Adopters of Fuel Cell Vehicles* di Scott Hardman. Dai 27.021 casi e 27 variabili originali la pulizia (OpenRefine e Python) ha portato a **10.688 casi e 16 variabili**. Tutti i rispondenti possiedono un veicolo a basse emissioni; il 54,2% un BEV.

| Gruppo | Variabili |
|---|---|
| Socio-demografiche | genere, classe d'età, classe di reddito, livello di istruzione |
| Abitative | casa di proprietà, casa indipendente, persone in famiglia, auto in famiglia |
| Sensibilità ambientale | importanza di ridurre le emissioni di gas serra |
| Classe veicolare | tipologia di auto attuale e precedente |
| Mobilità e percorrenza | viaggio più lungo, viaggi oltre 200 miglia, distanza casa-lavoro, VMT annuo |

## 1. Analisi statistica

### 1.1 Statistiche descrittive

| Variabile | Media | Dev. std | Q1 | Mediana | Q3 | Max |
|---|---|---|---|---|---|---|
| Reddito familiare (USD) | 204.117,00 | 116.606,00 | 125.000,00 | 175.000,00 | 275.000,00 | 500.000,00 |
| Persone in famiglia | 2,94 | 1,24 | 2,00 | 3,00 | 4,00 | 13,00 |
| Auto in famiglia | 2,52 | 0,92 | 2,00 | 2,00 | 3,00 | 5,00 |
| Importanza ridurre emissioni (-3/+3) | 1,62 | 1,64 | 0,98 | 2,51 | 2,72 | 3,00 |
| Viaggio più lungo (miglia) | 376,74 | 356,54 | 175,35 | 317,82 | 434,35 | 4.865,70 |
| Viaggi > 200 miglia (12 mesi) | 2,46 | 5,23 | 0,00 | 1,00 | 2,00 | 65,00 |
| Distanza casa-lavoro (miglia) | 18,78 | 22,54 | 6,43 | 14,12 | 24,52 | 488,96 |
| VMT annuo (miglia) | 14.279,60 | 15.965,60 | 8.395,50 | 11.557,00 | 15.947,00 | 378.000,00 |

### 1.2 Analisi bivariata

Quota di BEV per classe di reddito: Bassa 38,5%, Media 47,7%, Alta 54,4%, Molto alta 62,2%, Estremamente alta 69,2%.

| Variabile | Test | Effetto | Misura | p-value |
|---|---|---|---|---|
| VMT annuo (miglia) | Mann-Whitney U | 0,209 | r rank-biserial (BEV vs non BEV) | < 0,001 |
| Auto precedente | Chi-quadro | 0,171 | V di Cramér | < 0,001 |
| Reddito familiare (USD) | Mann-Whitney U | 0,169 | r rank-biserial (BEV vs non BEV) | < 0,001 |
| Viaggi > 200 miglia (12 mesi) | Mann-Whitney U | 0,152 | r rank-biserial (BEV vs non BEV) | < 0,001 |
| Classe di reddito | Chi-quadro | 0,147 | V di Cramér | < 0,001 |
| Persone in famiglia | Mann-Whitney U | 0,095 | r rank-biserial (BEV vs non BEV) | < 0,001 |
| Livello di istruzione | Chi-quadro | 0,089 | V di Cramér | < 0,001 |
| Importanza ridurre emissioni (-3/+3) | Mann-Whitney U | 0,083 | r rank-biserial (BEV vs non BEV) | < 0,001 |
| Classe d'età | Chi-quadro | 0,077 | V di Cramér | < 0,001 |
| Auto in famiglia | Mann-Whitney U | 0,075 | r rank-biserial (BEV vs non BEV) | < 0,001 |
| Casa di proprietà | Chi-quadro | 0,057 | V di Cramér | < 0,001 |
| Casa indipendente | Chi-quadro | 0,052 | V di Cramér | < 0,001 |
| Genere | Chi-quadro | 0,042 | V di Cramér | < 0,001 |
| Distanza casa-lavoro (miglia) | Mann-Whitney U | 0,036 | r rank-biserial (BEV vs non BEV) | 0,001 |
| Viaggio più lungo (miglia) | Mann-Whitney U | 0,014 | r rank-biserial (BEV vs non BEV) | 0,210 |

### 1.3 Regressione logistica

Pseudo-R² di McFadden 0,064; VIF massimo 2,62.

| Variabile | Odds ratio | IC 95% | p-value |
|---|---|---|---|
| Genere maschile | 1,245 | 1,135–1,367 | < 0,001 |
| Età <25 (vs 45-54) | 1,137 | 0,425–3,044 | 0,798 |
| Età 25-34 (vs 45-54) | 0,893 | 0,705–1,132 | 0,351 |
| Età 35-44 (vs 45-54) | 1,115 | 0,988–1,259 | 0,077 |
| Età 55-64 (vs 45-54) | 0,794 | 0,710–0,887 | < 0,001 |
| Età 65-74 (vs 45-54) | 0,708 | 0,625–0,802 | < 0,001 |
| Età 75-79 (vs 45-54) | 0,922 | 0,768–1,107 | 0,384 |
| Età >80 (vs 45-54) | 0,503 | 0,306–0,826 | 0,007 |
| Reddito basso (vs medio) | 0,720 | 0,557–0,931 | 0,012 |
| Reddito alto (vs medio) | 1,217 | 1,106–1,339 | < 0,001 |
| Reddito molto alto (vs medio) | 1,608 | 1,423–1,817 | < 0,001 |
| Reddito estremamente alto (vs medio) | 2,117 | 1,805–2,481 | < 0,001 |
| Licenza Media (vs Laurea 2L) | 0,412 | 0,163–1,042 | 0,061 |
| Diploma o Qualifica professionale (vs Laurea 2L) | 0,697 | 0,612–0,794 | < 0,001 |
| Laurea 1L (vs Laurea 2L) | 0,886 | 0,812–0,966 | 0,006 |
| Auto precedente BEV (vs PHEV) | 3,474 | 2,905–4,154 | < 0,001 |
| Auto precedente HEV (vs PHEV) | 0,980 | 0,847–1,133 | 0,781 |
| Auto precedente ICE (vs PHEV) | 1,509 | 1,327–1,715 | < 0,001 |
| Auto precedente GNC (vs PHEV) | 1,452 | 0,798–2,642 | 0,222 |
| Casa di proprietà | 1,253 | 1,105–1,421 | < 0,001 |
| Auto in famiglia (+1) | 1,150 | 1,098–1,206 | < 0,001 |
| Importanza ridurre emissioni (+1 punto) | 1,105 | 1,078–1,134 | < 0,001 |
| Viaggi > 200 miglia (+1) | 0,966 | 0,957–0,974 | < 0,001 |
| Viaggio più lungo (+100 mi) | 1,004 | 0,993–1,016 | 0,466 |
| Distanza casa-lavoro (+10 mi) | 0,981 | 0,963–1,000 | 0,048 |
| VMT annuo (+1.000 mi) | 0,985 | 0,981–0,989 | < 0,001 |

### 1.4 Ipotesi di ricerca

| Ipotesi | Variabile | Odds ratio | p-value | Esito |
|---|---|---|---|---|
| H1 | Reddito basso (vs medio) | 0,720 | 0,012 | Confermata |
| H1 | Reddito alto (vs medio) | 1,217 | < 0,001 | Confermata |
| H1 | Reddito molto alto (vs medio) | 1,608 | < 0,001 | Confermata |
| H1 | Reddito estremamente alto (vs medio) | 2,117 | < 0,001 | Confermata |
| H2.1 | Auto precedente PHEV (vs ICE) | 0,663 | < 0,001 | Contraria |
| H2.2 | Auto precedente ICE (vs PHEV) | 1,509 | < 0,001 | Aumenta la probabilità |
| H2.2 | Auto precedente HEV (vs PHEV) | 0,980 | 0,781 | Non significativo |
| H2.2 | Auto precedente GNC (vs PHEV) | 1,452 | 0,222 | Non significativo |
| H2.3 | Auto precedente BEV (vs PHEV) | 3,474 | < 0,001 | Confermata |
| H3 | Viaggio più lungo (+100 mi) | 1,004 | 0,466 | Non significativo |
| H3 | Viaggi > 200 miglia (+1) | 0,966 | < 0,001 | Confermata |
| H3 | Distanza casa-lavoro (+10 mi) | 0,981 | 0,048 | Confermata |
| H3 | VMT annuo (+1.000 mi) | 0,985 | < 0,001 | Confermata |
| H4 | Importanza ridurre emissioni (+1 punto) | 1,105 | < 0,001 | Confermata |

**H2.1 non è supportata:** chi aveva una PHEV ha odds di BEV 0,66 volte quelle di chi aveva un'auto ICE. **H2.2 (esplorativa):** rispetto alla PHEV, chi aveva un'auto ICE ha odds 1,51 volte maggiori; HEV e GNC non differiscono in modo significativo.

## 2. Analisi esplorativa

Analisi svolte sui 5.796 possessori di BEV.

- **PCA sulla mobilità:** le prime due componenti spiegano il 59,1% della varianza; alpha di Cronbach 0,33.
- **MCA variabili abitative:** Dim1 20,0%, Dim2 14,3%; indici *Stile di vita* (Dim1) e *Dimensione familiare* (Dim2).
- **MCA mobilità e percorrenza:** Dim1 13,8%, Dim2 10,6%; indici *Viaggi* (Dim1) e *Mobilità quotidiana* (Dim2).
- **MCA variabili socio-demografiche:** Dim1 8,6%, Dim2 7,8%; indici *Status socioeconomico* (Dim1) e *Distanza sostenibilità* (Dim2).
- **FAMD:** le prime due dimensioni spiegano il 12,9%; Dim1 è guidata da Casa indipendente: No, Casa di proprietà: No, Auto in famiglia.

## 3. Cluster

### 3.1 K-Mode: segmentazione finale

Come nella tesi, la segmentazione finale usa il K-Mode (k = 4) sulle 14 variabili categoriali dei possessori di BEV. Molte partizioni hanno un costo quasi uguale: con semi diversi i gruppi cambiano (ARI medio 0,13), quindi i profili vanno letti come segmenti descrittivi e non come gruppi naturali.

> **Confronto R / Python.** Con le stesse variabili, R (klaR, migliore di 20 avvii) trova costo 30.724 e cluster di 1.555 / 1.554 / 1.462 / 1.225 persone; Python (kmodes, 50 inizializzazioni) costo 30.833 e cluster di 1.911 / 1.783 / 1.219 / 883; la tesi riportava 1.438 / 1.270 / 1.473 / 1.615. Sia in R sia in Python tornano un gruppo di coppie con VMT basso e uno di famiglie con 2 figli, 3 auto e pendolarismo lungo; cambia la suddivisione per reddito, età e numero di viaggi.

| Cluster | Profilo | n | Età | Reddito (USD) | VMT | Viaggi > 200 mi | Casa-lavoro (mi) |
|---|---|---|---|---|---|---|---|
| Cluster 1 | Coppie · reddito alto · età 45-64 · VMT basso · Casa-lavoro 6,5-15 mi | 1911 | 50,0 | 217.412 | 9.957 | 1,9 | 13,3 |
| Cluster 2 | Famiglie con 2 figli · reddito alto · età 45-64 · VMT alto · Casa-lavoro 25-45 mi | 1783 | 48,3 | 231.029 | 16.743 | 2,7 | 24,5 |
| Cluster 3 | Coppie · reddito medio · età >65 · VMT medio · Casa-lavoro < 6,5 mi | 1219 | 55,2 | 179.799 | 12.086 | 1,4 | 15,1 |
| Cluster 4 | Famiglie con 2 figli · reddito molto alto · età 45-64 · VMT medio · Casa-lavoro 15-25 mi | 883 | 46,3 | 258.947 | 11.875 | 1,7 | 18,4 |

| Variabile | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| Genere | Maschio | Maschio | Maschio | Maschio |
| Classe di reddito | Alta | Alta | Media | Molto alta |
| Livello di istruzione | Laurea 2L o Dottorato | Laurea 2L o Dottorato | Laurea 2L o Dottorato | Laurea 2L o Dottorato |
| Casa di proprietà | Si | Si | Si | Si |
| Casa indipendente | Si | Si | Si | Si |
| Auto precedente | ICE | ICE | ICE | ICE |
| Numero persone in famiglia | Coppie | Famiglie con 2 figli | Coppie | Famiglie con 2 figli |
| Numero di auto in famiglia | Due auto | Tre auto | Due auto | Due auto |
| VMT | VMT basso | VMT alto | VMT medio | VMT medio |
| Viaggio lungo | Viaggio 350-550 mi | Viaggio 175-350 mi | Viaggio < 175 mi | Viaggio 175-350 mi |
| Numero viaggi lunghi | 1 viaggio lungo | 1 viaggio lungo | Nessun viaggio lungo | 1 viaggio lungo |
| Distanza casa-lavoro | Casa-lavoro 6,5-15 mi | Casa-lavoro 25-45 mi | Casa-lavoro < 6,5 mi | Casa-lavoro 15-25 mi |
| Sensibilità ambientale | Molto alta | Molto alta | Molto alta | Molto alta |
| Classe età raggruppata | 45-64 | 45-64 | >65 | 45-64 |

### 3.2 Verifica con K-Means e gerarchico

K-Means con k = 4 su età, indici compositi e auto precedente: silhouette 0,40, ma ARI con l'auto precedente 0,997: i cluster coincidono con il tipo di auto posseduta prima, ed è per questo che nella tesi è stato scartato. Senza quelle variabili: silhouette 0,21, stabilità 0,88. Gerarchico (Ward): correlazione cofenetica 0,54.

## 4. Machine learning

Split stratificato 8.550 / 2.138; screening di 15 modelli in 10-fold CV; ottimizzazione con RandomizedSearchCV su ROC-AUC.

| Modello | Accuracy | F1 | ROC-AUC | IC 95% |
|---|---|---|---|---|
| Voting (soft) | 0,662 | 0,712 | 0,725 | 0,705–0,746 |
| LightGBM | 0,669 | 0,715 | 0,725 | 0,705–0,746 |
| XGBoost | 0,662 | 0,714 | 0,725 | 0,704–0,746 |
| Gradient Boosting | 0,661 | 0,710 | 0,721 | 0,700–0,742 |
| Random Forest | 0,664 | 0,712 | 0,721 | 0,700–0,743 |
| Gradient Boosting (feature selezionate) | 0,661 | 0,709 | 0,720 | 0,699–0,741 |
| AdaBoost | 0,654 | 0,703 | 0,698 | 0,676–0,720 |
| MLP | 0,649 | 0,695 | 0,691 | 0,667–0,714 |
| SVM | 0,638 | 0,696 | 0,679 | 0,656–0,702 |
| Regressione logistica | 0,626 | 0,680 | 0,664 | 0,642–0,686 |

Variabili più importanti (SHAP, LightGBM): N. viaggi lunghi, VMT annuo, Reddito, BEV precedente, Sensibilità ambientale, Classe età.

Senza *auto precedente BEV* la ROC-AUC passa da 0,725 a 0,719.

## 5. Deep learning

La *ANN avanzata* riproduce il modello finale della tesi: 189 feature polinomiali di grado 2, LeakyReLU, regolarizzazione L1 + L2, BatchNormalization e AdamW. Qui il set di validazione è estratto dal training (nella tesi coincideva con il test set): accuracy 64,6%, ROC-AUC 0,672.

| Rete | Input | Strati | Attivazione | Parametri | Epoche | Accuracy | ROC-AUC | IC 95% |
|---|---|---|---|---|---|---|---|---|
| ANN regolarizzata | 18 | 64 → 32 | ReLU | 3.329 | 92 | 0,651 | 0,694 | 0,672–0,716 |
| ANN base | 18 | 16 → 64 → 128 → 64 | ReLU | 18.033 | 48 | 0,630 | 0,683 | 0,662–0,706 |
| ANN avanzata (tesi) | 189 | 64 → 128 → 256 → 128 → 64 | LeakyReLU | 94.977 | 189 | 0,646 | 0,672 | 0,648–0,695 |

## Limiti e conclusioni

- **Contesto geografico:** solo California e solo possessori di veicoli a basse emissioni.
- **Capacità esplicativa contenuta:** pseudo-R² 0,064, ROC-AUC massima 0,725.
- **Variabili mancanti:** nessuna variabile psicologica o culturale, nessuna misura concreta dell'attitudine verso la sostenibilità.
- **Fattori influenti:** reddito, miglia annue e numero di viaggi lunghi; conta anche la sensibilità ambientale.
- **Il passaggio non è graduale:** chi arriva da un'auto tradizionale sceglie il BEV più spesso di chi aveva una PHEV.
- **Transizione culturale, non solo tecnologica**, sostenuta da politiche mirate.
