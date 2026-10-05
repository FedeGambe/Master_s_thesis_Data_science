"""Genera i notebook di presentazione in ``notebooks/``.

I notebook non contengono logica di analisi: importano le funzioni da ``src/tesi_bev`` e
mostrano tabelle e grafici. Funzionano in locale e su Google Colab (la prima cella clona il
repository e installa le dipendenze mancanti).

Uso:
    python scripts/genera_notebook.py            # crea i notebook senza output
    python scripts/genera_notebook.py --esegui   # li esegue e salva gli output (grafici anche in PNG)
"""
import argparse
import os
import sys
from pathlib import Path

import nbformat as nbf

RADICE = Path(__file__).resolve().parents[1]
DIR_NB = RADICE / "notebooks"
REPO = "FedeGambe/Master_s_thesis_Data_science"
RAMO = "revisione-progetto"


def badge(nome):
    url = f"https://colab.research.google.com/github/{REPO}/blob/{RAMO}/notebooks/{nome}"
    return f"[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


SETUP = f'''# Configurazione dell'ambiente (locale o Google Colab)
import os, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    if not os.path.exists("Master_s_thesis_Data_science"):
        !git clone -q -b {RAMO} https://github.com/{REPO}.git
    %cd Master_s_thesis_Data_science
    !pip install -q -r requirements-colab.txt
elif os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, "src")

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import plotly.io as pio
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{{:.4f}}".format)
if os.environ.get("TESI_GRAFICI_STATICI"):
    pio.renderers.default = "png"  # usato solo per salvare output visibili su GitHub
    pio.defaults.default_width = 1000
'''


def md(testo):
    return nbf.v4.new_markdown_cell(testo.strip())


def code(testo):
    return nbf.v4.new_code_cell(testo.strip())


def intestazione(nome, titolo, descrizione):
    return [md(f"# {titolo}\n\n{badge(nome)}\n\n{descrizione}"), code(SETUP)]


# --------------------------------------------------------------------------------------------
NOTEBOOK = {}

NOTEBOOK["01_dati_e_analisi_descrittive.ipynb"] = lambda n: [
    *intestazione(n, "1. Dati e analisi descrittive",
                  "Caricamento del dataset, controlli di qualità, distribuzioni univariate e relazione di ogni "
                  "variabile con il possesso di un BEV (capitoli 1-3 della tesi)."),
    code("""
from tesi_bev import config as C, dati, statistica as S, grafici as G
df = dati.carica_dati()
print(f"{df.shape[0]} osservazioni, {df.shape[1]} variabili, {df.isna().sum().sum()} valori mancanti, "
      f"{df.duplicated().sum()} duplicati")
df.head()
"""),
    md("""
## 1.1 Popolazione e variabile dipendente

Tutti i rispondenti possiedono già un veicolo a basse emissioni (BEV, PHEV, FCEV o HEV): il modello quindi non
stima la probabilità di comprare *un'auto elettrica* in generale, ma la probabilità che il veicolo posseduto sia un
**BEV** rispetto alle altre alimentazioni alternative.
"""),
    code("""
pd.concat([S.frequenze(df, C.TIPO_AUTO_ATTUALE), S.frequenze(df, C.TARGET)], keys=["Auto attuale", "BEV dummy"])
"""),
    md("## 1.2 Variabili quantitative"),
    code("S.descrittive_numeriche(df)"),
    code("""
for col in [C.REDDITO, C.EMISSIONI, C.VMT, C.N_VIAGGI_LUNGHI]:
    G.distribuzione(df, col, log_x=col in (C.VMT,)).show()
"""),
    md("## 1.3 Quota di BEV per categoria"),
    code("""
for col, ordine in [(C.CLASSE_REDDITO, C.ORDINE_REDDITO), (C.CLASSE_ETA, C.ORDINE_ETA),
                    (C.ISTRUZIONE, C.ORDINE_ISTRUZIONE), (C.AUTO_PRECEDENTE, C.TIPI_AUTO)]:
    G.quota_bev(df, col, ordine).show()
"""),
    md("""
## 1.4 Test bivariati

Chi-quadro con V di Cramér per le variabili categoriali, Mann-Whitney con correlazione rank-biserial per quelle
numeriche (le distribuzioni sono fortemente asimmetriche, quindi un test non parametrico è più adatto del t-test).
Con oltre 10.000 osservazioni quasi tutto risulta significativo: conviene guardare la **dimensione dell'effetto**.
"""),
    code("S.test_bivariati(df).sort_values('effetto', key=abs, ascending=False)"),
    md("## 1.5 Correlazioni tra le feature dei modelli"),
    code("""
X, y = dati.prepara_ml(df)
G.matrice_correlazione(X).show()
"""),
]

NOTEBOOK["02_regressione_logistica.ipynb"] = lambda n: [
    *intestazione(n, "2. Regressione logistica e ipotesi di ricerca",
                  "Verifica delle ipotesi H1-H4 (capitoli 4-5 della tesi) con un modello logit. Rispetto alla "
                  "versione originale: VIF e pseudo-R² calcolati con l'intercetta, intervalli di confidenza degli "
                  "odds ratio, effetti marginali medi e variabili di percorrenza riscalate per avere OR leggibili."),
    md("""
## Ipotesi

| Codice | Ipotesi |
|---|---|
| **H1** | Un reddito più alto aumenta la probabilità di possedere un BEV |
| **H2.1** | Aver posseduto una PHEV favorisce il passaggio al BEV |
| **H2.2** | Come cambia la probabilità per chi aveva un'auto ICE, HEV o GNC? (esplorativa, senza direzione attesa) |
| **H2.3** | Chi ha già avuto un BEV tende a ricomprarlo |
| **H3** | Lunghe percorrenze riducono la probabilità di possedere un BEV |
| **H4** | Chi dà importanza alla riduzione delle emissioni sceglie più spesso un BEV |
"""),
    code("""
from tesi_bev import config as C, dati, statistica as S, grafici as G
df = dati.carica_dati()
X_completo, y = dati.prepara_logit(df, ridotto=False)
"""),
    md("""
## 2.1 Multicollinearità (VIF)

Il VIF è calcolato con l'intercetta: nessuna variabile supera 3. Il modello stimato è quello della tesi, che esclude
*Numero persone in famiglia* e *Casa indipendente* (correlate con *Numero di auto* e *Casa di proprietà*), e viene
confrontato con il modello completo: la capacità esplicativa è praticamente la stessa.
"""),
    code("S.vif(X_completo)"),
    md("## 2.2 Stima del modello"),
    code("""
X, y = dati.prepara_logit(df, ridotto=True)
risultato = S.logit(X, y)
pd.DataFrame({"completo": S.bonta_adattamento(S.logit(X_completo, y)), "ridotto": S.bonta_adattamento(risultato)})
"""),
    code("""
tabella = S.tabella_logit(risultato)
tabella
"""),
    code("G.forest_plot(tabella).show()"),
    md("""
## 2.3 Effetti marginali medi

L'odds ratio non è una variazione di probabilità (nella tesi un OR di 1,22 era letto come "+22% di probabilità").
L'effetto marginale medio indica di quanti **punti percentuali** cambia in media la probabilità di possedere un BEV.
"""),
    code("S.effetti_marginali(risultato).sort_values('effetto_marginale')"),
    md("## 2.4 Verifica delle ipotesi"),
    code("""
ipotesi = S.verifica_ipotesi(risultato)
ipotesi[["ipotesi", "variabile", "odds_ratio", "p_value", "direzione_attesa", "esito"]]
"""),
    md("""
**Lettura.** H1, H2.3 e H4 sono confermate. H3 è confermata da numero di viaggi lunghi, VMT e distanza casa-lavoro
(quest'ultima al limite della significatività), ma non dal viaggio più lungo. **H2.1 non è supportata**: chi aveva
una PHEV ha una probabilità *minore* di possedere un BEV rispetto a chi aveva un'auto tradizionale (ICE).
**H2.2** (esplorativa): rispetto alla PHEV, aver avuto un'auto ICE aumenta la probabilità di possedere un BEV, mentre
HEV e GNC non differiscono in modo significativo.
"""),
]

NOTEBOOK["03_analisi_esplorative.ipynb"] = lambda n: [
    *intestazione(n, "3. Analisi esplorative multivariate: PCA, MCA e FAMD",
                  "Versione Python degli script R (cartella `R/`). Le analisi sono svolte sui **soli possessori di "
                  "BEV**, come nella tesi, per costruire gli indici sintetici usati poi nel clustering. I segni "
                  "delle componenti possono essere invertiti rispetto a R: gli assi sono definiti a meno del segno."),
    code("""
from tesi_bev import config as C, dati, esplorativa as E, grafici as G
bev = dati.solo_bev(dati.crea_variabili_categoriali())
print(f"Possessori di BEV: {len(bev)}")
"""),
    md("""
## 3.1 Variabili categoriali derivate

Le variabili quantitative sono discretizzate con le soglie della tesi. Da notare che la classe *VMT molto alto*
contiene pochissime osservazioni, perché la soglia cade a metà tra il terzo quartile e il massimo.
"""),
    code("""
pd.concat({c: bev[c].value_counts() for c in ["VMT categoriale", "Sensibilità ambientale categoriale",
                                             "Numero viaggi lunghi categoriale"]}).to_frame("n")
"""),
    md("## 3.2 PCA sulle variabili di mobilità"),
    code("""
pca = E.pca(bev, C.VARIABILI_MOBILITA)
print(f"Alpha di Cronbach: {E.alpha_cronbach(bev[C.VARIABILI_MOBILITA]):.3f}")
G.scree(pca["varianza"], "PCA mobilità: varianza spiegata").show()
pca["loadings"]
"""),
    code("G.loadings(pca['loadings'], 'PCA mobilità: loadings PC1-PC2').show()"),
    md("""
L'alpha di Cronbach è bassa (≈0,33): le quattro variabili di mobilità misurano aspetti diversi e la prima componente
spiega solo un terzo della varianza. Gli indici *Viaggi PCA* e *Mobilità quotidiana PCA* vanno quindi interpretati
con cautela.

## 3.3 MCA tematiche
"""),
    code("""
mca = E.mca_gruppi(bev)
for nome, r in mca.items():
    print(f"MCA {nome}: Dim1+Dim2 = {r['autovalori']['cumulata_%'].iloc[1]:.1f}% dell'inerzia")
    G.mappa_categorie(r["categorie"], f"MCA {nome}").show()
"""),
    code("mca['sociodemografiche']['categorie'].sort_values('contributo_Dim1_%', ascending=False)"),
    md("## 3.4 FAMD (variabili miste)"),
    code("""
famd = E.famd(bev, n_componenti=10)
G.scree(famd["autovalori"], "FAMD: varianza spiegata").show()
famd["contributi"][["contributo_Dim1_%", "contributo_Dim2_%"]].sort_values("contributo_Dim1_%", ascending=False)
"""),
    md("""
## 3.5 Indici compositi

Le prime due dimensioni di ogni analisi diventano gli indici usati nel clustering. Correzione rispetto allo script R
originale: *Status socioeconomico* e *Distanza sostenibilità* erano copiati per errore dall'MCA sulla mobilità.
"""),
    code("E.indici_compositi(bev).describe()"),
    md("## 3.6 Confronto con R"),
    code("""
from pathlib import Path
if Path("risultati/r/pca_mobilita_varianza.csv").exists():
    r_pca = pd.read_csv("risultati/r/pca_mobilita_varianza.csv", index_col=0)
    confronto = pd.DataFrame({"Python": pca["varianza"]["varianza_%"].values, "R": r_pca["varianza_perc"].values},
                             index=pca["varianza"].index)
    display(confronto)
else:
    print("Eseguire 'Rscript R/esegui_tutto.R' per il confronto con R")
"""),
]

NOTEBOOK["04_cluster.ipynb"] = lambda n: [
    *intestazione(n, "4. Segmentazione dei possessori di BEV",
                  "Come nella tesi, la segmentazione finale usa il K-Mode sulle variabili categoriali (k = 4). "
                  "Il K-Means sugli indici compositi e il clustering gerarchico servono da verifica."),
    code("""
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tesi_bev import config as C, dati, esplorativa as E, cluster as K, grafici as G

bev = dati.solo_bev(dati.crea_variabili_categoriali())
df = K.dataset_cluster(bev, E.indici_compositi(bev))
K.VARIABILI_KMODE
"""),
    md("""
## 4.1 K-Mode: scelta del numero di cluster

Il K-Mode usa la distanza di Hamming (numero di caratteristiche diverse) e rappresenta ogni cluster con la modalità
più frequente di ciascuna variabile. Il costo cala in modo regolare: k = 4 è scelto, come nella tesi, per avere
profili ancora interpretabili.
"""),
    code("""
costo = K.costo_kmode(df[K.VARIABILI_KMODE], range(1, 9))
G.gomito_silhouette(costo.rename(columns={"costo": "wss"}).assign(silhouette=None)).show()
costo
"""),
    md("## 4.2 K-Mode con k = 4: i profili (circa 1 minuto)"),
    code("""
km_mode = K.kmode(df, k=4)
df["cluster_kmode"] = km_mode["etichette"]
print(f"Costo {km_mode['costo']:.0f} - dimensioni {df['cluster_kmode'].value_counts().sort_index().tolist()}")
km_mode["centri"].T
"""),
    code("""
profilo_num = ["Età", C.REDDITO, C.N_PERSONE, C.N_AUTO, C.VMT, C.N_VIAGGI_LUNGHI, C.DISTANZA_LAVORO, C.EMISSIONI]
df.groupby("cluster_kmode")[profilo_num].mean().round(1)
"""),
    md("""
## 4.3 Stabilità del K-Mode

Molte partizioni hanno un costo quasi uguale: ripetendo l'algoritmo con semi diversi i gruppi cambiano. Un ARI vicino
a 0 indica partizioni molto diverse, quindi i profili vanno letti come **segmenti descrittivi**, non come gruppi
naturali. Il calcolo richiede alcuni minuti.
"""),
    code("""
CALCOLA_STABILITA = False
if CALCOLA_STABILITA:
    print(f"ARI medio tra semi diversi: {K.stabilita_kmode(df, k=4):.3f}")
"""),
    md("## 4.4 Verifica con il K-Means (configurazione della tesi)"),
    code("""
X = df[K.VARIABILI_KMEANS]
scelta = K.gomito_silhouette(X, range(2, 9))
G.gomito_silhouette(scelta).show()
km = K.kmeans(X, k=4)
print(f"WSS = {km['wss']:.0f} (tesi, in R: 34.630) - silhouette = {km['silhouette']:.3f} (tesi: 0,395)")
coord = PCA(2).fit_transform(StandardScaler().fit_transform(X))
G.cluster_2d(coord, km["etichette"]).show()
"""),
    md("""
Le 5 variabili dummy dell'auto precedente, una volta standardizzate, dominano la distanza euclidea: i 4 cluster
coincidono quasi esattamente con il tipo di auto posseduta in precedenza. È il motivo per cui nella tesi il K-Means è
stato scartato a favore del K-Mode.
"""),
    code("""
print(f"ARI tra K-Means e auto precedente: {adjusted_rand_score(df[C.AUTO_PRECEDENTE], km['etichette']):.3f}")
print(f"ARI tra K-Means e K-Mode: {adjusted_rand_score(df['cluster_kmode'], km['etichette']):.3f}")
G.composizione_cluster(df, km["etichette"], C.AUTO_PRECEDENTE).show()
pd.crosstab(km["etichette"], df[C.AUTO_PRECEDENTE])
"""),
    md("## 4.5 K-Means senza le dummy dell'auto precedente"),
    code("""
variabili = [v for v in K.VARIABILI_KMEANS if not v.startswith("Auto precedente")] + ["Status socioeconomico", C.EMISSIONI]
X2 = df[variabili]
G.gomito_silhouette(K.gomito_silhouette(X2, range(2, 9))).show()
km2 = K.kmeans(X2, k=4)
print(f"Silhouette: {km2['silhouette']:.3f} - stabilità (ARI bootstrap): {K.stabilita_bootstrap(X2, 4, 20):.3f}")
K.profilo_cluster(df, km2["etichette"], ["Età", C.REDDITO, C.EMISSIONI, C.N_AUTO, C.VMT, C.N_VIAGGI_LUNGHI],
                  [C.CLASSE_REDDITO, C.ISTRUZIONE, C.CASA_INDIPENDENTE, C.AUTO_PRECEDENTE])
"""),
    md("""
Anche senza le dummy non emerge una struttura a gruppi netta, coerentemente con l'instabilità del K-Mode.

## 4.6 Clustering gerarchico (Ward)
"""),
    code("""
h = K.gerarchico(X, k=4)
print(f"Gerarchico Ward: correlazione cofenetica {h['correlazione_cofenetica']:.3f}, silhouette {h['silhouette']:.3f}, "
      f"ARI con K-Means {adjusted_rand_score(km['etichette'], h['etichette']):.3f}")
"""),
]

NOTEBOOK["05_machine_learning.ipynb"] = lambda n: [
    *intestazione(n, "5. Modelli predittivi di machine learning",
                  "Confronto di 15 algoritmi e ottimizzazione dei migliori, senza data leakage. I risultati sono "
                  "prodotti da `scripts/05_machine_learning.py` (circa 20 minuti) e salvati in `risultati/`: il "
                  "notebook li carica. Impostare `RIESEGUI = True` per ricalcolare tutto."),
    code("""
RIESEGUI = False

from pathlib import Path
import numpy as np
from tesi_bev import archivio as A, config as C, dati, grafici as G
if RIESEGUI or not Path("risultati/machine_learning/risultati_test.csv").exists():
    !python scripts/05_machine_learning.py
SEZ = "machine_learning"
"""),
    md("""
## Cosa è cambiato rispetto alla tesi

| Problema nella versione originale | Correzione |
|---|---|
| `scaler.fit_transform(X_test)`: scaler riaddestrato sul test | Scaler dentro una `Pipeline`, addestrato solo sui fold di training |
| Selezione delle feature su tutto il dataset | `SelectKBest` dentro la pipeline, `k` ottimizzato in CV |
| Stimatore del "fold migliore" usato sul test | `refit=True`: modello riaddestrato su tutto il training |
| Split diversi tra modelli (`random_state` 343 e 42) | Un unico split stratificato per tutti |
| Tabella finale con metriche su righe sbagliate | Metriche calcolate da un'unica funzione per modello |
| Solo accuracy | ROC-AUC per l'ottimizzazione, IC bootstrap, test di McNemar, calibrazione |
"""),
    code("A.carica_json(SEZ, 'split')"),
    md("## 5.1 Screening in cross-validation (solo training)"),
    code("""
screening = A.carica_tabella(SEZ, "screening_cv", indice=False)
G.box_screening(screening, "roc_auc").show()
A.carica_tabella(SEZ, "screening_riepilogo")
"""),
    md("## 5.2 Iperparametri ottimizzati"),
    code("""
ottimizzazione = A.carica_json(SEZ, "ottimizzazione")
pd.DataFrame({k: {"ROC-AUC CV": v.get("roc_auc_cv"), "tempo (s)": v.get("tempo_s"), "parametri": v.get("parametri", v.get("componenti"))}
              for k, v in ottimizzazione.items()}).T
"""),
    md("## 5.3 Risultati sul test set"),
    code("""
risultati = A.carica_tabella(SEZ, "risultati_test")
G.barre_metriche(risultati, ic=risultati).show()
risultati[["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_ic_basso", "roc_auc_ic_alto", "brier"]]
"""),
    md("""
**Il migliore è davvero migliore?** Il test di McNemar confronta gli errori del modello migliore con quelli di ogni
altro modello sullo stesso test set. Un p-value alto significa che la differenza non è statisticamente significativa.
"""),
    code("A.carica_tabella(SEZ, 'mcnemar_vs_migliore')"),
    code("""
roc = A.carica_json(SEZ, "curve_roc")
G.curve_roc({k: v for k, v in roc.items() if k in risultati.index[:5].tolist() + ["Regressione logistica"]}).show()
cal = A.carica_json(SEZ, "calibrazione")
G.calibrazione({k: cal[k] for k in risultati.index[:3]}).show()
"""),
    code("""
migliore = risultati.index[0]
G.matrice_confusione(risultati.loc[migliore], f"Matrice di confusione - {migliore}").show()
"""),
    md("## 5.4 Confronto con i risultati della tesi"),
    code("A.carica_tabella(SEZ, 'confronto_tesi_revisione')"),
    md("""
## 5.5 Quanto pesa l'auto precedente?

*Auto precedente: BEV* è il predittore più forte, ma anche il meno informativo dal punto di vista di marketing: chi
aveva già un BEV tende a ricomprarlo. Togliendolo si misura quanto le altre caratteristiche spiegano da sole.
"""),
    code("A.carica_tabella(SEZ, 'ablation_bev_precedente')[['accuracy', 'f1', 'roc_auc']]"),
    md("## 5.6 Interpretabilità: permutation importance e SHAP"),
    code("""
G.importanza(A.carica_tabella(SEZ, "importanza_permutazione", indice=False), "importanza",
             "Permutation importance (calo di ROC-AUC sul test)").show()
G.importanza(A.carica_tabella(SEZ, "importanza_shap", indice=False), "shap_medio_assoluto",
             "Importanza SHAP (|valore| medio)").show()
"""),
    code("""
import shap, matplotlib.pyplot as plt
valori = np.load("risultati/machine_learning/shap_valori.npy")
campione = pd.read_csv("risultati/machine_learning/shap_campione.csv")
shap.summary_plot(valori, campione.rename(columns=C.nome_breve), show=False)
plt.title(f"SHAP - {A.carica_json(SEZ, 'modello_finale')['modello_shap']}")
plt.show()
"""),
]

NOTEBOOK["06_reti_neurali.ipynb"] = lambda n: [
    *intestazione(n, "6. Reti neurali",
                  "Reti Keras addestrate sullo stesso split dei modelli di machine learning, compresa la rete avanzata "
                  "della tesi (189 feature polinomiali, LeakyReLU, L1 + L2, BatchNormalization, AdamW). Differenza "
                  "chiave rispetto alla tesi: l'early stopping usa un set di **validazione** estratto dal training; "
                  "nella versione originale usava il test set, rendendo ottimistica la stima finale. L'addestramento "
                  "delle tre reti richiede circa 2 minuti."),
    code("""
from tesi_bev import dati, modelli as M, valutazione as V, grafici as G
from tesi_bev.reti_neurali import ARCHITETTURE, ReteNeurale
X, y = dati.prepara_ml()
X_tr, X_te, y_tr, y_te = M.dividi(X, y)
pd.DataFrame(ARCHITETTURE).T
"""),
    code("""
reti, prob = {}, {}
for nome in ARCHITETTURE:
    reti[nome] = ReteNeurale(nome).fit(X_tr, y_tr)
    prob[nome] = reti[nome].predict_proba(X_te)[:, 1]
    G.storia_addestramento(reti[nome].storia_, f"{nome}: {len(reti[nome].storia_['loss'])} epoche").show()
"""),
    code("""
risultati = V.tabella_risultati(prob, y_te).join(V.tabella_ic(prob, y_te, n=500))
risultati[["accuracy", "f1", "roc_auc", "roc_auc_ic_basso", "roc_auc_ic_alto", "brier"]]
"""),
    md("## Confronto con i modelli ad alberi"),
    code("""
from pathlib import Path
if Path("risultati/machine_learning/risultati_test.csv").exists():
    ml = pd.read_csv("risultati/machine_learning/risultati_test.csv", index_col=0)
    display(pd.concat([ml.head(3), risultati])[["accuracy", "f1", "roc_auc", "roc_auc_ic_basso", "roc_auc_ic_alto"]])
"""),
    md("""
Su dati tabellari di questa dimensione le reti neurali non superano i modelli a gradient boosting. La rete avanzata,
con quasi 30 volte i parametri della rete regolarizzata, non migliora la ROC-AUC: le feature polinomiali aumentano la
complessità senza aggiungere informazione.
"""),
]

NOTEBOOK["07_dashboard.ipynb"] = lambda n: [
    *intestazione(n, "7. Dashboard di predizione",
                  "Dashboard interattiva (Dash) che usa il modello finale salvato in `modelli/`. In locale si può "
                  "avviare anche con `python app/dashboard.py`."),
    md("""
Correzioni rispetto alla dashboard originale:

* il genere era codificato al contrario rispetto al training (Maschio = 0 invece di 1)
* gli input non venivano standardizzati prima della trasformazione polinomiale
* veniva mostrata come "accuratezza della predizione" l'accuratezza globale del modello
* il preprocessing era duplicato a mano; ora la dashboard usa la stessa funzione `dati.codifica_ml` del training
"""),
    code("""
sys.path.insert(0, "app")
from dashboard import crea_app, carica_modello_finale, predici
modello, meta = carica_modello_finale()
print(f"Modello: {meta['modello']} - ROC-AUC test: {meta['metriche_test']['roc_auc']:.3f}")
"""),
    md("## Esempio di predizione"),
    code("""
esempio = {"Genere": "Maschio", "Classe d'età": "45-54", "Reddito familiare": 225000,
           "Livello di istruzione": "Laurea 2L o Dottorato", "Casa di proprietà": "Si", "Casa Indipendente": "Si",
           "Numero persone in famiglia": 3, "Numero di auto in famiglia": 2,
           "Importanza di ridurre le emissioni di gas serra": 2.5, "Tipologia di auto precedente": "ICE",
           "Viaggio più lungo negli ultimi 12 mesi": 300, "Numero di viaggi superiori a 200 miglia negli ultimi 12 mesi": 1,
           "Distanza casa-lavoro": 12, "VMT annuo": 11000}
print(f"Probabilità che il veicolo sia un BEV: {predici(modello, esempio):.1%}")
"""),
    md("## Avvio della dashboard"),
    code("""
app = crea_app(modello, meta)
if not os.environ.get("TESI_GRAFICI_STATICI"):  # non avviare il server durante la generazione automatica
    app.run(jupyter_mode="inline" if not IN_COLAB else "external", port=8050)
"""),
]


def costruisci(esegui=False, solo=None):
    DIR_NB.mkdir(exist_ok=True)
    for nome, fabbrica in NOTEBOOK.items():
        if solo and not nome.startswith(solo):
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = fabbrica(nome)
        nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                       "language_info": {"name": "python"}, "colab": {"provenance": []}}
        percorso = DIR_NB / nome
        if esegui:
            from nbclient import NotebookClient
            os.environ["TESI_GRAFICI_STATICI"] = "1"
            print(f"Esecuzione: {nome}")
            NotebookClient(nb, timeout=600, kernel_name="python3",
                           resources={"metadata": {"path": str(DIR_NB)}}).execute()
        nbf.write(nb, percorso)
        print(f"Scritto: {percorso.relative_to(RADICE)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--esegui", action="store_true")
    parser.add_argument("--solo", help="prefisso del notebook da generare, es. 05")
    argomenti = parser.parse_args()
    costruisci(argomenti.esegui, argomenti.solo)
    sys.exit(0)
