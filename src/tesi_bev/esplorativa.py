"""Analisi esplorative multivariate: PCA, MCA e FAMD.

Versione Python degli script R (cartella ``R/``). PCA con scikit-learn,
MCA e FAMD con ``prince``, che segue le stesse definizioni di FactoMineR.
I segni delle componenti possono differire da R: è normale, gli assi sono definiti a meno del segno.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import prince
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config as C
from .dati import VARIABILI_MCA


def alpha_cronbach(df: pd.DataFrame) -> float:
    """Alpha di Cronbach sulle variabili standardizzate (equivale a psych::alpha con std.alpha)."""
    z = (df - df.mean()) / df.std(ddof=1)
    k = z.shape[1]
    return float(k / (k - 1) * (1 - z.var(ddof=1).sum() / z.sum(axis=1).var(ddof=1)))


# --- PCA -------------------------------------------------------------------------------------

def pca(df: pd.DataFrame, colonne, n_componenti=None, inverti_segno=False):
    """PCA su variabili standardizzate (come ``prcomp(scale=TRUE)`` in R)."""
    Z = StandardScaler().fit_transform(df[colonne])
    modello = PCA(n_components=n_componenti).fit(Z)
    segno = -1 if inverti_segno else 1
    nomi = [f"PC{i + 1}" for i in range(modello.n_components_)]
    loadings = pd.DataFrame(segno * modello.components_.T, index=colonne, columns=nomi)
    varianza = pd.DataFrame({
        "autovalore": modello.explained_variance_,
        "varianza_%": modello.explained_variance_ratio_ * 100,
        "cumulata_%": np.cumsum(modello.explained_variance_ratio_) * 100,
    }, index=nomi)
    punteggi = pd.DataFrame(segno * modello.transform(Z), columns=nomi, index=df.index)
    return {"modello": modello, "loadings": loadings, "varianza": varianza, "punteggi": punteggi}


# --- MCA -------------------------------------------------------------------------------------

def mca(df: pd.DataFrame, colonne, n_componenti=2, seed=C.SEED):
    dati = df[colonne].astype(str)
    modello = prince.MCA(n_components=n_componenti, random_state=seed).fit(dati)
    autovalori = modello.eigenvalues_summary.copy()
    autovalori.columns = ["autovalore", "varianza_%", "cumulata_%"]
    for c in autovalori.columns[1:]:
        autovalori[c] = autovalori[c].astype(str).str.rstrip("%").str.replace(",", "").astype(float)
    nomi = [f"Dim{i + 1}" for i in range(n_componenti)]
    coord = modello.column_coordinates(dati)
    coord.columns = nomi
    contrib = modello.column_contributions_ * 100
    contrib.columns = [f"contributo_{n}_%" for n in nomi]
    punteggi = modello.row_coordinates(dati)
    punteggi.columns = nomi
    return {"modello": modello, "autovalori": autovalori, "categorie": coord.join(contrib), "punteggi": punteggi}


def mca_gruppi(df_bev: pd.DataFrame) -> dict:
    """Le tre MCA tematiche della tesi (abitative, mobilità, socio-demografiche) sui possessori di BEV."""
    return {nome: mca(df_bev, colonne) for nome, colonne in VARIABILI_MCA.items()}


# --- FAMD ------------------------------------------------------------------------------------

COLONNE_FAMD = [
    C.GENERE, C.CLASSE_ETA, C.CLASSE_REDDITO, C.ISTRUZIONE, C.CASA_PROPRIETA, C.CASA_INDIPENDENTE,
    C.AUTO_PRECEDENTE, C.N_PERSONE, C.N_AUTO, C.EMISSIONI, *C.VARIABILI_MOBILITA,
]


def famd(df: pd.DataFrame, colonne=None, n_componenti=10, seed=C.SEED):
    colonne = colonne or COLONNE_FAMD
    dati = df[colonne].copy()
    for c in dati.columns:
        dati[c] = dati[c].astype(str) if dati[c].dtype == object or str(dati[c].dtype) in ("string", "str") else dati[c].astype(float)
    modello = prince.FAMD(n_components=n_componenti, random_state=seed).fit(dati)
    autovalori = modello.eigenvalues_summary.copy()
    autovalori.columns = ["autovalore", "varianza_%", "cumulata_%"]
    for c in autovalori.columns[1:]:
        autovalori[c] = autovalori[c].astype(str).str.rstrip("%").str.replace(",", "").astype(float)
    contrib = modello.column_contributions_ * 100
    contrib.columns = [f"contributo_Dim{i + 1}_%" for i in range(contrib.shape[1])]
    punteggi = modello.row_coordinates(dati)
    punteggi.columns = [f"Dim{i + 1}" for i in range(punteggi.shape[1])]
    return {"modello": modello, "autovalori": autovalori, "contributi": contrib, "punteggi": punteggi}


# --- Indici compositi per il clustering ------------------------------------------------------

def indici_compositi(df_bev: pd.DataFrame) -> pd.DataFrame:
    """Costruisce gli indici sintetici usati nel clustering della tesi.

    * Stile di vita / Dimensione familiare: dimensioni 1-2 dell'MCA sulle variabili abitative
    * Viaggi / Mobilità quotidiana: dimensioni 1-2 dell'MCA sulla mobilità
    * Status socioeconomico / Distanza sostenibilità: dimensioni 1-2 dell'MCA socio-demografica
      (nello script R originale erano copiate per errore dall'MCA sulla mobilità)
    * Viaggi PCA / Mobilità quotidiana PCA: componenti 1-2 della PCA sulle 4 variabili di mobilità
    """
    gruppi = mca_gruppi(df_bev)
    p = pca(df_bev, C.VARIABILI_MOBILITA, n_componenti=2)["punteggi"]
    return pd.DataFrame({
        "Stile di vita": gruppi["abitative"]["punteggi"]["Dim1"],
        "Dimensione familiare": gruppi["abitative"]["punteggi"]["Dim2"],
        "Viaggi": gruppi["mobilita"]["punteggi"]["Dim1"],
        "Mobilità quotidiana": gruppi["mobilita"]["punteggi"]["Dim2"],
        "Status socioeconomico": gruppi["sociodemografiche"]["punteggi"]["Dim1"],
        "Distanza sostenibilità": gruppi["sociodemografiche"]["punteggi"]["Dim2"],
        "Viaggi PCA": p["PC1"],
        "Mobilità quotidiana PCA": p["PC2"],
    }, index=df_bev.index)
