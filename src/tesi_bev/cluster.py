"""Clustering dei possessori di BEV: K-Means, K-Mode, K-Prototype e gerarchico."""
from __future__ import annotations

import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config as C
from .dati import VARIABILI_KMODE

# Variabili del cluster finale della tesi ("data_clus3" nello script R)
VARIABILI_KMEANS = [
    "Età", "Dimensione familiare", "Stile di vita",
    *[f"Auto precedente: {t}" for t in ["HEV", "GNC", "BEV", "PHEV", "ICE"]],
    "Mobilità quotidiana PCA", "Viaggi PCA",
]


def dataset_cluster(df_bev_cat: pd.DataFrame, indici: pd.DataFrame) -> pd.DataFrame:
    """Unisce variabili originali, categoriali e indici compositi in un unico dataset per i cluster."""
    out = df_bev_cat.join(indici)
    for t in C.TIPI_AUTO:
        out[f"Auto precedente: {t}"] = (out[C.AUTO_PRECEDENTE] == t).astype(int)
    return out


# --- Scelta del numero di cluster ------------------------------------------------------------

def gomito_silhouette(X, k_valori=range(2, 11), seed=C.SEED, campione_silhouette=3000) -> pd.DataFrame:
    """Inerzia (WSS) e silhouette media di K-Means per diversi k."""
    Z = StandardScaler().fit_transform(X)
    righe = []
    for k in k_valori:
        km = KMeans(n_clusters=k, n_init=25, random_state=seed).fit(Z)
        sil = silhouette_score(Z, km.labels_, sample_size=min(campione_silhouette, len(Z)), random_state=seed)
        righe.append({"k": k, "wss": km.inertia_, "silhouette": sil})
    return pd.DataFrame(righe)


def costo_kmode(df_cat: pd.DataFrame, k_valori=range(1, 9), seed=C.SEED) -> pd.DataFrame:
    righe = []
    for k in k_valori:
        km = KModes(n_clusters=k, init="Huang", n_init=5, random_state=seed).fit(df_cat.astype(str))
        righe.append({"k": k, "costo": km.cost_})
    return pd.DataFrame(righe)


# --- Algoritmi -------------------------------------------------------------------------------

def kmeans(X: pd.DataFrame, k=4, seed=C.SEED):
    Z = StandardScaler().fit_transform(X)
    modello = KMeans(n_clusters=k, n_init=25, random_state=seed).fit(Z)
    centri = pd.DataFrame(modello.cluster_centers_, columns=X.columns)
    return {"etichette": modello.labels_, "centri_standardizzati": centri, "wss": modello.inertia_,
            "silhouette": silhouette_score(Z, modello.labels_)}


def kmode(df_cat: pd.DataFrame, k=4, seed=C.SEED, colonne=None, n_init=50):
    """K-Mode (distanza di Hamming) sulle variabili categoriali: la segmentazione finale della tesi.

    Molte partizioni hanno costo quasi uguale: con 50 inizializzazioni si tiene quella a costo minimo.
    I cluster sono riordinati per dimensione decrescente, così la numerazione è stabile tra esecuzioni.
    """
    colonne = colonne or VARIABILI_KMODE
    dati = df_cat[colonne].astype(str)
    modello = KModes(n_clusters=k, init="Huang", n_init=n_init, random_state=seed).fit(dati)
    ordine = pd.Series(modello.labels_).value_counts().index.tolist()
    rinomina = {vecchio: nuovo for nuovo, vecchio in enumerate(ordine)}
    etichette = np.array([rinomina[e] for e in modello.labels_])
    centri = pd.DataFrame(modello.cluster_centroids_, columns=colonne).iloc[ordine].reset_index(drop=True)
    return {"etichette": etichette, "centri": centri, "costo": modello.cost_}


def stabilita_kmode(df_cat: pd.DataFrame, k=4, semi=(1, 2, 3, 4, 5), colonne=None) -> float:
    """ARI medio tra la partizione di riferimento e quelle ottenute con altri seed (1 = sempre uguale)."""
    rif = kmode(df_cat, k, colonne=colonne)["etichette"]
    return float(np.mean([adjusted_rand_score(rif, kmode(df_cat, k, seed=s, colonne=colonne)["etichette"]) for s in semi]))


def distribuzioni_cluster(df: pd.DataFrame, etichette, colonne) -> dict:
    """Per ogni variabile: % delle modalità in ciascun cluster e nel totale."""
    d = df[colonne].astype(str).assign(Cluster=etichette)
    risultato = {}
    for c in colonne:
        tab = pd.crosstab(d[c], d["Cluster"], normalize="columns") * 100
        tab["Totale"] = d[c].value_counts(normalize=True) * 100
        risultato[c] = tab
    return risultato


def kprototype(df: pd.DataFrame, numeriche, categoriali, k=4, seed=C.SEED):
    dati = pd.concat([pd.DataFrame(StandardScaler().fit_transform(df[numeriche]), columns=numeriche, index=df.index),
                      df[categoriali].astype(str)], axis=1)
    idx_cat = list(range(len(numeriche), dati.shape[1]))
    modello = KPrototypes(n_clusters=k, init="Cao", n_init=3, random_state=seed).fit(dati.to_numpy(), categorical=idx_cat)
    return {"etichette": modello.labels_, "costo": modello.cost_}


def gerarchico(X: pd.DataFrame, k=4, metodo="ward"):
    Z = StandardScaler().fit_transform(X)
    legami = linkage(Z, method=metodo)
    cofenetica = cophenet(legami, pdist(Z))[0]
    etichette = fcluster(legami, t=k, criterion="maxclust") - 1
    return {"etichette": etichette, "legami": legami, "correlazione_cofenetica": float(cofenetica),
            "silhouette": silhouette_score(Z, etichette)}


def stabilita_bootstrap(X: pd.DataFrame, k=4, n_ripetizioni=30, seed=C.SEED) -> float:
    """ARI medio tra la partizione completa e quelle ottenute su campioni bootstrap (1 = perfettamente stabile)."""
    Z = StandardScaler().fit_transform(X)
    rif = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    rng = np.random.default_rng(seed)
    ari = []
    for _ in range(n_ripetizioni):
        idx = rng.choice(len(Z), len(Z), replace=True)
        km = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(1e9))).fit(Z[idx])
        ari.append(adjusted_rand_score(rif.predict(Z), km.predict(Z)))
    return float(np.mean(ari))


# --- Profilazione ----------------------------------------------------------------------------

def profilo_cluster(df: pd.DataFrame, etichette, numeriche, categoriali) -> pd.DataFrame:
    """Medie delle numeriche e modalità più frequente (con %) delle categoriali per ogni cluster."""
    d = df.assign(Cluster=etichette)
    righe = {}
    for cl, g in d.groupby("Cluster"):
        r = {"n": len(g), "quota_%": round(len(g) / len(d) * 100, 1)}
        for c in numeriche:
            r[c] = g[c].mean()
        for c in categoriali:
            freq = g[c].value_counts(normalize=True)
            r[c] = f"{freq.index[0]} ({freq.iloc[0] * 100:.0f}%)"
        righe[cl] = r
    return pd.DataFrame(righe).T
