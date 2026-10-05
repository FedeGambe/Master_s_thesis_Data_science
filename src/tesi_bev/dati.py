"""Caricamento e preparazione dei dati.

Un'unica fonte di verità per le trasformazioni: le stesse funzioni sono usate
da analisi statistiche, machine learning, reti neurali e dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# --- Codifiche -------------------------------------------------------------------------------

MAPPA_GENERE = {"Maschio": 1, "Femmina": 0}
MAPPA_SI_NO = {"Si": 1, "No": 0}
MAPPA_ETA = {v: i + 1 for i, v in enumerate(C.ORDINE_ETA)}
MAPPA_ISTRUZIONE = {v: i + 1 for i, v in enumerate(C.ORDINE_ISTRUZIONE)}
MAPPA_REDDITO = {v: i + 1 for i, v in enumerate(C.ORDINE_REDDITO)}
# Limite inferiore della classe d'età: è la variabile "Età" usata nel clustering della tesi
ETA_LIMITE_INFERIORE = {"<25": 18, "25-34": 25, "35-44": 35, "45-54": 45, "55-64": 55, "65-74": 65, "75-79": 75, ">80": 80}
ETA_RAGGRUPPATA = {
    "<25": "<35", "25-34": "<35", "35-44": "35-44", "45-54": "45-64",
    "55-64": "45-64", "65-74": ">65", "75-79": ">65", ">80": ">65",
}


def carica_dati(percorso=None) -> pd.DataFrame:
    """Carica il dataset originale (locale se presente, altrimenti da GitHub)."""
    percorso = percorso or C.FILE_DATASET
    try:
        df = pd.read_csv(percorso)
    except FileNotFoundError:
        df = pd.read_csv(C.URL_DATASET)
    df[C.TARGET] = df[C.TARGET].astype(int)
    return df


# --- Machine learning ------------------------------------------------------------------------

COLONNE_ML = [
    C.GENERE, C.CLASSE_ETA, C.REDDITO, C.ISTRUZIONE, C.CASA_PROPRIETA, C.CASA_INDIPENDENTE,
    C.N_PERSONE, C.N_AUTO, C.EMISSIONI, C.VIAGGIO_LUNGO, C.N_VIAGGI_LUNGHI, C.DISTANZA_LAVORO, C.VMT,
    *[f"Auto precedente: {t}" for t in C.TIPI_AUTO],
]


def codifica_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Trasforma il dataset originale nelle 18 feature numeriche usate dai modelli.

    Le categorie ordinali diventano interi, l'auto precedente diventa 5 dummy.
    Funziona anche su una singola riga (usata dalla dashboard).
    """
    X = pd.DataFrame(index=df.index)
    X[C.GENERE] = df[C.GENERE].map(MAPPA_GENERE)
    X[C.CLASSE_ETA] = df[C.CLASSE_ETA].map(MAPPA_ETA)
    X[C.REDDITO] = df[C.REDDITO].astype(float)
    X[C.ISTRUZIONE] = df[C.ISTRUZIONE].map(MAPPA_ISTRUZIONE)
    X[C.CASA_PROPRIETA] = df[C.CASA_PROPRIETA].map(MAPPA_SI_NO)
    X[C.CASA_INDIPENDENTE] = df[C.CASA_INDIPENDENTE].map(MAPPA_SI_NO)
    for col in [C.N_PERSONE, C.N_AUTO, C.EMISSIONI, C.VIAGGIO_LUNGO, C.N_VIAGGI_LUNGHI, C.DISTANZA_LAVORO, C.VMT]:
        X[col] = df[col].astype(float)
    for tipo in C.TIPI_AUTO:
        X[f"Auto precedente: {tipo}"] = (df[C.AUTO_PRECEDENTE] == tipo).astype(int)
    if X.isna().any().any():
        colonne = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Valori non riconosciuti nelle colonne: {colonne}")
    return X[COLONNE_ML].astype(float)


def prepara_ml(df: pd.DataFrame | None = None, escludi_bev_precedente: bool = False):
    """Restituisce (X, y) per i modelli predittivi."""
    df = carica_dati() if df is None else df
    X = codifica_ml(df)
    if escludi_bev_precedente:
        X = X.drop(columns=["Auto precedente: BEV"])
    return X, df[C.TARGET].astype(int)


# --- Regressione logistica -------------------------------------------------------------------

SCALE_LOGIT = {
    C.VIAGGIO_LUNGO: (100, "per 100 mi"),
    C.DISTANZA_LAVORO: (10, "per 10 mi"),
    C.VMT: (1000, "per 1.000 mi"),
}


def nome_logit(colonna: str) -> str:
    """Nome della colonna nella matrice della logistica (con l'unità se riscalata)."""
    return f"{colonna} ({SCALE_LOGIT[colonna][1]})" if colonna in SCALE_LOGIT else colonna


def prepara_logit(df: pd.DataFrame | None = None, ridotto: bool = True):
    """Matrice per la logistica: dummy rispetto alle categorie di riferimento della tesi.

    Con ``ridotto=True`` vengono escluse "Numero persone in famiglia" e "Casa Indipendente":
    è il modello riportato nelle Tabelle 4-5 della tesi (che mantiene "Casa di proprietà").
    """
    df = carica_dati() if df is None else df
    X = pd.DataFrame(index=df.index)
    X["Genere (Maschio)"] = df[C.GENERE].map(MAPPA_GENERE)
    for col, rif in C.RIFERIMENTI_LOGIT.items():
        ordine = {
            C.CLASSE_ETA: C.ORDINE_ETA, C.CLASSE_REDDITO: C.ORDINE_REDDITO,
            C.ISTRUZIONE: C.ORDINE_ISTRUZIONE, C.AUTO_PRECEDENTE: C.TIPI_AUTO,
        }[col]
        for cat in ordine:
            if cat != rif:
                X[f"{col}: {cat} vs {rif}"] = (df[col] == cat).astype(int)
    X[C.CASA_PROPRIETA] = df[C.CASA_PROPRIETA].map(MAPPA_SI_NO)
    X[C.CASA_INDIPENDENTE] = df[C.CASA_INDIPENDENTE].map(MAPPA_SI_NO)
    for col in [C.N_PERSONE, C.N_AUTO, C.EMISSIONI, C.N_VIAGGI_LUNGHI]:
        X[col] = df[col].astype(float)
    # Riscalate per avere odds ratio leggibili (con le miglia singole l'OR è ~1,0000)
    for col, (fattore, unita) in SCALE_LOGIT.items():
        X[f"{col} ({unita})"] = df[col].astype(float) / fattore
    if ridotto:
        X = X.drop(columns=[C.N_PERSONE, C.CASA_INDIPENDENTE])
    return X.astype(float), df[C.TARGET].astype(int)


# --- Variabili categoriali (MCA, K-Mode) ------------------------------------------------------

def _taglia(serie, bins, etichette):
    return pd.cut(serie, bins=bins, labels=etichette, right=False).astype(str)


def _bins_punti_medi(valori):
    """Soglie a metà strada tra punti di riferimento consecutivi (metodo usato nella tesi)."""
    return [(b - a) / 2 + a for a, b in zip(valori[:-1], valori[1:])]


def crea_variabili_categoriali(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Discretizza le variabili quantitative come nel capitolo 6 della tesi."""
    df = carica_dati() if df is None else df
    out = df.copy()
    out["Numero persone in famiglia categoriale"] = _taglia(
        df[C.N_PERSONE], [0, 1.1, 2.1, 3.1, 4.1, np.inf],
        ["Single", "Coppie", "Famiglie con 1 figlio", "Famiglie con 2 figli", "Famiglie con 3+ figli"])
    out["Numero di auto in famiglia categoriale"] = _taglia(
        df[C.N_AUTO], [0, 1.1, 2.1, 3.1, np.inf], ["Un'auto", "Due auto", "Tre auto", "Quattro o più auto"])

    soglie_emi = _bins_punti_medi([-3, -1.5, 0, 1.5, 3])
    out["Sensibilità ambientale categoriale"] = _taglia(
        df[C.EMISSIONI], [-np.inf, *soglie_emi, np.inf], ["Molto bassa", "Bassa", "Media", "Alta", "Molto alta"])

    out["Viaggio lungo categoriale"] = _taglia(
        df[C.VIAGGIO_LUNGO], [0, 175, 350, 550, np.inf],
        ["Viaggio < 175 mi", "Viaggio 175-350 mi", "Viaggio 350-550 mi", "Viaggio > 550 mi"])
    out["Numero viaggi lunghi categoriale"] = _taglia(
        df[C.N_VIAGGI_LUNGHI], [0, 0.1, 1.1, 2.1, np.inf],
        ["Nessun viaggio lungo", "1 viaggio lungo", "2 viaggi lunghi", "3+ viaggi lunghi"])
    out["Distanza casa-lavoro categoriale"] = _taglia(
        df[C.DISTANZA_LAVORO], [0, 6.5, 15, 25, 45, np.inf],
        ["Casa-lavoro < 6,5 mi", "Casa-lavoro 6,5-15 mi", "Casa-lavoro 15-25 mi", "Casa-lavoro 25-45 mi", "Casa-lavoro > 45 mi"])

    # Soglie a metà tra min, quartili e max del VMT (come nella tesi)
    soglie_vmt = _bins_punti_medi([6.72, 8391, 11557, 15960, 378000])
    out["VMT categoriale"] = _taglia(
        df[C.VMT], [0, *soglie_vmt, np.inf], ["VMT molto basso", "VMT basso", "VMT medio", "VMT alto", "VMT molto alto"])

    out["Classe età raggruppata"] = df[C.CLASSE_ETA].map(ETA_RAGGRUPPATA)
    out["Età"] = df[C.CLASSE_ETA].map(ETA_LIMITE_INFERIORE)
    out["BEV"] = df[C.TARGET].map({1: "Si", 0: "No"})
    return out


VARIABILI_MCA = {
    "abitative": [C.CASA_PROPRIETA, C.CASA_INDIPENDENTE, "Numero persone in famiglia categoriale",
                  "Numero di auto in famiglia categoriale"],
    "mobilita": ["VMT categoriale", "Viaggio lungo categoriale", "Numero viaggi lunghi categoriale",
                 "Distanza casa-lavoro categoriale"],
    "sociodemografiche": [C.GENERE, "Classe età raggruppata", C.CLASSE_REDDITO, C.ISTRUZIONE,
                          "Sensibilità ambientale categoriale"],
}

VARIABILI_KMODE = [
    C.GENERE, C.CLASSE_REDDITO, C.ISTRUZIONE, C.CASA_PROPRIETA, C.CASA_INDIPENDENTE, C.AUTO_PRECEDENTE,
    "Numero persone in famiglia categoriale", "Numero di auto in famiglia categoriale", "VMT categoriale",
    "Viaggio lungo categoriale", "Numero viaggi lunghi categoriale", "Distanza casa-lavoro categoriale",
    "Sensibilità ambientale categoriale", "Classe età raggruppata",
]


def solo_bev(df: pd.DataFrame) -> pd.DataFrame:
    """Sotto-campione dei possessori di BEV, su cui la tesi costruisce MCA, FAMD e cluster."""
    return df[df[C.TARGET] == 1].reset_index(drop=True)
