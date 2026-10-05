"""Esportazione del modello finale per la dashboard HTML (``docs/dashboard.html``).

La pipeline (StandardScaler + LightGBM) viene tradotta in un JSON che ``app/modello_bev.js``
valuta nel browser: codifica degli input, standardizzazione, somma delle foglie degli alberi
e contributi SHAP (TreeSHAP esatto). Così la pagina funziona senza Python né server.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from . import config as C, dati, modelli as M

DIR_APP = C.RADICE / "app"
FILE_LOGICA = DIR_APP / "modello_bev.js"
FILE_MODELLO_HTML = DIR_APP / "modello_dashboard.html"
FILE_PAGINA = C.DIR_DOCS / "dashboard.html"
FILE_FAVICON = C.DIR_DOCS / "favicon.svg"

CAMPI = {  # id del campo nella pagina -> colonna del dataset originale (come in app/dashboard.py)
    "genere": C.GENERE, "eta": C.CLASSE_ETA, "reddito": C.REDDITO, "istruzione": C.ISTRUZIONE,
    "casa_prop": C.CASA_PROPRIETA, "casa_ind": C.CASA_INDIPENDENTE, "persone": C.N_PERSONE, "auto": C.N_AUTO,
    "emissioni": C.EMISSIONI, "auto_prec": C.AUTO_PRECEDENTE, "viaggio": C.VIAGGIO_LUNGO,
    "n_viaggi": C.N_VIAGGI_LUNGHI, "distanza": C.DISTANZA_LAVORO, "vmt": C.VMT,
}
NUMERICI = ["persone", "auto", "emissioni", "viaggio", "n_viaggi", "distanza", "vmt"]
QUANTILI_ESEMPI = {"alta": 0.9, "media": 0.5, "bassa": 0.1}


def _esempi(modello, df) -> dict:
    """Tre clienti reali del test set con probabilità predetta vicina al 90°, 50° e 10° percentile."""
    X, y = dati.prepara_ml(df)
    _, X_te, _, _ = M.dividi(X, y)
    prob = pd.Series(modello.predict_proba(X_te)[:, 1], index=X_te.index)
    esempi = {}
    for nome, q in QUANTILI_ESEMPI.items():
        riga = df.loc[(prob - prob.quantile(q)).abs().idxmin()]
        esempi[nome] = {id_: (riga[col].item() if hasattr(riga[col], "item") else riga[col]) for id_, col in CAMPI.items()}
    return esempi


def _campi_pagina(modello, df) -> dict:
    """Opzioni, intervalli ed esempi ricavati dal dataset."""
    return {
        "campi": CAMPI,
        "opzioni": {"genere": list(dati.MAPPA_GENERE), "eta": C.ORDINE_ETA, "istruzione": C.ORDINE_ISTRUZIONE,
                    "reddito": sorted(int(r) for r in df[C.REDDITO].unique()), "auto_prec": C.TIPI_AUTO},
        "intervalli": {id_: [math.floor(df[CAMPI[id_]].min()), math.ceil(df[CAMPI[id_]].max())] for id_ in NUMERICI},
        "esempi": _esempi(modello, df),
    }


def _regole_codifica() -> list[dict]:
    """Per ogni colonna di ``dati.COLONNE_ML`` dice come ottenerla dal profilo (stesse regole di ``codifica_ml``)."""
    mappe = {C.GENERE: dati.MAPPA_GENERE, C.CLASSE_ETA: dati.MAPPA_ETA, C.ISTRUZIONE: dati.MAPPA_ISTRUZIONE,
             C.CASA_PROPRIETA: dati.MAPPA_SI_NO, C.CASA_INDIPENDENTE: dati.MAPPA_SI_NO}
    regole = []
    for col in dati.COLONNE_ML:
        if col in mappe:
            regole.append({"tipo": "mappa", "sorgente": col, "mappa": mappe[col]})
        elif col.startswith("Auto precedente: "):
            regole.append({"tipo": "dummy", "sorgente": C.AUTO_PRECEDENTE, "valore": col.removeprefix("Auto precedente: ")})
        else:
            regole.append({"tipo": "numero", "sorgente": col})
    return regole


def _appiattisci(radice: dict) -> list[list]:
    """Albero LightGBM -> array paralleli [feature, soglia, sinistro, destro, valore, copertura].

    Nelle foglie la feature è -1. La copertura (numero di osservazioni di training nel nodo)
    serve a TreeSHAP.
    """
    f, s, sx, dx, v, c = [], [], [], [], [], []

    def visita(nodo) -> int:
        i = len(f)
        for lista in (f, s, sx, dx, v, c):
            lista.append(0)
        if "split_index" in nodo:
            if nodo["decision_type"] != "<=" or nodo["missing_type"] != "None":
                raise ValueError("Split non supportato dalla dashboard HTML")
            f[i], s[i], v[i], c[i] = nodo["split_feature"], nodo["threshold"], 0, nodo["internal_count"]
            sx[i] = visita(nodo["left_child"])
            dx[i] = visita(nodo["right_child"])
        else:
            f[i], s[i], sx[i], dx[i], v[i], c[i] = -1, 0, -1, -1, nodo["leaf_value"], nodo.get("leaf_count", 1)
        return i

    visita(radice)
    return [f, s, sx, dx, v, c]


def dati_modello(modello, meta: dict) -> dict:
    """Tutto ciò che serve alla pagina per riprodurre ``modello.predict_proba`` e i contributi SHAP."""
    nomi = [n for n, _ in modello.steps]
    if nomi != ["scaler", "modello"] or type(modello[-1]).__name__ != "LGBMClassifier":
        raise ValueError(f"La dashboard HTML supporta solo StandardScaler + LightGBM (trovato: {nomi}, "
                         f"{type(modello[-1]).__name__}). Usa app/dashboard.py.")
    albero = modello[-1].booster_.dump_model()
    if albero["objective"] != "binary sigmoid:1" or albero["num_tree_per_iteration"] != 1:
        raise ValueError(f"Obiettivo LightGBM non supportato: {albero['objective']}")
    scaler = modello.named_steps["scaler"]
    m = meta["metriche_test"]
    df = dati.carica_dati()
    return {
        "colonne": dati.COLONNE_ML,
        "nomi_brevi": [C.nome_breve(c) for c in dati.COLONNE_ML],
        "codifica": _regole_codifica(),
        "media": scaler.mean_.tolist(),
        "scala": scaler.scale_.tolist(),
        "alberi": [_appiattisci(t["tree_structure"]) for t in albero["tree_info"]],
        "info": {
            "modello": meta["modello"], "soglia": meta["soglia"], "roc_auc": m["roc_auc"], "accuracy": m["accuracy"],
            "quota_bev": float(df[C.TARGET].mean()), "n": len(df),
        },
        **_campi_pagina(modello, df),
    }


def link_favicon() -> str:
    """Tag ``<link rel="icon">`` con la favicon SVG incorporata (stessa icona di docs/risultati.html)."""
    svg = FILE_FAVICON.read_text(encoding="utf-8").strip()
    return f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,{quote(svg, safe=" =/:")}">'


def scrivi_pagina(modello, meta: dict, percorso: Path = FILE_PAGINA) -> Path:
    """Inserisce logica JS e modello JSON nel template e scrive una pagina HTML autonoma."""
    contenuto = json.dumps(dati_modello(modello, meta), ensure_ascii=False, separators=(",", ":"))
    pagina = (FILE_MODELLO_HTML.read_text(encoding="utf-8")
              .replace("<!--__FAVICON__-->", link_favicon())
              .replace("/*__LOGICA__*/", FILE_LOGICA.read_text(encoding="utf-8"))
              .replace("/*__MODELLO__*/null", contenuto.replace("</", "<\\/")))
    percorso.write_text(pagina, encoding="utf-8")
    return percorso
