"""Salvataggio e caricamento dei risultati (tabelle CSV, JSON e modelli)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C


def slug(testo: str) -> str:
    """Nome di file sicuro: lettere, numeri, trattini e underscore."""
    return re.sub(r"[^\w\-]+", "_", testo).strip("_")


def _percorso(sezione: str, nome: str) -> Path:
    radice, est = nome.rsplit(".", 1)
    p = C.DIR_RISULTATI / sezione / f"{slug(radice)}.{est}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def salva_tabella(df: pd.DataFrame, sezione: str, nome: str, indice=True) -> Path:
    p = _percorso(sezione, f"{nome}.csv")
    df.to_csv(p, index=indice, float_format="%.6g")
    return p


def carica_tabella(sezione: str, nome: str, indice=True) -> pd.DataFrame:
    return pd.read_csv(C.DIR_RISULTATI / sezione / f"{slug(nome)}.csv", index_col=0 if indice else None)


def _serializzabile(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="index")
    if isinstance(o, pd.Series):
        return o.to_dict()
    raise TypeError(type(o))


def salva_json(obj, sezione: str, nome: str) -> Path:
    p = _percorso(sezione, f"{nome}.json")
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_serializzabile), encoding="utf-8")
    return p


def carica_json(sezione: str, nome: str):
    return json.loads((C.DIR_RISULTATI / sezione / f"{slug(nome)}.json").read_text(encoding="utf-8"))


def esiste(sezione: str, nome: str) -> bool:
    return any((C.DIR_RISULTATI / sezione).glob(f"{slug(nome)}.*"))


def salva_modello(modello, nome: str) -> Path:
    C.DIR_MODELLI.mkdir(parents=True, exist_ok=True)
    p = C.DIR_MODELLI / f"{nome}.joblib"
    joblib.dump(modello, p, compress=3)
    return p


def carica_modello(nome: str):
    return joblib.load(C.DIR_MODELLI / f"{nome}.joblib")
