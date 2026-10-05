"""Metriche, intervalli di confidenza, test di confronto e interpretabilità."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score,
                             roc_curve)

from . import config as C

METRICHE_PRINCIPALI = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]


def metriche(y, prob, soglia=0.5) -> dict:
    pred = (prob >= soglia).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob),
        "log_loss": log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def tabella_risultati(probabilita: dict, y, soglia=0.5) -> pd.DataFrame:
    """Una riga per modello: le metriche sono calcolate tutte dalla stessa funzione,
    così non possono finire su righe sbagliate (bug della tabella originale)."""
    t = pd.DataFrame({nome: metriche(y, p, soglia) for nome, p in probabilita.items()}).T
    return t.sort_values("roc_auc", ascending=False)


def bootstrap_ic(y, prob, metrica="roc_auc", n=1000, livello=0.95, seed=C.SEED, soglia=0.5):
    """Intervallo di confidenza bootstrap (percentile) di una metrica sul test set."""
    y, prob = np.asarray(y), np.asarray(prob)
    rng = np.random.default_rng(seed)
    valori = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        valori.append(metriche(y[idx], prob[idx], soglia)[metrica])
    a = (1 - livello) / 2
    return float(np.quantile(valori, a)), float(np.quantile(valori, 1 - a))


def tabella_ic(probabilita: dict, y, metriche_ic=("accuracy", "roc_auc", "f1"), n=1000) -> pd.DataFrame:
    righe = {}
    for nome, p in probabilita.items():
        r = {}
        for m in metriche_ic:
            basso, alto = bootstrap_ic(y, p, m, n=n)
            r[f"{m}_ic_basso"], r[f"{m}_ic_alto"] = basso, alto
        righe[nome] = r
    return pd.DataFrame(righe).T


def mcnemar(y, pred_a, pred_b) -> dict:
    """Test di McNemar (con correzione di continuità) tra due classificatori sullo stesso test set."""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    giusto_a, giusto_b = a == y, b == y
    n01 = int(np.sum(giusto_a & ~giusto_b))
    n10 = int(np.sum(~giusto_a & giusto_b))
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10, "chi2": chi2, "p_value": float(stats.chi2.sf(chi2, 1))}


def confronto_mcnemar(probabilita: dict, y, riferimento: str, soglia=0.5) -> pd.DataFrame:
    rif = (probabilita[riferimento] >= soglia).astype(int)
    righe = {}
    for nome, p in probabilita.items():
        if nome != riferimento:
            righe[nome] = mcnemar(y, rif, (p >= soglia).astype(int))
    return pd.DataFrame(righe).T


def soglia_youden(y, prob) -> float:
    fpr, tpr, soglie = roc_curve(y, prob)
    return float(soglie[np.argmax(tpr - fpr)])


def curva_roc(y, prob) -> pd.DataFrame:
    fpr, tpr, soglie = roc_curve(y, prob)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "soglia": soglie})


def curva_calibrazione(y, prob, n_bin=10) -> pd.DataFrame:
    frazione, media = calibration_curve(y, prob, n_bins=n_bin, strategy="quantile")
    return pd.DataFrame({"probabilita_media_predetta": media, "frazione_positivi": frazione})


def importanza_permutazione(modello, X, y, scoring="roc_auc", n_ripetizioni=10, seed=C.SEED) -> pd.DataFrame:
    r = permutation_importance(modello, X, y, scoring=scoring, n_repeats=n_ripetizioni, random_state=seed, n_jobs=-1)
    return pd.DataFrame({"variabile": X.columns, "importanza": r.importances_mean, "dev_std": r.importances_std}) \
        .sort_values("importanza", ascending=False, ignore_index=True)


def valori_shap(pipeline_addestrata, X, n_campione=1000, seed=C.SEED):
    """Valori SHAP per una pipeline (scaler + modello ad alberi). Restituisce (valori, X_campione)."""
    import shap
    Xc = X.sample(min(n_campione, len(X)), random_state=seed)
    preprocessing, modello = pipeline_addestrata[:-1], pipeline_addestrata[-1]
    colonne = list(preprocessing.get_feature_names_out(X.columns))
    Xs = pd.DataFrame(preprocessing.transform(Xc), columns=colonne, index=Xc.index)
    Xc = Xc[colonne]
    spiegatore = shap.TreeExplainer(modello)
    valori = spiegatore.shap_values(Xs)
    if isinstance(valori, list):
        valori = valori[1]
    elif valori.ndim == 3:
        valori = valori[:, :, 1]
    return valori, Xc


def importanza_shap(valori, X) -> pd.DataFrame:
    return pd.DataFrame({"variabile": X.columns, "shap_medio_assoluto": np.abs(valori).mean(axis=0)}) \
        .sort_values("shap_medio_assoluto", ascending=False, ignore_index=True)
