"""Modelli di machine learning.

Ogni modello è una ``Pipeline`` (standardizzazione + stimatore): lo scaler viene
addestrato solo sui dati di training di ciascun fold, eliminando il data leakage
presente nella versione originale (``fit_transform`` sul test set).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config as C


def pipeline(stimatore, selezione_k=None) -> Pipeline:
    passi = [("scaler", StandardScaler())]
    if selezione_k is not None:
        passi.append(("selezione", SelectKBest(f_classif, k=selezione_k)))
    passi.append(("modello", stimatore))
    return Pipeline(passi)


def dividi(X, y, test_size=C.TEST_SIZE, seed=C.SEED):
    """Split train/test stratificato: lo stesso per TUTTI i modelli."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def cv_stratificata(n_fold=C.N_FOLD, seed=C.SEED):
    return StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)


# --- Screening iniziale ----------------------------------------------------------------------

def modelli_screening(seed=C.SEED) -> dict:
    """I 13 modelli confrontati nella tesi + XGBoost e LightGBM, con iperparametri di default."""
    return {
        "Regressione logistica": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": BernoulliNB(),
        "Albero decisionale": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(random_state=seed, n_jobs=-1),
        "SVM": SVC(probability=True, random_state=seed),
        "Ridge": RidgeClassifier(),
        "SGD": SGDClassifier(random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "AdaBoost": AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=seed),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.01),  # le 5 dummy auto precedente sono collineari
        "MLP": MLPClassifier(max_iter=500, random_state=seed),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1),
    }


def screening(X_train, y_train, modelli=None, cv=None) -> pd.DataFrame:
    """Cross-validation di ogni modello: restituisce una riga per fold e modello."""
    modelli = modelli or modelli_screening()
    cv = cv or cv_stratificata()
    righe = []
    for nome, stimatore in modelli.items():
        metriche = {"accuracy": "accuracy", "f1": "f1"}
        if hasattr(stimatore, "predict_proba") or hasattr(stimatore, "decision_function"):
            metriche["roc_auc"] = "roc_auc"
        r = cross_validate(pipeline(stimatore), X_train, y_train, cv=cv, scoring=metriche, n_jobs=-1)
        for i in range(len(r["fit_time"])):
            righe.append({"modello": nome, "fold": i + 1, **{m: r[f"test_{m}"][i] for m in metriche}})
    return pd.DataFrame(righe)


# --- Ottimizzazione degli iperparametri -------------------------------------------------------

def spazi_ricerca(seed=C.SEED) -> dict:
    """Stimatore e spazio degli iperparametri per i modelli ottimizzati (prefisso ``modello__``)."""
    return {
        "Regressione logistica": (LogisticRegression(max_iter=2000, solver="saga"), {
            "C": loguniform(1e-3, 1e2), "l1_ratio": uniform(0, 1), "penalty": ["elasticnet"]}),
        "Random Forest": (RandomForestClassifier(random_state=seed, n_jobs=-1), {
            "n_estimators": randint(200, 600), "max_depth": [None, 6, 10, 16], "min_samples_leaf": randint(1, 20),
            "max_features": ["sqrt", 0.5, 0.8]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=seed), {
            "n_estimators": randint(100, 400), "learning_rate": loguniform(0.01, 0.2), "max_depth": randint(2, 5),
            "subsample": uniform(0.5, 0.5), "min_samples_split": randint(2, 20)}),
        "AdaBoost": (AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=seed), {
            "n_estimators": randint(50, 400), "learning_rate": loguniform(0.05, 1.5)}),
        "SVM": (SVC(probability=True, random_state=seed), {
            "C": loguniform(0.1, 30), "gamma": loguniform(1e-3, 0.3)}),
        "MLP": (MLPClassifier(max_iter=1000, early_stopping=True, random_state=seed), {
            "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "alpha": loguniform(1e-5, 1e-1),
            "learning_rate_init": loguniform(1e-4, 1e-2)}),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=seed, n_jobs=1), {
            "n_estimators": randint(100, 600), "learning_rate": loguniform(0.01, 0.2), "max_depth": randint(2, 6),
            "subsample": uniform(0.5, 0.5), "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": randint(1, 20),
            "reg_lambda": loguniform(1e-2, 10)}),
        "LightGBM": (LGBMClassifier(random_state=seed, verbose=-1, n_jobs=1), {
            "n_estimators": randint(100, 600), "learning_rate": loguniform(0.01, 0.2), "num_leaves": randint(8, 64),
            "min_child_samples": randint(10, 100), "subsample": uniform(0.5, 0.5), "subsample_freq": [1],
            "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(1e-2, 10)}),
    }


ITERAZIONI = {"SVM": 12, "MLP": 20}


def ottimizza(nome, X_train, y_train, n_iter=40, scoring="roc_auc", cv_fold=5, seed=C.SEED, selezione_k=None):
    """RandomizedSearchCV sulla pipeline completa; con ``refit=True`` il modello finale è
    riaddestrato su tutto il training (non si sceglie il "fold migliore")."""
    stimatore, spazio = spazi_ricerca(seed)[nome]
    spazio = {f"modello__{k}": v for k, v in spazio.items()}
    if selezione_k == "cerca":
        spazio["selezione__k"] = list(range(6, X_train.shape[1] + 1))
    ricerca = RandomizedSearchCV(
        pipeline(stimatore, selezione_k=("all" if selezione_k == "cerca" else selezione_k)),
        spazio, n_iter=ITERAZIONI.get(nome, n_iter), scoring=scoring,
        cv=cv_stratificata(cv_fold, seed), random_state=seed, n_jobs=-1, refit=True)
    inizio = time.time()
    ricerca.fit(X_train, y_train)
    ricerca.tempo_secondi_ = time.time() - inizio
    return ricerca


def voting(stimatori: dict) -> VotingClassifier:
    """Voting "soft" sui migliori modelli già ottimizzati (le pipeline vengono clonate e riaddestrate)."""
    return VotingClassifier(list(stimatori.items()), voting="soft", n_jobs=-1)


def probabilita(modello, X) -> np.ndarray:
    return modello.predict_proba(X)[:, 1]
