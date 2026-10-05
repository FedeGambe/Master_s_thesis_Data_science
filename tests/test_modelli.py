import numpy as np
from sklearn.linear_model import LogisticRegression

from tesi_bev import dati, modelli as M, valutazione as V


def test_split_stratificato_e_riproducibile():
    X, y = dati.prepara_ml()
    a = M.dividi(X, y)
    b = M.dividi(X, y)
    assert (a[0].index == b[0].index).all()
    assert abs(a[2].mean() - a[3].mean()) < 0.01


def test_pipeline_scaler_addestrato_solo_sul_training():
    X, y = dati.prepara_ml()
    X_tr, X_te, y_tr, _ = M.dividi(X, y)
    p = M.pipeline(LogisticRegression(max_iter=500)).fit(X_tr, y_tr)
    assert np.allclose(p.named_steps["scaler"].mean_, X_tr.mean().values)


def test_metriche_coerenti():
    y = np.array([0, 0, 1, 1, 1])
    prob = np.array([0.1, 0.6, 0.7, 0.8, 0.4])
    m = V.metriche(y, prob)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 1
    assert abs(m["accuracy"] - 0.6) < 1e-9


def test_tabella_risultati_righe_per_modello():
    """Ogni riga deve contenere le metriche del proprio modello (bug B1 della tesi)."""
    y = np.array([0, 1, 0, 1, 1, 0])
    prob = {"perfetto": np.array([0, 1, 0, 1, 1, 0.0]), "casuale": np.array([0.5] * 6)}
    t = V.tabella_risultati(prob, y)
    assert t.loc["perfetto", "roc_auc"] == 1.0
    assert t.loc["casuale", "roc_auc"] == 0.5


def test_mcnemar_identici():
    y = np.array([0, 1, 1, 0])
    assert V.mcnemar(y, y, y)["p_value"] == 1.0
