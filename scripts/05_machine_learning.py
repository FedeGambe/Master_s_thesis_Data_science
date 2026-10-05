"""Fase 5: modelli di machine learning senza data leakage.

1. split train/test stratificato unico per tutti i modelli
2. screening di 15 modelli in 10-fold CV sul solo training
3. ottimizzazione degli iperparametri (RandomizedSearchCV, ROC-AUC) con refit sul training completo
4. valutazione sul test set: metriche, IC bootstrap 95%, test di McNemar, calibrazione
5. ablation senza "auto precedente BEV" e interpretabilità (permutation importance + SHAP)
6. salvataggio della pipeline migliore per la dashboard

Uso: python scripts/05_machine_learning.py [--veloce]
"""
import _percorso  # noqa: F401

import argparse
import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from tesi_bev import archivio, config as C, dati, modelli as M, valutazione as V

SEZIONE = "machine_learning"
MODELLI_OTTIMIZZATI = ["Regressione logistica", "Random Forest", "Gradient Boosting", "AdaBoost",
                       "SVM", "MLP", "XGBoost", "LightGBM"]
ALBERI = {"Gradient Boosting", "Random Forest", "XGBoost", "LightGBM"}

# Risultati riportati nella tesi originale (test set, con data leakage) per il confronto prima/dopo
RISULTATI_TESI = {
    "Gradient Boosting": {"accuracy": 0.6604, "f1": 0.7017, "roc_auc": 0.7112},
    "AdaBoost": {"accuracy": 0.6543, "f1": 0.6968, "roc_auc": 0.6976},
    "Random Forest": {"accuracy": 0.6483, "f1": 0.6931, "roc_auc": 0.7118},
    "SVM": {"accuracy": 0.6333, "f1": 0.6998, "roc_auc": 0.6828},
    "MLP": {"accuracy": 0.6506, "f1": 0.6992, "roc_auc": 0.6945},
    "Regressione logistica": {"accuracy": 0.6207, "f1": 0.6362, "roc_auc": 0.6647},
    "Voting (soft)": {"accuracy": 0.6586, "f1": 0.6935, "roc_auc": 0.7112},
    "Gradient Boosting (feature selezionate)": {"accuracy": 0.6740, "f1": 0.7184, "roc_auc": 0.7384},
}


def main(veloce=False):
    X, y = dati.prepara_ml()
    X_tr, X_te, y_tr, y_te = M.dividi(X, y)
    archivio.salva_json({"n_train": len(X_tr), "n_test": len(X_te), "quota_BEV_train": y_tr.mean(),
                         "quota_BEV_test": y_te.mean(), "feature": list(X.columns), "seed": C.SEED}, SEZIONE, "split")

    # 1. Screening
    print("Screening dei modelli...")
    scr = M.screening(X_tr, y_tr, cv=M.cv_stratificata(5 if veloce else C.N_FOLD))
    archivio.salva_tabella(scr, SEZIONE, "screening_cv", indice=False)
    riepilogo_scr = scr.groupby("modello").agg(["mean", "std"]).drop(columns="fold")
    riepilogo_scr.columns = [f"{m}_{s}" for m, s in riepilogo_scr.columns]
    archivio.salva_tabella(riepilogo_scr.sort_values("roc_auc_mean", ascending=False), SEZIONE, "screening_riepilogo")

    # 2. Ottimizzazione
    ottimizzati, info_ricerca = {}, {}
    for nome in MODELLI_OTTIMIZZATI:
        print(f"Ottimizzazione: {nome}")
        ricerca = M.ottimizza(nome, X_tr, y_tr, n_iter=8 if veloce else 40, cv_fold=3 if veloce else 5)
        ottimizzati[nome] = ricerca.best_estimator_
        info_ricerca[nome] = {"roc_auc_cv": ricerca.best_score_, "tempo_s": ricerca.tempo_secondi_,
                              "parametri": {k.replace("modello__", ""): v for k, v in ricerca.best_params_.items()}}

    print("Ottimizzazione: Gradient Boosting (feature selezionate)")
    ricerca = M.ottimizza("Gradient Boosting", X_tr, y_tr, n_iter=8 if veloce else 40, cv_fold=3 if veloce else 5,
                          selezione_k="cerca")
    nome_sel = "Gradient Boosting (feature selezionate)"
    ottimizzati[nome_sel] = ricerca.best_estimator_
    selezionate = list(ricerca.best_estimator_.named_steps["selezione"].get_feature_names_out(X.columns))
    info_ricerca[nome_sel] = {"roc_auc_cv": ricerca.best_score_, "tempo_s": ricerca.tempo_secondi_,
                              "parametri": {k.replace("modello__", ""): v for k, v in ricerca.best_params_.items()},
                              "feature_selezionate": selezionate}

    migliori_cv = sorted(MODELLI_OTTIMIZZATI, key=lambda n: info_ricerca[n]["roc_auc_cv"], reverse=True)[:3]
    print(f"Voting soft su: {migliori_cv}")
    vc = M.voting({n: clone(ottimizzati[n]) for n in migliori_cv}).fit(X_tr, y_tr)
    ottimizzati["Voting (soft)"] = vc
    info_ricerca["Voting (soft)"] = {"componenti": migliori_cv}
    archivio.salva_json(info_ricerca, SEZIONE, "ottimizzazione")

    # 3. Valutazione sul test set
    prob = {nome: M.probabilita(m, X_te) for nome, m in ottimizzati.items()}
    risultati = V.tabella_risultati(prob, y_te)
    ic = V.tabella_ic(prob, y_te, n=300 if veloce else 1000)
    risultati = risultati.join(ic)
    archivio.salva_tabella(risultati, SEZIONE, "risultati_test")
    migliore = risultati.index[0]
    archivio.salva_tabella(V.confronto_mcnemar(prob, y_te, migliore), SEZIONE, "mcnemar_vs_migliore")
    archivio.salva_json({n: V.curva_roc(y_te, p).iloc[::max(1, len(y_te) // 300)].to_dict(orient="list")
                         for n, p in prob.items()}, SEZIONE, "curve_roc")
    archivio.salva_json({n: V.curva_calibrazione(y_te, p).to_dict(orient="list") for n, p in prob.items()},
                        SEZIONE, "calibrazione")
    pd.DataFrame(prob).assign(y=y_te.values).to_csv(C.DIR_RISULTATI / SEZIONE / "probabilita_test.csv", index=False)

    confronto = pd.DataFrame(RISULTATI_TESI).T.add_suffix("_tesi").join(
        risultati[["accuracy", "f1", "roc_auc"]].add_suffix("_revisione"), how="outer")
    archivio.salva_tabella(confronto, SEZIONE, "confronto_tesi_revisione")

    # 4. Ablation: stesso modello senza la variabile "Auto precedente: BEV"
    modello_base = migliore if migliore != "Voting (soft)" else migliori_cv[0]
    senza = ["Auto precedente: BEV"]
    m_abl = clone(ottimizzati[modello_base])
    if "selezione" in m_abl.named_steps:
        m_abl.set_params(selezione__k="all")
    m_abl.fit(X_tr.drop(columns=senza), y_tr)
    p_abl = M.probabilita(m_abl, X_te.drop(columns=senza))
    ablation = pd.DataFrame({f"{modello_base} (tutte le variabili)": V.metriche(y_te, prob[modello_base]),
                             f"{modello_base} senza 'Auto precedente: BEV'": V.metriche(y_te, p_abl)}).T
    archivio.salva_tabella(ablation, SEZIONE, "ablation_bev_precedente")

    # 5. Interpretabilità
    archivio.salva_tabella(V.importanza_permutazione(ottimizzati[modello_base], X_te, y_te), SEZIONE,
                           "importanza_permutazione", indice=False)
    albero = modello_base if modello_base in ALBERI else next(n for n in risultati.index if n in ALBERI)
    valori, Xc = V.valori_shap(ottimizzati[albero], X_te)
    archivio.salva_tabella(V.importanza_shap(valori, Xc), SEZIONE, "importanza_shap", indice=False)
    np.save(C.DIR_RISULTATI / SEZIONE / "shap_valori.npy", valori)
    Xc.to_csv(C.DIR_RISULTATI / SEZIONE / "shap_campione.csv", index=False)

    # 6. Modello per la dashboard
    archivio.salva_modello(ottimizzati[modello_base], "modello_finale")
    metadati = {
        "modello": modello_base, "feature": list(X.columns), "soglia": 0.5,
        "soglia_youden": V.soglia_youden(y_te, prob[modello_base]),
        "metriche_test": {k: float(v) for k, v in risultati.loc[modello_base].items()},
        "modello_shap": albero, "versioni": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
    }
    archivio.salva_json(metadati, SEZIONE, "modello_finale")
    (C.DIR_MODELLI / "modello_finale.json").write_text(json.dumps(metadati, indent=2, ensure_ascii=False, default=float))

    print(risultati[["accuracy", "f1", "roc_auc", "roc_auc_ic_basso", "roc_auc_ic_alto"]].round(4).to_string())
    print(f"Migliore: {migliore}. Modello salvato per la dashboard: {modello_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--veloce", action="store_true", help="meno iterazioni e fold (per test rapidi)")
    main(parser.parse_args().veloce)
