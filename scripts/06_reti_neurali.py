"""Fase 6: reti neurali con validazione separata dal test set.

Usa lo stesso split train/test dei modelli di machine learning, così i risultati sono confrontabili.
"""
import _percorso  # noqa: F401

import pandas as pd

from tesi_bev import archivio, dati, modelli as M, valutazione as V
from tesi_bev.reti_neurali import ARCHITETTURE, ReteNeurale

SEZIONE = "reti_neurali"


def main():
    X, y = dati.prepara_ml()
    X_tr, X_te, y_tr, y_te = M.dividi(X, y)

    prob, storie = {}, {}
    for nome in ARCHITETTURE:
        print(f"Addestramento: {nome}")
        rete = ReteNeurale(nome).fit(X_tr, y_tr)
        prob[nome] = rete.predict_proba(X_te)[:, 1]
        storie[nome] = {**rete.storia_, "epoche_eseguite": len(rete.storia_["loss"]),
                        "n_feature": rete.n_feature_, "n_parametri": int(rete.modello_.count_params())}

    risultati = V.tabella_risultati(prob, y_te).join(V.tabella_ic(prob, y_te))
    archivio.salva_tabella(risultati, SEZIONE, "risultati_test")
    archivio.salva_json(storie, SEZIONE, "storia_addestramento")
    archivio.salva_json({n: {**p, **{k: storie[n][k] for k in ("epoche_eseguite", "n_feature", "n_parametri")}}
                         for n, p in ARCHITETTURE.items()},
                        SEZIONE, "architetture")
    pd.DataFrame(prob).assign(y=y_te.values).to_csv(
        archivio.C.DIR_RISULTATI / SEZIONE / "probabilita_test.csv", index=False)
    print(risultati[["accuracy", "f1", "roc_auc", "roc_auc_ic_basso", "roc_auc_ic_alto"]].round(4).to_string())


if __name__ == "__main__":
    main()
