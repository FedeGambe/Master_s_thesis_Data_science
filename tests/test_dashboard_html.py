"""La dashboard HTML (app/modello_bev.js) deve dare le stesse probabilità e gli stessi SHAP del modello Python."""
import json
import shutil
import subprocess

import numpy as np
import pytest

from tesi_bev import archivio, config as C, dati, esporta_web, modelli as M

pytestmark = pytest.mark.skipif(shutil.which("node") is None or not (C.DIR_MODELLI / "modello_finale.joblib").exists(),
                                reason="servono Node.js e il modello salvato")

SCRIPT_NODE = """
const fs = require("fs");
const m = require(process.argv[1]);
const d = JSON.parse(fs.readFileSync(0, "utf8"));
const r = d.profili.map(p => m.predici(d.modello, p));
process.stdout.write(JSON.stringify({prob: r.map(x => x.probabilita), shap: r.map(x => x.contributi),
                                     base: m.valoreAtteso(d.modello)}));
"""


def test_js_coincide_con_python():
    modello = archivio.carica_modello("modello_finale")
    meta = json.loads((C.DIR_MODELLI / "modello_finale.json").read_text(encoding="utf-8"))
    df = dati.carica_dati()
    X, y = dati.prepara_ml(df)
    _, X_te, _, _ = M.dividi(X, y)
    righe = df.loc[X_te.index[:300]]
    sorgenti = {r["sorgente"] for r in esporta_web._regole_codifica()}
    profili = json.loads(righe[sorted(sorgenti)].to_json(orient="records", force_ascii=False))

    ingresso = json.dumps({"modello": esporta_web.dati_modello(modello, meta), "profili": profili})
    uscita = subprocess.run(["node", "-e", SCRIPT_NODE, str(esporta_web.FILE_LOGICA)], input=ingresso,
                            capture_output=True, text=True, check=True).stdout
    js = json.loads(uscita)

    Xr = dati.codifica_ml(righe)
    prob_py = modello.predict_proba(Xr)[:, 1]
    contrib_py = modello[-1].booster_.predict(modello[:-1].transform(Xr), pred_contrib=True)
    assert np.abs(np.array(js["prob"]) - prob_py).max() < 1e-9
    assert np.abs(np.array(js["shap"]) - contrib_py[:, :-1]).max() < 1e-9
    assert abs(js["base"] - contrib_py[0, -1]) < 1e-9
