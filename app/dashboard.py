"""Dashboard di predizione: probabilità che il veicolo di un cliente sia un BEV.

Avvio:  python app/dashboard.py   ->   http://127.0.0.1:8050

Il modello è la pipeline salvata da ``scripts/05_machine_learning.py`` in ``modelli/``.
Gli input passano per ``dati.codifica_ml``, la stessa funzione usata in addestramento:
così non possono esserci differenze di codifica (es. il genere invertito della versione originale).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dash_bootstrap_components as dbc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from dash import Dash, Input, Output, dcc, html  # noqa: E402

from tesi_bev import archivio, config as C, dati, esporta_web, modelli as M  # noqa: E402


# --- Modello ---------------------------------------------------------------------------------

def carica_modello_finale():
    """Carica la pipeline salvata; se le versioni delle librerie non sono compatibili
    (es. su Colab) la riaddestra con gli iperparametri trovati in ottimizzazione."""
    meta = json.loads((C.DIR_MODELLI / "modello_finale.json").read_text(encoding="utf-8"))
    try:
        return archivio.carica_modello("modello_finale"), meta
    except Exception as errore:  # pickle non compatibile con la versione installata
        print(f"Modello salvato non caricabile ({type(errore).__name__}): riaddestramento in corso...")
        parametri = archivio.carica_json("machine_learning", "ottimizzazione")[meta["modello"]]["parametri"]
        stimatore, _ = M.spazi_ricerca()[meta["modello"]]
        modello = M.pipeline(stimatore.set_params(**parametri))
        X, y = dati.prepara_ml()
        X_tr, _, y_tr, _ = M.dividi(X, y)
        return modello.fit(X_tr, y_tr), meta


def predici(modello, profilo: dict) -> float:
    return float(modello.predict_proba(dati.codifica_ml(pd.DataFrame([profilo])))[0, 1])


def contributi_shap(modello, profilo: dict) -> pd.Series:
    """Contributo di ogni variabile alla predizione (in log-odds), per modelli ad alberi."""
    import shap
    X = dati.codifica_ml(pd.DataFrame([profilo]))
    Xs = modello[:-1].transform(X)
    valori = shap.TreeExplainer(modello[-1]).shap_values(Xs)
    valori = valori[1] if isinstance(valori, list) else valori
    valori = valori[:, :, 1] if np.ndim(valori) == 3 else valori
    return pd.Series(np.ravel(valori), index=[C.nome_breve(c) for c in X.columns]).sort_values(key=abs)


# --- Interfaccia -----------------------------------------------------------------------------

def _menu(id_, etichetta, opzioni, valore):
    return dbc.Col([dbc.Label(etichetta), dcc.Dropdown(id=id_, options=opzioni, value=valore, clearable=False)], md=6, className="mb-3")


def _numero(id_, etichetta, valore, minimo, massimo, passo):
    return dbc.Col([dbc.Label(etichetta), dbc.Input(id=id_, type="number", value=valore, min=minimo, max=massimo, step=passo)],
                   md=6, className="mb-3")


CAMPI = {  # id del componente -> colonna del dataset originale
    "genere": C.GENERE, "eta": C.CLASSE_ETA, "reddito": C.REDDITO, "istruzione": C.ISTRUZIONE,
    "casa_prop": C.CASA_PROPRIETA, "casa_ind": C.CASA_INDIPENDENTE, "persone": C.N_PERSONE, "auto": C.N_AUTO,
    "emissioni": C.EMISSIONI, "auto_prec": C.AUTO_PRECEDENTE, "viaggio": C.VIAGGIO_LUNGO,
    "n_viaggi": C.N_VIAGGI_LUNGHI, "distanza": C.DISTANZA_LAVORO, "vmt": C.VMT,
}
REDDITI = sorted(int(r) for r in dati.carica_dati()[C.REDDITO].unique())  # classi presenti nel dataset


def crea_app(modello, meta) -> Dash:
    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Predizione BEV")
    app.index_string = app.index_string.replace("{%favicon%}", esporta_web.link_favicon())
    si_no = [{"label": "Sì", "value": "Si"}, {"label": "No", "value": "No"}]
    m = meta["metriche_test"]

    app.layout = dbc.Container([
        html.H2("Chi sceglie un'auto elettrica?", className="mt-4"),
        html.P(["Probabilità che il veicolo a basse emissioni di un cliente sia un ", html.B("BEV"),
                f" (e non PHEV, HEV o FCEV). Modello: {meta['modello']}, ROC-AUC sul test {m['roc_auc']:.3f}, "
                f"accuracy {m['accuracy']:.1%}."], className="text-muted"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Profilo socio-demografico"),
                dbc.Row([
                    _menu("genere", "Genere", ["Maschio", "Femmina"], "Maschio"),
                    _menu("eta", "Classe d'età", C.ORDINE_ETA, "45-54"),
                    _menu("reddito", "Reddito familiare (USD)", [{"label": f"{r:,}".replace(",", "."), "value": r} for r in REDDITI], 175000),
                    _menu("istruzione", "Livello di istruzione", C.ORDINE_ISTRUZIONE, "Laurea 1L"),
                    _menu("casa_prop", "Casa di proprietà", si_no, "Si"),
                    _menu("casa_ind", "Casa indipendente", si_no, "Si"),
                    _numero("persone", "Persone in famiglia", 3, 1, 13, 1),
                    _numero("auto", "Auto in famiglia", 2, 1, 5, 1),
                ]),
                html.H5("Atteggiamento e mobilità", className="mt-2"),
                dbc.Row([
                    dbc.Col([dbc.Label("Importanza di ridurre le emissioni (-3 … +3)"),
                             dcc.Slider(id="emissioni", min=-3, max=3, step=0.25, value=1.5,
                                        marks={i: str(i) for i in range(-3, 4)})], md=12, className="mb-3"),
                    _menu("auto_prec", "Tipologia di auto precedente", C.TIPI_AUTO, "ICE"),
                    _numero("viaggio", "Viaggio più lungo (miglia, 12 mesi)", 300, 0, 5000, 10),
                    _numero("n_viaggi", "Viaggi > 200 miglia (12 mesi)", 1, 0, 65, 1),
                    _numero("distanza", "Distanza casa-lavoro (miglia)", 12, 0, 500, 1),
                    _numero("vmt", "Miglia percorse all'anno (VMT)", 11000, 0, 100000, 500),
                ]),
            ])), lg=7),
            dbc.Col([
                dbc.Card(dbc.CardBody([dcc.Graph(id="indicatore", config={"displayModeBar": False}),
                                       html.Div(id="esito", className="text-center")])),
                dbc.Card(dbc.CardBody([html.H6("Perché questa predizione"),
                                       dcc.Graph(id="spiegazione", config={"displayModeBar": False})]), className="mt-3"),
            ], lg=5),
        ], className="g-3"),
        html.P("Le probabilità sono stime di un modello con capacità predittiva moderata (ROC-AUC ≈ 0,72): "
               "vanno lette come indicazioni di tendenza, non come certezze sul singolo cliente.",
               className="text-muted small mt-3 mb-5"),
    ], fluid="lg")

    @app.callback(Output("indicatore", "figure"), Output("esito", "children"), Output("spiegazione", "figure"),
                  [Input(k, "value") for k in CAMPI])
    def aggiorna(*valori):
        if any(v is None for v in valori):
            return go.Figure(), "Completa tutti i campi", go.Figure()
        profilo = dict(zip(CAMPI.values(), valori))
        p = predici(modello, profilo)
        indicatore = go.Figure(go.Indicator(
            mode="gauge+number", value=p * 100, number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Probabilità BEV"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2f6f5e"},
                   "threshold": {"line": {"color": "#c8813a", "width": 3}, "value": 54.2}}))
        indicatore.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
        esito = [html.B("Più probabile un BEV" if p >= meta["soglia"] else "Più probabile un altro tipo di veicolo"),
                 html.Br(), html.Small("La linea arancione indica la quota media di BEV nel campione (54,2%).",
                                       className="text-muted")]
        try:
            contrib = contributi_shap(modello, profilo).tail(8)
            spiegazione = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h",
                                           marker_color=np.where(contrib.values > 0, "#2f6f5e", "#c8813a")))
            spiegazione.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=30),
                                      xaxis_title="← verso altro veicolo | verso BEV →")
        except Exception:
            spiegazione = go.Figure()
        return indicatore, esito, spiegazione

    return app


if __name__ == "__main__":
    modello, meta = carica_modello_finale()
    crea_app(modello, meta).run(debug=False)
