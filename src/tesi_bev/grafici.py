"""Grafici Plotly con uno stile comune a tutti i notebook."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config as C

COLORI_TARGET = {"BEV": "#2f6f5e", "Non BEV": "#c8813a"}
PALETTE = ["#2f6f5e", "#c8813a", "#4f6fa8", "#a8556b", "#7a8b3b", "#6b5ca5", "#3b8b9b", "#9b6b3b"]
TEMPLATE = "plotly_white"


def _stile(fig, titolo=None, altezza=450):
    fig.update_layout(template=TEMPLATE, title=titolo, height=altezza, colorway=PALETTE,
                      font=dict(family="Inter, Arial, sans-serif", size=13),
                      title_font=dict(size=17), margin=dict(l=60, r=30, t=60 if titolo else 30, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def _etichetta_target(df):
    return df[C.TARGET].map({1: "BEV", 0: "Non BEV"})


# --- Descrittive -----------------------------------------------------------------------------

def distribuzione(df, colonna, per_target=True, nbins=40, log_x=False):
    d = df.assign(Gruppo=_etichetta_target(df))
    fig = px.histogram(d, x=colonna, color="Gruppo" if per_target else None, nbins=nbins, barmode="overlay",
                       opacity=0.7, color_discrete_map=COLORI_TARGET, log_x=log_x, histnorm="percent")
    fig.update_yaxes(title="% osservazioni")
    return _stile(fig, colonna)


def quota_bev(df, colonna, ordine=None):
    g = df.groupby(colonna)[C.TARGET].agg(["mean", "size"]).reset_index()
    if ordine:
        g[colonna] = pd.Categorical(g[colonna], ordine, ordered=True)
        g = g.sort_values(colonna)
    g["quota"] = g["mean"] * 100
    fig = px.bar(g, x=colonna, y="quota", text=g["quota"].round(1), hover_data={"size": True})
    fig.add_hline(y=df[C.TARGET].mean() * 100, line_dash="dot", annotation_text="media campione")
    fig.update_traces(marker_color=PALETTE[0])
    fig.update_yaxes(title="% possessori BEV", range=[0, 100])
    return _stile(fig, f"Quota di BEV per {colonna}")


def matrice_correlazione(df, colonne=None, titolo="Matrice di correlazione"):
    corr = df[colonne or df.columns].corr()
    nomi = [C.nome_breve(c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=nomi, y=nomi, zmin=-1, zmax=1, colorscale="RdBu",
                               text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=9)))
    return _stile(fig, titolo, altezza=650)


# --- Logistica -------------------------------------------------------------------------------

def forest_plot(tabella_logit, titolo="Odds ratio con IC 95%"):
    t = tabella_logit.drop(index="const", errors="ignore").iloc[::-1]
    colori = np.where(t["significativo"], PALETTE[0], "#9a9a9a")
    fig = go.Figure(go.Scatter(
        x=t["odds_ratio"], y=t.index, mode="markers", marker=dict(color=colori, size=9),
        error_x=dict(type="data", symmetric=False, array=t["OR_ic_97.5%"] - t["odds_ratio"],
                     arrayminus=t["odds_ratio"] - t["OR_ic_2.5%"], color="#888"),
        hovertemplate="%{y}<br>OR %{x:.3f}<extra></extra>"))
    fig.add_vline(x=1, line_dash="dash")
    fig.update_xaxes(type="log", title="Odds ratio (scala log)", tickvals=[0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4])
    return _stile(fig, titolo, altezza=max(450, 26 * len(t)))


# --- Esplorative -----------------------------------------------------------------------------

def scree(autovalori, titolo="Varianza spiegata"):
    a = autovalori.reset_index(drop=True)
    x = [f"Dim {i + 1}" for i in range(len(a))]
    fig = go.Figure([go.Bar(x=x, y=a["varianza_%"], name="Varianza %", marker_color=PALETTE[0]),
                     go.Scatter(x=x, y=a["cumulata_%"], name="Cumulata %", mode="lines+markers",
                                marker_color=PALETTE[1])])
    fig.update_yaxes(title="%", range=[0, 105])
    return _stile(fig, titolo)


def mappa_categorie(categorie, titolo="Mappa delle categorie (MCA)"):
    c = categorie.reset_index().rename(columns={"index": "categoria"})
    c["variabile"] = c.iloc[:, 0].astype(str).str.split("__").str[0]
    c["modalita"] = c.iloc[:, 0].astype(str).str.split("__").str[-1]
    fig = px.scatter(c, x="Dim1", y="Dim2", color="variabile", text="modalita")
    fig.update_traces(textposition="top center", marker_size=10)
    fig.add_hline(y=0, line_color="#bbb")
    fig.add_vline(x=0, line_color="#bbb")
    return _stile(fig, titolo, altezza=600)


def loadings(loadings_df, titolo="Loadings PCA"):
    l = loadings_df.iloc[:, :2].copy()
    l.index = [C.nome_breve(i) for i in l.index]
    fig = go.Figure()
    for nome, r in l.iterrows():
        fig.add_trace(go.Scatter(x=[0, r.iloc[0]], y=[0, r.iloc[1]], mode="lines+markers+text", text=["", nome],
                                 textposition="top center", showlegend=False, line=dict(color=PALETTE[0])))
    fig.update_xaxes(range=[-1, 1], title=l.columns[0])
    fig.update_yaxes(range=[-1, 1], title=l.columns[1], scaleanchor="x")
    return _stile(fig, titolo, altezza=550)


# --- Cluster ---------------------------------------------------------------------------------

def gomito_silhouette(tabella):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Metodo del gomito (WSS)", "Silhouette media"))
    fig.add_trace(go.Scatter(x=tabella["k"], y=tabella["wss"], mode="lines+markers", name="WSS"), 1, 1)
    fig.add_trace(go.Scatter(x=tabella["k"], y=tabella["silhouette"], mode="lines+markers", name="Silhouette"), 1, 2)
    fig.update_xaxes(title="k")
    return _stile(fig, None, altezza=400)


def cluster_2d(coordinate, etichette, titolo="Cluster sulle prime due componenti"):
    d = pd.DataFrame(coordinate[:, :2], columns=["Dim1", "Dim2"]).assign(Cluster=pd.Series(etichette).astype(str))
    fig = px.scatter(d, x="Dim1", y="Dim2", color="Cluster", opacity=0.5)
    return _stile(fig, titolo, altezza=550)


def composizione_cluster(df, etichette, colonna):
    d = pd.crosstab(pd.Series(etichette, name="Cluster"), df[colonna].reset_index(drop=True), normalize="index") * 100
    fig = px.bar(d, barmode="stack")
    fig.update_yaxes(title="%")
    return _stile(fig, f"Composizione dei cluster per {colonna}")


# --- Modelli ---------------------------------------------------------------------------------

def box_screening(df_screening, metrica="roc_auc"):
    ordine = df_screening.groupby("modello")[metrica].mean().sort_values(ascending=False).index
    fig = px.box(df_screening, x="modello", y=metrica, category_orders={"modello": list(ordine)}, points="all")
    fig.update_traces(marker_color=PALETTE[0])
    return _stile(fig, f"Screening dei modelli: {metrica} in cross-validation", altezza=480)


def barre_metriche(tabella, metriche=("accuracy", "f1", "roc_auc"), ic=None):
    fig = go.Figure()
    for i, m in enumerate(metriche):
        err = None
        if ic is not None and f"{m}_ic_basso" in ic:
            err = dict(type="data", symmetric=False, array=ic.loc[tabella.index, f"{m}_ic_alto"] - tabella[m],
                       arrayminus=tabella[m] - ic.loc[tabella.index, f"{m}_ic_basso"])
        fig.add_trace(go.Bar(x=tabella.index, y=tabella[m], name=m, error_y=err, marker_color=PALETTE[i]))
    fig.update_yaxes(range=[0.5, 0.8])
    return _stile(fig, "Metriche sul test set (con IC 95% bootstrap)", altezza=480)


def curve_roc(curve: dict):
    fig = go.Figure()
    for nome, c in curve.items():
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], mode="lines", name=nome))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#aaa"), showlegend=False))
    fig.update_xaxes(title="Tasso falsi positivi")
    fig.update_yaxes(title="Tasso veri positivi")
    return _stile(fig, "Curve ROC sul test set", altezza=550)


def calibrazione(curve: dict):
    fig = go.Figure()
    for nome, c in curve.items():
        fig.add_trace(go.Scatter(x=c["probabilita_media_predetta"], y=c["frazione_positivi"], mode="lines+markers", name=nome))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#aaa"), showlegend=False))
    fig.update_xaxes(title="Probabilità predetta")
    fig.update_yaxes(title="Frazione osservata di BEV")
    return _stile(fig, "Calibrazione delle probabilità", altezza=500)


def importanza(tabella, colonna_valore, titolo, n=15):
    t = tabella.head(n).iloc[::-1]
    fig = go.Figure(go.Bar(x=t[colonna_valore], y=[C.nome_breve(v) for v in t["variabile"]], orientation="h",
                           marker_color=PALETTE[0]))
    return _stile(fig, titolo, altezza=max(400, 28 * len(t)))


def matrice_confusione(m: dict, titolo="Matrice di confusione"):
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    fig = go.Figure(go.Heatmap(z=z, x=["Pred. non BEV", "Pred. BEV"], y=["Reale non BEV", "Reale BEV"],
                               text=z, texttemplate="%{text}", colorscale="Greens", showscale=False))
    fig.update_yaxes(autorange="reversed")
    return _stile(fig, titolo, altezza=380)


def storia_addestramento(storia: dict, titolo="Addestramento della rete"):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "AUC"))
    for chiave, col in [("loss", 1), ("val_loss", 1), ("auc", 2), ("val_auc", 2)]:
        if chiave in storia:
            fig.add_trace(go.Scatter(y=storia[chiave], mode="lines", name=chiave), 1, col)
    fig.update_xaxes(title="Epoca")
    return _stile(fig, titolo, altezza=420)
