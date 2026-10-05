"""Fase 7: pagina dei risultati (HTML) e documento Markdown.

Legge le tabelle prodotte dalle fasi 1-6 in ``risultati/`` e scrive:
    docs/risultati.html   pagina autonoma con indice e grafici interattivi
    docs/RISULTATI.md     stessi contenuti in Markdown
"""
import _percorso  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from tesi_bev import archivio as A, config as C, dati
from tesi_bev.dati import nome_logit

MODELLO_HTML = Path(__file__).with_name("modello_pagina_risultati.html")
ALBERI = {"Voting (soft)", "LightGBM", "XGBoost", "Gradient Boosting", "Random Forest",
          "Gradient Boosting (feature selezionate)", "AdaBoost"}

ETICHETTE_LOGIT = {
    "const": "Intercetta",
    "Genere (Maschio)": "Genere maschile",
    **{f"{C.CLASSE_ETA}: {c} vs 45-54": f"Età {c} (vs 45-54)" for c in C.ORDINE_ETA},
    **{f"{C.CLASSE_REDDITO}: {c} vs Media": f"Reddito {m} (vs medio)" for c, m in
       zip(C.ORDINE_REDDITO, ["basso", "medio", "alto", "molto alto", "estremamente alto"])},
    **{f"{C.ISTRUZIONE}: {c} vs Laurea 2L o Dottorato": f"{c} (vs Laurea 2L)" for c in C.ORDINE_ISTRUZIONE},
    **{f"{C.AUTO_PRECEDENTE}: {c} vs PHEV": f"Auto precedente {c} (vs PHEV)" for c in C.TIPI_AUTO},
    f"{C.AUTO_PRECEDENTE}: PHEV vs ICE": "Auto precedente PHEV (vs ICE)",
    C.CASA_INDIPENDENTE: "Casa indipendente",
    C.CASA_PROPRIETA: "Casa di proprietà",
    C.N_PERSONE: "Persone in famiglia",
    C.N_AUTO: "Auto in famiglia (+1)",
    C.EMISSIONI: "Importanza ridurre emissioni (+1 punto)",
    C.N_VIAGGI_LUNGHI: "Viaggi > 200 miglia (+1)",
    nome_logit(C.VIAGGIO_LUNGO): "Viaggio più lungo (+100 mi)",
    nome_logit(C.DISTANZA_LAVORO): "Distanza casa-lavoro (+10 mi)",
    nome_logit(C.VMT): "VMT annuo (+1.000 mi)",
}
NOMI_VARIABILI = {
    C.REDDITO: "Reddito familiare (USD)", C.N_PERSONE: "Persone in famiglia", C.N_AUTO: "Auto in famiglia",
    C.EMISSIONI: "Importanza ridurre emissioni (-3/+3)", C.VIAGGIO_LUNGO: "Viaggio più lungo (miglia)",
    C.N_VIAGGI_LUNGHI: "Viaggi > 200 miglia (12 mesi)", C.DISTANZA_LAVORO: "Distanza casa-lavoro (miglia)",
    C.VMT: "VMT annuo (miglia)", C.CLASSE_REDDITO: "Classe di reddito", C.CLASSE_ETA: "Classe d'età",
    C.ISTRUZIONE: "Livello di istruzione", C.AUTO_PRECEDENTE: "Auto precedente", C.GENERE: "Genere",
    C.CASA_INDIPENDENTE: "Casa indipendente", C.CASA_PROPRIETA: "Casa di proprietà",
}
NOMI_INDICI = {"abitative": "variabili abitative", "mobilita": "mobilità e percorrenza",
               "sociodemografiche": "variabili socio-demografiche"}
DIM_INDICI = {"abitative": ("Stile di vita", "Dimensione familiare"), "mobilita": ("Viaggi", "Mobilità quotidiana"),
              "sociodemografiche": ("Status socioeconomico", "Distanza sostenibilità")}


def num(v, d=3):
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def nome_var(v):
    return NOMI_VARIABILI.get(v, C.nome_breve(v))


def righe(df):
    """DataFrame -> lista di dizionari serializzabili (NaN -> None)."""
    return json.loads(df.to_json(orient="records", force_ascii=False))


def quote(colonna):
    t = A.carica_tabella("statistica", f"quota_bev_{colonna}")
    return [{"etichetta": str(i), "valore": r.quota_BEV, "n": int(r.n)} for i, r in t.iterrows()]


ETICHETTE_BREVI = {
    C.GENERE: "Genere", C.CLASSE_ETA: "Età", "Classe età raggruppata": "Età", C.CLASSE_REDDITO: "Reddito",
    C.ISTRUZIONE: "Istruzione", C.CASA_PROPRIETA: "Casa di proprietà", C.CASA_INDIPENDENTE: "Casa indipendente",
    C.AUTO_PRECEDENTE: "Auto precedente",
}
ABBREVIAZIONI = {"Diploma o Qualifica professionale": "Diploma", "Laurea 2L o Dottorato": "Laurea 2L"}


def etichetta_categoria(variabile, modalita):
    modalita = ABBREVIAZIONI.get(modalita, modalita)
    if variabile.endswith("categoriale"):
        return modalita
    return f"{ETICHETTE_BREVI.get(variabile, nome_var(variabile))}: {modalita}"


def categoria_mca(indice):
    return etichetta_categoria(*indice.split("__"))


def variabile_famd(indice):
    """prince FAMD chiama le modalità "Variabile_modalità"; le numeriche restano col nome originale."""
    for v in ETICHETTE_BREVI:
        if indice.startswith(v + "_"):
            return etichetta_categoria(v, indice[len(v) + 1:])
    return nome_var(indice)


# --- Raccolta dei dati per capitolo ----------------------------------------------------------

# Istogrammi: (passo dei bin, unità) per le quantitative; None = valori interi singoli
BIN_DISTRIBUZIONI = {
    C.REDDITO: None, C.N_PERSONE: None, C.N_AUTO: None, C.EMISSIONI: (0.5, ""),
    C.VIAGGIO_LUNGO: (100, " mi"), C.N_VIAGGI_LUNGHI: None, C.DISTANZA_LAVORO: (5, " mi"), C.VMT: (2500, " mi"),
}
ORDINI_CATEGORIALI = {
    C.GENERE: ["Maschio", "Femmina"], C.CLASSE_ETA: C.ORDINE_ETA, C.CLASSE_REDDITO: C.ORDINE_REDDITO,
    C.ISTRUZIONE: C.ORDINE_ISTRUZIONE, C.CASA_PROPRIETA: ["Si", "No"], C.CASA_INDIPENDENTE: ["Si", "No"],
    C.AUTO_PRECEDENTE: C.TIPI_AUTO,
}


def _percentuali(serie_gruppo, etichette_gruppo, etichette):
    conteggi = etichette_gruppo.value_counts()
    return [round(float(conteggi.get(e, 0)) / len(serie_gruppo) * 100, 2) for e in etichette]


def distribuzioni():
    """Distribuzione di ogni variabile, separata per BEV e non BEV (% sul proprio gruppo)."""
    df = dati.carica_dati()
    bev = df[C.TARGET] == 1
    risultato = []
    for col, passo in BIN_DISTRIBUZIONI.items():
        x = df[col]
        nota = ""
        if passo is None:
            limite = int(x.quantile(0.99)) if col == C.N_VIAGGI_LUNGHI else None
            valori = sorted(x.unique()) if limite is None else list(range(0, limite + 1))
            etichette = [f"{v:,.0f}".replace(",", ".") for v in valori]
            classi = x.map(lambda v: f"{v:,.0f}".replace(",", ".") if limite is None or v <= limite else f"> {limite}")
            if limite is not None:
                etichette.append(f"> {limite}")
                nota = f"L'ultima barra raccoglie i valori oltre {limite} (1% delle osservazioni)."
        else:
            passo_v, unita = passo
            inizio = np.floor(x.min() / passo_v) * passo_v
            fine = np.ceil(x.quantile(0.99) / passo_v) * passo_v if col != C.EMISSIONI else 3.0
            bordi = np.arange(inizio, fine + passo_v / 2, passo_v)
            fmt_b = (lambda v: num(v, 1)) if passo_v < 1 else (lambda v: f"{v:,.0f}".replace(",", "."))
            etichette = [fmt_b(a) for a in bordi[:-1]]
            ultimo = len(bordi) - 2 if col == C.EMISSIONI else len(bordi) - 1  # il massimo (3,0) va nell'ultima barra
            idx = np.clip(np.digitize(x, bordi) - 1, 0, ultimo)
            classi = pd.Series([etichette[i] if i < len(etichette) else f"> {fmt_b(bordi[-1])}" for i in idx], index=x.index)
            if col != C.EMISSIONI:
                etichette.append(f"> {fmt_b(bordi[-1])}")
                nota = f"Barre da {fmt_b(passo_v)}{unita}; l'ultima raccoglie l'1% di valori più alti (oltre {fmt_b(bordi[-1])}{unita})."
            else:
                nota = "Scala da -3 (nessuna importanza) a +3 (massima importanza), barre da 0,5 punti."
        risultato.append({"nome": nome_var(col), "tipo": "quantitativa", "etichette": etichette, "nota": nota,
                          "bev": _percentuali(x[bev], classi[bev], etichette),
                          "altro": _percentuali(x[~bev], classi[~bev], etichette),
                          "mediana_bev": float(x[bev].median()), "mediana_altro": float(x[~bev].median())})
    for col, ordine in ORDINI_CATEGORIALI.items():
        etichette = [ABBREVIAZIONI.get(e, e) for e in ordine]
        classi = df[col].map(lambda v: ABBREVIAZIONI.get(v, v))
        risultato.append({"nome": nome_var(col), "tipo": "categoriale", "etichette": etichette, "nota": "",
                          "bev": _percentuali(df[col][bev], classi[bev], etichette),
                          "altro": _percentuali(df[col][~bev], classi[~bev], etichette)})
    return risultato


def statistica():
    descr = A.carica_tabella("statistica", "descrittive_numeriche")
    biv = A.carica_tabella("statistica", "test_bivariati", indice=False)
    logit = A.carica_tabella("statistica", "logit_ridotto")
    ame = A.carica_tabella("statistica", "effetti_marginali_ridotto")
    vif = A.carica_tabella("statistica", "vif", indice=False)
    ipotesi = A.carica_tabella("statistica", "verifica_ipotesi", indice=False)
    return {
        "descrittive": [{"variabile": nome_var(i), **{k: r[k] for k in ["media", "dev_std", "min", "q1", "mediana", "q3", "max", "asimmetria"]}}
                        for i, r in descr.iterrows()],
        "bivariati": sorted([{"nome": nome_var(r.variabile), "test": r.test, "misura": r.misura_effetto, "v": abs(r.effetto),
                              "segno": float(np.sign(r.effetto)), "p": r.p_value, "tipo": r.tipo} for r in biv.itertuples()],
                            key=lambda r: -r["v"]),
        "quote": {
            "reddito": quote(C.CLASSE_REDDITO), "eta": quote(C.CLASSE_ETA),
            "istruzione": [dict(q, etichetta=q["etichetta"].replace(" o Qualifica professionale", "").replace(" o Dottorato", ""))
                           for q in quote(C.ISTRUZIONE)],
            "auto": quote(C.AUTO_PRECEDENTE),
        },
        "vif": [{"variabile": ETICHETTE_LOGIT.get(r.variabile, r.variabile), "vif": r.VIF} for r in vif.itertuples()],
        "logit": [{"nome": ETICHETTE_LOGIT.get(i, i), "or": r.odds_ratio, "basso": r["OR_ic_2.5%"],
                   "alto": r["OR_ic_97.5%"], "p": r.p_value, "sig": bool(r.significativo)}
                  for i, r in logit.iterrows() if i != "const"],
        "ame": [{"nome": ETICHETTE_LOGIT.get(i, i), "v": r.effetto_marginale, "basso": r["ic_2.5%"], "alto": r["ic_97.5%"],
                 "p": r.p_value} for i, r in ame.iterrows()],
        "ipotesi": [{"codice": r.ipotesi, "descrizione": r.descrizione, "variabile": ETICHETTE_LOGIT.get(r.variabile, r.variabile),
                     "or": r.odds_ratio, "p": r.p_value, "esito": r.esito,
                     "esplorativa": r.direzione_attesa.startswith("nessuna")} for r in ipotesi.itertuples()],
        "bonta": A.carica_json("statistica", "bonta_adattamento"),
        "distribuzioni": distribuzioni(),
    }


def esplorativa():
    pv = A.carica_tabella("esplorativa", "pca_mobilita_varianza")
    pl = A.carica_tabella("esplorativa", "pca_mobilita_loadings")
    mca = {}
    for g in ["abitative", "mobilita", "sociodemografiche"]:
        av = A.carica_tabella("esplorativa", f"mca_{g}_autovalori")
        cat = A.carica_tabella("esplorativa", f"mca_{g}_categorie")
        cat.index = [categoria_mca(i) for i in cat.index]
        mca[g] = {"titolo": NOMI_INDICI[g], "dimensioni": DIM_INDICI[g], "varianza": av["varianza_%"].tolist()[:2],
                  "dim1": [{"nome": i, "v": r["contributo_Dim1_%"], "coord": r.Dim1}
                           for i, r in cat.sort_values("contributo_Dim1_%", ascending=False).head(6).iterrows()],
                  "dim2": [{"nome": i, "v": r["contributo_Dim2_%"], "coord": r.Dim2}
                           for i, r in cat.sort_values("contributo_Dim2_%", ascending=False).head(6).iterrows()]}
    fa = A.carica_tabella("esplorativa", "famd_autovalori")
    fc = A.carica_tabella("esplorativa", "famd_contributi")
    return {
        "pca": {"varianza": [{"c": i, "var": r["varianza_%"], "cum": r["cumulata_%"]} for i, r in pv.iterrows()],
                "loadings": [{"variabile": nome_var(i), **{c: r[c] for c in ["PC1", "PC2", "PC3", "PC4"]}} for i, r in pl.iterrows()],
                "alpha": A.carica_json("esplorativa", "affidabilita")["alpha_cronbach_mobilita"]},
        "mca": mca,
        "famd": {"varianza": [{"c": f"Dim {i + 1}", "var": r["varianza_%"], "cum": r["cumulata_%"]} for i, (_, r) in enumerate(fa.iterrows())],
                 "dim1": [{"nome": variabile_famd(i), "v": r["contributo_Dim1_%"]} for i, r in fc.sort_values("contributo_Dim1_%", ascending=False).head(8).iterrows()],
                 "dim2": [{"nome": variabile_famd(i), "v": r["contributo_Dim2_%"]} for i, r in fc.sort_values("contributo_Dim2_%", ascending=False).head(8).iterrows()]},
    }


def profilo(nome):
    p = A.carica_tabella("cluster", f"profilo_{nome}")
    return [{"cluster": f"Cluster {i + 1}", "n": int(r.n), "quota": r["quota_%"], "eta": r["Età"], "reddito": r[C.REDDITO],
             "emissioni": r[C.EMISSIONI], "vmt": r[C.VMT], "viaggi": r[C.N_VIAGGI_LUNGHI], "auto_prec": r[C.AUTO_PRECEDENTE],
             "istruzione": r[C.ISTRUZIONE], "casa": r[C.CASA_INDIPENDENTE]} for i, (_, r) in enumerate(p.iterrows())]


# Variabili che riassumono ciascun profilo K-Mode nel titolo della scheda
SINTESI_KMODE = ["Numero persone in famiglia categoriale", "Classe Reddito Familiare", "Classe età raggruppata",
                 "VMT categoriale", "Distanza casa-lavoro categoriale"]


def cluster():
    rie = A.carica_json("cluster", "riepilogo")
    comp = A.carica_tabella("cluster", "auto_precedente_per_cluster_tesi")
    km = A.carica_tabella("cluster", "centroidi_kmode")
    medie = A.carica_tabella("cluster", "kmode_medie")
    dim = rie["kmode"]["dimensioni"]
    sintesi = [" · ".join(f"reddito {v.lower()[:-1]}o" if c == C.CLASSE_REDDITO else f"età {v}" if c.startswith("Classe età") else v
                          for c in SINTESI_KMODE for v in [str(km.iloc[i][c])]) for i in range(len(km))]
    km.columns = [nome_var(c).replace(" categoriale", "") for c in km.columns]
    return {
        "kmode_profili": [{"cluster": f"Cluster {i + 1}", "sintesi": sintesi[i], "n": dim[i], "quota": dim[i] / sum(dim) * 100,
                           "eta": r["Età"], "reddito": r[C.REDDITO], "persone": r[C.N_PERSONE], "auto": r[C.N_AUTO],
                           "vmt": r[C.VMT], "viaggi": r[C.N_VIAGGI_LUNGHI], "distanza": r[C.DISTANZA_LAVORO],
                           "emissioni": r[C.EMISSIONI], "bev_prec": r["BEV precedente_%"]}
                          for i, (_, r) in enumerate(medie.iterrows())],
        "riepilogo": rie,
        "scelta_k": righe(A.carica_tabella("cluster", "scelta_k_tesi", indice=False)),
        "scelta_k_senza": righe(A.carica_tabella("cluster", "scelta_k_senza_auto_precedente", indice=False)),
        "composizione": {"tipi": list(comp.columns), "righe": [{"n": n, "quote": comp.iloc[i].tolist()}
                                                              for i, n in enumerate(rie["tesi"]["dimensioni"])]},
        "profilo": profilo("tesi"), "profilo_senza": profilo("senza_auto_precedente"),
        "kmode_costo": righe(A.carica_tabella("cluster", "scelta_k_kmode", indice=False)),
        "kmode_centri": [{"variabile": c, **{f"c{i + 1}": km.iloc[i][c] for i in range(len(km))}} for c in km.columns],
    }


def curva(y, p, punti=120):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, punti).astype(int))
    return {"x": fpr[idx].round(4).tolist(), "y": tpr[idx].round(4).tolist()}


def machine_learning():
    S = "machine_learning"
    scr = A.carica_tabella(S, "screening_riepilogo")
    ott = A.carica_json(S, "ottimizzazione")
    test = A.carica_tabella(S, "risultati_test")
    prob = pd.read_csv(C.DIR_RISULTATI / S / "probabilita_test.csv")
    cal = A.carica_json(S, "calibrazione")
    migliore = test.index[0]
    selezionati = list(test.index[:3]) + ["Regressione logistica"]
    abl = A.carica_tabella(S, "ablation_bev_precedente")

    def parametri(v):
        if "parametri" in v:
            return ", ".join(f"{p}={num(x, 3) if isinstance(x, float) else x}" for p, x in v["parametri"].items())
        return "media delle probabilità di " + ", ".join(v.get("componenti", []))

    return {
        "split": A.carica_json(S, "split"),
        "screening": [{"nome": i, "v": r.roc_auc_mean, "basso": r.roc_auc_mean - r.roc_auc_std, "alto": r.roc_auc_mean + r.roc_auc_std,
                       "acc": r.accuracy_mean, "f1": r.f1_mean} for i, r in scr.iterrows()],
        "parametri": [{"modello": k, "auc_cv": v.get("roc_auc_cv"), "parametri": parametri(v)} for k, v in ott.items()],
        "feature_selezionate": len(ott["Gradient Boosting (feature selezionate)"]["feature_selezionate"]),
        "test": [{"nome": i, **{k: r[k] for k in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
                                                  "roc_auc_ic_basso", "roc_auc_ic_alto", "brier"]},
                  "gruppo": "alberi" if i in ALBERI else "altro"} for i, r in test.iterrows()],
        "roc": {n: curva(prob["y"], prob[n]) for n in selezionati},
        "calibrazione": {n: {"x": cal[n]["probabilita_media_predetta"], "y": cal[n]["frazione_positivi"]}
                         for n in [migliore, "Regressione logistica"]},
        "mcnemar": [{"modello": i, "n01": int(r.n01), "n10": int(r.n10), "chi2": r.chi2, "p": r.p_value}
                    for i, r in A.carica_tabella(S, "mcnemar_vs_migliore").sort_values("p_value").iterrows()],
        "migliore": migliore,
        "confusione": {k: int(test.loc[migliore, k]) for k in ["tn", "fp", "fn", "tp"]},
        "shap": [{"nome": C.nome_breve(r.variabile), "v": r.shap_medio_assoluto}
                 for r in A.carica_tabella(S, "importanza_shap", indice=False).head(12).itertuples()],
        "permutazione": [{"nome": C.nome_breve(r.variabile), "v": r.importanza}
                         for r in A.carica_tabella(S, "importanza_permutazione", indice=False).head(12).itertuples()],
        "ablation": [{"nome": i, "accuracy": r.accuracy, "f1": r.f1, "roc_auc": r.roc_auc} for i, r in abl.iterrows()],
        "modello_shap": A.carica_json(S, "modello_finale")["modello_shap"],
    }


def deep_learning(ml):
    S = "reti_neurali"
    arch = A.carica_json(S, "architetture")
    storie = A.carica_json(S, "storia_addestramento")
    ris = A.carica_tabella(S, "risultati_test")
    prob = pd.read_csv(C.DIR_RISULTATI / S / "probabilita_test.csv")
    prob_ml = pd.read_csv(C.DIR_RISULTATI / "machine_learning" / "probabilita_test.csv")
    return {
        "architetture": [{"nome": k, "input": v["n_feature"], "strati": " → ".join(map(str, v["strati"])),
                          "attivazione": "LeakyReLU" if v.get("attivazione") == "leaky_relu" else "ReLU",
                          "regolarizzazione": ("L1 + L2 " if v.get("l1") else "L2 ") + num(v["l2"], 3) if v.get("l2") else "–",
                          "batch_norm": bool(v.get("batch_norm")), "dropout": v["dropout"],
                          "ottimizzatore": "AdamW" if v.get("ottimizzatore") == "adamw" else "Adam",
                          "parametri": v["n_parametri"], "epoche": v["epoche_eseguite"]} for k, v in arch.items()],
        "storie": {k: {c: v[c] for c in ["loss", "val_loss", "auc", "val_auc"]} for k, v in storie.items()},
        "test": [{"nome": i, **{k: r[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "roc_auc_ic_basso",
                                                  "roc_auc_ic_alto", "brier"]}} for i, r in ris.iterrows()],
        "roc": {ml["migliore"]: curva(prob_ml["y"], prob_ml[ml["migliore"]]), **{n: curva(prob["y"], prob[n]) for n in ris.index}},
    }


def kmode_r():
    """Risultato del K-Mode in R (klaR), se R/04_cluster.R è stato eseguito."""
    f = C.DIR_RISULTATI / "r" / "kmode_k4.csv"
    if not f.exists():
        return {"kmode_r_costo": "–", "kmode_r_dim": "–"}
    r = pd.read_csv(f).iloc[0]
    return {"kmode_r_costo": num(r["costo"], 0),
            "kmode_r_dim": " / ".join(f"{int(d):,}".replace(",", ".") for d in r["dimensioni"].split(" / "))}


def raccogli():
    qualita = A.carica_json("dati", "qualita")
    st, es, cl, ml = statistica(), esplorativa(), cluster(), machine_learning()
    dl = deep_learning(ml)
    logit = {r["nome"]: r for r in st["logit"]}
    direzionali = {r["codice"] for r in st["ipotesi"] if not r["esplorativa"]}
    n_conf = len({r["codice"] for r in st["ipotesi"] if r["esito"] == "Confermata"})
    best = ml["test"][0]
    rie = cl["riepilogo"]
    valori = {
        "n": f"{qualita['righe']:,}".replace(",", "."), "pct_bev": f"{num(qualita['quota_BEV_%'], 1)}%",
        "media_bev": qualita["quota_BEV_%"], "n_bev": f"{qualita['n_BEV']:,}".replace(",", "."),
        "r2": num(st["bonta"]["ridotto"]["pseudo_R2_McFadden"], 3),
        "or_ice": num(logit["Auto precedente ICE (vs PHEV)"]["or"], 2),
        "or_phev_ice": num(1 / logit["Auto precedente ICE (vs PHEV)"]["or"], 2),
        "or_bev": num(logit["Auto precedente BEV (vs PHEV)"]["or"], 2),
        "or_reddito_top": num(logit["Reddito estremamente alto (vs medio)"]["or"], 2),
        "vif_max": num(max(r["vif"] for r in st["vif"]), 2), "alpha": num(es["pca"]["alpha"], 2),
        "pca_cum2": num(es["pca"]["varianza"][1]["cum"], 1), "famd_cum2": num(es["famd"]["varianza"][1]["cum"], 1),
        "sil_tesi": num(rie["tesi"]["silhouette"], 2), "ari_tesi": num(rie["tesi"]["ARI_con_auto_precedente"], 3),
        "stab_tesi": num(rie["tesi"]["stabilita_ARI_bootstrap"], 2),
        "kmode_stab": num(rie["kmode"]["stabilita_ARI_semi"], 2), "kmode_costo": num(rie["kmode"]["costo"], 0),
        "kmode_ari_auto": num(rie["kmode"]["ARI_con_auto_precedente"], 3),
        "ari_kmeans_kmode": num(rie["tesi"]["ARI_con_kmode"], 3),
        "kmode_dim": " / ".join(f"{d:,}".replace(",", ".") for d in rie["kmode"]["dimensioni"]),
        **kmode_r(),
        "sil_senza": num(rie["senza_auto_precedente"]["silhouette"], 2),
        "stab_senza": num(rie["senza_auto_precedente"]["stabilita_ARI_bootstrap"], 2),
        "cofenetica": num(rie["gerarchico_ward"]["correlazione_cofenetica"], 2),
        "ari_ward": num(rie["gerarchico_ward"]["ARI_vs_kmeans_tesi"], 2),
        "auc_best": num(best["roc_auc"], 3), "acc_best": f"{num(best['accuracy'] * 100, 1)}%", "migliore": ml["migliore"],
        "auc_ic": f"{num(best['roc_auc_ic_basso'], 3)}–{num(best['roc_auc_ic_alto'], 3)}",
        "n_train": f"{ml['split']['n_train']:,}".replace(",", "."), "n_test": f"{ml['split']['n_test']:,}".replace(",", "."),
        "modello_shap": ml["modello_shap"], "n_feature_sel": ml["feature_selezionate"],
        "abl_con": num(ml["ablation"][0]["roc_auc"], 3), "abl_senza": num(ml["ablation"][1]["roc_auc"], 3),
        "auc_dl": num(max(r["roc_auc"] for r in dl["test"]), 3),
        "auc_dl_avanzata": num(next(r["roc_auc"] for r in dl["test"] if r["nome"] == "ANN avanzata (tesi)"), 3),
        "acc_dl_avanzata": f"{num(next(r['accuracy'] for r in dl['test'] if r['nome'] == 'ANN avanzata (tesi)') * 100, 1)}%",
        "n_poly": next(a["input"] for a in dl["architetture"] if a["nome"] == "ANN avanzata (tesi)"),
    }
    cifre = [
        {"v": valori["n"], "t": "possessori di veicoli a basse emissioni"},
        {"v": valori["pct_bev"], "t": "possiede un BEV"},
        {"v": f"{n_conf} su {len(direzionali)}", "t": "ipotesi confermate (almeno in parte), più una esplorativa"},
        {"v": valori["auc_best"], "t": f"ROC-AUC del modello migliore ({ml['migliore']})"},
    ]
    return {"valori": valori, "cifre": cifre, "stat": st, "espl": es, "cluster": cl, "ml": ml, "dl": dl}


# --- Markdown --------------------------------------------------------------------------------

def tabella_md(intestazioni, righe_md):
    return ["| " + " | ".join(intestazioni) + " |", "|" + "|".join("---" for _ in intestazioni) + "|",
            *["| " + " | ".join(map(str, r)) + " |" for r in righe_md], ""]


def pval(p):
    return "< 0,001" if p < 0.001 else num(p, 3)


def scrivi_markdown(d, percorso):
    v, st, es, cl, ml, dl = d["valori"], d["stat"], d["espl"], d["cluster"], d["ml"], d["dl"]
    arch = {a["nome"]: a for a in dl["architetture"]}
    md = [
        "# Analisi data-driven per l'adozione di veicoli elettrici", "",
        "*Analisi statistiche e predittive sugli acquirenti di auto elettriche*", "",
        "Laureando: **Federico Gamberini** (mat. 178147) · Relatore: **Prof. Roberto Cavicchioli**  ",
        "Laurea Magistrale in Management e Comunicazione d'Impresa, Università degli Studi di Modena e Reggio Emilia, "
        "Dipartimento di Comunicazione ed Economia, A.A. 2023/2024", "",
        "> Versione interattiva con grafici: [`docs/risultati.html`](risultati.html).", "",
        "## Indice", "",
        "- [Introduzione](#introduzione)",
        "- [1. Analisi statistica](#1-analisi-statistica)",
        "- [2. Analisi esplorativa](#2-analisi-esplorativa)",
        "- [3. Cluster](#3-cluster)",
        "- [4. Machine learning](#4-machine-learning)",
        "- [5. Deep learning](#5-deep-learning)",
        "- [Limiti e conclusioni](#limiti-e-conclusioni)", "",
        "## Introduzione", "",
        "**Contesto.** Transizione del settore automotive verso la riduzione delle emissioni, con normative ambientali e "
        "incentivi in aumento. **Obiettivo.** Individuare i fattori socio-demografici e le variabili chiave nella scelta di "
        "un veicolo elettrico. **Implicazioni.** Strategie di marketing per la mobilità elettrica e politiche pubbliche per "
        "la sostenibilità.", "",
        "**Dataset.** *Sociodemographic data for BEV owning households in California* (UC Davis), derivato dalla ricerca "
        "*Understanding the Early Adopters of Fuel Cell Vehicles* di Scott Hardman. Dai 27.021 casi e 27 variabili originali la "
        f"pulizia (OpenRefine e Python) ha portato a **{v['n']} casi e 16 variabili**. Tutti i rispondenti possiedono un veicolo "
        f"a basse emissioni; il {v['pct_bev']} un BEV.", "",
        *tabella_md(["Gruppo", "Variabili"], [
            ["Socio-demografiche", "genere, classe d'età, classe di reddito, livello di istruzione"],
            ["Abitative", "casa di proprietà, casa indipendente, persone in famiglia, auto in famiglia"],
            ["Sensibilità ambientale", "importanza di ridurre le emissioni di gas serra"],
            ["Classe veicolare", "tipologia di auto attuale e precedente"],
            ["Mobilità e percorrenza", "viaggio più lungo, viaggi oltre 200 miglia, distanza casa-lavoro, VMT annuo"]]),
        "## 1. Analisi statistica", "",
        "### 1.1 Statistiche descrittive", "",
        *tabella_md(["Variabile", "Media", "Dev. std", "Q1", "Mediana", "Q3", "Max"],
                    [[r["variabile"], num(r["media"], 2), num(r["dev_std"], 2), num(r["q1"], 2), num(r["mediana"], 2),
                      num(r["q3"], 2), num(r["max"], 2)] for r in st["descrittive"]]),
        "### 1.2 Analisi bivariata", "",
        "Quota di BEV per classe di reddito: " + ", ".join(f"{q['etichetta']} {num(q['valore'], 1)}%" for q in st["quote"]["reddito"]) + ".", "",
        *tabella_md(["Variabile", "Test", "Effetto", "Misura", "p-value"],
                    [[r["nome"], r["test"], num(r["v"], 3), r["misura"], pval(r["p"])] for r in st["bivariati"]]),
        "### 1.3 Regressione logistica", "",
        f"Pseudo-R² di McFadden {v['r2']}; VIF massimo {v['vif_max']}.", "",
        *tabella_md(["Variabile", "Odds ratio", "IC 95%", "p-value"],
                    [[r["nome"], num(r["or"]), f"{num(r['basso'])}–{num(r['alto'])}", pval(r["p"])] for r in st["logit"]]),
        "### 1.4 Ipotesi di ricerca", "",
        *tabella_md(["Ipotesi", "Variabile", "Odds ratio", "p-value", "Esito"],
                    [[r["codice"], r["variabile"], num(r["or"]), pval(r["p"]), r["esito"]] for r in st["ipotesi"]]),
        f"**H2.1 non è supportata:** chi aveva una PHEV ha odds di BEV {v['or_phev_ice']} volte quelle di chi aveva un'auto ICE. "
        f"**H2.2 (esplorativa):** rispetto alla PHEV, chi aveva un'auto ICE ha odds {v['or_ice']} volte maggiori; HEV e GNC "
        "non differiscono in modo significativo.", "",
        "## 2. Analisi esplorativa", "",
        f"Analisi svolte sui {v['n_bev']} possessori di BEV.", "",
        f"- **PCA sulla mobilità:** le prime due componenti spiegano il {v['pca_cum2']}% della varianza; alpha di Cronbach {v['alpha']}.",
        *[f"- **MCA {g['titolo']}:** Dim1 {num(g['varianza'][0], 1)}%, Dim2 {num(g['varianza'][1], 1)}%; indici "
          f"*{g['dimensioni'][0]}* (Dim1) e *{g['dimensioni'][1]}* (Dim2)." for g in es["mca"].values()],
        f"- **FAMD:** le prime due dimensioni spiegano il {v['famd_cum2']}%; Dim1 è guidata da "
        + ", ".join(r["nome"] for r in es["famd"]["dim1"][:3]) + ".", "",
        "## 3. Cluster", "",
        "### 3.1 K-Mode: segmentazione finale", "",
        f"Come nella tesi, la segmentazione finale usa il K-Mode (k = 4) sulle {len(cl['kmode_centri'])} variabili categoriali "
        f"dei possessori di BEV. Molte partizioni hanno un costo quasi uguale: con semi diversi i gruppi cambiano (ARI medio "
        f"{v['kmode_stab']}), quindi i profili vanno letti come segmenti descrittivi e non come gruppi naturali.", "",
        f"> **Confronto R / Python.** Con le stesse variabili, R (klaR, migliore di 20 avvii) trova costo {v['kmode_r_costo']} "
        f"e cluster di {v['kmode_r_dim']} persone; Python (kmodes, 50 inizializzazioni) costo {v['kmode_costo']} e cluster di "
        f"{v['kmode_dim']}; la tesi riportava 1.438 / 1.270 / 1.473 / 1.615. Sia in R sia in Python tornano un gruppo di coppie "
        "con VMT basso e uno di famiglie con 2 figli, 3 auto e pendolarismo lungo; cambia la suddivisione per reddito, età e "
        "numero di viaggi.", "",
        *tabella_md(["Cluster", "Profilo", "n", "Età", "Reddito (USD)", "VMT", "Viaggi > 200 mi", "Casa-lavoro (mi)"],
                    [[r["cluster"], r["sintesi"], r["n"], num(r["eta"], 1), num(r["reddito"], 0), num(r["vmt"], 0),
                      num(r["viaggi"], 1), num(r["distanza"], 1)] for r in cl["kmode_profili"]]),
        *tabella_md(["Variabile", "C1", "C2", "C3", "C4"],
                    [[r["variabile"], r["c1"], r["c2"], r["c3"], r["c4"]] for r in cl["kmode_centri"]]),
        "### 3.2 Verifica con K-Means e gerarchico", "",
        f"K-Means con k = 4 su età, indici compositi e auto precedente: silhouette {v['sil_tesi']}, ma ARI con l'auto "
        f"precedente {v['ari_tesi']}: i cluster coincidono con il tipo di auto posseduta prima, ed è per questo che nella tesi "
        f"è stato scartato. Senza quelle variabili: silhouette {v['sil_senza']}, stabilità {v['stab_senza']}. Gerarchico (Ward): "
        f"correlazione cofenetica {v['cofenetica']}.", "",
        "## 4. Machine learning", "",
        f"Split stratificato {v['n_train']} / {v['n_test']}; screening di 15 modelli in 10-fold CV; ottimizzazione con "
        "RandomizedSearchCV su ROC-AUC.", "",
        *tabella_md(["Modello", "Accuracy", "F1", "ROC-AUC", "IC 95%"],
                    [[r["nome"], num(r["accuracy"]), num(r["f1"]), num(r["roc_auc"]),
                      f"{num(r['roc_auc_ic_basso'])}–{num(r['roc_auc_ic_alto'])}"] for r in ml["test"]]),
        f"Variabili più importanti (SHAP, {v['modello_shap']}): " + ", ".join(r["nome"] for r in ml["shap"][:6]) + ".", "",
        f"Senza *auto precedente BEV* la ROC-AUC passa da {v['abl_con']} a {v['abl_senza']}.", "",
        "## 5. Deep learning", "",
        f"La *ANN avanzata* riproduce il modello finale della tesi: {v['n_poly']} feature polinomiali di grado 2, LeakyReLU, "
        "regolarizzazione L1 + L2, BatchNormalization e AdamW. Qui il set di validazione è estratto dal training (nella tesi "
        f"coincideva con il test set): accuracy {v['acc_dl_avanzata']}, ROC-AUC {v['auc_dl_avanzata']}.", "",
        *tabella_md(["Rete", "Input", "Strati", "Attivazione", "Parametri", "Epoche", "Accuracy", "ROC-AUC", "IC 95%"],
                    [[r["nome"], arch[r["nome"]]["input"], arch[r["nome"]]["strati"], arch[r["nome"]]["attivazione"],
                      f"{arch[r['nome']]['parametri']:,}".replace(",", "."), arch[r["nome"]]["epoche"], num(r["accuracy"]),
                      num(r["roc_auc"]), f"{num(r['roc_auc_ic_basso'])}–{num(r['roc_auc_ic_alto'])}"] for r in dl["test"]]),
        "## Limiti e conclusioni", "",
        "- **Contesto geografico:** solo California e solo possessori di veicoli a basse emissioni.",
        f"- **Capacità esplicativa contenuta:** pseudo-R² {v['r2']}, ROC-AUC massima {v['auc_best']}.",
        "- **Variabili mancanti:** nessuna variabile psicologica o culturale, nessuna misura concreta dell'attitudine verso la sostenibilità.",
        "- **Fattori influenti:** reddito, miglia annue e numero di viaggi lunghi; conta anche la sensibilità ambientale.",
        "- **Il passaggio non è graduale:** chi arriva da un'auto tradizionale sceglie il BEV più spesso di chi aveva una PHEV.",
        "- **Transizione culturale, non solo tecnologica**, sostenuta da politiche mirate.", "",
    ]
    percorso.write_text("\n".join(md), encoding="utf-8")


def main():
    d = raccogli()
    C.DIR_DOCS.mkdir(exist_ok=True)
    scrivi_markdown(d, C.DIR_DOCS / "RISULTATI.md")
    html = MODELLO_HTML.read_text(encoding="utf-8").replace(
        "/*DATI*/null", json.dumps(d, ensure_ascii=False, default=float))
    (C.DIR_DOCS / "risultati.html").write_text(html, encoding="utf-8")
    print("Scritti: docs/risultati.html, docs/RISULTATI.md")


if __name__ == "__main__":
    main()
