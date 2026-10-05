"""Analisi statistiche: descrittive, test bivariati, VIF e regressione logistica."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config as C
from .dati import nome_logit


# --- Descrittive -----------------------------------------------------------------------------

def descrittive_numeriche(df: pd.DataFrame, colonne=None) -> pd.DataFrame:
    colonne = colonne or C.VARIABILI_NUMERICHE
    d = df[colonne].describe(percentiles=[0.25, 0.5, 0.75]).T
    d["asimmetria"] = df[colonne].skew()
    return d.rename(columns={"count": "n", "mean": "media", "std": "dev_std", "min": "min",
                             "25%": "q1", "50%": "mediana", "75%": "q3", "max": "max"})


def frequenze(df: pd.DataFrame, colonna: str) -> pd.DataFrame:
    n = df[colonna].value_counts(dropna=False)
    return pd.DataFrame({"n": n, "%": (n / n.sum() * 100).round(2)})


# --- Bivariate rispetto al target ------------------------------------------------------------

def cramer_v(tabella: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(tabella, correction=False)[0]
    n = tabella.to_numpy().sum()
    r, k = tabella.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def test_bivariati(df: pd.DataFrame, target: str = C.TARGET) -> pd.DataFrame:
    """Chi-quadro + V di Cramér per le categoriali, Mann-Whitney + r rank-biserial per le numeriche."""
    righe = []
    for col in C.VARIABILI_CATEGORIALI:
        tab = pd.crosstab(df[col], df[target])
        chi2, p, gdl, _ = stats.chi2_contingency(tab)
        righe.append({"variabile": col, "tipo": "categoriale", "test": "Chi-quadro",
                      "statistica": chi2, "p_value": p, "effetto": cramer_v(tab), "misura_effetto": "V di Cramér"})
    for col in C.VARIABILI_NUMERICHE:
        a, b = df.loc[df[target] == 1, col], df.loc[df[target] == 0, col]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r = 1 - 2 * u / (len(a) * len(b))  # rank-biserial: >0 se i valori sono più alti tra i NON BEV
        righe.append({"variabile": col, "tipo": "numerica", "test": "Mann-Whitney U",
                      "statistica": u, "p_value": p, "effetto": -r, "misura_effetto": "r rank-biserial (BEV vs non BEV)",
                      "mediana_BEV": a.median(), "mediana_non_BEV": b.median()})
    return pd.DataFrame(righe)


def quota_bev_per_categoria(df: pd.DataFrame, colonna: str, ordine=None) -> pd.DataFrame:
    g = df.groupby(colonna)[C.TARGET].agg(["mean", "size"]).rename(columns={"mean": "quota_BEV", "size": "n"})
    g["quota_BEV"] = (g["quota_BEV"] * 100).round(2)
    return g.reindex(ordine) if ordine else g


# --- Multicollinearità -----------------------------------------------------------------------

def vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF calcolato con l'intercetta (la versione della tesi la ometteva)."""
    Xc = sm.add_constant(X, has_constant="add")
    valori = [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]
    return pd.DataFrame({"variabile": X.columns, "VIF": valori}).sort_values("VIF", ascending=False, ignore_index=True)


# --- Regressione logistica -------------------------------------------------------------------

def logit(X: pd.DataFrame, y: pd.Series):
    return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=False)


def tabella_logit(risultato) -> pd.DataFrame:
    ic = risultato.conf_int()
    t = pd.DataFrame({
        "coefficiente": risultato.params,
        "errore_std": risultato.bse,
        "p_value": risultato.pvalues,
        "odds_ratio": np.exp(risultato.params),
        "OR_ic_2.5%": np.exp(ic[0]),
        "OR_ic_97.5%": np.exp(ic[1]),
    })
    t["significativo"] = t["p_value"] < 0.05
    return t


def effetti_marginali(risultato) -> pd.DataFrame:
    """Effetti marginali medi: variazione della probabilità di possedere un BEV (in punti percentuali)."""
    m = risultato.get_margeff(at="overall", method="dydx")
    t = m.summary_frame()
    t.columns = ["effetto_marginale", "errore_std", "z", "p_value", "ic_2.5%", "ic_97.5%"]
    for c in ["effetto_marginale", "ic_2.5%", "ic_97.5%"]:
        t[c] = t[c] * 100
    return t


def bonta_adattamento(risultato) -> dict:
    return {
        "n": int(risultato.nobs),
        "pseudo_R2_McFadden": float(risultato.prsquared),
        "log_likelihood": float(risultato.llf),
        "log_likelihood_nullo": float(risultato.llnull),
        "LR_test_p_value": float(risultato.llr_pvalue),
        "AIC": float(risultato.aic),
        "BIC": float(risultato.bic),
    }


# Ipotesi della tesi e coefficienti che le misurano
# Prefisso per i confronti letti al contrario rispetto al riferimento del modello
# (es. "PHEV vs ICE" si ricava dal coefficiente "ICE vs PHEV" cambiato di segno).
INVERSO = "inverso:"

# codice -> (descrizione, variabili, direzione attesa: +1 aumenta, -1 riduce, 0 ipotesi esplorativa)
IPOTESI = {
    "H1": ("Un reddito più alto aumenta la probabilità di possedere un BEV",
           [f"{C.CLASSE_REDDITO}: {c} vs Media" for c in ["Bassa", "Alta", "Molto alta", "Estremamente alta"]], +1),
    "H2.1": ("Aver posseduto una PHEV favorisce il passaggio al BEV, rispetto a chi aveva un'auto tradizionale (ICE)",
             [f"{INVERSO}{C.AUTO_PRECEDENTE}: ICE vs PHEV"], +1),
    "H2.2": ("Come cambia la probabilità di possedere un BEV per chi aveva un'auto ICE, HEV o GNC (rispetto a PHEV)",
             [f"{C.AUTO_PRECEDENTE}: {c} vs PHEV" for c in ["ICE", "HEV", "GNC"]], 0),
    "H2.3": ("Chi ha già avuto un BEV tende a ricomprarlo",
             [f"{C.AUTO_PRECEDENTE}: BEV vs PHEV"], +1),
    "H3": ("Lunghe percorrenze riducono la probabilità di possedere un BEV",
           [nome_logit(c) for c in C.VARIABILI_MOBILITA], -1),
    "H4": ("Chi dà importanza alla riduzione delle emissioni sceglie più spesso un BEV",
           [C.EMISSIONI], +1),
}


def _nome_inverso(variabile: str) -> str:
    """"Tipologia di auto precedente: ICE vs PHEV" -> "Tipologia di auto precedente: PHEV vs ICE"."""
    prefisso, confronto = variabile.split(": ")
    a, b = confronto.split(" vs ")
    return f"{prefisso}: {b} vs {a}"


def verifica_ipotesi(risultato, alfa: float = 0.05) -> pd.DataFrame:
    """Confronta segno e significatività dei coefficienti con la direzione attesa di ogni ipotesi.

    Per H1 la variabile "Bassa vs Media" ha direzione attesa opposta alle altre.
    Le ipotesi esplorative (direzione 0) riportano solo se l'effetto aumenta o riduce la probabilità.
    """
    tab = tabella_logit(risultato)
    righe = []
    for codice, (descrizione, variabili, segno) in IPOTESI.items():
        for v in variabili:
            inverso = v.startswith(INVERSO)
            colonna = v[len(INVERSO):] if inverso else v
            coef, p = tab.loc[colonna, "coefficiente"], tab.loc[colonna, "p_value"]
            if inverso:
                coef = -coef
            atteso = -segno if "Bassa vs" in v else segno
            if p >= alfa:
                esito = "Non significativo"
            elif atteso == 0:
                esito = "Aumenta la probabilità" if coef > 0 else "Riduce la probabilità"
            elif np.sign(coef) == atteso:
                esito = "Confermata"
            else:
                esito = "Contraria"
            righe.append({"ipotesi": codice, "descrizione": descrizione,
                          "variabile": _nome_inverso(colonna) if inverso else colonna,
                          "odds_ratio": float(np.exp(coef)), "p_value": p,
                          "direzione_attesa": {1: "+", -1: "-", 0: "nessuna (esplorativa)"}[int(atteso)], "esito": esito})
    return pd.DataFrame(righe)
