"""Fase 4: segmentazione dei possessori di BEV.

Come nella tesi, la segmentazione finale usa il K-Mode sulle variabili categoriali (k = 4).
Il K-Means sugli indici compositi e il clustering gerarchico servono da verifica: il K-Means
a 4 cluster è dominato dalle dummy dell'auto precedente, e per questo nella tesi era stato scartato.
"""
import _percorso  # noqa: F401

import pandas as pd
from sklearn.metrics import adjusted_rand_score

from tesi_bev import archivio, cluster as K, config as C

SEZIONE = "cluster"
VARIABILI_SENZA_AUTO = [v for v in K.VARIABILI_KMEANS if not v.startswith("Auto precedente")] + [
    "Status socioeconomico", C.EMISSIONI]
PROFILO_NUM = ["Età", C.REDDITO, C.EMISSIONI, C.N_PERSONE, C.N_AUTO, C.VMT, C.N_VIAGGI_LUNGHI, C.DISTANZA_LAVORO]
PROFILO_CAT = [C.AUTO_PRECEDENTE, C.CLASSE_REDDITO, C.ISTRUZIONE, C.CASA_INDIPENDENTE, C.GENERE]


def main():
    df = pd.read_csv(C.DIR_DATI_ELABORATI / "dataset_cluster.csv")
    riepilogo = {}

    # 1. K-Mode: segmentazione finale
    archivio.salva_tabella(K.costo_kmode(df[K.VARIABILI_KMODE]), SEZIONE, "scelta_k_kmode", indice=False)
    km = K.kmode(df, k=4)
    df["cluster_kmode"] = km["etichette"]
    archivio.salva_tabella(km["centri"], SEZIONE, "centroidi_kmode")
    for variabile, tabella in K.distribuzioni_cluster(df, km["etichette"], K.VARIABILI_KMODE).items():
        archivio.salva_tabella(tabella, SEZIONE, f"kmode_distribuzione_{variabile}")
    medie = df.groupby("cluster_kmode")[PROFILO_NUM].mean()
    medie["BEV precedente_%"] = df.groupby("cluster_kmode")[C.AUTO_PRECEDENTE].apply(lambda s: (s == "BEV").mean() * 100)
    archivio.salva_tabella(medie, SEZIONE, "kmode_medie")
    riepilogo["kmode"] = {
        "variabili": K.VARIABILI_KMODE, "k": 4, "costo": km["costo"],
        "dimensioni": pd.Series(km["etichette"]).value_counts().sort_index().tolist(),
        "stabilita_ARI_semi": K.stabilita_kmode(df, k=4),
        "ARI_con_auto_precedente": adjusted_rand_score(df[C.AUTO_PRECEDENTE], km["etichette"]),
    }

    # 2. Verifica con K-Means (configurazione della tesi e variante senza auto precedente)
    for nome, variabili in [("tesi", K.VARIABILI_KMEANS), ("senza_auto_precedente", VARIABILI_SENZA_AUTO)]:
        X = df[variabili]
        archivio.salva_tabella(K.gomito_silhouette(X, range(2, 9)), SEZIONE, f"scelta_k_{nome}", indice=False)
        r = K.kmeans(X, k=4)
        df[f"cluster_{nome}"] = r["etichette"]
        archivio.salva_tabella(K.profilo_cluster(df, r["etichette"], PROFILO_NUM, PROFILO_CAT), SEZIONE, f"profilo_{nome}")
        archivio.salva_tabella(pd.crosstab(r["etichette"], df[C.AUTO_PRECEDENTE], normalize="index") * 100,
                               SEZIONE, f"auto_precedente_per_cluster_{nome}")
        riepilogo[nome] = {
            "variabili": variabili, "k": 4, "wss": r["wss"], "silhouette": r["silhouette"],
            "stabilita_ARI_bootstrap": K.stabilita_bootstrap(X, k=4, n_ripetizioni=20),
            "dimensioni": pd.Series(r["etichette"]).value_counts().sort_index().tolist(),
            "ARI_con_auto_precedente": adjusted_rand_score(df[C.AUTO_PRECEDENTE], r["etichette"]),
            "ARI_con_kmode": adjusted_rand_score(df["cluster_kmode"], r["etichette"]),
        }

    # 3. Gerarchico (Ward) sulle stesse variabili del K-Means
    h = K.gerarchico(df[K.VARIABILI_KMEANS], k=4)
    riepilogo["gerarchico_ward"] = {"correlazione_cofenetica": h["correlazione_cofenetica"], "silhouette": h["silhouette"],
                                    "ARI_vs_kmeans_tesi": adjusted_rand_score(df["cluster_tesi"], h["etichette"])}

    archivio.salva_json(riepilogo, SEZIONE, "riepilogo")
    df[["cluster_kmode", "cluster_tesi", "cluster_senza_auto_precedente"]].to_csv(
        C.DIR_DATI_ELABORATI / "etichette_cluster.csv", index=False)
    r = riepilogo["kmode"]
    print(f"K-Mode: dimensioni {r['dimensioni']}, costo {r['costo']:.0f}, stabilità tra seed {r['stabilita_ARI_semi']:.3f}")
    print(km["centri"].T.to_string())
    for nome in ["tesi", "senza_auto_precedente"]:
        r = riepilogo[nome]
        print(f"K-Means {nome}: silhouette {r['silhouette']:.3f}, ARI con auto precedente {r['ARI_con_auto_precedente']:.3f}")


if __name__ == "__main__":
    main()
