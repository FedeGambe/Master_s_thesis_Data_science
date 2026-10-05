"""Fase 3: PCA, MCA e FAMD sui possessori di BEV + indici compositi per il clustering."""
import _percorso  # noqa: F401

from tesi_bev import archivio, cluster as K, config as C, dati, esplorativa as E

SEZIONE = "esplorativa"


def main():
    df_bev = dati.solo_bev(dati.crea_variabili_categoriali())

    p = E.pca(df_bev, C.VARIABILI_MOBILITA)
    archivio.salva_tabella(p["varianza"], SEZIONE, "pca_mobilita_varianza")
    archivio.salva_tabella(p["loadings"], SEZIONE, "pca_mobilita_loadings")
    archivio.salva_json({"alpha_cronbach_mobilita": E.alpha_cronbach(df_bev[C.VARIABILI_MOBILITA])}, SEZIONE, "affidabilita")

    for nome, risultato in E.mca_gruppi(df_bev).items():
        archivio.salva_tabella(risultato["autovalori"], SEZIONE, f"mca_{nome}_autovalori")
        archivio.salva_tabella(risultato["categorie"], SEZIONE, f"mca_{nome}_categorie")

    f = E.famd(df_bev, n_componenti=10)
    archivio.salva_tabella(f["autovalori"], SEZIONE, "famd_autovalori")
    archivio.salva_tabella(f["contributi"], SEZIONE, "famd_contributi")

    indici = E.indici_compositi(df_bev)
    dataset = K.dataset_cluster(df_bev, indici)
    dataset.to_csv(C.DIR_DATI_ELABORATI / "dataset_cluster.csv", index=False)
    print(f"PCA mobilità: PC1+PC2 = {p['varianza']['cumulata_%'].iloc[1]:.1f}% della varianza")
    print(f"Dataset per il clustering: {dataset.shape} -> {C.DIR_DATI_ELABORATI / 'dataset_cluster.csv'}")


if __name__ == "__main__":
    main()
