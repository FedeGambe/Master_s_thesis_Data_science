"""Fase 1: controlli di qualità e creazione dei dataset elaborati (usati anche dagli script R)."""
import _percorso  # noqa: F401

from tesi_bev import archivio, config as C, dati


def main():
    df = dati.carica_dati()
    C.DIR_DATI_ELABORATI.mkdir(parents=True, exist_ok=True)

    qualita = {
        "righe": len(df),
        "colonne": df.shape[1],
        "valori_mancanti": int(df.isna().sum().sum()),
        "righe_duplicate": int(df.duplicated().sum()),
        "quota_BEV_%": round(df[C.TARGET].mean() * 100, 2),
        "n_BEV": int(df[C.TARGET].sum()),
        "n_non_BEV": int((1 - df[C.TARGET]).sum()),
        "tipologia_auto_attuale": df[C.TIPO_AUTO_ATTUALE].value_counts().to_dict(),
    }
    archivio.salva_json(qualita, "dati", "qualita")

    X, y = dati.prepara_ml(df)
    X.assign(**{C.TARGET: y}).to_csv(C.DIR_DATI_ELABORATI / "dataset_ml.csv", index=False)

    Xl, yl = dati.prepara_logit(df, ridotto=False)
    Xl.assign(**{C.TARGET: yl}).to_csv(C.DIR_DATI_ELABORATI / "dataset_logit.csv", index=False)

    cat = dati.crea_variabili_categoriali(df)
    cat.to_csv(C.DIR_DATI_ELABORATI / "dataset_categoriale.csv", index=False)
    for col in [c for c in cat.columns if c.endswith("categoriale")]:
        archivio.salva_tabella(cat[col].value_counts().to_frame("n"), "dati", f"frequenze_{col}")

    print(f"Dataset: {qualita['righe']} righe, {qualita['valori_mancanti']} NaN, "
          f"{qualita['righe_duplicate']} duplicati, BEV {qualita['quota_BEV_%']}%")
    print(f"File scritti in {C.DIR_DATI_ELABORATI}")


if __name__ == "__main__":
    main()
