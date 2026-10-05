import pandas as pd
import pytest

from tesi_bev import config as C, dati


@pytest.fixture(scope="module")
def df():
    return dati.carica_dati()


def test_dataset_integro(df):
    assert df.shape == (10688, 18)
    assert df.isna().sum().sum() == 0
    assert set(df[C.TARGET].unique()) == {0, 1}


def test_codifica_ml(df):
    X, y = dati.prepara_ml(df)
    assert list(X.columns) == dati.COLONNE_ML
    assert X.notna().all().all()
    # Maschio = 1, come in addestramento (la dashboard originale usava la codifica opposta)
    assert X.loc[df[C.GENERE] == "Maschio", C.GENERE].eq(1).all()
    # le dummy dell'auto precedente sono esclusive
    assert X.filter(like="Auto precedente").sum(axis=1).eq(1).all()


def test_codifica_singola_riga_coerente(df):
    """La dashboard codifica una riga alla volta: deve dare lo stesso risultato della codifica in blocco."""
    X = dati.codifica_ml(df)
    riga = dati.codifica_ml(df.iloc[[123]])
    pd.testing.assert_frame_equal(riga, X.iloc[[123]])


def test_valore_sconosciuto_genera_errore(df):
    riga = df.iloc[[0]].copy()
    riga[C.GENERE] = "Altro"
    with pytest.raises(ValueError):
        dati.codifica_ml(riga)


def test_logit_riferimenti(df):
    X, _ = dati.prepara_logit(df)
    assert f"{C.CLASSE_REDDITO}: Bassa vs Media" in X.columns
    assert C.N_PERSONE not in X.columns  # modello ridotto


def test_variabili_categoriali_senza_nan(df):
    cat = dati.crea_variabili_categoriali(df)
    for col in [c for c in cat.columns if c.endswith("categoriale")]:
        assert not cat[col].isin(["nan"]).any(), col
