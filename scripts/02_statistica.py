"""Fase 2: statistiche descrittive, test bivariati, VIF e regressione logistica (ipotesi H1-H4)."""
import _percorso  # noqa: F401

from tesi_bev import archivio, config as C, dati, statistica as S

SEZIONE = "statistica"


def main():
    df = dati.carica_dati()

    archivio.salva_tabella(S.descrittive_numeriche(df), SEZIONE, "descrittive_numeriche")
    archivio.salva_tabella(S.test_bivariati(df), SEZIONE, "test_bivariati", indice=False)
    for col, ordine in [(C.CLASSE_REDDITO, C.ORDINE_REDDITO), (C.CLASSE_ETA, C.ORDINE_ETA),
                        (C.ISTRUZIONE, C.ORDINE_ISTRUZIONE), (C.AUTO_PRECEDENTE, C.TIPI_AUTO),
                        (C.GENERE, None), (C.CASA_INDIPENDENTE, None), (C.CASA_PROPRIETA, None)]:
        archivio.salva_tabella(S.quota_bev_per_categoria(df, col, ordine), SEZIONE, f"quota_bev_{col}")

    # VIF sul modello completo: con l'intercetta nessuna variabile supera 5
    X_completo, y = dati.prepara_logit(df, ridotto=False)
    archivio.salva_tabella(S.vif(X_completo), SEZIONE, "vif", indice=False)

    risultati = {}
    for nome, ridotto in [("completo", False), ("ridotto", True)]:
        X, y = dati.prepara_logit(df, ridotto=ridotto)
        r = S.logit(X, y)
        archivio.salva_tabella(S.tabella_logit(r), SEZIONE, f"logit_{nome}")
        archivio.salva_tabella(S.effetti_marginali(r), SEZIONE, f"effetti_marginali_{nome}")
        risultati[nome] = (r, S.bonta_adattamento(r))

    archivio.salva_json({k: v[1] for k, v in risultati.items()}, SEZIONE, "bonta_adattamento")
    ipotesi = S.verifica_ipotesi(risultati["ridotto"][0])
    archivio.salva_tabella(ipotesi, SEZIONE, "verifica_ipotesi", indice=False)

    print("Pseudo-R² McFadden (ridotto):", round(risultati["ridotto"][1]["pseudo_R2_McFadden"], 4))
    print(ipotesi[["ipotesi", "variabile", "odds_ratio", "p_value", "esito"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
