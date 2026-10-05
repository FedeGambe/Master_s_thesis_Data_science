"""Configurazione centrale: percorsi, seed e nomi delle colonne.

Tutti i percorsi sono relativi alla radice del repository, così il codice
funziona allo stesso modo in locale, in CI e su Google Colab.
"""
from pathlib import Path

RADICE = Path(__file__).resolve().parents[2]

DIR_DATI_ORIGINALI = RADICE / "dati" / "originali"
DIR_DATI_ELABORATI = RADICE / "dati" / "elaborati"
DIR_RISULTATI = RADICE / "risultati"
DIR_MODELLI = RADICE / "modelli"
DIR_DOCS = RADICE / "docs"

FILE_DATASET = DIR_DATI_ORIGINALI / "dataset_originale.csv"

# Copia remota usata quando il file locale non è disponibile (es. notebook aperto da solo su Colab).
REPO_GITHUB = "FedeGambe/Master_s_thesis_Data_science"
RAMO_GITHUB = "revisione-progetto"
URL_DATASET = (
    f"https://raw.githubusercontent.com/{REPO_GITHUB}/{RAMO_GITHUB}/dati/originali/dataset_originale.csv"
)

SEED = 343          # stesso seed dello split usato nella tesi
TEST_SIZE = 0.2
N_FOLD = 10

TARGET = "BEV dummy"

# Colonne del dataset originale
GENERE = "Genere"
CLASSE_ETA = "Classe d'età"
REDDITO = "Reddito familiare"
CLASSE_REDDITO = "Classe Reddito Familiare"
ISTRUZIONE = "Livello di istruzione"
CASA_PROPRIETA = "Casa di proprietà"
CASA_INDIPENDENTE = "Casa Indipendente"
N_PERSONE = "Numero persone in famiglia"
N_AUTO = "Numero di auto in famiglia"
EMISSIONI = "Importanza di ridurre le emissioni di gas serra"
AUTO_PRECEDENTE = "Tipologia di auto precedente"
AUTO_ATTUALE = "Auto attuale"
TIPO_AUTO_ATTUALE = "Tipologia di auto attuale"
VIAGGIO_LUNGO = "Viaggio più lungo negli ultimi 12 mesi"
N_VIAGGI_LUNGHI = "Numero di viaggi superiori a 200 miglia negli ultimi 12 mesi"
DISTANZA_LAVORO = "Distanza casa-lavoro"
VMT = "VMT annuo"

VARIABILI_NUMERICHE = [REDDITO, N_PERSONE, N_AUTO, EMISSIONI, VIAGGIO_LUNGO, N_VIAGGI_LUNGHI, DISTANZA_LAVORO, VMT]
VARIABILI_CATEGORIALI = [GENERE, CLASSE_ETA, CLASSE_REDDITO, ISTRUZIONE, CASA_PROPRIETA, CASA_INDIPENDENTE, AUTO_PRECEDENTE]
VARIABILI_MOBILITA = [VIAGGIO_LUNGO, N_VIAGGI_LUNGHI, DISTANZA_LAVORO, VMT]

# Ordine delle categorie (usato per codifiche ordinali e grafici)
ORDINE_ETA = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", "75-79", ">80"]
ORDINE_REDDITO = ["Bassa", "Media", "Alta", "Molto alta", "Estremamente alta"]
ORDINE_ISTRUZIONE = ["Licenza Media", "Diploma o Qualifica professionale", "Laurea 1L", "Laurea 2L o Dottorato"]
TIPI_AUTO = ["PHEV", "BEV", "HEV", "ICE", "GNC"]

# Categorie di riferimento della regressione logistica (come nella tesi)
RIFERIMENTI_LOGIT = {
    CLASSE_ETA: "45-54",
    CLASSE_REDDITO: "Media",
    ISTRUZIONE: "Laurea 2L o Dottorato",
    AUTO_PRECEDENTE: "PHEV",
}

# Nomi brevi per grafici e tabelle
NOMI_BREVI = {
    GENERE: "Genere (M)",
    CLASSE_ETA: "Classe età",
    REDDITO: "Reddito",
    ISTRUZIONE: "Istruzione",
    CASA_PROPRIETA: "Casa prop.",
    CASA_INDIPENDENTE: "Casa indip.",
    N_PERSONE: "N. persone fam.",
    N_AUTO: "N. auto fam.",
    EMISSIONI: "Sensibilità ambientale",
    VIAGGIO_LUNGO: "Viaggio più lungo",
    N_VIAGGI_LUNGHI: "N. viaggi lunghi",
    DISTANZA_LAVORO: "Distanza casa-lavoro",
    VMT: "VMT annuo",
    **{f"Auto precedente: {t}": f"{t} precedente" for t in TIPI_AUTO},
}


def nome_breve(colonna: str) -> str:
    return NOMI_BREVI.get(colonna, colonna)
