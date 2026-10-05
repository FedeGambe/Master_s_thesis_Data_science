# Funzioni e percorsi comuni agli script R.
# Eseguire dalla radice del repository, ad esempio:  Rscript R/esegui_tutto.R
# I dataset in dati/elaborati/ vengono creati da:    python scripts/01_prepara_dati.py
#                                                    python scripts/03_esplorativa.py

suppressPackageStartupMessages({
  library(dplyr)
  library(FactoMineR)
  library(factoextra)
  library(cluster)
})

set.seed(343)

DIR_DATI <- file.path("dati", "elaborati")
DIR_RISULTATI <- file.path("risultati", "r")
DIR_FIGURE <- file.path(DIR_RISULTATI, "figure")
dir.create(DIR_FIGURE, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(file.path(DIR_DATI, "dataset_categoriale.csv"))) {
  stop("Dataset elaborati mancanti: eseguire prima 'python scripts/01_prepara_dati.py'")
}

leggi <- function(nome) read.csv(file.path(DIR_DATI, nome), encoding = "UTF-8")

salva <- function(df, nome) {
  write.csv(df, file.path(DIR_RISULTATI, paste0(nome, ".csv")), fileEncoding = "UTF-8")
}

salva_grafico <- function(grafico, nome, larghezza = 9, altezza = 6) {
  ggplot2::ggsave(file.path(DIR_FIGURE, paste0(nome, ".png")), grafico, width = larghezza, height = altezza, dpi = 150)
}

autovalori <- function(risultato) {
  e <- as.data.frame(risultato$eig)
  colnames(e) <- c("autovalore", "varianza_perc", "cumulata_perc")
  e
}

# Possessori di BEV con le variabili categoriali (equivale al vecchio data_eda.csv filtrato)
dati_bev <- function() {
  d <- leggi("dataset_categoriale.csv")
  d[d$BEV.dummy == 1, ]
}
