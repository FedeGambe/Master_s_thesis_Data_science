# MCA tematiche sui possessori di BEV: variabili abitative, mobilità, socio-demografiche.
# Le prime due dimensioni di ciascuna MCA diventano gli indici compositi usati nel clustering.
source(file.path("R", "00_comune.R"))

bev <- dati_bev()

gruppi <- list(
  abitative = c("Casa.di.proprietà", "Casa.Indipendente",
                "Numero.persone.in.famiglia.categoriale", "Numero.di.auto.in.famiglia.categoriale"),
  mobilita = c("VMT.categoriale", "Viaggio.lungo.categoriale",
               "Numero.viaggi.lunghi.categoriale", "Distanza.casa.lavoro.categoriale"),
  sociodemografiche = c("Genere", "Classe.età.raggruppata", "Classe.Reddito.Familiare",
                        "Livello.di.istruzione", "Sensibilità.ambientale.categoriale")
)

punteggi <- list()
for (nome in names(gruppi)) {
  dati_gruppo <- bev[, gruppi[[nome]]] %>% mutate(across(everything(), as.factor))
  risultato <- MCA(dati_gruppo, graph = FALSE)
  salva(autovalori(risultato), paste0("mca_", nome, "_autovalori"))
  categorie <- cbind(risultato$var$coord[, 1:2], risultato$var$contrib[, 1:2])
  colnames(categorie) <- c("Dim1", "Dim2", "contributo_Dim1", "contributo_Dim2")
  salva(as.data.frame(categorie), paste0("mca_", nome, "_categorie"))
  punteggi[[nome]] <- risultato$ind$coord[, 1:2]

  salva_grafico(fviz_mca_var(risultato, repel = TRUE, choice = "var.cat") +
                  ggplot2::ggtitle(paste("MCA", nome)), paste0("mca_", nome, "_mappa"))
  salva_grafico(fviz_contrib(risultato, choice = "var", axes = 1, top = 15) +
                  ggplot2::ggtitle(paste("MCA", nome, "- contributi Dim 1")), paste0("mca_", nome, "_contributi"))
}

# Indici compositi. Correzione rispetto allo script originale: lo status socioeconomico
# usava per errore i punteggi della MCA sulla mobilità invece di quella socio-demografica.
indici <- data.frame(
  Stile.di.vita = punteggi$abitative[, 1],
  Dimensione.familiare = punteggi$abitative[, 2],
  Viaggi = punteggi$mobilita[, 1],
  Mobilità.quotidiana = punteggi$mobilita[, 2],
  Status.socioeconomico = punteggi$sociodemografiche[, 1],
  Distanza.sostenibilità = punteggi$sociodemografiche[, 2]
)
salva(indici, "indici_mca")
cat("MCA completate:", paste(names(gruppi), collapse = ", "), "\n")
