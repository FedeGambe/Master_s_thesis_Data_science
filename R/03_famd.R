# FAMD (variabili miste) sui possessori di BEV.
source(file.path("R", "00_comune.R"))

bev <- dati_bev()
variabili <- c("Genere", "Classe.d.età", "Classe.Reddito.Familiare", "Livello.di.istruzione",
               "Casa.di.proprietà", "Casa.Indipendente", "Tipologia.di.auto.precedente",
               "Numero.persone.in.famiglia", "Numero.di.auto.in.famiglia",
               "Importanza.di.ridurre.le.emissioni.di.gas.serra", "Viaggio.più.lungo.negli.ultimi.12.mesi",
               "Numero.di.viaggi.superiori.a.200.miglia.negli.ultimi.12.mesi", "Distanza.casa.lavoro", "VMT.annuo")
dati_famd <- bev[, variabili] %>% mutate(across(where(is.character), as.factor))

risultato <- FAMD(dati_famd, ncp = 10, graph = FALSE)
salva(autovalori(risultato), "famd_autovalori")
salva(as.data.frame(risultato$var$contrib), "famd_contributi")

salva_grafico(fviz_eig(risultato, addlabels = TRUE, ncp = 10) + ggplot2::ggtitle("FAMD: varianza spiegata"), "famd_scree")
salva_grafico(fviz_famd_var(risultato, repel = TRUE) + ggplot2::ggtitle("FAMD: variabili"), "famd_variabili")
salva_grafico(fviz_contrib(risultato, "var", axes = 1) + ggplot2::ggtitle("FAMD: contributi Dim 1"), "famd_contributi_dim1")
salva_grafico(fviz_contrib(risultato, "var", axes = 2) + ggplot2::ggtitle("FAMD: contributi Dim 2"), "famd_contributi_dim2")

cat("FAMD - varianza spiegata Dim1+Dim2:", round(autovalori(risultato)$cumulata_perc[2], 2), "%\n")
