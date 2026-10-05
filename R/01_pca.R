# PCA sulle variabili quantitative (possessori di BEV) e sulle sole variabili di mobilità.
source(file.path("R", "00_comune.R"))
library(psych)

bev <- dati_bev()

quantitative <- bev %>% select(
  Reddito.familiare, Numero.persone.in.famiglia, Numero.di.auto.in.famiglia,
  Importanza.di.ridurre.le.emissioni.di.gas.serra, Viaggio.più.lungo.negli.ultimi.12.mesi,
  Numero.di.viaggi.superiori.a.200.miglia.negli.ultimi.12.mesi, Distanza.casa.lavoro, VMT.annuo)

# PCA su tutte le quantitative
pca_tutte <- prcomp(quantitative, scale. = TRUE)
varianza <- data.frame(
  componente = paste0("PC", seq_along(pca_tutte$sdev)),
  autovalore = pca_tutte$sdev^2,
  varianza_perc = 100 * pca_tutte$sdev^2 / sum(pca_tutte$sdev^2))
varianza$cumulata_perc <- cumsum(varianza$varianza_perc)
salva(varianza, "pca_quantitative_varianza")
salva(as.data.frame(pca_tutte$rotation), "pca_quantitative_loadings")

# PCA sulle 4 variabili di mobilità: da qui nascono gli indici "Viaggi PCA" e "Mobilità quotidiana PCA"
mobilita <- quantitative %>% select(
  Viaggio.più.lungo.negli.ultimi.12.mesi, Numero.di.viaggi.superiori.a.200.miglia.negli.ultimi.12.mesi,
  Distanza.casa.lavoro, VMT.annuo)
alpha_mobilita <- psych::alpha(mobilita, check.keys = FALSE, warnings = FALSE)
salva(data.frame(alpha_cronbach_std = alpha_mobilita$total$std.alpha), "pca_mobilita_alpha")

pca_mob <- prcomp(mobilita, scale. = TRUE)
var_mob <- data.frame(
  componente = paste0("PC", 1:4),
  autovalore = pca_mob$sdev^2,
  varianza_perc = 100 * pca_mob$sdev^2 / sum(pca_mob$sdev^2))
var_mob$cumulata_perc <- cumsum(var_mob$varianza_perc)
salva(var_mob, "pca_mobilita_varianza")
salva(as.data.frame(pca_mob$rotation), "pca_mobilita_loadings")

salva_grafico(fviz_eig(pca_mob, addlabels = TRUE) + ggplot2::ggtitle("PCA mobilità: varianza spiegata"), "pca_mobilita_scree")
salva_grafico(fviz_pca_var(pca_mob, repel = TRUE) + ggplot2::ggtitle("PCA mobilità: variabili"), "pca_mobilita_variabili")

cat("PCA mobilità - varianza spiegata PC1+PC2:", round(var_mob$cumulata_perc[2], 2), "%\n")
