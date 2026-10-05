# Clustering dei possessori di BEV: K-Mode con klaR (segmentazione finale della tesi),
# verificato con K-Means e gerarchico (Ward).
# Usa dati/elaborati/dataset_cluster.csv, creato da: python scripts/03_esplorativa.py
source(file.path("R", "00_comune.R"))

dati <- leggi("dataset_cluster.csv")
variabili <- c("Età", "Dimensione.familiare", "Stile.di.vita",
               "Auto.precedente..HEV", "Auto.precedente..GNC", "Auto.precedente..BEV",
               "Auto.precedente..PHEV", "Auto.precedente..ICE", "Mobilità.quotidiana.PCA", "Viaggi.PCA")
X <- scale(dati[, variabili])

# Scelta del numero di cluster: gomito e silhouette (silhouette su un campione per velocità)
campione <- sample(nrow(X), 3000)
distanze_campione <- dist(X[campione, ])
scelta_k <- data.frame(k = 2:8, wss = NA, silhouette = NA)
for (i in seq_len(nrow(scelta_k))) {
  km <- kmeans(X, centers = scelta_k$k[i], nstart = 25)
  scelta_k$wss[i] <- km$tot.withinss
  scelta_k$silhouette[i] <- mean(silhouette(km$cluster[campione], distanze_campione)[, 3])
}
salva(scelta_k, "kmeans_scelta_k")

km <- kmeans(X, centers = 4, nstart = 25)
sil <- mean(silhouette(km$cluster[campione], distanze_campione)[, 3])
salva(data.frame(wss = km$tot.withinss, silhouette = sil, dimensioni = paste(table(km$cluster), collapse = " / ")), "kmeans_k4")
salva(as.data.frame(km$centers), "kmeans_k4_centri")
salva(as.data.frame.matrix(prop.table(table(km$cluster, dati$Tipologia.di.auto.precedente), 1) * 100),
      "kmeans_k4_auto_precedente")

salva_grafico(fviz_cluster(km, data = X, geom = "point", ellipse.type = "convex", ggtheme = ggplot2::theme_minimal()) +
                ggplot2::ggtitle("K-Means, k = 4"), "kmeans_k4")

# Gerarchico (Ward) su un campione: su 5.800 osservazioni il dendrogramma non è leggibile
hc <- hclust(dist(X[campione, ]), method = "ward.D2")
cofenetica <- cor(dist(X[campione, ]), cophenetic(hc))
salva(data.frame(correlazione_cofenetica = cofenetica, dimensioni = paste(table(cutree(hc, k = 4)), collapse = " / ")),
      "gerarchico_ward")
salva_grafico(fviz_dend(hc, k = 4, show_labels = FALSE, rect = TRUE) + ggplot2::ggtitle("Gerarchico (Ward), k = 4"),
              "gerarchico_dendrogramma")

# K-Mode sulle variabili categoriali (segmentazione finale della tesi, pacchetto klaR)
variabili_kmode <- make.names(c(
  "Genere", "Classe Reddito Familiare", "Livello di istruzione", "Casa di proprietà", "Casa Indipendente",
  "Tipologia di auto precedente", "Numero persone in famiglia categoriale", "Numero di auto in famiglia categoriale",
  "VMT categoriale", "Viaggio lungo categoriale", "Numero viaggi lunghi categoriale",
  "Distanza casa-lavoro categoriale", "Sensibilità ambientale categoriale", "Classe età raggruppata"))
dati_kmode <- as.data.frame(lapply(dati[, variabili_kmode], as.character))
# Molte partizioni hanno costo quasi uguale: si tiene la migliore su 20 avvii
migliore <- NULL
for (avvio in 1:20) {
  set.seed(avvio)
  prova <- klaR::kmodes(dati_kmode, modes = 4, iter.max = 20)
  if (is.null(migliore) || sum(prova$withindiff) < sum(migliore$withindiff)) migliore <- prova
}
ordine <- order(-migliore$size)
salva(data.frame(costo = sum(migliore$withindiff), dimensioni = paste(migliore$size[ordine], collapse = " / ")), "kmode_k4")
salva(migliore$modes[ordine, ], "kmode_k4_modalita")

cat("K-Mode k=4: costo =", sum(migliore$withindiff), "- dimensioni =", paste(migliore$size[ordine], collapse = " / "), "\n")
cat("K-Means k=4: WSS =", round(km$tot.withinss, 1), "- silhouette =", round(sil, 3), "\n")
