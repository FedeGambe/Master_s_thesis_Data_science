library(cluster)
library(ggplot2)      
library(dplyr)
library(corrplot)
library(corrr)
library(FactoMineR)
library(factoextra)
library(klaR)






set.seed(42)

# Caricare il dataset
data <- read.csv("~/Desktop/EDA/data_per_cluster.csv")
data$Casa.di.proprietà.num <- data$Casa.di.proprietà
data$Casa.di.proprietà.num <- replace(data$Casa.di.proprietà.num, data$Casa.di.proprietà.num == "Si", 1)
data$Casa.di.proprietà.num <- replace(data$Casa.di.proprietà.num, data$Casa.di.proprietà.num == "No", 0)

data$Casa.Indipendente.num <- data$Casa.Indipendente
data$Casa.Indipendente.num <- replace(data$Casa.Indipendente.num, data$Casa.Indipendente.num == "Si", 1)
data$Casa.Indipendente.num <- replace(data$Casa.Indipendente.num, data$Casa.Indipendente.num == "No", 0)
# Convert character columns to numeric
data$Casa.di.proprietà.num <- as.numeric(data$Casa.di.proprietà.num)
data$Casa.Indipendente.num <- as.numeric(data$Casa.Indipendente.num)

str(data)

# Datatsst originale
data_org <- data %>% 
  select( Genere_dummy,Età, Reddito.familiare, Livello.di.istruzione.num, Casa.di.proprietà.num, Casa.Indipendente.num, Numero.persone.in.famiglia, 
          Numero.di.auto.in.famiglia, Importanza.di.ridurre.le.emissioni.di.gas.serra, Auto.precedente.HEV, Auto.precedente.GNC, Auto.precedente.BEV,
          Auto.precedente.PHEV, Auto.precedente.ICE, Viaggio.più.lungo.negli.ultimi.12.mesi, 
          Numero.di.viaggi.superiori.a.200.miglia.negli.ultimi.12.mesi, Distanza.casa.lavoro, VMT.annuo
  )
data_org_scaled <- scale(data_org)

#Dataset Cluster 1: rimpiazzate le var abitative
data_clus1 <- data %>% 
  select( Genere_dummy,Età, Reddito.familiare, Livello.di.istruzione.num, Dimensione.familiare, Stile.di.Vita, 
          Importanza.di.ridurre.le.emissioni.di.gas.serra, Auto.precedente.HEV, Auto.precedente.GNC, Auto.precedente.BEV,
          Auto.precedente.PHEV, Auto.precedente.ICE, Viaggio.più.lungo.negli.ultimi.12.mesi, 
          Numero.di.viaggi.superiori.a.200.miglia.negli.ultimi.12.mesi, Distanza.casa.lavoro, VMT.annuo
  )
data_clus1_scaled <- scale(data_clus1) # Scala le variabili

#Dataset cluster 2: rimpiazzare le var di mobilità, tolte il livello di istruzione e reddito 
data_clus2 <- data %>% 
  select( Genere_dummy,Età, Dimensione.familiare, Stile.di.Vita, 
          Importanza.di.ridurre.le.emissioni.di.gas.serra, Auto.precedente.HEV, Auto.precedente.GNC, Auto.precedente.BEV,
          Auto.precedente.PHEV, Auto.precedente.ICE, Mobilità.quotidiana.pca, Viaggi.pca, status.socioeconomico, distanza.sostenibilità
  )
data_clus2_scaled <- scale(data_clus2)

#Dataset cluster 3: tolto il genere
data_clus3 <- data %>% 
  select( Età, Dimensione.familiare, Stile.di.Vita, 
          Auto.precedente.HEV, Auto.precedente.GNC, Auto.precedente.BEV,
          Auto.precedente.PHEV, Auto.precedente.ICE, Mobilità.quotidiana.pca, Viaggi.pca
  )
data_clus3_scaled <- scale(data_clus3)

######################################################
#KMODE
##########################################
par(mfrow = c(1, 2)) 
wss_plot <- function(data, max_k) {
  wss <- numeric(max_k)
  for (k in 1:max_k) {
    clustering <- kmodes(data, modes=k)
    wss[k] <- sum(clustering$withinss)
  }
  return(wss)
}
# Genera il grafico del WSS per k da 1 a 10
wss_values <- wss_plot(data_org, 10)
plot(1:10, wss_values, type="b", pch=19, col="blue", xlab="Numero di cluster", ylab="WSS",
     main="Grafico WSS per K-mode")

######################################################
#KMEANS
##########################################

#PLOTS 10 Cluster
par(mfrow = c(4, 2)) 
#data_org
wss <- (nrow(data_org_scaled)-1)*sum(apply(data_org_scaled, 2, var))
for (i in 2:10) wss[i] <- sum(kmeans(data_org_scaled, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(10)
for (i in 2:10) {
  km <- kmeans(data_org_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_org_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:10, avg_sil[2:10], type = "b",xlab = "Numero di cluster",ylab = "Silhouette media",main = "Silhouette")
mtext("Main Title for the Entire Plot Layout", side = 3, line = 0, outer = TRUE, cex = 1.5)
#clus1
wss <- (nrow(data_clus1_scaled)-1)*sum(apply(data_clus1_scaled, 2, var))
for (i in 2:10) wss[i] <- sum(kmeans(data_clus1_scaled, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(10)
for (i in 2:10) {
  km <- kmeans(data_clus1_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_clus1_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:10, avg_sil[2:10], type = "b",xlab = "Numero di cluster",ylab = "Silhouette media",main = "Silhouette")
mtext("Main Title for the Entire Plot Layout", side = 3, line = 0, outer = TRUE, cex = 1.5)
#clus2
wss <- (nrow(data_clus2_scaled)-1)*sum(apply(data_clus2_scaled, 2, var))
for (i in 2:10) wss[i] <- sum(kmeans(data_clus2_scaled, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(10)
for (i in 2:10) {
  km <- kmeans(data_clus2_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_clus2_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:10, avg_sil[2:10], type = "b",xlab = "Numero di cluster",ylab = "Silhouette media",main = "Silhouette")
#clus3
wss <- (nrow(data_clus3_scaled)-1)*sum(apply(data_clus3_scaled, 2, var))
for (i in 2:10) wss[i] <- sum(kmeans(data_clus3_scaled, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(10)
for (i in 2:10) {
  km <- kmeans(data_clus3_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_clus3_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:10, avg_sil[2:10], type = "b",xlab = "Numero di cluster",ylab = "Silhouette media",main = "Silhouette")



#Cluster Orig ________________________________________________________________________________________________
##Grafici con 30 cluster
par(mfrow = c(2, 2)) 
wss <- (nrow(data_org_scaled)-1)*sum(apply(data_org_scaled, 2, var))
for (i in 2:30) wss[i] <- sum(kmeans(data_org_scaled, centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(30)
for (i in 2:30) {
  km <- kmeans(data_org_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_org_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:30, avg_sil[2:30], type = "b",
     xlab = "Numero di cluster",
     ylab = "Silhouette media",
     main = "Silhouette")

##Grafici con 10 cluster
#par(mfrow = c(1, 2)) 
wss <- (nrow(data_org_scaled)-1)*sum(apply(data_org_scaled, 2, var))
for (i in 2:10) wss[i] <- sum(kmeans(data_org_scaled, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(10)
for (i in 2:10) {
  km <- kmeans(data_org_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_org_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:10, avg_sil[2:10], type = "b",xlab = "Numero di cluster",ylab = "Silhouette media",main = "Silhouette")


#Cluster 1  ________________________________________________________________________________________________
##Grafici con 30 cluster
par(mfrow = c(2, 2)) 
# Determina il numero ottimale di cluster (Elbow method)
wss <- (nrow(data_clus1_scaled)-1)*sum(apply(data_clus1_scaled, 2, var))
for (i in 2:30) wss[i] <- sum(kmeans(data_clus1_scaled, centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")

## Calcola la somma dei quadrati intra-cluster per ogni numero di cluster
#wss <- numeric(30)
#for (i in 1:30) {km <- kmeans(data_bev_scaled, centers = i, nstart = 25)wss[i] <- km$tot.withinss}
## Grafico Elbow
#plot(1:30, wss, type = "b", pch = 19, frame = FALSE,xlab = "Numero di cluster",ylab = "Somma dei quadrati intra-cluster",main = "Elbow")

# Calcola e plotta la silhouette per vari numeri di cluster
avg_sil <- numeric(30)
for (i in 2:30) {
  km <- kmeans(data_clus1_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_clus1_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
# Grafico della silhouette media per numero di cluster
plot(2:30, avg_sil[2:30], type = "b",
     xlab = "Numero di cluster",
     ylab = "Silhouette media",
     main = "Silhouette")

##Grafici con 10 cluster
#par(mfrow = c(1, 2)) 
wss <- (nrow(data_clus1_scaled)-1)*sum(apply(data_clus1_scaled, 2, var))
for (i in 2:10) wss[i] <- sum(kmeans(data_clus1_scaled, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(10)
for (i in 2:10) {
  km <- kmeans(data_clus1_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_clus1_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:10, avg_sil[2:10], type = "b",xlab = "Numero di cluster",ylab = "Silhouette media",main = "Silhouette")



#Cluster 2 ________________________________________________________________________________________________
##Grafici con 30 cluster
par(mfrow = c(2, 2)) 
wss <- (nrow(data_clus2_scaled)-1)*sum(apply(data_clus2_scaled, 2, var))
for (i in 2:30) wss[i] <- sum(kmeans(data_clus2_scaled, centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(30)
for (i in 2:30) {
  km <- kmeans(data_clus2_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_clus2_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:30, avg_sil[2:30], type = "b",
     xlab = "Numero di cluster",
     ylab = "Silhouette media",
     main = "Silhouette")

##Grafici con 10 cluster
#par(mfrow = c(1, 2)) 
wss <- (nrow(data_clus2_scaled)-1)*sum(apply(data_clus2_scaled, 2, var))
for (i in 2:10) wss[i] <- sum(kmeans(data_clus2_scaled, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(10)
for (i in 2:10) {
  km <- kmeans(data_clus2_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_clus2_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:10, avg_sil[2:10], type = "b",xlab = "Numero di cluster",ylab = "Silhouette media",main = "Silhouette")

#Cluster 3 ________________________________________________________________________________________________
##Grafici con 30 cluster
par(mfrow = c(2, 2)) 
wss <- (nrow(data_clus3_scaled)-1)*sum(apply(data_clus3_scaled, 2, var))
for (i in 2:30) wss[i] <- sum(kmeans(data_clus3_scaled, centers=i)$withinss)
plot(1:30, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(30)
for (i in 2:30) {
  km <- kmeans(data_clus3_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_clus3_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:30, avg_sil[2:30], type = "b",
     xlab = "Numero di cluster",
     ylab = "Silhouette media",
     main = "Silhouette")

##Grafici con 10 cluster
#par(mfrow = c(1, 2)) 
wss <- (nrow(data_clus3_scaled)-1)*sum(apply(data_clus3_scaled, 2, var))
for (i in 2:10) wss[i] <- sum(kmeans(data_clus3_scaled, centers=i)$withinss)
plot(1:10, wss, type="b", xlab="Numero di cluster", ylab="Somma dei quadrati intra-cluster", main = "Elbow")
avg_sil <- numeric(10)
for (i in 2:10) {
  km <- kmeans(data_clus3_scaled, centers = i, nstart = 25)
  sil <- silhouette(km$cluster, dist(data_clus3_scaled))
  avg_sil[i] <- mean(sil[, 3])
}
plot(2:10, avg_sil[2:10], type = "b",xlab = "Numero di cluster",ylab = "Silhouette media",main = "Silhouette")
mtext("Elbow e Silhouette -> Data V4", outer = TRUE, cex = 1.5)



#CLUSTERING
# Esegui k-means clustering (scegliendo il numero di cluster ottimale)
set.seed(123) # per riproducibilità
km <- kmeans(data_clus3_scaled, centers=4, nstart = 20) # modifica 'centers' con il numero scelto

fviz_cluster(km, data = data_clus3_scaled,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
             ellipse.type = "euclid",  # Concentration ellipse
             star.plot = TRUE,  # Add segments from centroids to items
             repel = TRUE,  # Avoid label overplotting (slow)
             ggtheme = theme_minimal(),
             geom = "point",  # Plot only the points (no labels)
             labelsize = 0  # Remove labels
)
km$tot.withinss #->34629.87
sil_result <- silhouette(km$cluster, dist(data_clus3_scaled))
avg_sil_width <- mean(sil_result[, 3])
cat("Average Silhouette Width for 4 clusters:", avg_sil_width, "\n") #0.3953377

# Aggiungi i cluster al dataset: Data BEV
data_clus3_scaled$Cluster <- km$cluster

# Aggiungi i cluster al dataset: Data Ricercatore
data$Cluster <- km$cluster


# Visualizza i risultati
print(km$centers)
table(data$Cluster)


# Esegui l'analisi FAMD su cui prioettare i cluster
data_or <- read.csv("~/Desktop/EDA/data_eda.csv") 
data_famd <- data_or %>%
  select( -Reddito.familiare, -Numero.persone.in.famiglia..categoriale, -Numero.persone.in.famiglia.categoriale, 
          -Numero.di.auto.in.famiglia.categoriale, -VMT.categoriale,-Viaggio.lungo.categoriale, 
          -Numero.viaggi.lunghi.all.anno.categoriale, -Distanza.casa.lavoro.categoriale, 
          -Sensibilità.ambientale.categoriale, -X, -Auto.attuale, -Tipologia.di.auto.attuale)
data_bev1 <- subset(data_famd, BEV.dummy == 'Si')
res.famd <- FAMD(data_bev1, ncp = 10, graph = FALSE)
coord <- res.famd$ind$coord[, 1:2]

fviz_cluster(list(data = res.famd$ind$coord[, 1:2], cluster = km$cluster),
             geom = "point",          # Plot points
             ellipse.type = "convex", # Add convex hulls for clusters
             main = "FAMD con K-means Clustering")



fviz_cluster(km, data = data_clus3_scaled,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
             ellipse.type = "euclid", # Concentration ellipse
             star.plot = TRUE, # Add segments from centroids to items
             repel = TRUE, # Avoid label overplotting (slow)
             ggtheme = theme_minimal()
)

######################################################
#GERARCHICO 
##########################################
data_dummy <- data_ric %>% 
  select(
    -`Genere_`, -`Età`, -`Classe.d.età`, -`Reddito.familiare`, 
    -`Classe.Reddito.Familiare`, -`Livello.di.istruzione`, 
    -`Auto`, -`Tipologia`, -`X`
  )
data_bev <- subset(data_dummy, BEV.dummy == 1)
data_bev$BEV.dummy <- NULL
data_bev_scaled <- scale(data_bev)

res.dist <- dist(data_bev_scaled, method = "euclidean")
res.hc <- hclust(d = res.dist, method = "ward.D2")
fviz_dend(res.hc, cex = 0.5)
res.coph <- cophenetic(res.hc)
cor(res.dist, res.coph)

# Cut tree into 4 groups
grp <- cutree(res.hc, k = 4)
#head(grp, n = 4)
table(grp)

fviz_dend(res.hc, k = 4, # Cut in four groups
          cex = 0.5, # label size
          k_colors = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
          color_labels_by_k = TRUE, # color labels by groups
          rect = TRUE # Add rectangle around groups
)

fviz_cluster(list(data = data_bev_scaled, cluster = grp),
             palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07"),
             ellipse.type = "convex", # Concentration ellipse
             repel = TRUE, # Avoid label overplotting (slow)
             show.clust.cent = FALSE, ggtheme = theme_minimal())
        

##Hierarchical clustering con diversi linkage methods
hc.complete <- hclust(dist(data_bev_scaled), method = "complete")
cor(res.dist, cophenetic(hc.complete))
hc.average <- hclust(dist(data_bev_scaled), method = "average")
cor(res.dist, cophenetic(hc.average))
hc.single <- hclust(dist(data_bev_scaled), method = "single")
cor(res.dist, cophenetic(hc.single))

par(mfrow = c(1, 3))  ##Plottiamo i grafici
plot(hc.complete, main = "Complete Linkage", xlab = "", sub = "", cex = .9)
plot(hc.average , main = "Average Linkage", xlab = "", sub = "", cex = .9)
plot(hc.single, main = "Single Linkage", xlab = "", sub = "", cex = .9)
