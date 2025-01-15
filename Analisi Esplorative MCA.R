library(FactoMineR)
library(factoextra)
library(dplyr)
library(gridExtra)


# Selezionare le variabili dal dataset
data <- read.csv("~/Desktop/EDA/data_eda.csv") 
  colnames(data)
data <- data %>%
  select(-X)

data_mca <- data %>%
  select( -Reddito.familiare, -Numero.persone.in.famiglia, -Numero.di.auto.in.famiglia,
          -Importanza.di.ridurre.le.emissioni.di.gas.serra, -Viaggio.più.lungo.negli.ultimi.12.mesi, -Numero.di.viaggi.superiori.a.200.miglia.negli.ultimi.12.mesi,
          -Distanza.casa.lavoro, -VMT.annuo, -Auto.attuale) #togliere le var categoriali
data_mca <- data_mca %>%
  mutate(Classe_eta_raggruppata = case_when( #rapruppare le classi di età
    Classe.d.età %in% c("<25", "25-34") ~ " <35",
    Classe.d.età %in% c("35-44") ~ "35-44",
    Classe.d.età %in% c("45-54", "55-64") ~ "45-64",
    Classe.d.età %in% c("65-74", "75-79", ">80") ~ ">65", 
    TRUE ~ NA_character_  # Per gestire eventuali valori mancanti
  ))
data_mca_filtered <- data_mca[data_mca$BEV.dummy == "Si", ]
table(data_mca_filtered$Classe_eta_raggruppata) #verificare i risultati

data_mca_filtered <- subset(data_mca_filtered, select = -Classe.d.età) #nuovo dataset con classi d'età ragruppate
data_eda2 <- data[data$BEV.dummy == "Si", ]



####____________ VAR ABITATIVE
data_cas <- data_mca_filtered %>%
  select(Casa.di.proprietà, Casa.Indipendente, Numero.persone.in.famiglia.categoriale, Numero.di.auto.in.famiglia.categoriale)
data_cas <- data_cas %>%
  mutate(across(everything(), as.factor))
mca_result_abit <- MCA(data_cas, graph = FALSE)

#autovalori
eig.val_abit <- mca_result_abit$eig  # Accedi direttamente ai valori propri
eig.val_abit <- as.data.frame(eig.val_abit) # Se eig.val è una matrice, converti in data frame
print(eig.val_abit)
colnames(eig.val_abit) <- c("eigenvalue", "variance.percent", "cumulative.variance.percent")
par(mfrow = c(1, 2))
plot(eig.val_abit$variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza  Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 20, col = "red", lty = 2)
plot(eig.val_abit$cumulative.variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza Accumulata Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 70, col = "red", lty = 1)
abline(h = 80, col = "red", lty = 2)
abline(h = 90, col = "red", lty = 1)

#Plot
fviz_mca_var(mca_result_abit, repel = TRUE) + 
  ggtitle("MCA delle Variabili abiatative ")
mca_result_abit <- MCA(data_cas[, c("Casa.di.proprietà", "Casa.Indipendente", "Numero.persone.in.famiglia.categoriale", "Numero.di.auto.in.famiglia.categoriale")], graph = FALSE)
fviz_mca_var(mca_result_abit, 
             repel = TRUE, 
             col.var = "contrib",  
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")) +  
  ggtitle("MCA delle Variabili abitative") +
  theme(legend.position = "right")
mca_scores_abit <- mca_result_abit$ind$coord
head(mca_scores_abit)

#Contributi
plot1 <- fviz_contrib(mca_result_abit, choice = "var", axes = 1, top = 15) +
  ggtitle("Contributions: Dim1") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
plot2 <- fviz_contrib(mca_result_abit, choice = "var", axes = 2, top = 15) +
  ggtitle("Contributions: Dim2") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
grid.arrange(plot1, plot2, ncol = 2)

var_contrib_abit <- get_mca_var(mca_result_abit)$contrib
var_contrib_abit <- var_contrib_abit[, 1:2]
var_coord_abit <- get_mca_var(mca_result_abit)$coord
var_coord_abit <- var_coord_abit[, 1:2]
complete_table_abit  <- cbind(var_coord_abit, var_contrib_abit)
print (complete_table_abit)

#Aggiunta nuove dimensioni
#data_new <- data %>%
  #select(-Casa.di.proprietà, -Casa.Indipendente, -Numero.persone.in.famiglia..categoriale, -Numero.di.auto.in.famiglia..categoriale)

data_eda2 <- data_eda2 %>%
  mutate( Stile.di.Vita = mca_scores_abit[,1],  # Prima componente
         Dimensione.familiare = mca_scores_abit[,2])  # Seconda componente


####____________ VAR ABITATIVE.2
data_cas2 <- data_mca_filtered %>%
  select(Casa.di.proprietà, Casa.Indipendente, Numero.persone.in.famiglia.categoriale, Numero.di.auto.in.famiglia.categoriale, Classe_eta_raggruppata)
data_cas2 <- data_cas2 %>%
  mutate(across(everything(), as.factor))
mca_result_abit2 <- MCA(data_cas2, graph = FALSE)

#autovalori
eig.val_abit2 <- mca_result_abit2$eig  # Accedi direttamente ai valori propri
eig.val_abit2 <- as.data.frame(eig.val_abit2) # Se eig.val è una matrice, converti in data frame
print(eig.val_abit2)
colnames(eig.val_abit2) <- c("eigenvalue", "variance.percent", "cumulative.variance.percent")
par(mfrow = c(1, 2))
plot(eig.val_abit2$variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza  Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 20, col = "red", lty = 2)
plot(eig.val_abit2$cumulative.variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza Accumulata Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 70, col = "red", lty = 1)
abline(h = 80, col = "red", lty = 2)
abline(h = 90, col = "red", lty = 1)

#Plot
fviz_mca_var(mca_result_abit2, repel = TRUE) + 
  ggtitle("MCA delle Variabili abiatative ")
mca_result_abit2 <- MCA(data_cas2[, c("Casa.di.proprietà", "Casa.Indipendente", "Numero.persone.in.famiglia.categoriale", "Numero.di.auto.in.famiglia.categoriale", "Classe_eta_raggruppata")], graph = FALSE)
fviz_mca_var(mca_result_abit2, 
             repel = TRUE, 
             col.var = "contrib",  
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")) +  
  ggtitle("MCA delle Variabili abitative") +
  theme(legend.position = "right")
mca_scores2 <- mca_result_abit$ind$coord
head(mca_scores2)

#Contributi
plot1 <- fviz_contrib(mca_result_abit, choice = "var", axes = 1, top = 15) +
  ggtitle("Contributions: Dim1") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
plot2 <- fviz_contrib(mca_result_abit, choice = "var", axes = 2, top = 15) +
  ggtitle("Contributions: Dim2") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
grid.arrange(plot1, plot2, ncol = 2)

var_contrib_abit2 <- get_mca_var(mca_result_abit2)$contrib
var_contrib_abit2 <- var_contrib_abit2[, 1:2]
var_coord_abit2 <- get_mca_var(mca_result_abit2)$coord
var_coord_abit2 <- var_coord_abit2[, 1:2]
complete_table_abit2  <- cbind(var_coord_abit2, var_contrib_abit2)
print (complete_table_abit2)



####____________ VAR VIAGGI E MOBILITà
data_viag <- data_mca_filtered %>%
  select(VMT.categoriale, Viaggio.lungo.categoriale, Numero.viaggi.lunghi.all.anno.categoriale, Distanza.casa.lavoro.categoriale)
data_viag <- data_viag %>%
  mutate(across(everything(), as.factor))
mca_result_viag <- MCA(data_viag, graph = FALSE)

#autovalori
eig.val_viag <- mca_result_viag$eig  # Accedi direttamente ai valori propri
eig.val_viag <- as.data.frame(eig.val_viag) # Se eig.val è una matrice, converti in data fram
print(eig.val_viag)
colnames(eig.val_viag) <- c("eigenvalue", "variance.percent", "cumulative.variance.percent")
par(mfrow = c(1, 2))
plot(eig.val_viag$variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza  Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 20, col = "red", lty = 2)
plot(eig.val_viag$cumulative.variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza Accumulata Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 70, col = "red", lty = 1)
abline(h = 80, col = "red", lty = 2)
abline(h = 90, col = "red", lty = 1)

# Visualizzare le variabili
fviz_mca_var(mca_result_viag, repel = TRUE) + 
  ggtitle("MCA delle Variabili legate alla mobilità")
# Visualizzare sia le variabili che gli individui
mca_result_viag <- MCA(data_viag[, c("VMT.categoriale","Viaggio.lungo.categoriale", "Numero.viaggi.lunghi.all.anno.categoriale", "Distanza.casa.lavoro.categoriale")], graph = FALSE)
# Creazione del grafico delle variabili con l'aggiunta della label
fviz_mca_var(mca_result_viag, 
             repel = TRUE, 
             col.var = "cos2",  
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")) +  
  ggtitle("MCA delle Variabili legate alla mobilità") +
  theme(legend.position = "right")
# Ottenere i punteggi delle componenti (coordinate degli individui)
mca_scores_viag <- mca_result_viag$ind$coord
# Visualizzare i primi punteggi per assicurarsi che siano corretti
head(mca_scores_viag)
# Aggiungere i punteggi delle prime due componenti al dataset originale

class(mca_scores_viag)


#Contributi
plot1 <- fviz_contrib(mca_result_viag, choice = "var", axes = 1, top = 15) +
  ggtitle("Contributions: Dim1") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
plot2 <- fviz_contrib(mca_result_viag, choice = "var", axes = 2, top = 15) +
  ggtitle("Contributions: Dim2") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
grid.arrange(plot1, plot2, ncol = 2)

var_contrib_viag <- get_mca_var(mca_result_viag)$contrib
var_contrib_viag <- var_contrib_viag[, 1:2]
var_coord_viag <- get_mca_var(mca_result_viag)$coord
var_coord_viag <- var_coord_viag[, 1:2]
complete_table_viag  <- cbind(var_coord_viag, var_contrib_viag)
print (complete_table_viag)


#data_eda2 <- data[data$BEV.dummy == "Si", ]
data_eda2 <- data_eda2 %>%
  mutate(Viaggi = mca_scores_viag[,1],  # Prima componente
         Mobilità.quotidiana = mca_scores_viag[,2])  # Seconda componente




##____________ VAR DEMOGRAFICHE

data_dem <- data_mca_filtered %>%
  select(Genere, Classe_eta_raggruppata, Classe.Reddito.Familiare, Livello.di.istruzione, Sensibilità.ambientale.categoriale)
data_dem <- data_dem %>%
  mutate(across(everything(), as.factor))
mca_result_dem <- MCA(data_dem, graph = FALSE)

eig.val_dem <- mca_result_dem$eig  # Accedi direttamente ai valori propri
eig.val_dem <- as.data.frame(eig.val_dem) # Se eig.val è una matrice, converti in data frame
colnames(eig.val_dem) <- c("eigenvalue", "variance.percent", "cumulative.variance.percent")
par(mfrow = c(1, 2))
plot(eig.val_dem$variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza  Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 20, col = "red", lty = 2)
plot(eig.val_dem$cumulative.variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza Accumulata Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 70, col = "red", lty = 1)
abline(h = 80, col = "red", lty = 2)
abline(h = 90, col = "red", lty = 1)

fviz_mca_var(mca_result_dem, repel = TRUE) + 
  ggtitle("MCA delle Variabili Socio demografiche")
mca_result_dem <- MCA(data_dem[, c("Genere", "Classe_eta_raggruppata", "Classe.Reddito.Familiare", "Livello.di.istruzione", "Sensibilità.ambientale.categoriale")], graph = FALSE)
fviz_mca_var(mca_result_dem, 
             repel = TRUE, 
             col.var = "cos2",  
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")) +  
  ggtitle("MCA delle Variabili Socio demografiche") +
  theme(legend.position = "right")
mca_scores_dem <- mca_result_dem$ind$coord
head(mca_scores_dem)

class(mca_scores_dem)


# Crea i due grafici con margini aumentati
plot1 <- fviz_contrib(mca_result_dem, choice = "var", axes = 1, top = 15) +
  ggtitle("Contributions: Dim1") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
plot2 <- fviz_contrib(mca_result_dem, choice = "var", axes = 2, top = 15) +
  ggtitle("Contributions: Dim2") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
# Stampa i grafici affiancati
grid.arrange(plot1, plot2, ncol = 2)

var_contrib_dem <- get_mca_var(mca_result_dem)$contrib
var_contrib_dem <- var_contrib_dem[, 1:2]
var_coord_dem <- get_mca_var(mca_result_dem)$coord
var_coord_dem <- var_coord_dem[, 1:2]
complete_table_dem  <- cbind(var_coord_dem, var_contrib_dem)
print (complete_table_dem)

#data_eda2 <- data[data$BEV.dummy == "Si", ]
data_eda2 <- data_eda2 %>%
  mutate(status.socioeconomico = mca_scores_viag[,1],  # Prima componente
         distanza.sostenibilità = mca_scores_viag[,2])  # Seconda componente




##MCA TOTALE
data_mca_filtered <- data_mca_filtered %>%
  mutate(across(everything(), as.factor))

#data_mca_filtered <- data_mca_filtered %>%
 # select(-Numero.persone.in.famiglia..categoriale)
# Eseguire la MCA
mca_result <- MCA(data_mca_filtered, graph = FALSE)

eig.val <- mca_result$eig  # Accedi direttamente ai valori propri
eig.val <- as.data.frame(eig.val) # Se eig.val è una matrice, converti in data frame
colnames(eig.val) <- c("eigenvalue", "variance.percent", "cumulative.variance.percent")
print(eig.val)
par(mfrow = c(1, 2))
plot(eig.val$variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza  Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 20, col = "red", lty = 2)
plot(eig.val$cumulative.variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza Accumulata Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 70, col = "red", lty = 1)
abline(h = 80, col = "red", lty = 2)
abline(h = 90, col = "red", lty = 1)

# Visualizzare le variabili
fviz_mca_var(mca_result, repel = TRUE) + 
  ggtitle("MCA delle Variabili legate alla mobilità")
# Visualizzare sia le variabili che gli individui
#mca_result_viag <- MCA(data_viag[, c("VMT.categoriale","Viaggio.lungo.categoriale", "Numero.viaggi.lunghi.all.anno.categoriale", "Distanza.casa.lavoro.categoriale")], graph = FALSE)
# Creazione del grafico delle variabili con l'aggiunta della label
fviz_mca_var(mca_result, 
             repel = TRUE, 
             col.var = "cos2",  
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")) +  
  ggtitle("MCA delle Variabili legate alla mobilità") +
  theme(legend.position = "right")

# Crea i due grafici con margini aumentati
plot1 <- fviz_contrib(mca_result, choice = "var", axes = 1, top = 15) +
  ggtitle("Contributions: Dim1") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
plot2 <- fviz_contrib(mca_result, choice = "var", axes = 2, top = 15) +
  ggtitle("Contributions: Dim2") +
  theme(plot.margin = unit(c(1, 1, 1, 1), "cm"))  # Aumenta i margini
# Stampa i grafici affiancati
grid.arrange(plot1, plot2, ncol = 2)




###PCA
data_filtered <- data[data$BEV.dummy == "Si", ]
data_pca <- data_filtered %>%
  select(Viaggio.più.lungo.negli.ultimi.12.mesi,Numero.di.viaggi.superiori.a.200.miglia.negli.ultimi.12.mesi,
         Distanza.casa.lavoro, VMT.annuo)

library(psych)
# Calculate Cronbach's alpha
alpha_result <- alpha(data_pca)
# Display the result
print(alpha_result)

correlation_matrix <- cor(data_pca)
#corrplot(cor(data_pca), order = "hclust")  ##Matrice di correlazione
print(correlation_matrix)
data_pca_scaled <- scale(data_pca)

pca_result <- prcomp(data_pca_scaled, center = TRUE, scale. = TRUE)
summary(pca_result)
plot(pca_result, main = "Scree plot")


loadings <- pca_result$rotation
print(loadings)


pca_result$sdev  ##Qui stiamo richiamando la dev. std. delle componenti principali
pr.var <- pca_result$sdev^2 #la var è il quadrato della dev std
pr.var
#ora si calcola la var cumulata in cui si rapporta la var sulla somma dalla varianza
pve <- pr.var / sum(pr.var)*100 ##In proporzione, quanto ogni PC contribuisce a spiegare la varianza totale
pve
#la prima PCI spiega il 62% e la seconda 24,74% -> ed è già tantissimo
##Disegniamo i grafici
par(mfrow = c(1, 2)) #per settare il plot, in questo caso 2
plot(pve , xlab = "Componenti Principali", ylab = "Varianza Spiegata (%)", ylim = c(0, 100), type = "b",pch = 19, col = "black")
plot(cumsum(pve), xlab = "Componenti Principali", ylab = "Varianza Accumulata Spiegata (%)", ylim = c(0, 100), type = "b",pch = 19, col = "black")
abline(h = 70, col = "red", lty = 1)
abline(h = 80, col = "red", lty = 2)
abline(h = 90, col = "red", lty = 1)


# Oppure creare un grafico a dispersione dei primi due componenti principali
par(mfrow = c(1, 3))
biplot(pca_result, scale = 0)
#biplot(pca_result, xlim=c(-7,7), ylim=c(-7,7), scale=0, col=c("white"))
biplot(pca_result, xlim=c(-1,1), ylim=c(-1,1),scale = 0, pch = NA) #biplot perché vogliamo creare un doppio plot, con le osservazioni e vettori
custom_labels <- c("Viaggio.più.lungo", "N.viaggi.lunghi", "Distanza.casa.lavoro", "VMT")
arrows(0, 0, loadings[, 1], loadings[, 2], col="red", length=0.1)
text(loadings[, 1] * 1.1, loadings[, 2] * 1.1, 
     labels = custom_labels, col = "red", cex = 1.2)  # Adjust cex for smaller text
pca_data <- data.frame(pca_result$x)
plot(pca_data$PC1, pca_data$PC2, main = "PCA: PC1 vs PC2", xlab = "PC1", ylab = "PC2")

pca_scores <- pca_result$x
pca_scores1e2 <- pca_scores [ , 1:2]
data_eda2 <-data_eda2 %>%
  mutate(Viaggi.pca = pca_scores1e2[,1],  # Prima componente
         Mobilità.quotidiana.pca = pca_scores1e2[,2])  # Seconda componente



##___________________ NUOVO DATAFRAME
write.csv(data_eda2, file = "~/Desktop/EDA/data_eda2.csv", row.names = FALSE)

#Data con var dummy x cluster
data_clus <- read.csv("~/Desktop/EDA/data_ricercatore.csv") 
data_clus <- data_clus[data_clus$BEV.dummy == "1", ]

#aggiungiamo le dimensioni 
data_clus <- data_clus %>%
  mutate( Stile.di.Vita = mca_scores_abit[,1],  # Prima componente
          Dimensione.familiare = mca_scores_abit[,2])  # Seconda componente
data_clus <- data_clus %>%
  mutate(status.socioeconomico = mca_scores_viag[,1],  # Prima componente
         distanza.sostenibilità = mca_scores_viag[,2])  # Seconda componente
data_clus <- data_clus %>%
  mutate(Viaggi = mca_scores_viag[,1],  # Prima componente
         Mobilità.quotidiana = mca_scores_viag[,2])  # Seconda componente
data_clus <-data_clus %>%
  mutate(Viaggi.pca = pca_scores1e2[,1],  # Prima componente
         Mobilità.quotidiana.pca = pca_scores1e2[,2])  # Seconda componente

write.csv(data_clus, file = "~/Desktop/EDA/data_per_cluster.csv", row.names = FALSE)

