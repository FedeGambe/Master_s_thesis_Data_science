library(FactoMineR)
library(factoextra)
library(dplyr)    # alternatively, this also loads %>%

data_or <- read.csv("~/Desktop/EDA/data_eda_2.csv") 
colnames(data_or)


data <- data_or %>%
  select( -Reddito.familiare, -Numero.persone.in.famiglia..categoriale, -Numero.persone.in.famiglia.categoriale, 
          -Numero.di.auto.in.famiglia.categoriale, -VMT.categoriale,-Viaggio.lungo.categoriale, 
          -Numero.viaggi.lunghi.all.anno.categoriale, -Distanza.casa.lavoro.categoriale, 
          -Sensibilità.ambientale.categoriale, -X, -Auto.attuale, -Tipologia.di.auto.attuale)

str(data)

#FAMD
#trasfomrate in fattori
#data <- data %>%mutate_if(is.character, as.factor)
data_bev1 <- subset(data, BEV.dummy == 'Si')

#Eingvalues
#eig.val <- get_eigenvalue(res.famd)
#head(eig.val)
#fviz_screeplot(res.famd)
#fviz_eig(res.famd, geom="line", addlabels=T)

##Varinza : focus sui primi 10
res.famd <- FAMD(data_bev1, ncp = 10, graph = FALSE)
print(res.famd)
eig.val <- res.famd$eig  # Accedi direttamente ai valori propri
eig.val <- as.data.frame(eig.val) # Se eig.val è una matrice, converti in data frame
colnames(eig.val) <- c("eigenvalue", "variance.percent", "cumulative.variance.percent")
par(mfrow = c(1, 2))
plot(eig.val$variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza  Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 20, col = "red", lty = 2)
plot(eig.val$cumulative.variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza Accumulata Spiegata (%)", 
     pch = 19, col = "black")


##Varinza totale
res.famd1 <- FAMD(data_bev1, ncp = 29, graph = FALSE)
print(res.famd1)
eig.val1 <- res.famd1$eig  # Accedi direttamente ai valori propri
eig.val1 <- as.data.frame(eig.val1) # Se eig.val è una matrice, converti in data frame
colnames(eig.val1) <- c("eigenvalue", "variance.percent", "cumulative.variance.percent")
par(mfrow = c(1, 2))
plot(eig.val1$variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza  Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 20, col = "red", lty = 2)
plot(eig.val1$cumulative.variance.percent, type = "b", 
     xlab = "Dimensioni", ylab = "Varianza Accumulata Spiegata (%)", 
     pch = 19, col = "black")
abline(h = 70, col = "red", lty = 1)
abline(h = 80, col = "red", lty = 2)
abline(h = 90, col = "red", lty = 1)

fviz_eig(res.famd, geom="line", addlabels=T)
print (eig.val1)

#GRAPH
var <- get_famd_var(res.famd)
var

# Coordinates of variables
head(var$coord)
# Cos2: quality of representation on the factore map
head(var$cos2)
# Contributions to the dimensions
head(var$contrib)

# Plot of variables
fviz_famd_var(res.famd, repel = TRUE)
# Contribution to the first dimension
fviz_contrib(res.famd,"var", axes = 1)
# Contribution to the second dimension
fviz_contrib(res.famd,"var", axes = 2)


quanti.var <- get_famd_var(res.famd, "quanti.var")
quanti.var
fviz_famd_var (res.famd, "quanti.var", repel = TRUE, col.var = "black")  
fviz_famd_var(res.famd, "quanti.var" , col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
              repel = TRUE)

quali.var <- get_famd_var(res.famd, "quali.var")
quali.var
fviz_famd_var (res.famd, "quali.var", repel = TRUE, col.var = "black")  
fviz_famd_var(res.famd, "quali.var" , col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
              repel = TRUE)

ind <- get_famd_ind(res.famd)
ind

fviz_famd_ind (res.famd, col.ind = "cos2", 
               gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
               repel = TRUE)

fviz_mfa_ind(res.famd, habillage = "Label", 
             palette = c("#00AFBB" , , "#FC4E07" ), 
             addEllipses = TRUE, 
             ellipse.type = "confidence", 
             repel = TRUE # Avoid text overlapping
)

fviz_ellipses(res.famd, c("Label","XXXX"), repel = TRUE)

fviz_famd_var(res.famd, repel = TRUE) + 
  ggtitle("FAMD - Mappa delle Variabili")














#####################################
get_eigenvalue(famd_result)
fviz_eig(famd_result)

## FAMD
res.famd <- FAMD(data, 
                 sup.var = 20,  ## Set the target variable "Churn" as a supplementary variable
                 graph = FALSE, 
                 ncp=25)

## Inspect principal components
get_eigenvalue(res.famd)
fviz_eig(res.famd)
## Set figure size
options(repr.plot.width = 14, repr.plot.height = 12)

## Plot individual observations
fviz_famd_ind(res.famd, label = "none", 
              habillage = "BEV.dummy", palette = c("#00AFBB", "#FC4E07"), # color by groups 
              repel = TRUE, alpha.ind = 0.5) + 
  xlim(-5, 6) + ylim (-4.5, 4.5) +
  theme(text = element_text(size=20), axis.text.x = element_text(size=20), axis.text.y = element_text(size=20))

fviz_famd_ind(res.famd, repel = TRUE) + 
  ggtitle("FAMD - Mappa delle Variabili")

#MFA
# Supponiamo che le variabili numeriche siano nei blocchi numerici e quelle categoriche nei blocchi categorici

ncol(data)
numeriche <- data[, c(7,8,9,12,13,14,15)]   # Se le prime due colonne sono numeriche
categoriche <- data[, c(1,2,3,4,5,6,10,11)]  # Se le successive tre sono categoriche
data_separato <- cbind(numeriche, categoriche)

res.mfa <- MFA(data_separato, 
              group = c(7, 8),        # Numero di variabili per ogni blocco
              type = c("n", "c"),     # 'n' per numeriche e 'c' per categoriche
              name.group = c("numeriche", "categoriche"), 
              ncp = 15)                # Numero di componenti principali da considerare

# Supponiamo che le variabili categoriche siano nelle colonne 1:7 e numeriche nelle colonne 8:16
# Converti le variabili categoriche in fattori
data_separato[, 1:7] <- lapply(data_separato[, 1:7], factor)

# Verifica che le variabili siano state correttamente convertite in fattori
sapply(data_separato[, 1:7], class)  # Dovresti vedere "factor" per queste colonne

res.mfa <- MFA(data_separato, 
              group = c(7, 8),        # 7 variabili numeriche e 9 variabili categoriche
              type = c("n", "c"),     # 'n' per numeriche, 'c' per categoriche
              name.group = c("Numeriche", "Categoriche"), 
              ncp = 15)                # Numero di componenti principali da considerare


# Controlla i tipi delle colonne
sapply(data_separato, class)

print(res.mfa)

library("factoextra") 
eig.val <- get_eigenvalue(res.mfa) 
head(eig.val)
fviz_screeplot(res.mfa)
