library(ggplot2)      
library(dplyr)
library(corrplot)
library(corrr)
library(DT)

data <- read.csv("~/Desktop/EDA/data_eda_2.csv") 
colnames(data)

data_pca <- data %>%
  select( Reddito.familiare, Numero.persone.in.famiglia, Numero.di.auto.in.famiglia,
          Importanza.di.ridurre.le.emissioni.di.gas.serra, Viaggio.più.lungo.negli.ultimi.12.mesi, Numero.di.viaggi.superiori.a.200.miglia.negli.ultimi.12.mesi,
          Distanza.casa.lavoro, VMT.annuo)
##Metodo 1

apply(data_pca , 2, mean)  ##Apply ci serve per stimare la media (i.e. applicare la funzione "mean") alle colonne del dataset, 
#che sono indicate dal parametro "2" - "1" invece sta per le righe -> 2: colonne , 1: righe
apply(data_pca , 2, var) ##Qua stimiamo la varianza
colnames(data_pca)
# Abbrevia i nomi delle colonne nel dataset
colnames(data_pca) <- abbreviate(colnames(data_pca), minlength = 10)  # Puoi cambiare minlength per abbreviazioni più brevi
# Calcola e visualizza la matrice di correlazione
corrplot(cor(data_pca), order = "hclust")

##Stimiamo le componenti principali
#prcomp: principal component
pr.out <- prcomp(data_pca , scale = TRUE)  ##Le variabili vengono standardizzate con "scale" = TRUE) 
names(pr.out) #names estrae tutto ciò che c'è all'interno di un'oggetto
#all'interno di pr.out ci sono: standard deviation, rotation, center, scale e X -> a noi interessa rotation

pr.out$rotation  ##In "rotation" troviamo i principal component loadings -> si usa il dollaro per richiamare ciò che c'è all'interno
##Plotting delle prime due componenti principali, ci sono 3 plot di grandezza diversa. Il primo non ci sono limiti alla sua grandezza, il seconda si concentra solo sui valori -1 e +1, il terzo su -2 e +2
#biplot è un doppio plot, con asse cartesiano e i vettori, anche se non è utile per la visualzzazione, perché i vettori non servono sl "cliente"
biplot(pr.out, scale = 0)
biplot(pr.out, xlim=c(-1,1), ylim=c(-1,1), scale = 0) #x e y lim è il limite
biplot(pr.out, xlim=c(-2,2), ylim=c(-2,2), scale = 0)
biplot(pr.out, xlim=c(-3,-2), ylim=c(-3,2), scale = 0)

##Cosa succede se cambiamo i segni? 
#Es. in questa analisi abbiamo trovato PCI negativi, quindi sul grafico ad un valore negativo corrisponde un valore positivo
#questo accade poi anche nella regressione. Quindi si tende a cambiare segno, per dire che alti valori di PCI 1 corrisondono altì reati
#Per evitare infraentendimenti 
pr.out$rotation = -pr.out$rotation #qui li cambiamo il segno
pr.out$x = -pr.out$x
biplot(pr.out , scale = 0)

##Varianza spiegata da ogni componente
# con al dev std possiamo calcolare la varianza
pr.out$sdev  ##Qui stiamo richiamando la dev. std. delle componenti principali
pr.var <- pr.out$sdev^2 #la var è il quadrato della dev std
pr.var
#ora si calcola la var cumulata in cui si rapporta la var sulla somma dalla varianza
pve <- pr.var / sum(pr.var)  ##In proporzione, quanto ogni PC contribuisce a spiegare la varianza totale
pve
#la prima PCI spiega il 62% e la seconda 24,74% -> ed è già tantissimo
##Disegniamo i grafici
par(mfrow = c(1, 2)) #per settare il plot, in questo caso 2
plot(pve , xlab = "Principal Component", ylab = "Proportion of Variance Explained", ylim = c(0, 1), type = "b")
plot(cumsum(pve), xlab = "Principal Component", ylab = "Cumulative Proportion of Variance Explained", ylim = c(0, 1), type = "b")

#Metodo 2

##BEST SUBSET METHOD

library(leaps) #dentro leaps c'è best subset
regfit.full <- regsubsets(BEV.dummy ~ ., data)   ##La funzione "regsubsets" (nella library leaps) ci fa la best subset selection
##Il codice "Salary ~ ." vuol dire che il salario é in funzione di una serie di regressori: tilde, punto -> indica che prende tutti i regressori così da capire quali sono i migliori
summary(regfit.full)
#Outpu: -> algoritmo di soluzione
#di default arriva a 8 iterazioni, ovvero 8 variabili
##Le variabili che mostrano degli asterischi (*) sono quelle che vengono incluse nel modello, il modello inizia con 1 regressore, poi con 2,... 
##Di default "regsubsets" ci restituisce al massimo 8 regressori, ma con "nvmax" possiamo aumentare

#best subset non è molto stabile -> PROBLEMA, es. il modello con 7 variabili CRBI esce. Per questo calcoliamo il Best con tutte le variabili meno la dip, in questo caso 19

regfit.full <- regsubsets(Salary ~ ., data = Hitters ,nvmax = 19)
reg.summary <- summary(regfit.full)
reg.summary
names(reg.summary)   ##La funzione "summary" vi restituisce R2, RSS, R2 corretto, Cp, and BIC -> serve per vedere che informazioni ci sono

reg.summary$rsq  ## con il dollaro si richiama tutto il rsq dal sumary ->Guardiamo come evolve R2 se aumentiamo i regressori
reg.summary$adjr2 #vediao che co dopo 11 regressori r2adj cala 
##Plottiamo i risultati di RSS e R2 corretto
##Se inseriamo il parametro type = "l" stiamo dicendo a R di usare le linee per il plotting

#_____________Grafici di Rss, R2 adj, Cp e Bic
par(mfrow = c(2, 2)) #impostare i grafici
plot(reg.summary$rss , xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(reg.summary$adjr2 , xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")

which.max(reg.summary$adjr2)   ##La funzione "which.max" si usa per trovare il punto di massimo -> il punto di massimo è 11
points(11, reg.summary$adjr2[11], col = "red", cex = 2, pch = 20) #-> per mettere un punto sul grafico 
##Possiamo fare la stessa cosa anche con Cp e BIC, ma stavolta cerchiamo il punto di minimo con "which.min"
plot(reg.summary$cp, xlab = "Number of Variables", ylab = "Cp", type = "l")
which.min(reg.summary$cp)
points(10, reg.summary$cp[10], col = "red", cex = 2, pch = 20)
which.min(reg.summary$bic)
plot(reg.summary$bic , xlab = "Number of Variables", ylab = "BIC", type = "l")
points(6, reg.summary$bic[6], col = "red", cex = 2,  pch = 20)
#Output: ci dice che il numero di regressori milgiori sono tra 6 e 11, poi tocca al ricercatore capire quanti inserirne

#_____________Grafici di quali variabili
##All'interno di regsubsets() abbiamo un comando legato a plot che ci permette di mostrare il miglior modello dato un certo numero di predittori
#dev.off fa il risetTAGGIO del grafico
dev.off()  ##Questo comando fa sí che RStudio apra una nuova graphics device, nel caso i comandi "plot" successivi non funzionino
plot(regfit.full , scale = "r2")
plot(regfit.full , scale = "adjr2") #le caselle in nero più alte sono le variabili inserite con il r2 max: es, interc, Hits, Walks, CAtBat,..
plot(regfit.full , scale = "Cp")
plot(regfit.full , scale = "bic") #come nel caso precedente, dobbiamo guardare ol var con la casella nera più alta

##Quali sono le variabili che danno un modello con il BIC piú basso?
##AtBat, Hits, Walks, CRBI, DivisionW, e PutOuts
##Vediamo i coefficienti associati con questo modello
coef(regfit.full ,6) #,6 per vedere 6 coefficenti delle var in quanto con il Bic il n migliore è 6



##FORWARD E BACKWARD STEPWISE SELECTION

##Usiamo la stessa funzione regsubsets, solamente specificando il parametro "forward" o "backword" all'interno della parentesi
regfit.fwd <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19, method = "forward")
summary(regfit.fwd)
regfit.bwd <- regsubsets(Salary ~ ., data = Hitters,nvmax = 19, method = "backward")
summary(regfit.bwd)

##Proviamo a richiamare i risultati ottenuti con i tre metodi usando 7 variabili (best subset, forward, backward): i risultati sono diversi
coef(regfit.full , 7) #bestsubset
coef(regfit.fwd , 7) #forward
coef(regfit.bwd , 7) #backward
#gli ultimi due sono diversi dal bs per via della loro metodologia, metre fw e bw differeisocno poco perché il metodo è simile

