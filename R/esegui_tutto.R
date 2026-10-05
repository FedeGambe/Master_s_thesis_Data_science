# Esegue tutte le analisi R in sequenza. Dalla radice del repository:  Rscript R/esegui_tutto.R
for (script in c("01_pca.R", "02_mca.R", "03_famd.R", "04_cluster.R")) {
  cat("\n==>", script, "\n")
  source(file.path("R", script), local = new.env())
}
