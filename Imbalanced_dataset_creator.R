
#After using the code of M.Toby Hocking to create imbalance for binary
# Use this to create imbalance only on one label
library(data.table)
unb_path="~/data_Classif_unbalanced/FashionMNIST.csv"
MNIST_unb <- fread(unb_path)
MNIST <-fread("~/data_Classif/FashionMNIST.csv")
for(seed in 1:5){
  seed_cols <- grep(paste0("seed", seed), names(MNIST_unb), value = TRUE)
  label_set <- 2 * ((1:seed) - 1)
  MNIST_unb[, (seed_cols) := lapply(.SD, function(col) {
    idx <- which(col == "" & !(MNIST$y %in% label_set))
    col[idx] <- "unbalanced"
    col
  }), .SDcols = seed_cols]
}

#Verify distribution of data 
subset_label_counts <- list()
for (col_name in names(MNIST_unb)) {
  for (subset_type in c("balanced", "unbalanced")) {
    row_idx <- which(MNIST_unb[[col_name]] == subset_type)
    label_counts <- MNIST[row_idx, .N, by = y]
    label_row <- dcast(label_counts, . ~ y, value.var = "N", fill = 0)[, . := NULL]
    label_row[, subset := paste0(col_name, "_", subset_type)]
    subset_label_counts[[length(subset_label_counts) + 1]] <- label_row
  }
}
result_dt <- rbindlist(subset_label_counts, fill = TRUE)
setcolorder(result_dt, c("subset", setdiff(names(result_dt), "subset")))
#Save the new unbalanced dataset
fwrite(MNIST_unb, unb_path)
