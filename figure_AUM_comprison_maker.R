library(data.table)
library(ggplot2)
score_out <-fread("~/R-AUM_Multiclass/score_conv_grid_search.csv")
score_out_minus_bad <-score_out[-c(10,46)]
summary_dt <- score_out_minus_bad[, .(
  mean_auc_micro = mean(auc_micro),
  sd_auc_micro = sd(auc_micro),
  mean_auc_macro = mean(auc_macro),
  sd_auc_macro = sd(auc_macro)
), by = .(test.subset, train.subsets, algorithm)]
long_dt <- melt(summary_dt,
                measure = patterns(mean = "^mean_auc", sd = "^sd_auc"),
                variable.name = "metric",
)
long_dt[, metric := factor(metric, labels = c("auc_micro", "auc_macro"))]
long_dt[, test.subset := paste0("test = ", test.subset)]


ggplot(long_dt, aes(x = mean, y = algorithm, color = metric)) +
  geom_point(position = position_dodge(width = 0.5), size = 1) +
  geom_errorbarh(
    aes(xmin = mean - sd, xmax = mean + sd),
    position = position_dodge(width = 0.5),
    height = 0.25
  ) +
  facet_grid(test.subset ~ train.subsets) +
  xlim(0.9,1)+
  labs(
    title = "AUC mean ± SD by Algorithm (3 folds), imbalance ~ 0.1%",
    x = "AUC",
    y = "Algorithm",
    color = "Metric"
  )