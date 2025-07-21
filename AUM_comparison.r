source("R-AUM_Multiclass/utils_AUM.R")
library(data.table)
(SOAK <- mlr3resampling::ResamplingSameOtherSizesCV$new())
unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/MNIST.csv")
task.list <- list()
data.csv <- sub("_unbalanced", "", unb.csv.vec)
MNIST_dt <- fread(file=data.csv)
subset_dt <- fread(unb.csv.vec) 
task_dt <- data.table(subset_dt, MNIST_dt)[, label := factor(y)]
feature.names <- grep("^[0-9]+$", names(task_dt), value=TRUE)
subset.name.vec <- names(subset_dt)
subset.name <- "seed1_prop0.01"
(data.name <- gsub(".*/|[.]csv$", "", unb.csv.vec))
subset_vec <- task_dt[[subset.name]]
task_id <- paste0(data.name,"_",subset.name)
itask <- mlr3::TaskClassif$new(
  task_id, task_dt[subset_vec != ""], target="label")
itask$col_roles$stratum <- c("y",subset.name)
itask$col_roles$subset <- subset.name
itask$col_roles$feature <- feature.names
task.list[[task_id]] <- itask

SOAK$param_set$values$subsets <- "SO"


Micro_AUC = R6::R6Class("Micro_AUC",
  inherit = mlr3::MeasureClassif,
  public = list(
    initialize = function() { 
      super$initialize(
        id = "auc_micro",
        packages = "torch",
        properties = character(),
        task_properties = "multiclass",
        predict_type = "prob",
        range = c(0, 1),
        minimize = FALSE
      )
    }
  ),

  private = list(
    # define score as private method
    .score = function(prediction, ...) {
      pred_tensor=torch::torch_tensor(prediction$prob)
      label_tensor=torch::torch_tensor(prediction$truth)
      ROC_AUC_micro<-function(pred_tensor,label_tensor){
        n_class=pred_tensor$size(2)
        one_hot_labels = torch::nnf_one_hot(label_tensor, num_classes=n_class) 
        is_positive = one_hot_labels
        is_negative =1-one_hot_labels
        fn_diff = -is_positive$flatten()
        fp_diff = is_negative$flatten()
        thresh_tensor = -pred_tensor$flatten()
        fn_denom = is_positive$sum()
        fp_denom = is_negative$sum()
        sorted_indices = torch::torch_argsort(thresh_tensor)
        sorted_fp_cum = fp_diff[sorted_indices]$cumsum(dim=1) / fp_denom
        sorted_fn_cum = -fn_diff[sorted_indices]$flip(1)$cumsum(dim=1)$flip(1) / fn_denom
        
        sorted_thresh = thresh_tensor[sorted_indices]
        sorted_is_diff = sorted_thresh$diff() != 0
        sorted_fp_end = torch::torch_cat(c(sorted_is_diff, torch::torch_tensor(TRUE)))
        sorted_fn_end = torch::torch_cat(c(torch::torch_tensor(TRUE), sorted_is_diff))
        
        uniq_thresh = sorted_thresh[sorted_fp_end]
        uniq_fp_after = sorted_fp_cum[sorted_fp_end]
        uniq_fn_before = sorted_fn_cum[sorted_fn_end]
        
        FPR = torch::torch_cat(c(torch::torch_tensor(0.0), uniq_fp_after))
        FNR = torch::torch_cat(c(uniq_fn_before, torch::torch_tensor(0.0)))
        roc <- list(
          FPR=FPR,
          FNR=FNR,
          TPR=1 - FNR,
          "min(FPR,FNR)"=torch::torch_minimum(FPR, FNR),
          min_constant=torch::torch_cat(c(torch::torch_tensor(-1), uniq_thresh)),
          max_constant=torch::torch_cat(c(uniq_thresh, torch::torch_tensor(0))))
        FPR_diff = roc$FPR[2:N]-roc$FPR[1:-2]
        TPR_sum = roc$TPR[2:N]+roc$TPR[1:-2]
        return(torch::torch_sum(FPR_diff*TPR_sum/2.0))
      }
      auc=ROC_AUC_micro(pred_tensor, label_tensor)
      as.numeric(auc)
    }
  )
)
Macro_AUC = R6::R6Class("Macro_AUC",
  inherit = mlr3::MeasureClassif,
  public = list(
    initialize = function() { 
      super$initialize(
        id = "auc_macro",
        packages = "torch",
        properties = character(),
        task_properties = "multiclass",
        predict_type = "prob",
        range = c(0, 1),
        minimize = FALSE
      )
    }
  ),
  private = list(
    .score = function(prediction, ...) {
      pred_tensor=torch::torch_tensor(prediction$prob)
      label_tensor=torch::torch_tensor(prediction$truth)
      ROC_AUC_macro<-function(pred_tensor,label_tensor){
        n_class=pred_tensor$size(2)
        one_hot_labels = torch::nnf_one_hot(label_tensor, num_classes = n_class)
        is_positive = one_hot_labels
        is_negative =1-one_hot_labels
        fn_diff = -is_positive
        fp_diff = is_negative
        thresh_tensor = -pred_tensor
        fn_denom = is_positive$sum(dim = 1)
        fp_denom = is_negative$sum(dim = 1)
        sorted_indices = torch::torch_argsort(thresh_tensor, dim = 1)
        sorted_fp_cum = torch::torch_gather(fp_diff, dim=1, index=sorted_indices)$cumsum(1)/fp_denom
        sorted_fn_cum = -torch::torch_gather(fn_diff, dim=1, index=sorted_indices)$flip(1)$cumsum(1)$flip(1)/fn_denom
        sorted_thresh = torch::torch_gather(thresh_tensor, dim=1, index=sorted_indices)
        zeros_vec=torch::torch_zeros(1,n_class)
        FPR = torch::torch_cat(c(zeros_vec, sorted_fp_cum))
        FNR = torch::torch_cat(c(sorted_fn_cum, zeros_vec))
        roc<- list(
          FPR_all_classes= FPR,
          FNR_all_classes= FNR,
          TPR_all_classes= 1 - FNR,
          "min(FPR,FNR)"= torch::torch_minimum(FPR, FNR),
          min_constant = torch::torch_cat(c(-torch::torch_ones(1,n_class), sorted_thresh)),
          max_constant = torch::torch_cat(c(sorted_thresh, zeros_vec))
        )
        FPR_diff = roc$FPR_all_classes[2:N,] - roc$FPR_all_classes[1:-2,]
        TPR_sum = roc$TPR_all_classes[2:N,] + roc$TPR_all_classes[1:-2,]
        sum=torch::torch_sum(FPR_diff * TPR_sum / 2.0,dim=1)
        mask = torch::torch_isnan(sum)$logical_not()
        sum_valid = sum[mask]
        mean_valid = sum_valid$mean()
      }
      
      auc=ROC_AUC_macro(pred_tensor, label_tensor)
      as.numeric(auc)
    }
                        )
)
auc_micro <- Micro_AUC$new()
auc_macro<-Macro_AUC$new()
measure_list <- c(auc_micro,auc_macro)

nn_AUM_micro_loss <- torch::nn_module(
  "nn_AUM_micro_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward =Proposed_AUM_micro
)
nn_AUM_macro_loss <- torch::nn_module(
  "nn_AUM_macro_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward =Proposed_AUM_macro
)
n.epochs<-200
make_torch_learner <- function(id,loss){
  po_list <- c(
    list(
      mlr3pipelines::po(
        "select",
        selector = mlr3pipelines::selector_type(c("numeric", "integer"))),
      mlr3torch::PipeOpTorchIngressNumeric$new()),
    list(
      mlr3pipelines::po("nn_head"),
      mlr3pipelines::po(
        "torch_loss",
        loss),
      mlr3pipelines::po(
        "torch_optimizer",
        mlr3torch::t_opt("sgd", lr=0.2)),
      mlr3pipelines::po(
        "torch_callbacks",
        mlr3torch::t_clbk("history")),
      mlr3pipelines::po(
        "torch_model_classif",
        batch_size = 100000,
        patience=n.epochs,
        measures_valid=measure_list,
        measures_train=measure_list,
        predict_type="prob",
        epochs = paradox::to_tune(upper = n.epochs, internal = TRUE)))
    )
    graph <- Reduce(mlr3pipelines::concat_graphs, po_list)
    glearner <- mlr3::as_learner(graph)
    mlr3::set_validate(glearner, validate = 0.5)
    mlr3tuning::auto_tuner(
      learner = glearner,
      tuner = mlr3tuning::tnr("internal"),
      resampling = mlr3::rsmp("insample"),
      measure = mlr3::msr("internal_valid_score", minimize = TRUE),
      term_evals = 1,
      id=id,
      store_models = TRUE)
}
learner.list<-list(
    make_torch_learner("linear_Cross_entropy",torch::nn_cross_entropy_loss),
    make_torch_learner("linear_Macro_average",nn_AUM_macro_loss),
    make_torch_learner("linear_Micro_average",nn_AUM_micro_loss)
)

(bench.grid <- mlr3::benchmark_grid(
  task.list,
  learner.list,
  SOAK))
reg.dir <- "2025-07-16-AUM"
cache.RData <- paste0(reg.dir,".RData")
if(file.exists(cache.RData)){
  load(cache.RData)
}else{
  if(FALSE){#code below only works on the cluster.
    unlink(reg.dir, recursive=TRUE)
    reg = batchtools::makeExperimentRegistry(
      file.dir = reg.dir,
      seed = 1,
      packages = "mlr3verse"
    )
    mlr3batchmark::batchmark(
      bench.grid, store_models = TRUE, reg=reg)
    job.table <- batchtools::getJobTable(reg=reg)
    chunks <- data.frame(job.table, chunk=1)
    batchtools::submitJobs(chunks, resources=list(
      walltime = 2*60*60,#seconds
      memory = 6000,#megabytes per cpu
      ncpus=1,  #>1 for multicore/parallel jobs.
      ntasks=1, #>1 for MPI jobs.
      chunks.as.arrayjobs=TRUE), reg=reg)
    batchtools::getStatus(reg=reg)
    jobs.after <- batchtools::getJobTable(reg=reg)
    table(jobs.after$error)
    ids <- jobs.after[is.na(error), job.id]
    bench.result <- mlr3batchmark::reduceResultsBatchmark(ids, reg = reg)
  }else{
    ## In the code below, we declare a multisession future plan to
    ## compute each benchmark iteration in parallel on this computer
    ## (data set, learning algorithm, cross-validation fold). For a
    ## few dozen iterations, using the multisession backend is
    ## probably sufficient (I have 12 CPUs on my work PC).
    if(require(future))plan("multisession")
    bench.result <- mlr3::benchmark(bench.grid, store_models = TRUE)
  }
  save(bench.result, file=cache.RData)
}
score_dt <- mlr3resampling::score(bench.result, measure_list)
score_out <- score_dt[, .(
  task_id, test.subset, train.subsets, test.fold, algorithm, auc_micro,auc_macro)]
summary_dt <- score_out[, .(
  mean_auc_micro = mean(auc_micro),
  sd_auc_micro = sd(auc_micro),
  mean_auc_macro = mean(auc_macro),
  sd_auc_macro = sd(auc_macro)
), by = .(test.subset, train.subsets, algorithm)]
long_dt <- melt(summary_dt,
                measure = patterns(mean = "^mean_auc", sd = "^sd_auc"),
                variable.name = "metric",
                value.name = c("mean", "sd")
)
long_dt[, metric := factor(metric, labels = c("auc_micro", "auc_macro"))]
long_dt[, test.subset := paste0("test = ", test.subset)]


ggplot(long_dt, aes(x = mean, y = algorithm, color = metric)) +
  geom_point(position = position_dodge(width = 0.5), size = 2) +
  geom_errorbarh(
    aes(xmin = mean - sd, xmax = mean + sd),
    position = position_dodge(width = 0.5),
    height = 0.25
  ) +
  facet_grid(test.subset ~ train.subsets) +
  labs(
    title = "AUC Mean ± SD by Algorithm",
    x = "AUC",
    y = "Algorithm",
    color = "Metric"
  )
