###Loading necessary
source("R-AUM_Multiclass/utils_AUM.R")
library(data.table)

##Creating task 
(SOAK <- mlr3resampling::ResamplingSameOtherSizesCV$new())
unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/EMNIST.csv")
task.list <- list()
data.csv <- sub("_unbalanced", "", unb.csv.vec)
MNIST_dt <- fread(file=data.csv)
subset_dt <- fread(unb.csv.vec) 
task_dt <- data.table(subset_dt, MNIST_dt)[, label := factor(y)]
feature.names <- grep("^[0-9]+$", names(task_dt), value=TRUE)
subset.name.vec <- names(subset_dt)
subset.name <- "seed1_prop0.001"
(data.name <- gsub(".*/|[.]csv$", "", unb.csv.vec))
subset_vec <- task_dt[[subset.name]]
task_id <- paste0(data.name,"_",subset.name)
itask <- mlr3::TaskClassif$new(
  task_id, task_dt[subset_vec != ""], target="label")
itask$col_roles$stratum <- c("y",subset.name)
itask$col_roles$subset <- subset.name
itask$col_roles$feature <- feature.names
task.list[[task_id]] <- itask




####keeping only same and other
SOAK$param_set$values$subsets <- "SO"

####Defining custom measures
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
measure_list <- c(auc_macro,auc_micro)
## END defining custom measures

##Defining custom losses
weighted_ce <- function(input,target) {
  n_classes <- input$size(2)
  counts <- torch::torch_bincount(target, minlength = n_classes)
  
  weights <- 1 / (counts + 1e-8)
  weights <- weights / weights$sum()
  
  torch::nnf_cross_entropy(input,target, weight = weights)
  }
nn_weighted_CE_loss <- torch::nn_module(
  "nn_weighted_CE_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward =weighted_ce
)
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
n.pixels=28
n.epochs<-300
make_torch_learner <- function(id,loss,lr_list){
  po_list <- c(
    list(
      mlr3pipelines::po(
        "select",
        selector = mlr3pipelines::selector_type(c("numeric", "integer"))),
      mlr3torch::PipeOpTorchIngressNumeric$new()),
    list(
      mlr3pipelines::po(
      "nn_reshape",
      shape=c(-1,1,n.pixels,n.pixels)),
    mlr3pipelines::po(
      "nn_conv2d_1",
      out_channels = 20,
      kernel_size = 6),
    mlr3pipelines::po("nn_relu_1", inplace = TRUE),
    mlr3pipelines::po(
      "nn_max_pool2d_1",
      kernel_size = 4),
    mlr3pipelines::po("nn_flatten"),
    mlr3pipelines::po(
      "nn_linear",
      out_features = 50),
    mlr3pipelines::po("nn_relu_2", inplace = TRUE)),
    list(
      mlr3torch::nn("linear", out_features=10),
      mlr3pipelines::po("nn_head"),
      mlr3pipelines::po(
        "torch_loss",
        loss),
      mlr3pipelines::po(
        "torch_optimizer",
        mlr3torch::t_opt("sgd", lr = paradox::to_tune(
          levels = lr_list 
        ))),
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
      tuner = mlr3tuning::tnr("grid_search"),
      resampling = mlr3::rsmp("insample"),
      measure = mlr3::msr("internal_valid_score", minimize = FALSE),
      term_evals = length(lr_list),
      id = id,
      store_models = TRUE
    )
}
lr_list=c(10^seq(-1,1),5*10^seq(-1,1))
learner.list<-list(
    make_torch_learner("conv_CE_unweighted",torch::nn_cross_entropy_loss,lr_list),
    make_torch_learner("conv_Macro_AUM",nn_AUM_macro_loss,lr_list),
    make_torch_learner("conv_Micro_AUM",nn_AUM_micro_loss,lr_list),
    make_torch_learner("conv_CE_weighted",nn_weighted_CE_loss,lr_list)
)
### END defining custom losses

(bench.grid <- mlr3::benchmark_grid(
  task.list,
  learner.list,
  SOAK))

reg.dir <- "2025-07-24-AUM"
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
      walltime = 4*60*60,#seconds
      memory = 16000,#megabytes per cpu
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
##Plotting
library(ggplot2)
score_dt <- mlr3resampling::score(bench.result, measure_list)
score_out <- score_dt[, .(
  task_id, test.subset, train.subsets, test.fold, algorithm, auc_micro,auc_macro)]
fwrite(score_out, "~/R-AUM_Multiclass/score_conv_grid_search.csv")


