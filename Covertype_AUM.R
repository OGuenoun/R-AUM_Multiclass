source("R-AUM_Multiclass/utils_AUM.R")
library(data.table)
cover_not_ready_dt<-fread("covertype.csv")
soil_cols <- paste0("Soil_Type", 1:40)
wilderness_cols <- paste0("Wilderness_Area", 1:4)
normalisable_vars <- setdiff(names(cover_not_ready_dt), c(soil_cols, wilderness_cols, "class"))
features <- setdiff(names(cover_not_ready_dt),"class")
cover_dt=cover_not_ready_dt[, (normalisable_vars) := lapply(.SD, scale), .SDcols = normalisable_vars]
cover_ds<-cover_dt[,label:=factor(class)]
task <- mlr3::TaskClassif$new(
  "Cover_type", cover_ds, target="label")
task$col_roles$feature <- features
task$col_roles$stratum <- "class"
####Defining custom measures
Micro_AUC = R6::R6Class("Micro_AUC",
                        inherit = mlr3::MeasureClassif,
                        public = list(
                          AUC=ROC_AUC_micro,
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
                            auc=self$AUC(pred_tensor, label_tensor)
                            as.numeric(auc)
                          }
                        )
)
Macro_AUC = R6::R6Class("Macro_AUC",
                        inherit = mlr3::MeasureClassif,
                        public = list(
                          AUC=ROC_AUC_macro,
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
                            auc=self$AUC(pred_tensor, label_tensor)
                            as.numeric(auc)
                          }
                        )
)
Micro_AUM = R6::R6Class("Macro_AUC",
                        inherit = mlr3::MeasureClassif,
                        public = list(
                          AUM=Proposed_AUM_micro,
                          initialize = function() { 
                            super$initialize(
                              id = "aum_micro",
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
                            auc=self$AUM(pred_tensor, label_tensor)
                            as.numeric(auc)
                          }
                        )
)
Macro_AUM = R6::R6Class("Macro_AUC",
                        inherit = mlr3::MeasureClassif,
                        public = list(
                          AUM=Proposed_AUM_macro,
                          initialize = function() { 
                            super$initialize(
                              id = "aum_macro",
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
                            aum=self$AUM(pred_tensor, label_tensor)
                            as.numeric(aum)
                          }
                        )
)
auc_micro <- Micro_AUC$new()
auc_macro<-Macro_AUC$new()
aum_micro <- Micro_AUM$new()
aum_macro<-Macro_AUM$new()
measure_list <- c(auc_macro,auc_micro,aum_macro,aum_micro,mlr3::msr("classif.logloss"))
##Defining custom losses
weighted_ce <- function(input,target) {
  n_classes <- input$size(2)
  counts <- torch::torch_bincount(target, minlength = n_classes)
  
  weights <- 1 / (counts + 1e-8)
  weights <- weights / weights$sum()
  
  torch::nnf_cross_entropy(input,target, weight = weights)
}

nn_weighted_ce_loss <- torch::nn_module(
  "nn_weighted_ce_loss",
  inherit = torch::nn_mse_loss,
  public=list(
    weights=weights,
    initialize = function() {
      super$initialize()
    }
  )
  ,
  forward = weighted_ce
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
n.epochs=3000
make_torch_learner <- function(id,loss,lr){
  po_list <- c(
    list(
      mlr3pipelines::po(
        "select",
        selector = mlr3pipelines::selector_type(c("numeric", "integer"))),
      mlr3torch::PipeOpTorchIngressNumeric$new()),
    list(mlr3pipelines::po(
      "nn_linear_1",
      out_features = 100),
      mlr3pipelines::po("nn_relu_1", inplace = TRUE),
      mlr3pipelines::po(
        "nn_linear_2",
        out_features = 200),
      mlr3pipelines::po("nn_relu_2", inplace = TRUE)
    ),
    list(
      mlr3torch::nn("linear", out_features=7),
      mlr3pipelines::po("nn_head"),
      mlr3pipelines::po(
        "torch_loss",
        loss),
      mlr3pipelines::po(
        "torch_optimizer",
        mlr3torch::t_opt("sgd", lr =lr)),
      mlr3pipelines::po(
        "torch_callbacks",
        mlr3torch::t_clbk("history")),
      mlr3pipelines::po(
        "torch_model_classif",
        batch_size = 200000,
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
    measure = mlr3::msr("internal_valid_score", minimize = FALSE),
    term_evals = 1,
    id = id,
    store_models = TRUE
  )
}
if(FALSE){
  pred_logits <- torch_tensor(matrix(
    c( 1.2, -0.3,  0.1,
       -0.8,  0.4,  1.1,
       0.2,  0.1, -0.5,
       2.0, -1.0,  0.0,
       -0.2,  1.4,  0.3,
       0.1, -0.2,  0.0,
       -1.2,  0.3,  1.0,
       0.0,  0.0,  0.0),
    ncol = 3, byrow = TRUE))
  
  pred_probs <- torch::nnf_softmax(pred_logits, dim = 2)  
  
  labels <- torch_tensor(c(1, 3, 1, 1, 1, 1, 3, 3), dtype = torch_long())
  auc <- ROC_AUC_macro(pred_probs,labels)
}
lr_list=c(0.001,0.01,0.1,1)
batches_lis=c(10,100,1000)
learner.list<-c()
for(lr in lr_list){
  learner.list<-c(learner.list,c(
    make_torch_learner(paste0("lr",lr,"dense_CE_unweighted"),torch::nn_cross_entropy_loss,lr),
    make_torch_learner(paste0("lr",lr,"dense_Macro_AUM"),nn_AUM_macro_loss,lr),
    make_torch_learner(paste0("lr",lr,"dense_Micro_AUM"),nn_AUM_micro_loss,lr),
    make_torch_learner(paste0("lr",lr,"dense_CE_weighted"),nn_weighted_ce_loss,lr))
  )
  
}

### END defining custom losses

kfoldcv <- mlr3::rsmp("cv")
kfoldcv$param_set$values$folds <- 3




(bench.grid <- mlr3::benchmark_grid(
  task,
  learner.list,
  kfoldcv))

reg.dir <- "~/links/scratch/2025-08-18-covertype-dense"
cache.RData <- paste0(reg.dir,".RData")
keep_history <- function(x){
  learners <- x$learner_state$model$marshaled$tuning_instance$archive$learners
  x$learner_state$model <- if(is.function(learners)){
    L <- learners(1)[[1]]
    x$history <- L$model$torch_model_classif$model$callbacks$history
  }
  x
}
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
      walltime = 7*60*60,#seconds
      memory = 18000,#megabytes per cpu
      ncpus=1,  #>1 for multicore/parallel jobs.
      ntasks=1, #>1 for MPI jobs.
      chunks.as.arrayjobs=TRUE), reg=reg)
    batchtools::getStatus(reg=reg)
    jobs.after <- batchtools::getJobTable(reg=reg)
    table(jobs.after$error)
    ids <- jobs.after[is.na(error), job.id]
    bench.result <- mlr3batchmark::reduceResultsBatchmark(ids, reg = reg,fun=keep_history)
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
library(ggplot2)
score_dt <-bench.result$score(c(auc_macro,auc_micro))
score_dt[, lr := as.numeric(sub("lr([0-9.]+).*", "\\1", learner_id))]
score_dt[, learner_name := sub("lr[0-9.]+", "", learner_id)]
summary_dt <- score_dt[, .(
  mean_auc_micro = mean(auc_micro),
  sd_auc_micro = sd(auc_micro),
  mean_auc_macro = mean(auc_macro),
  sd_auc_macro = sd(auc_macro)
), by = .( learner_name,task_id,lr)]
best_lr_summ=summary_dt[
  , .SD[which.max(mean_auc_macro)], by = .(learner_name,task_id)
]
