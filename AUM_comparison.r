source("R-AUM_Multiclass/utils_AUM.R")
library(data.table)
MNIST_dt <- fread("~/data_Classif/MNIST.csv")
MNIST_dt <- fread("~/data_Classif/MNIST.csv")
data.table(
  name=names(MNIST_dt),
  first_row=unlist(MNIST_dt[1]),
  last_row=unlist(MNIST_dt[.N]))
MNIST_dt[, label := factor(y)]
MNIST_dt_others <- MNIST_dt[label != 1]
percentages <- c(1, 0.1, 0.01)

imbalanced_sets <- lapply(percentages, function(pct) {
  dt_1_subset <- MNIST_dt[label == 1][sample(.N, floor(.N * pct))]
  result <- rbindlist(list(MNIST_dt_others, dt_1_subset))
  return(result)
})
rm(MNIST_dt_others,MNIST_dt)
make_task<-function(id,dataset){
  mtask <- mlr3::TaskClassif$new(
    id, dataset, target="label")
  mtask$col_roles$stratum <- "label"
  
  mtask$col_roles$feature <- grep("^[0-9]+$", names(dataset), value=TRUE)
  return(mtask)
}

list_tasks=list(
  make_task("MNIST_balanced",imbalanced_sets[[1]]),
  make_task("MNIST_imbalance_0.1",imbalanced_sets[[2]]),
  make_task("MNIST_imbalance_0.01",imbalanced_sets[[3]])
)

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
      probs <- prediction$prob
      truth <- prediction$truth
      Proposed_AUM_micro(pred_tensor = probs,label_tensor = truth)
    }
  )
)
auc_micro <- Micro_AUC$new()
measure_list <- c(mlr3::msrs("classif.mauc_aunu"), auc_micro)
nn_AUM_micro_loss <- torch::nn_module(
  "nn_AUM_micro_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward = function(input, target) {
    print(table(as.integer(target)))
    Proposed_AUM_micro(input, target)
  }
)
nn_AUM_macro_loss <- torch::nn_module(
  "nn_AUM_macro_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward = function(input, target) {
    Proposed_AUM_macro(input, target)
  }
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
        batch_size = 10000,
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
kfoldcv <- mlr3::rsmp("cv")
kfoldcv$param_set$values$folds <- 4

(bench.grid <- mlr3::benchmark_grid(
  list_tasks,
  learner.list,
  kfoldcv))
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
      walltime = 60*60,#seconds
      memory = 2000,#megabytes per cpu
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