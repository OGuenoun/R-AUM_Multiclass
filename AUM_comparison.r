source("R-AUM_Multiclass/utils_AUM.R")
library(data.table)
(SOAK <- mlr3resampling::ResamplingSameOtherSizesCV$new())
unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/MNIST.csv")
task.list <- list()
data.csv <- sub("_unbalanced", "", unb.csv)
MNIST_dt <- fread(file=data.csv)
subset_dt <- fread(unb.csv) 
task_dt <- data.table(subset_dt, MNIST_dt)[, label := factor(y)]
feature.names <- grep("^[0-9]+$", names(task_dt), value=TRUE)
subset.name.vec <- names(subset_dt)
subset.name.vec <- c("seed1_prop0.01","seed2_prop0.001","seed1_prop0.1")
(data.name <- gsub(".*/|[.]csv$", "", unb.csv))
for(subset.name in subset.name.vec){
  subset_vec <- task_dt[[subset.name]]
  task_id <- paste0(data.name,"_",subset.name)
  itask <- mlr3::TaskClassif$new(
    task_id, task_dt[subset_vec != ""], target="label")
  itask$col_roles$stratum <- "y"
  itask$col_roles$subset <- subset.name
  itask$col_roles$feature <- feature.names
  task.list[[task_id]] <- itask
}
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
      probs <- prediction$prob
      truth <- prediction$truth
      Proposed_AUC_micro(pred_tensor = probs,label_tensor = truth)
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