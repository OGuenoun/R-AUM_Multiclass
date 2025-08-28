###Loading necessary
source("R-AUM_Multiclass/utils_AUM.R")
library(data.table)

##Creating task 
(SOAK <- mlr3resampling::ResamplingSameOtherSizesCV$new())
unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/FashionMNIST.csv")
task.list <- list()
data.csv <- sub("_unbalanced", "3", unb.csv.vec)
MNIST_dt <- fread(file=data.csv)
subset_dt <- fread(unb.csv.vec) 
task_dt <- data.table(subset_dt, MNIST_dt)
feature.names <- grep("^[0-9]+$", names(task_dt), value=TRUE)
subset.name.vec <- names(subset_dt)
subset.name.vec <- c("seed2_prop0.001","seed3_prop0.001")
(data.name <- gsub(".*/|[.]csv$", "", unb.csv.vec))
target_mapping=list(
  "y_agg3"="seed3_prop0.001",
  "y_agg4"="seed2_prop0.001",
  "y_agg5"="seed2_prop0.001")

for (targ in names(target_mapping)) {
  subset.name <- target_mapping[[targ]]
  subset_vec <- task_dt[[subset.name]]
  n_classes <- as.integer(sub("^y_agg", "", targ))
  new_col <- paste0("n_classes", n_classes)
  task_id <- paste0(new_col,"_",data.name,"_",subset.name)
  task_dt[, (col) := as.factor(get(col))]
  itask <- mlr3::TaskClassif$new(
    task_id, task_dt[subset_vec != ""], target=targ)
  itask$col_roles$stratum <- c(targ, subset.name)
  itask$col_roles$subset <- subset.name
  itask$col_roles$feature <- feature.names
  task.list[[task_id]] <- itask
}
if(FALSE){
  for(subset.name in subset.name.vec){
    subset_vec <- task_dt[[subset.name]]
    task_id <- paste0(data.name,"_",subset.name)
    itask <- mlr3::TaskClassif$new(
      task_id, task_dt[subset_vec != ""], target="label")
    itask$col_roles$stratum <- c("label", subset.name)
    itask$col_roles$subset <- subset.name
    itask$col_roles$feature <- feature.names
    task.list[[task_id]] <- itask
  }
}





####keeping only same and other
SOAK$param_set$values$subsets <- "SO"

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
## END defining custom measures
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
nn_weighted_AUM_micro_loss <- torch::nn_module(
  "nn_AUM_micro_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward =Proposed_AUM_micro_weighted
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
n.epochs=5000
make_torch_learner <- function(id,loss,lr,n_classes){
  po_list <- c(
    list(
      mlr3pipelines::po(
        "select",
        selector = mlr3pipelines::selector_type(c("numeric", "integer"))),
      mlr3torch::PipeOpTorchIngressNumeric$new()),
    list(
      mlr3torch::nn("linear", out_features=n_classes),
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
lr_list=c(0.001,0.01,0.1)
n_classes=c(3,4,5)

bench.grid <- data.table(
  task        = character(),
  learner      = character(),
  resampling     = character()
)
for(task in task.list){
  n_classes=length(task$class_names)
  learner.list<-c()
  for(lr in lr_list){
    learner.list<-c(learner.list,c(
      make_torch_learner(paste0("lr",lr,"linear_CE_unweighted_",n_classes),torch::nn_cross_entropy_loss,lr,n_classes),
      make_torch_learner(paste0("lr",lr,"linear_Micro_AUM_weighted_",n_classes),nn_weighted_AUM_micro_loss,lr,n_classes),
      make_torch_learner(paste0("lr",lr,"linear_Macro_AUM_",n_classes),nn_AUM_macro_loss,lr,n_classes),
      make_torch_learner(paste0("lr",lr,"linear_Micro_AUM_",n_classes),nn_AUM_micro_loss,lr,n_classes),
      make_torch_learner(paste0("lr",lr,"linear_CE_weighted_",n_classes),nn_weighted_ce_loss,lr,n_classes))
    )
    
  }
  (bench.grid <-rbind(bench.grid, mlr3::benchmark_grid(
    task,
    learner.list,
    SOAK)))
}


### END defining custom losses








reg.dir <- "~/links/scratch/2025-08-25-classes"
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
      walltime = 8*60*60,#seconds
      memory = 16000,#megabytes per cpu
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
##Plotting
library(ggplot2)
score_dt <- mlr3resampling::score(bench.result, c(auc_macro,auc_micro))
score_out <- score_dt[, .(
  task_id, test.subset, train.subsets, test.fold, learner_id, auc_micro,auc_macro,learner,iteration)]
score_out[, lr := as.numeric(sub("lr([0-9.]+).*", "\\1", learner_id))]
score_out[, learner_name := sub("lr[0-9.]+", "", learner_id)]
#score_out[,batch_size:=sub(".*batch_([0-9]+)", "batch_size=\\1", learner_id)]
score_out[,n_classes:=sub("[a-zA-Z_]+([0-9])","n_classes=\\1",learner_name)]

summary_dt <- score_out[, .(
  mean_auc_micro = mean(auc_micro),
  sd_auc_micro = sd(auc_micro),
  mean_auc_macro = mean(auc_macro),
  sd_auc_macro = sd(auc_macro)
), by = .(test.subset, train.subsets, learner_name,task_id,lr)]
best_lr_summ=summary_dt[
  , .SD[which.max(mean_auc_macro)], by = .(learner_name,task_id,test.subset,train.subsets)
]





#fwrite(best_lr_summ,"~/R-AUM_Multiclass/AUC_results/Fashion_subclasses_linear.csv")


#best_lr_conv=fread("~/R-AUM_Multiclass/AUC_results/Fashion_subclasses_conv.csv")
#all_best_lr_summ=rbind(best_lr_summ,best_lr_conv)

long_dt <- melt(best_lr_summ,
                measure = patterns(mean = "^mean_auc", sd = "^sd_auc"),
                variable.name = "metric",
)
long_dt[, metric := factor(metric, labels = c("auc_micro", "auc_macro"))]

long_dt[, test.subset := paste0("test = ", test.subset)]
long_dt[, train.subsets := paste0("train = ", train.subsets)]
long_dt[,task_id:= sub("n_classes[0-9]_FashionMNIST_seed[1234]_prop([01.]+)", "imbalance=\\1", task_id)]
long_dt[,n_classes:= sub("[A-Za-z_]+([0-9])","n_classes=\\1",learner_name)]
long_dt[,learner_name:= sub("[0-9]+", "", learner_name)]

#long_dt[,learner_name:= sub("linear", "conv", learner_name)]

ggplot(long_dt, aes(x = mean, y = learner_name, color = metric)) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbarh(
    aes(xmin = mean - sd, xmax = mean + sd),
    position = position_dodge(width = 0.5),
    height = 0.25
  ) +
  facet_grid(test.subset ~ train.subsets+n_classes) +
  labs(
    title = " FashionMNIST,imbalance=0.1% ,AUC mean ± SD by Algorithm (3 folds)",
    x = "AUC",
    y = "Algorithm",
    color = "Metric"
  )
score_dt[, learner_name := sub("lr[0-9.]+", "", learner_id)]
best_lr_out=score_dt[
  , .SD[which.max(auc_macro)], by = .(learner_name,task_id,test.subset,train.subsets,iteration)
]

ROC_AUC_macro_first<-function(pred_tensor,label_tensor){
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
  counts <- torch::torch_bincount(label_tensor, minlength = n_class)
  present <- counts > 0
  sum=torch::torch_sum(FPR_diff * TPR_sum / 2.0,dim=1)
  sum[1]
}

best_lr_out[, AUC_subsampled :=
              vapply(prediction_test, function(p) {
                auc <- ROC_AUC_macro_first(
                  torch::torch_tensor(p$prob, dtype = torch::torch_float()),
                  torch::torch_tensor(p$truth,    dtype = torch::torch_long())
                )
                auc <- auc$item()
                as.numeric(auc)
              }, numeric(1))
]
summary_dt <- best_lr_out[, .(
  mean_auc_subsampled = mean(AUC_subsampled),
  sd_auc_subsampled = sd(AUC_subsampled)
), by = .(test.subset, train.subsets, learner_name,task_id)]
long_sub <- melt(summary_dt,
                 measure = patterns(mean = "^mean_auc", sd = "^sd_auc"),
                 variable.name = "metric",
)

long_sub[, test.subset := paste0("test = ", test.subset)]
long_sub[, train.subsets := paste0("train = ", train.subsets)]
long_sub[,task_id:= sub("n_classes[0-9]_FashionMNIST_seed[1234]_prop([01.]+)", "imbalance=\\1", task_id)]
long_sub[,n_classes:= sub("[A-Za-z_]+([0-9])","n_classes=\\1",learner_name)]
long_sub[,learner_name:= sub("[0-9]+", "", learner_name)]
ggplot(long_sub, aes(x = mean, y = learner_name)) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbarh(
    aes(xmin = mean - sd, xmax = mean + sd),
    position = position_dodge(width = 0.5),
    height = 0.25
  ) +
  facet_grid(test.subset ~ train.subsets+n_classes) +
  labs(
    title = " FashionMNIST,imbalance=0.1% ,AUC mean ± SD by Algorithm (3 folds)",
    x = "AUC",
    y = "Algorithm",
    color = "Metric"
  )





#ROCs
score_dt <- mlr3resampling::score(bench.result, c(auc_macro,auc_micro))
best_row_macro <- score_dt[grepl("Macro_AUM", learner_id)& grepl("_3", learner_id) & test.subset=="balanced" & train.subsets=="same"][which.max(auc_macro)]
predictions_macro= best_row_macro$prediction_test[[1]]
best_row_micro <- score_dt[grepl("Micro_AUM", learner_id)& grepl("_3", learner_id) & test.subset=="balanced" & train.subsets=="same"][which.max(auc_macro)]
predictions_micro= best_row_micro$prediction_test[[1]]
best_row_ce <- score_dt[grepl("CE_weighted", learner_id)& grepl("_3", learner_id) & test.subset=="balanced" & train.subsets=="same"][which.max(auc_macro)]
predictions_ce_weighted= best_row_ce$prediction_test[[1]]
pred_macro_dt=data.table(predictions_macro$truth,predictions_macro$prob)
pred_macro_dt[ ,loss := "macro_aum"]
pred_micro_dt=data.table(predictions_micro$truth,predictions_micro$prob)
pred_micro_dt[ ,loss := "micro_aum"]
pred_ce_dt=data.table(predictions_ce_weighted$truth,predictions_ce_weighted$prob)
pred_ce_dt[ ,loss := "cross-entropy weighted"]
pred_loss=rbind(pred_macro_dt,pred_micro_dt,pred_ce_dt)

setnames(pred_loss, old = "V1", new = "label")
pred_loss[, label := paste0("true class=", label)]
fwrite(pred_aum,"~/R-AUM_Multiclass/scores_issue/AUM_pred_scores.csv")
long_pred_dt <- melt(
  pred_loss,
  measure.vars = c("0", "1", "2"),
  variable.name = "prediction_for_class",
  value.name = "Value"
)
ggplot(long_pred_dt, aes(x = Value, color=prediction_for_class)) +
  geom_histogram( position = "identity", bins = 30) +
  labs(title = "Histograms of predictions from models optimized on different loss functions",
       x = "Value",
       y = "Count") +
  facet_grid(label ~ loss,scales = "free")
