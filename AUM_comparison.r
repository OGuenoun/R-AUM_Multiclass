###Loading necessary
source("R-AUM_Multiclass/utils_AUM.R")
library(data.table)

##Creating task 
(SOAK <- mlr3resampling::ResamplingSameOtherSizesCV$new())
unb.csv.vec <- Sys.glob("~/data_Classif_unbalanced/FashionMNIST.csv")
task.list <- list()
data.csv <- sub("_unbalanced", "", unb.csv.vec)
MNIST_dt <- fread(file=data.csv)
subset_dt <- fread(unb.csv.vec) 
task_dt <- data.table(subset_dt, MNIST_dt)[, label := factor(y)]
feature.names <- grep("^[0-9]+$", names(task_dt), value=TRUE)
subset.name.vec <- names(subset_dt)
subset.name.vec <- c("seed1_prop0.01")
(data.name <- gsub(".*/|[.]csv$", "", unb.csv.vec))
for(subset.name in subset.name.vec){
  subset_vec <- task_dt[[subset.name]]
  task_id <- paste0(data.name,"_",subset.name)
  itask <- mlr3::TaskClassif$new(
    task_id, task_dt[subset_vec != ""], target="label")
  itask$col_roles$stratum <- c("y", subset.name)
  itask$col_roles$subset <- subset.name
  itask$col_roles$feature <- feature.names
  task.list[[task_id]] <- itask
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
counts <- torch::torch_tensor(c(0.01,1,1,1,1,1,1,1,1,1))
weights <- 1 / (counts + 1e-8)
weights <- weights / weights$sum()
##Defining custom losses

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
  forward = function(input, target) {
    torch::nnf_cross_entropy(input, target, weight = self$weights)
  }
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
make_torch_learner <- function(id,loss,lr,batch_sz){
  n.epochs<-batch_sz/2
  po_list <- c(
    list(
      mlr3pipelines::po(
        "select",
        selector = mlr3pipelines::selector_type(c("numeric", "integer"))),
      mlr3torch::PipeOpTorchIngressNumeric$new()),
    list(mlr3pipelines::po(
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
        mlr3torch::t_opt("sgd", lr =lr)),
      mlr3pipelines::po(
        "torch_callbacks",
        mlr3torch::t_clbk("history")),
      mlr3pipelines::po(
        "torch_model_classif",
        batch_size = batch_sz,
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
lr_list=c(0.01,0.1,1)
batches_lis=c(10,100,1000)
learner.list<-c()
for(lr in lr_list){
  for(batch in batches_lis){
    learner.list<-c(learner.list,c(
      make_torch_learner(paste0("lr",lr,"linear_CE_unweighted_","batch_",batch),torch::nn_cross_entropy_loss,lr,batch),
      make_torch_learner(paste0("lr",lr,"linear_Macro_AUM_","batch_",batch),nn_AUM_macro_loss,lr,batch),
      make_torch_learner(paste0("lr",lr,"linear_Micro_AUM_","batch_",batch),nn_AUM_micro_loss,lr,batch),
      make_torch_learner(paste0("lr",lr,"linear_CE_weighted_","batch_",batch),nn_weighted_ce_loss,lr,batch))
    )
  }
  
}

### END defining custom losses






(bench.grid <- mlr3::benchmark_grid(
  task.list,
  learner.list,
  SOAK))

reg.dir <- "~/links/scratch/2025-08-06-bynclasses-linear"
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
      walltime = 5*60*60,#seconds
      memory = 15000,#megabytes per cpu
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


summary_dt <- score_out[, .(
  mean_auc_micro = mean(auc_micro),
  sd_auc_micro = sd(auc_micro),
  mean_auc_macro = mean(auc_macro),
  sd_auc_macro = sd(auc_macro)
), by = .(test.subset, train.subsets, learner_name,task_id,lr,batch_size)]
best_lr_summ=summary_dt[
  , .SD[which.max(mean_auc_macro)], by = .(learner_name,task_id,test.subset,train.subsets,batch_size)
]

#fwrite(best_lr_summ,"~/R-AUM_Multiclass/AUC_results/Fashion_subclasses_linear.csv")


best_lr_conv=fread("~/R-AUM_Multiclass/AUC_results/Fashion_subclasses_conv.csv")
all_best_lr_summ=rbind(best_lr_summ,best_lr_conv)

long_dt <- melt(best_lr_summ,
                measure = patterns(mean = "^mean_auc", sd = "^sd_auc"),
                variable.name = "metric",
)
long_dt[, metric := factor(metric, labels = c("auc_micro", "auc_macro"))]

long_dt[, test.subset := paste0("test = ", test.subset)]
long_dt[, train.subsets := paste0("train = ", train.subsets)]
long_dt[,task_id:= sub("FashionMNIST_seed([1234])_prop0.01", "subsampled_classes=\\1", task_id)]
long_dt[,learner_name:= sub("_batch_[0-9]+", "", learner_name)]

ggplot(long_dt, aes(x = mean, y = learner_name, color = metric)) +
  geom_point(position = position_dodge(width = 0.5)) +
  geom_errorbarh(
    aes(xmin = mean - sd, xmax = mean + sd),
    position = position_dodge(width = 0.5),
    height = 0.25
  ) +
  facet_grid(test.subset ~ train.subsets+batch_size) +
  labs(
    title = " FashionMNIST,imbalance = 1% ,AUC mean ± SD by Algorithm (3 folds)",
    x = "AUC",
    y = "Algorithm",
    color = "Metric"
  )
history_dt <- rbindlist(lapply(1:nrow(score_out), function(i) {
  row <- score_out[i]
  learner <- row$learner[[1]]
  hist <- learner$model
  hist_dt <- as.data.table(hist)
  hist_dt[, learner_name := row$learner_name]
  hist_dt[, task_id := row$task_id]
  hist_dt[, test.subset := row$test.subset]
  hist_dt[, train.subsets := row$train.subsets]
  hist_dt[, test.fold := row$test.fold]
  return(hist_dt)
}), fill = TRUE)
history_dt[, `:=`(
  train_measure_col = fcase(
    grepl("CE", learner_name), "train.classif.logloss",
    grepl("Micro_AUM", learner_name), "train.aum_micro",
    grepl("Macro_AUM", learner_name), "train.aum_macro",
    default = "train.auc_macro"
  ),
  valid_measure_col = fcase(
    grepl("CE", learner_name), "valid.classif.logloss",
    grepl("Micro_AUM", learner_name), "valid.aum_micro",
    grepl("Macro_AUM", learner_name), "valid.aum_macro",
    default = "valid.auc_macro"
  )
)]
history_dt[, train_value := mapply(function(row, col) row[[col]], split(history_dt, seq_len(nrow(history_dt))), train_measure_col)]
history_dt[, valid_value := mapply(function(row, col) row[[col]], split(history_dt, seq_len(nrow(history_dt))), valid_measure_col)]

(history_long <- nc::capture_melt_single(
  history_dt,
  set=nc::alevels(valid="validation", train="subtrain"),
  ".",
  measure=nc::alevels("auc_macro", "auc_micro")))
history_long[,task_id:= sub("MNIST_seed[12]_prop", "", task_id)]
history_long[,learner_name:= sub("conv_", "", learner_name)]
ggplot()+
  theme_bw()+
  geom_line(aes(
    epoch, value, color=set),
    data=history_long[measure=="auc_macro"])+
  facet_grid(learner_name+train.subsets~task_id+test.subset+test.fold)+
  scale_x_continuous("epoch")

#fwrite(history_long,"~/R-AUM_Multiclass/FashionMNIST_Learning_curves.csv")

##Training time
time_dt<-copy(score_out)
time_dt[, train_time := sapply(learner, function(l) l$timings[[1]]/3600)]
time_dt[, loss_function := fifelse(
  grepl("Micro_AUM", learner_name), "Micro_AUM",
  fifelse(grepl("Macro_AUM", learner_name), "Macro_AUM",
          fifelse(grepl("CE_weighted", learner_name), "CE_weighted",
                  fifelse(grepl("CE_unweighted", learner_name), "CE_unweighted", NA_character_)
          )))]
ggplot() +
  geom_histogram(aes(
    train_time),
    data=time_dt) +
  facet_grid(~loss_function)+
  labs(title = "Train time histogram , 532 linear model", x = "Hours", y = "Number of models ") 


#ROCs
best_row <- score_dt[grepl("Macro_AUM", learner_id)][which.max(auc_macro)]
predictions= best_row$prediction_test[[1]]
pred_tensor=torch::torch_tensor(predictions$prob)
label_tensor=torch::torch_tensor(predictions$truth)
roc_micro=ROC_curve_micro(pred_tensor,label_tensor)
roc_micro_dt <- data.table(
  sapply(roc_micro, function(t) torch::as_array(t))
)
ggplot(roc_micro_dt, aes(x = FPR , y = TPR)) +
  geom_line()+
  labs(
    title = " ROC micro for an AUM_macro optimized model",
    x = "FPR",
    y = "TPR",
  )

roc_macro=ROC_curve_macro(pred_tensor,label_tensor)

fpr_mat <- torch::as_array(roc_macro[[1]])  # matrix of shape [N, 10]
tpr_mat <- torch::as_array(roc_macro[[3]])
n_classes <- dim(fpr_mat)[2]
n_points <- dim(fpr_mat)[1]

roc_dt_macro <- data.table(
  fpr = as.vector(fpr_mat),
  tpr = as.vector(tpr_mat),
  class = rep(1:n_classes, each = n_points)
)
ggplot(roc_dt_macro, aes(x = fpr, y = tpr, color = factor(class))) +
  geom_line() +
  labs(title = "ROC Curves (One-vs-Rest)", x = "FPR", y = "TPR", color = "Class") +
  theme_minimal()
##Scatter plot , first one : AUC micro vs AUC macro
best_lr_out=score_out[
  , .SD[which.max(auc_macro)], by = .(learner_name,task_id,test.subset,train.subsets,iteration)
]
best_lr_aum <- best_lr_out[learner_name %like% "AUM", .(learner_name, auc_micro,auc_macro,iteration,test.fold,test.subset,train.subsets,task_id)]
macro <- best_lr_aum[learner_name=="linear_Macro_AUM",
                     .(iteration, test.fold, test.subset, train.subsets,
                       auc_micro_macroAUM =auc_micro),    # use mean if duplicates
                     by=.(iteration, test.fold, test.subset, train.subsets,task_id)]

micro <- best_lr_aum[learner_name=="linear_Micro_AUM",
                     .(iteration, test.fold, test.subset, train.subsets,
                       auc_micro_microAUM = auc_micro,task_id),
                     by=.(iteration, test.fold, test.subset, train.subsets)]
comp <- macro[micro, on=.(iteration, test.fold, test.subset, train.subsets,task_id)]
ggplot(comp, aes(x = auc_micro_microAUM , y = auc_micro_macroAUM)) +
  geom_point()+
  labs(
    title = " AUC micro values",
    x = "Training on micro AUM",
    y = "Training on macro AUM",
  )+
  xlim(0.5,1)+
  ylim(0.5,1)+
  coord_equal()
