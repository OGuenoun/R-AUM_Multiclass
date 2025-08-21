
ROC_curve_micro <- function(pred_tensor, label_tensor){
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
}
Proposed_AUM_micro<-function(pred_tensor,label_tensor){
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
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2]
  constant_diff = roc$min_constant[2:N]$diff()
  return(torch::torch_sum(min_FPR_FNR * constant_diff))
}
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

ROC_curve_macro<-function(pred_tensor, label_tensor){
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
    }
Proposed_AUM_macro<-function(pred_tensor,label_tensor){
  n_class=pred_tensor$size(2)
  one_hot_labels = torch::nnf_one_hot(label_tensor, num_classes = n_class)
  is_positive = one_hot_labels
  is_negative =1-one_hot_labels
  fn_diff = -is_positive
  fp_diff = is_negative
  thresh_tensor = -pred_tensor
  fn_denom = is_positive$sum(dim = 1)$clamp(min=1)
  fp_denom = is_negative$sum(dim = 1)$clamp(min=1)
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
  label_int <- label_tensor$to(dtype = torch::torch_int())
  actual_n_classes=torch::torch_bincount(label_int)$size(1)
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2,]
  constant_diff = roc$min_constant[2:N,]$diff(dim=1)
  sum = torch::torch_sum(min_FPR_FNR * constant_diff,dim=1)
  mean=torch::torch_sum(sum)/actual_n_classes
}
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
  counts <- torch::torch_bincount(label_tensor, minlength = n_class)
  present <- counts > 0
  sum=torch::torch_sum(FPR_diff * TPR_sum / 2.0,dim=1)
  mean_valid = sum[present]$mean()
}
Proposed_AUM_micro_weighted<-function(pred_tensor,label_tensor){
  n_class=pred_tensor$size(2)
  N=pred_tensor$size(1)
  counts <- torch::torch_bincount(label_tensor, minlength = n_class)
  Pweights <- 1 / (counts + 1e-8)
  Pweights <- Pweights / Pweights$sum()
  Nweights <-1/(N-counts+1e-8)
  Nweights <-Nweights/ Nweights$sum()
  one_hot_labels = torch::nnf_one_hot(label_tensor, num_classes=n_class) 
  is_positive = (one_hot_labels)*Pweights
  is_negative =(1-one_hot_labels)*Nweights
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
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2]
  constant_diff = roc$min_constant[2:N]$diff()
  return(torch::torch_sum(min_FPR_FNR * constant_diff))
}
four_labels <- torch::torch_tensor(c(1, 3, 2, 2), dtype = torch::torch_long())

# Predictions
four_pred <- torch::torch_tensor(matrix(
  c(0.4, 0.3, 0.3,
    0.2, 0.1, 0.7,
    0.5, 0.2, 0.3,
    0.3, 0.4, 0.3),
  nrow = 4, ncol = 3,
  byrow = TRUE
))
(Proposed_AUM_micro_weighted(four_pred,four_labels))
