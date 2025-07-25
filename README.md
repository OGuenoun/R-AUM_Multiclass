# Multi-class AUM for imbalanced datasets using R
I used [AUM_comparison.r](AUM_comparison.r) to train NNs on the cluster . I generated [score_conv_grid_search.csv](score_conv_grid_search.csv) which contains test results for CNNs using different loss functions

<p align="center">
  <img src="AUM_comparison_1percent.png" alt="Description" width="400"/>
</p>
<p align="center">
  <img src="AUM_comparison_0,1percent.png" alt="Description" width="400"/>
</p>

## Analysis

- When training on imbalaced datasets , both AUMs are better than unweighted cross entropy. 
- Micro AUM optimizes Macro AUC , but Macro AUM doesn't optimize Micro AUC
-When employing Macro AUC as the evaluation metric(which is generally more appropriate for imbalanced datasets) we observe that the performance gap between Macro AUM and Micro AUM widens with increasing class imbalance, with Macro AUM having better performance under higher imbalance . ( 1% Micro AUM was better, 0.1% Macro AUM was better )