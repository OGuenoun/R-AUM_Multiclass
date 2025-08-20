library(data.table)
library(ggplot2)
conv_time=fread("~/R-AUM_Multiclass/Training_Histo/time_conv_bynclasses_1500ep.csv")
conv_time[,arch:="conv"]
linear_time=fread("~/R-AUM_Multiclass/Training_Histo/time_linear_bynclasses_1500ep.csv")
linear_time[,arch:="linear"]
time_dt=rbind(conv_time,linear_time)
ggplot() +
  geom_histogram(aes(
    train_time),
    data=time_dt) +
  facet_grid(arch~loss_function)+
  scale_x_log10()+
  labs(title = "Train time histogram , 108 model for each subfigure", x = "Hours", y = "Number of models ") 
