library(xlsx)
library(showtext)
data<-read.csv("E:/data2.csv")
showtext_auto(enable = TRUE)
newdata <- data[order(data$test_proba_svm),]
newdata
nrow(newdata)
newdata$number<-c(1:1157)
library(ggplot2)
newdata$CT_CAC_above400 <- factor(newdata$CT_CAC_above400, levels=c("1", "0"))
p1<-ggplot(data=newdata,aes(x=number,y=test_proba_svm)) + 
  geom_point(aes(color=CT_CAC_above400))+
  scale_colour_manual(values=c("indianred2","mediumaquamarine"))+
  theme_light()+
  geom_vline(xintercept=289, col="grey25", lty=3)+
  geom_vline(xintercept=578, col="grey25", lty=3)+
  geom_vline(xintercept=867, col="grey25", lty=3)+
  labs(x = "Patients sorted in order of risk", y = "Predicted risk of severe CAC")+
  guides(color=guide_legend(override.aes = list(size=2)))+
  theme(legend.position="none")
p1





