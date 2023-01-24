library(xlsx)
library(survival)
library(compareC)
library(survIDINRI)
#Import data of prognostic analysis, this part of original data are available from the corresponding author on reasonable request.
data<-read.xlsx("D:/data3.xlsx",1)

#For primary end points:
#compared ML-CAC scores performance with CT-CAC scores 
fit_ML_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                      ML_CAC_Q2+ ML_CAC_Q3 + ML_CAC_Q4, data = data)
fit_CT_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                      CT_CAC_2+ CT_CAC_3 + CT_CAC_4, data = data)
data$ML_CAC <- fit_ML_CAC$linear.predictors 
data$CT_CAC <- fit_CT_CAC$linear.predictors 
compareC(data$primary_end_point_time, data$primary_end_point, -data$ML_CAC, -data$CT_CAC)
#∆C-index over a basic traditional Cox model
fit_basic_model <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                           + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR, data = data)
fit_basic_model_ML_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                                   + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                                 + ML_CAC_Q2+ ML_CAC_Q3 + ML_CAC_Q4, data = data)
fit_basic_model_CT_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                                  + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                                + CT_CAC_2+ CT_CAC_3 + CT_CAC_4, data = data)
data$basic_model <- fit_basic_model$linear.predictors 
data$basic_model_ML_CAC  <- fit_basic_model_ML_CAC$linear.predictors 
data$basic_model_CT_CAC  <- fit_basic_model_CT_CAC$linear.predictors 
compareC(data$primary_end_point_time, data$primary_end_point, -data$basic_model, -data$basic_model_ML_CAC)  #ML_CAC
compareC(data$primary_end_point_time, data$primary_end_point, -data$basic_model, -data$basic_model_CT_CAC)  #CT_CAC
#Improvement in the continuous NRI amd IDI over a basic traditional Cox model
indata0=as.matrix(subset(data, select=c(primary_end_point_time, primary_end_point,age,gender_female,smoker,hypertension,diabetes,SBP,Glu,LDLC,eGFR)))
indata1=as.matrix(subset(data, select=c(primary_end_point_time, primary_end_point,age,gender_female,smoker,hypertension,diabetes,SBP,Glu,LDLC,eGFR,
                                        ML_CAC_Q2, ML_CAC_Q3, ML_CAC_Q4)))
indata2=as.matrix(subset(data, select=c(primary_end_point_time, primary_end_point,age,gender_female,smoker,hypertension,diabetes,SBP,Glu,LDLC,eGFR,
                                        CT_CAC_2, CT_CAC_3, CT_CAC_4)))                                        
covs0<-as.matrix(indata0[,c(-1,-2)])
covs1<-as.matrix(indata1[,c(-1,-2)])
covs2<-as.matrix(indata2[,c(-1,-2)])
set.seed(1234)
x1 <- IDI.INF(indata0[ ,1:2], covs0, covs1, 36,  npert=1000) #for ML_CAC 
IDI.INF.OUT(x1) #for ML_CAC
x2 <- IDI.INF(indata0[ ,1:2], covs0, covs2, 36,  npert=1000) #for CT_CAC 
IDI.INF.OUT(x2) #for CT_CAC

#For the secondary end point, all the codes remain unchanged except for the time and status. 

#Improvement of CT-CAC scores in the C-index, continuous NRI amd IDI over ML-CAC scores for the primary end point:
fit_ML_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                      ML_CAC_Q2+ ML_CAC_Q3 + ML_CAC_Q4, data = data)
fit_ML_CAC_CT_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                             ML_CAC_Q2+ ML_CAC_Q3 + ML_CAC_Q4 
                           + CT_CAC_2+ CT_CAC_3 + CT_CAC_4, data = data)
data$ML_CAC <- fit_ML_CAC$linear.predictors 
data$ML_CAC_CT_CAC <- fit_ML_CAC_CT_CAC $linear.predictors 
compareC(data$primary_end_point_time, data$primary_end_point, -data$ML_CAC, -data$ML_CAC_CT_CAC)

indata3=as.matrix(subset(data, select=c(primary_end_point_time, primary_end_point,ML_CAC_Q2, ML_CAC_Q3, ML_CAC_Q4)))
indata4=as.matrix(subset(data, select=c(primary_end_point_time, primary_end_point,ML_CAC_Q2, ML_CAC_Q3, ML_CAC_Q4,
                                        CT_CAC_2, CT_CAC_3, CT_CAC_4)))
covs3<-as.matrix(indata3[,c(-1,-2)])
covs4<-as.matrix(indata4[,c(-1,-2)])
set.seed(1234)
x3 <- IDI.INF(indata3[ ,1:2], covs3, covs4, 36,  npert=1000) 
IDI.INF.OUT(x3) 









