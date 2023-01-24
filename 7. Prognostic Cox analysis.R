library(xlsx)
library(survival)
#Import data of prognostic analysis, this part of original data are available from the corresponding author on reasonable request.
data<-read.xlsx("D:/data3.xlsx",1)
#Cox proportional hazards regression analysis of ML-CAC score in different models
#For the primary end points:
#Crude model
fit_crude_Q234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                          + ML_CAC_Q2 + ML_CAC_Q3 + ML_CAC_Q4, data = data)
summary(fit_crude_Q234) 
fit_crude_quartiles <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                               + ML_CAC_quartiles, data = data)
summary(fit_crude_quartiles) 
#Model1
fit_model1_Q234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                           + age + gender_female
                         + ML_CAC_Q2 + ML_CAC_Q3 + ML_CAC_Q4, data = data)
summary(fit_model1_Q234) 
fit_model1_quartiles <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                + age + gender_female 
                              + ML_CAC_quartiles, data = data)
summary(fit_model1_quartiles) 
#Model2
fit_model2_Q234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                           + age + gender_female 
                          + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                          + ML_CAC_Q2 + ML_CAC_Q3 + ML_CAC_Q4, data = data)
summary(fit_model2_Q234) 
fit_model2_quartiles <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                + age + gender_female
                              + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                              + ML_CAC_quartiles, data = data)
summary(fit_model2_quartiles) 
#Model3
fit_model3_Q234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                           + age + gender_female
                         + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                         + Revascularization
                         + ML_CAC_Q2 + ML_CAC_Q3 + ML_CAC_Q4, data = data)
summary(fit_model3_Q234) 
fit_model3_quartiles <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                + age + gender_female 
                                + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                                + Revascularization
                                + ML_CAC_quartiles, data = data)
summary(fit_model3_quartiles) 




#Cox proportional hazards regression analysis of CT_CAC score in different models:
#For the primary end point:
#Crude model
fit_crude_234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                         + CT_CAC_2 + CT_CAC_3 + CT_CAC_4, data = data)
summary(fit_crude_234) 
fit_crude_categories <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                + CT_CAC_categories, data = data)
summary(fit_crude_categories) 
#Model1
fit_model1_234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                          + age + gender_female
                        + CT_CAC_2 + CT_CAC_3 + CT_CAC_4, data = data)
summary(fit_model1_234) 
fit_model1_categories <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                 + age + gender_female 
                               + CT_CAC_categories, data = data)
summary(fit_model1_categories) 
#Model2
fit_model2_234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                          + age + gender_female
                        + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                        + CT_CAC_2 + CT_CAC_3 + CT_CAC_4, data = data)
summary(fit_model2_234) 
fit_model2_categories <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                 + age + gender_female
                               + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                               + CT_CAC_categories, data = data)
summary(fit_model2_categories) 
#Model3
fit_model3_234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                          + age + gender_female
                        + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                        + Revascularization
                        + CT_CAC_2 + CT_CAC_3 + CT_CAC_4, data = data)
summary(fit_model3_234) 
fit_model3_categories <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                 + age + gender_female 
                               + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
                               + Revascularization
                               + CT_CAC_categories, data = data)
summary(fit_model3_categories) 




#For the second primary end point, all the codes remain unchanged except for the time and status. 




