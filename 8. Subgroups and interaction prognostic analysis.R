library(xlsx)
library(survival)
library(survIDINRI)
#Import data of prognostic analysis, this part of original data are available from the corresponding author on reasonable request.
data<-read.xlsx("D:/data3.xlsx",1)


#Subgroup interaction P value:
fit1 <-coxph(Surv(primary_end_point_time, primary_end_point) ~
        + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
      + ML_CAC_quartiles + Age_above60*ML_CAC_quartiles, data = data)
summary(fit1) # Age_above60

fit2 <-coxph(Surv(primary_end_point_time, primary_end_point) ~
               + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
             + ML_CAC_quartiles + gender_female*ML_CAC_quartiles, data = data)
summary(fit2) # gender

fit3 <-coxph(Surv(primary_end_point_time, primary_end_point) ~
               + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
             + ML_CAC_quartiles + Revascularization*ML_CAC_quartiles, data = data)
summary(fit3) # Revascularization

fit4 <-coxph(Surv(primary_end_point_time, primary_end_point) ~
               + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
             + ML_CAC_quartiles + BMI_above25*ML_CAC_quartiles, data = data)
summary(fit4) # BMI_above25

fit5 <-coxph(Surv(primary_end_point_time, primary_end_point) ~
               + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
             + ML_CAC_quartiles + smoker*ML_CAC_quartiles, data = data)
summary(fit5) # smoker

fit6 <-coxph(Surv(primary_end_point_time, primary_end_point) ~
               + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
             + ML_CAC_quartiles + hypertension*ML_CAC_quartiles, data = data)
summary(fit6) # hypertension

fit7 <-coxph(Surv(primary_end_point_time, primary_end_point) ~
               + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
             + ML_CAC_quartiles + diabetes*ML_CAC_quartiles, data = data)
summary(fit7) # diabetes

fit8 <-coxph(Surv(primary_end_point_time, primary_end_point) ~
               + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
             + ML_CAC_quartiles + hyperlipidemia*ML_CAC_quartiles, data = data)
summary(fit8) # hyperlipidemia



#adjusted HR for trend and 95%CI of ML-CAC:
#Here, an examle of Age_groups is shown
newdata1 <- data[data$Age_above60==1,]
fit9 <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                 + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
               + ML_CAC_quartiles, data = newdata1)
summary(fit9)#HR for Age_above60

newdata2 <- data[data$Age_above60==0,]
fit10 <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                + age + gender_female + smoker + hypertension + diabetes + SBP + Glu + LDLC + eGFR
              + ML_CAC_quartiles, data = newdata2)
summary(fit10)#HR for Age_below60



#Forest plot of ML-CAC for subgroup prognostic analysis
library(forestplot)
result<-read.xlsx("E:/forestplot1.xlsx",1,header = FALSE) #import the sorted data  
fig <- forestplot(result[,c(1:2,6:7)],           
                  mean=result[,3],
                  lower=result[,4],
                  upper=result[,5],
                  zero=1,                        
                  boxsize=0.3,
                  graph.pos = 4,
                  hrzl_lines=list('1'=gpar(lty=1,lwd=2),
                                  '26'=gpar(lwd=2,lty=1)),       
                  graphwidth=unit(.25,'npc'),
                  xticks=c(0,2,4,6),                          
                  is.summary=c(F,
                               F,F,F,
                               F,F,F,
                               F,F,F,
                               F,F,F,
                               F,F,F,
                               F,F,F,
                               F,F,F,
                               F,F,F),                       
                  txt_gp=fpTxtGp(label = gpar(cex=1),
                                 ticks = gpar(cex=1.1),
                                 xlab = gpar(cex=1),
                                 title = gpar(cex=2)),
                  lwd.zero=2,
                  lwd.ci=2,
                  lwd.xaxis=1,xlab='Hazard ratio',              
                  lty.ci=1,
                  ci.vertices=T,
                  ci.vertices.height=0.2,
                  clip=c(0,4),
                  ineheight=unit(8,'mm'),
                  line.margin=unit(8,'mm'),
                  colgap=unit(6,'mm'),
                  col=fpColors(zero = '#4D4D4D',
                               box = '#D00088',
                               lines = '#0066FF'),
                  fn.ci_norm='fpDrawCircleCI'
)
fig


