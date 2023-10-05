library(glmnet)
library(xlsx)
#Import the train data with 55 variables, and this part of original data are available from the corresponding author on reasonable request.
data<-read.xlsx("D:/whole_sorted_train_data.xlsx",1)

#LASSO lambda selection and the corresponding mean AUC using 10-fold cross-validation:
cvfit <- cv.glmnet(x=model.matrix(~.,data[,-c(1)]),
                   y=data$CT_CAC_above400,
                   family = "binomial",
                   grouped=FALSE,
                   nfolds = 10,
                   set.seed(42),
                   nlambda = 100,
                   alpha=1,
                   type.measure = c("auc"))
plot(cvfit)
abline(v=log(0.0318), col="black", lty=3) #plot Of 10 features selected


#LASSO coefficient profiles of the potential predictors:
lasso = glmnet(x=model.matrix(~.,data[,-c(1)]),
               y=data$CT_CAC_above400,
               family = "binomial",
               alpha=1)
plot(lasso, xvar = "lambda")
abline(v=log(cvfit$lambda.1se), col="black", lty=3 )
abline(v=log(cvfit$lambda.min), col="black", lty=3 )
abline(v=log(0.0318), col="black", lty=3 )



#two sets of features selected
coef(cvfit, s = "lambda.1se") #lambda.1se selected
coef(cvfit, s = 0.0318) #10 features selected


