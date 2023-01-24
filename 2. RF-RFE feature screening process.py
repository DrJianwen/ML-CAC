# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 00:59:07 2022

@author: Admin
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
#Import the train data with 54 variables, and this part of original data are available from the corresponding author on reasonable request.
data=pd.read_excel("D:\whole_sorted_train_data.xlsx")
train_features = data.drop(columns=['CT_CAC_above400'])
train_target = data['CT_CAC_above400']


#RF-RFE-CV, 10-fold cross-validation was used to plot the relationship between the numbers of features and mean AUC values. 
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)   
rfc_clf = RandomForestClassifier(random_state = 42,n_jobs=-1)
rfecv = RFECV(
    estimator=rfc_clf,
    step=1,
    cv=cv,
    scoring='roc_auc')
rfecv.fit(train_features, train_target)
rfecv.show()


#RF-RFE:
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
rfc_clf = RandomForestClassifier(random_state = 42,n_jobs=-1)
rfe_20 = RFE(estimator=rfc_clf, n_features_to_select=20, step=1)
rfe_20.fit(train_features, train_target)
rfe_20.ranking_
feature_idx_20 = rfe_20.support_
feature_name =train_features.columns[feature_idx_20]  #20 features selected by RF-RFE

rfe_10 = RFE(estimator=rfc_clf, n_features_to_select=20, step=1)
rfe_10.fit(train_features, train_target)
rfe_10.ranking_
feature_idx_10 = rfe_10.support_
feature_name =train_features.columns[feature_idx_10]  #10 features selected by RF-RFE










