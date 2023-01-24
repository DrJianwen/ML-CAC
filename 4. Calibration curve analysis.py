# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 13:19:42 2023

@author: Admin
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
#Data preprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data=pd.read_excel("D:\data.xlsx")
features = data.drop(columns=['ID', 'CT_CAC_score', 'CT_CAC_above400'])
target = data['CT_CAC_above400']
train_features, test_features, train_target, test_target = train_test_split(
    features, target, 
    test_size = 0.2, random_state = 42, stratify=target)
train_features['diabetes_duration'].fillna(train_features[train_features['diabetes_duration']!=0]['diabetes_duration'].median(),inplace=True)
train_features['hypertension_duration'].fillna(train_features[train_features['hypertension_duration']!=0]['hypertension_duration'].median(),inplace=True)
test_features['diabetes_duration'].fillna(train_features[train_features['diabetes_duration']!=0]['diabetes_duration'].median(),inplace=True)
test_features['hypertension_duration'].fillna(train_features[train_features['hypertension_duration']!=0]['hypertension_duration'].median(),inplace=True)
train_features.fillna(train_features.median(),inplace=True)
test_features.fillna(train_features.median(),inplace=True)
list_numerical = features.drop(['gender_female',  'history_CVD', 'diabetes', 'hypertension'], axis=1).columns
transfer = StandardScaler().fit(train_features[list_numerical]) 
train_features[list_numerical] = transfer.transform(train_features[list_numerical])
test_features[list_numerical] = transfer.transform(test_features[list_numerical])

#ML-models
svc_linear =  SVC(kernel = "linear", C=0.19397, probability=True, class_weight='balanced', random_state=42)
log =  LogisticRegression(C=0.35144,solver='newton-cg', class_weight='balanced',random_state=42)  
rfc_clf =  RandomForestClassifier(class_weight='balanced', n_estimators=1910, max_features=5,max_depth=8,
                                 min_samples_split=88,min_samples_leaf=28,max_leaf_nodes=44,
                                oob_score=True, random_state=42, n_jobs=-1)
xgb=xgboost.XGBClassifier (booster='gbtree',objective='binary:logistic',random_state=42, n_jobs=-1, n_estimators=370,
                           learning_rate=0.00236, max_depth=5, gamma=1.9, min_child_weight=4,subsample=0.5,
                           colsample_bytree=0.7, scale_pos_weight=1.5)
mlp_clf = MLPClassifier(hidden_layer_sizes=(34,36,38),alpha=1.54351, learning_rate_init=0.00023,random_state = 42)
clf_list = [
    (log, "LR",'orange'), (svc_linear, "SVM",'red'),(rfc_clf, "RF",'purple'),(xgb, "XGB",'Royalblue'),(mlp_clf, "MLP",'seagreen')]

#Plot
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(7, 2)
ax_calibration_curve = fig.add_subplot(gs[:4, :2])
calibration_displays = {}
for i, (clf, name,color) in enumerate(clf_list):
    clf.fit(train_features, train_target)
    display = CalibrationDisplay.from_estimator(
        clf,
        test_features,
        test_target,
        n_bins=5,
        name=name,
        ax=ax_calibration_curve,
        color=color,
    )
    calibration_displays[name] = display
ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration curve (test set)")
ax_calibration_curve.set_xlabel('Predicted probability')
ax_calibration_curve.set_ylabel('Fraction of positives')
# Add histogram
grid_positions = [(4, 0), (4, 1), (5, 0), (5, 1), (6, 0)]
for i, (_, name,colors) in enumerate(clf_list):
    row, col = grid_positions[i]
    ax = fig.add_subplot(gs[row, col])

    ax.hist(
        calibration_displays[name].y_prob,
        range=(0, 1),
        bins=5,
        label=name,
        color=colors,
    )
    ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")
plt.tight_layout()
plt.show()
