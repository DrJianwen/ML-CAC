# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 20:44:26 2023

@author: Admin
"""
#This part of data are publicly available online
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


#Tuning hyperparameters using Optuna 
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler        
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.neural_network import MLPClassifier
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)   
#logistic regression(LR):  
def objective(trial):
    param  = {
        "solver": trial.suggest_categorical("solver", ['newton-cg', 'lbfgs','liblinear','sag','saga']),
        "C" : trial.suggest_loguniform('C', 1e-5, 1)}               
    cv_scores = []    
    for idx, (train_idx, test_idx) in enumerate(cv.split(train_features, train_target)):          
        X_train, X_test = train_features.iloc[train_idx], train_features.iloc[test_idx]
        y_train, y_test = train_target.iloc[train_idx], train_target.iloc[test_idx]
        model = LogisticRegression(class_weight='balanced', random_state=42, n_jobs=-1, **param)
        model.fit(X_train,y_train)               
        test_proba = model.predict_proba(X_test)
        cv_scores.append(roc_auc_score(y_test,test_proba[:,1]))
    return np.mean(cv_scores)       
study = optuna.create_study(sampler=TPESampler(), direction="maximize")
study.optimize(objective, n_trials=100)     
print("Best trial:", study.best_trial)

#linear support vector machine(SVM):
def objective(trial):
    param  = {
        "C" : trial.suggest_loguniform('C', 1e-5, 1)}              
    cv_scores = []   
    for idx, (train_idx, test_idx) in enumerate(cv.split(train_features, train_target)):          
        X_train, X_test = train_features.iloc[train_idx], train_features.iloc[test_idx]
        y_train, y_test = train_target.iloc[train_idx], train_target.iloc[test_idx]
        model =  SVC(class_weight='balanced',kernel = "linear",probability=True,random_state=42, **param)
        model.fit(X_train,y_train)             
        test_proba = model.predict_proba(X_test)
        cv_scores.append(roc_auc_score(y_test,test_proba[:,1]))
    return np.mean(cv_scores)        
study = optuna.create_study(sampler=TPESampler(), direction="maximize")
study.optimize(objective, n_trials=100)    
print("Best trial:", study.best_trial)

#random forest(RF):
def objective(trial):
    param  = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=10),
        "max_features": trial.suggest_int("max_features", 5, 15, step=1),
        "max_depth": trial.suggest_int("max_depth", 5, 15, step=1),
        "min_samples_split": trial.suggest_int("min_samples_split", 30, 100, step=2),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50, step=2),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 20, 50, step=2)}   
    cv_scores = [] 
    for idx, (train_idx, test_idx) in enumerate(cv.split(train_features, train_target)):          
        X_train, X_test = train_features.iloc[train_idx], train_features.iloc[test_idx]
        y_train, y_test = train_target.iloc[train_idx], train_target.iloc[test_idx]
        model = RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1, class_weight='balanced', **param)
        model.fit(X_train,y_train)              
        test_proba = model.predict_proba(X_test)
        cv_scores.append(roc_auc_score(y_test,test_proba[:,1]))
    return np.mean(cv_scores)
study = optuna.create_study(sampler=TPESampler(), direction="maximize")
study.optimize(objective, n_trials=100) 
print("Best trial:", study.best_trial)

#extreme gradient boosting (XGBoost) (decision trees as base learner):
import warnings
def objective(trial):
    param  = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000, step=10),
        "learning_rate" : trial.suggest_loguniform('learning_rate_rate', 1e-5, 1),
        "max_depth": trial.suggest_int("max_depth", 5, 15, step=1),       
        "gamma": trial.suggest_float("gamma", 0, 5, step=0.1),        
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5, step=1),          
        "subsample": trial.suggest_float("subsample", 0.5, 1, step=0.1),         
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, step=0.1),     
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 3, step=0.1)}     
    cv_scores = []    
    for idx, (train_idx, test_idx) in enumerate(cv.split(train_features, train_target)):          
        X_train, X_test = train_features.iloc[train_idx], train_features.iloc[test_idx]
        y_train, y_test = train_target.iloc[train_idx], train_target.iloc[test_idx]
        model = xgboost.XGBClassifier (booster='gbtree',objective='binary:logistic',random_state=42, n_jobs=-1, **param)
        model.fit(X_train,y_train)               
        test_proba = model.predict_proba(X_test)
        cv_scores.append(roc_auc_score(y_test,test_proba[:,1]))
    return np.mean(cv_scores)        
study = optuna.create_study(sampler=TPESampler(), direction="maximize")
study.optimize(objective, n_trials=100)    
print("Best trial:", study.best_trial)

#multilayer perceptron (MLP):
def objective(trial):
    param= {
        "alpha": trial.suggest_loguniform('alpha', 1e-2, 10), 
        "learning_rate_init" : trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-2)} 
    n_layers = trial.suggest_int('n_layers', 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_{i}', 2, 40, step=2))       
    cv_scores = []   
    for idx, (train_idx, test_idx) in enumerate(cv.split(train_features, train_target)):          
        X_train, X_test = train_features.iloc[train_idx], train_features.iloc[test_idx]
        y_train, y_test = train_target.iloc[train_idx], train_target.iloc[test_idx]      
        model = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=tuple(layers), random_state=42, **param)
        model.fit(X_train,y_train)              
        test_proba = model.predict_proba(X_test)
        cv_scores.append(roc_auc_score(y_test,test_proba[:,1]))
    return np.mean(cv_scores)        
study = optuna.create_study(sampler=TPESampler(), direction="maximize")
study.optimize(objective, n_trials=100)      
print("Best trial:", study.best_trial)    


#The final ML models used in this study after tuning hyperparameters
log = LogisticRegression(C=0.35144,solver='newton-cg', class_weight='balanced',random_state=42)  

svc_linear = SVC(kernel = "linear", C=0.19397, probability=True, class_weight='balanced', random_state=42)

rfc_clf = RandomForestClassifier(class_weight='balanced', n_estimators=1910, max_features=5,max_depth=8,
                                 min_samples_split=88,min_samples_leaf=28,max_leaf_nodes=44,
                                oob_score=True, random_state=42, n_jobs=-1)

xgb=xgboost.XGBClassifier (booster='gbtree',objective='binary:logistic',random_state=42, n_jobs=-1, n_estimators=370,
                           learning_rate=0.00236, max_depth=5, gamma=1.9, min_child_weight=4,subsample=0.5,
                           colsample_bytree=0.7, scale_pos_weight=1.5)

mlp_clf = MLPClassifier(hidden_layer_sizes=(34,36,38),alpha=1.54351, learning_rate_init=0.00023,random_state = 42)



# Calculate mean AUC, F1-score, Balanced accuracy, Brier_score and their SE on 5 repetitions of 10-fold stratified cross-validation in the train set.
from scipy.stats import sem
from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
model = log  #Here, an examle of Logistic Regression is shown
def evaluate_model_auc(X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
	scores_auc = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
	return scores_auc
def evaluate_model_F1(X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
	scores_F1 = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1)
	return scores_F1
def evaluate_model_balanced_accuracy(X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
	scores_balanced_accuracy = cross_val_score(model, X, y, scoring='balanced_accuracy', cv=cv, n_jobs=-1)
	return scores_balanced_accuracy
def evaluate_model_brier_score(X, y):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
	scores_brier_score = cross_val_score(model, X, y, scoring='neg_brier_score', cv=cv, n_jobs=-1)
	return scores_brier_score
scores_auc = evaluate_model_auc(train_features, train_target)
scores_F1 = evaluate_model_F1(train_features, train_target)
scores_balanced_accuracy = evaluate_model_balanced_accuracy(train_features, train_target)
scores_brier_score = evaluate_model_brier_score(train_features, train_target)
print('AUC(SE): %.3f (%.3f)' % (mean(scores_auc), sem(scores_auc)))
print('F1 score(SE): %.3f (%.3f)' % (mean(scores_F1), sem(scores_F1)))
print('Balanced_accuracy(SE): %.3f (%.3f)' % (mean(scores_balanced_accuracy), sem(scores_balanced_accuracy)))
print('Brier_score(SE): %.3f (%.3f)' % (-mean(scores_brier_score), sem(scores_brier_score)))


# Calculate AUC, F1-score, Balanced accuracy, Brier_score on the test set
model=log     #Here, an examle of logistic regression is shown
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.metrics import *
from sklearn.metrics import brier_score_loss as neg_brier_score
def make_data(centers=3, cluster_std=[1.0, 3.0, 2.5], n_samples=150, n_features=2):
    X, y = make_blobs(n_samples, n_features)
    return X, y
def sensitivity(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return tp/(tp+fn)
def specificity(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return tn/(tn+fp)
def bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    auc_values = []
    for b in tqdm(range(nsamples)):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5))
def bootstrap_specifcity(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    values = []
    for b in tqdm(range(nsamples)):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict(X_test)
        value = specificity(y_test.ravel(), pred.ravel())
        values.append(value)
    return np.percentile(values, (2.5, 97.5))
def bootstrap_sensitivity(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    values = []
    for b in tqdm(range(nsamples)):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict(X_test)
        value = sensitivity(y_test.ravel(), pred.ravel())
        values.append(value)
    return np.percentile(values, (2.5, 97.5))
def bootstrap_balanced(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    values = []
    for b in tqdm(range(nsamples)):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict(X_test)
        value = balanced_accuracy_score(y_test.ravel(), pred.ravel())
        values.append(value)
    return np.percentile(values, (2.5, 97.5))
def bootstrap_f1_score(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    values = []
    for b in tqdm(range(nsamples)):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict(X_test)
        value = f1_score(y_test.ravel(), pred.ravel())
        values.append(value)
    return np.percentile(values, (2.5, 97.5))
def bootstrap_neg_brier_score(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    values = []
    for b in tqdm(range(nsamples)):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        value = neg_brier_score(y_test.ravel(), pred.ravel())
        values.append(value)
    return np.percentile(values, (2.5, 97.5))
if __name__ == "__main__":
    X_train=train_features.to_numpy()
    X_test=test_features.to_numpy()
    y_train=train_target.to_numpy()
    y_test=test_target.to_numpy()   
    lr_l1 = model
    lr_l1.fit(X_train, y_train)
    y_pred = lr_l1.predict(X_test)
    y_pred_score = lr_l1.predict_proba(X_test)    
    iter=10
    print('AUC:',roc_auc_score(y_test, y_pred_score[:,1]))
    print('AUC 95%CI:', bootstrap_auc(lr_l1, X_train, y_train, X_test, y_test,iter))  
    print('F1 score:',f1_score(y_test, y_pred))
    print('F1 score 95%CI:',bootstrap_f1_score(lr_l1, X_train, y_train, X_test, y_test,iter))   
    print('Balanced accuracy:',balanced_accuracy_score(y_test, y_pred))
    print('Balanced accuracy 95%CI:',bootstrap_balanced(lr_l1, X_train, y_train, X_test, y_test,iter))  
    print('Brier_score: ',neg_brier_score(y_test, y_pred_score[:,1]))
    print('Brier_score 95%CI:', bootstrap_neg_brier_score(lr_l1, X_train, y_train, X_test, y_test,iter))
    
    