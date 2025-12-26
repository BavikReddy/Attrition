#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shap
import warnings
import json
import argparse
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import lightgbm
from sklearn.feature_selection import VarianceThreshold
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif
from sklearn.feature_selection import RFE
from sklearn import svm
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime, date, timedelta
# from sqlalchemy.engine import create_engine
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load
from sklearn.metrics import accuracy_score,precision_score, recall_score,f1_score,classification_report,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier


# In[2]:


def feature_selection(x_train_2021,y_train_2021):

    # to avoid errors in the code further
    x_test_2021 = x_train_2021.copy()
    
    #REMOVE constant features
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(x_train_2021)
    constant_list = [not temp for temp in constant_filter.get_support()]
    print("Features with constant value (Zero Variance)- {0}:\n {1}".format(len(x_train_2021.columns[constant_list]),x_train_2021.columns[constant_list]))

    for col in x_train_2021.columns[constant_list]:
        print(col, ':',x_train_2021[col].unique())
    x_train_2021.drop(x_train_2021.columns[constant_list],axis=1,inplace=True)
    
#     scaler = StandardScaler()
#     x_train_2021.iloc[:,:] = scaler.fit_transform(x_train_2021.loc[:,:])
    
    # number of desired features
    num_feats=150
    
    #lasso regression
    lasso_selector = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear', class_weight='balanced'))
    lasso_selector.fit(x_train_2021, y_train_2021)
    lasso_support = lasso_selector.get_support()
    lasso_feature = x_train_2021.loc[:,lasso_support].columns.tolist()
    print(str(len(lasso_feature)), 'selected features')
    
    #select from model logistic regression
    lr_selector = SelectFromModel(LogisticRegression(C=0.01, solver='lbfgs', max_iter=1e5, class_weight='balanced'))
    lr_selector.fit(x_train_2021, y_train_2021)
    lr_support = lr_selector.get_support()
    lr_feature = x_train_2021.loc[:,lr_support].columns.tolist()
    print(str(len(lr_feature)), 'selected features')
    
     #select from model lightgbm
    lgbm_selector = SelectFromModel(lightgbm.LGBMClassifier())
    lgbm_selector.fit(x_train_2021, y_train_2021)
    lgbm_support = lgbm_selector.get_support()
    lgbm_feature = x_train_2021.loc[:,lgbm_support].columns.tolist()
    print(str(len(lgbm_feature)), 'selected features')
    
    #select from model SVM
    svm_selector = SelectFromModel(svm.LinearSVC(random_state=100, tol=1e-5, verbose=1) )
    svm_selector.fit(x_train_2021, y_train_2021)
    svm_support = svm_selector.get_support()
    svm_feature = x_train_2021.loc[:,svm_support].columns.tolist()
    print(str(len(svm_feature)), 'selected features')
    
    #Mutual information gain
    mutual_info=mutual_info_classif(x_train_2021,y_train_2021)
    mutual_data=pd.Series(mutual_info,index=x_train_2021.columns)
    mutual_data=mutual_data.sort_values(ascending=False)
    mutual_feature=mutual_data.index[:num_feats]
    mutual_feature=list(mutual_feature)
    feature_name=x_train_2021.columns
    mutual_support = [True if i in mutual_feature else False for i in feature_name]
    
     #select kbest - f_calssif
    kbest_selector = SelectKBest(f_classif, k=num_feats)
    kbest_selector.fit(x_train_2021, y_train_2021)
    kbest_support = kbest_selector.get_support()
    kbest_feature = x_train_2021.loc[:,kbest_support].columns.tolist()
    print(str(len(kbest_feature)), 'selected features')
    
     #select kbest - mutual_info_classif
    kbest_selector2 = SelectKBest(mutual_info_classif, k=num_feats)
    kbest_selector2.fit(x_train_2021, y_train_2021)
    kbest_support2 = kbest_selector2.get_support()
    kbest_feature2 = x_train_2021.loc[:,kbest_support2].columns.tolist()
    print(str(len(kbest_feature2)), 'selected features')
    
    #recursive feature elimination-LogisticRegression
    rfe_selector_L = RFE(estimator=LogisticRegression(C=0.01, solver='lbfgs', max_iter=1e5, class_weight='balanced'), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector_L.fit(x_train_2021, y_train_2021)
    rfe_support_L = rfe_selector_L.get_support()
    rfe_feature_L = x_train_2021.loc[:,rfe_support_L].columns.tolist()
    print(str(len(rfe_feature_L)), 'selected features')
    
    #recrsive feature elimination-Lightgbm
    rfe_selector_Lgbm = RFE(estimator=lightgbm.LGBMClassifier(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector_Lgbm.fit(x_train_2021, y_train_2021)
    rfe_support_Lgbm = rfe_selector_Lgbm.get_support()
    rfe_feature_Lgbm = x_train_2021.loc[:,rfe_support_Lgbm].columns.tolist()
    print(str(len(rfe_feature_Lgbm)), 'selected features')
    feature_name=x_train_2021.columns
    
     #recrsive feature elimination-SVM
    rfe_selector_svm = RFE(estimator=svm.LinearSVC(random_state=100, tol=1e-5), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector_svm.fit(x_train_2021, y_train_2021)
    rfe_support_svm = rfe_selector_svm.get_support()
    rfe_feature_svm = x_train_2021.loc[:,rfe_support_svm].columns.tolist()
    print(str(len(rfe_feature_svm)), 'selected features')
    
    # put all feature selection together
    feature_selection_df = pd.DataFrame({'Feature':feature_name,'lasso_select_from_model':lasso_support,'logistic_select_from_model':lr_support,                                        'lgbm_select_from_model':lgbm_support,'svm_select_from_model':svm_support,'mutual_info':mutual_support,'kbest':kbest_support,'rfe_logistic':rfe_support_L,'rfe_lgbm':rfe_support_Lgbm,'rfe_svm':rfe_support_svm})
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    
    filename_featureselection="feature_selection_expresscheck.csv"
    featureselection_path=os.path.join(base_dir,"feature_selection_expresscheck",filename_featureselection)
    print("featureselection_path:",featureselection_path)
    # feature_selection_df.to_excel(featureselection_path)
    feature_selection_df.to_csv(featureselection_path, header=True, index=False)
    
    return feature_selection_df


# In[3]:


def model_selection(feature_selection_df,x_train,y_train,x_test,y_test):
    all_features = feature_selection_df
    
    col=all_features[all_features.eq('product_type_most_frequent_FUEL_ONLY').any(1)]
    all_features.drop(col.index,axis=0,inplace=True)
    
    lasso=all_features[all_features['lasso_select_from_model']]['Feature'].to_list()
    logistic=all_features[all_features['logistic_select_from_model']]['Feature'].to_list()
    lgbm=all_features[all_features['lgbm_select_from_model']]['Feature'].to_list()
    svm=all_features[all_features['svm_select_from_model']]['Feature'].to_list()
    mutual_info=all_features[all_features['mutual_info']]['Feature'].to_list()
    kbest=all_features[all_features['kbest']]['Feature'].to_list()
    rfe_logistic=all_features[all_features['rfe_logistic']]['Feature'].to_list()
    rfe_lgbm=all_features[all_features['rfe_lgbm']]['Feature'].to_list()
    rfe_svm=all_features[all_features['rfe_svm']]['Feature'].to_list()  
    total_7=all_features[all_features['Total']>=7]['Feature'].to_list()
    total_6=all_features[all_features['Total']>=6]['Feature'].to_list()
    total_5=all_features[all_features['Total']>=5]['Feature'].to_list()
    
    featuresets = [
                   (x_train[total_7], x_test[total_7], 'total_7'),
                   (x_train[total_6], x_test[total_6], 'total_6'),
                   (x_train[total_5], x_test[total_5], 'total_5'),
                   (x_train[lasso], x_test[lasso], 'lasso'), 
                   (x_train[logistic], x_test[logistic], 'logistic'), 
                   (x_train[svm], x_test[svm], 'svm'), 
                   (x_train[lgbm], x_test[lgbm], 'lgbm'),
                   (x_train[mutual_info], x_test[mutual_info], 'mutual_info'),
                   (x_train[kbest], x_test[kbest], 'kbest'),
                   (x_train[rfe_logistic], x_test[rfe_logistic],'rfe_logistic'),
                   (x_train[rfe_lgbm], x_test[rfe_lgbm], 'rfe_lgbm'),
                   (x_train[rfe_svm], x_test[rfe_svm], 'rfe_svm')
    ]
    
    models = [
                LogisticRegression(C=0.01, solver='lbfgs', max_iter=1e5, class_weight='balanced'),
                XGBClassifier(booster='gblinear', random_state=100),
                LGBMClassifier( objective='binary', is_unbalance=True, metric='roc_auc',random_state=100),
                LogisticRegression(penalty="l1", solver='liblinear', class_weight='balanced'),
                RandomForestClassifier(n_estimators=100, max_features=1.0, class_weight='balanced',random_state=100),
    ]
    
    model_comparison=pd.DataFrame(columns=['algorithm','Features','accuracy','precision','recall','f1-score'])
    for x_tr, x_tst, n in featuresets:
        for model in models:

            print('\n***************************************\n')
            print('\n----------------{0} - {1}-----------------\n'.format(type(model).__name__, n))
            print("\n Number of columns {0}".format(x_tr.shape))
            print("Train shape",x_tr.shape)
            print("Test shape",x_tst.shape)
            x= x_tr.copy()
            y= x_tst.copy()

#             #Fix the Distribution
#             power = PowerTransformer(method='yeo-johnson', standardize=True)
#             x_tr = power.fit_transform(x_tr)
#             x_tst = power.fit_transform(x_tst)

#             #Scaling
#             scaler = StandardScaler()
#             x_tr = scaler.fit_transform(x_tr)
#             x_tst = scaler.fit_transform(x_tst)

#             x_tr = pd.DataFrame(data=x_tr, columns=x.columns)
#             x_tst = pd.DataFrame(data=y, columns=y.columns)

            model.fit(x_tr, y_train)
            print('\n--------------------predicted results----------------\n')
            y_pred = model.predict(x_tst)
            print('Accuracy score: ', accuracy_score(y_test, y_pred))
            print('Precision: ', precision_score(y_test, y_pred))
            print('Recall: ', recall_score(y_test, y_pred))
            print('f1_score: ', f1_score(y_test, y_pred))
            print('classification report:',classification_report(y_test, y_pred))
            accuracy = accuracy_score(y_test, y_pred)
            precision= precision_score(y_test, y_pred)
            recall=recall_score(y_test, y_pred)
            f1score=f1_score(y_test, y_pred)

            model_comparison=model_comparison.append({'algorithm':type(model).__name__,'Features':n,'accuracy':accuracy,                                                     'precision':precision,'recall':recall,'f1-score':f1score},ignore_index=True)
            model_comparison=model_comparison.sort_values('f1-score',ascending= False)
    
    filename_modelselection="model_selection_expresscheck.csv"
    modelselection_path=os.path.join(base_dir,"model_selection_expresscheck",filename_modelselection)
    print("modelselection_path:",modelselection_path)
    model_comparison.to_csv(modelselection_path, header=True, index=False)
    
    return model_comparison


# In[4]:


def lgbm_hyperopt(x_train,y_train,x_test,y_test):
    
    # using hyperopt for lgbm
    seed=100
    scores ={}
    def objective(params):
        scores ={}
        nl=int(params['num_leaves'])
        mcs=int(params['min_child_samples'])
        md=int(params['max_depth'])
        lr=params['learning_rate']
        ra=params['reg_alpha']
        model=LGBMClassifier(objective='binary', is_unbalance=True, metric='roc_auc',random_state=100, 
                           learning_rate=lr, max_depth=md, min_child_samples=mcs, num_leaves=nl,reg_alpha=ra)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)

        a = classification_report(y_test, y_pred, output_dict=True)
        score = 1-a['1']['f1-score']
        return score

    def optimize(trial):
        params={'num_leaves':hp.uniform('num_leaves',10,50),
               'min_child_samples':hp.uniform('min_child_samples',5,15),
               'max_depth':hp.uniform('max_depth',0,20),
               'learning_rate':hp.uniform('learning_rate',0.01,0.9),
               'reg_alpha':hp.uniform('reg_alpha',0.01,0.9)}
        best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
        return best

    trial=Trials()
    best=optimize(trial)
    
    return best


# In[5]:


def xgb_hyperopt(x_train,y_train,x_test,y_test):
    
    # using hyperopt for xgb
    seed=100
    scores ={}
    def objective(params):
        scores ={}
        lr=params['learning_rate']
        model=XGBClassifier(booster='gblinear', random_state=100,learning_rate=lr)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)

        a = classification_report(y_test, y_pred, output_dict=True)
        score = 1-a['1']['f1-score']
        return score

    def optimize(trial):
        params={'learning_rate':hp.uniform('learning_rate',0.01,0.9)}
        best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
        return best

    trial=Trials()
    best=optimize(trial)
    
    return best


# In[6]:


def randomforest_hyperopt(x_train,y_train,x_test,y_test):
    
    # randomforest using hyperopt
    seed=100
    scores ={}
    n_estimators=[100, 200, 300, 400,500,600,700,800,900]
    criterion=["gini", "entropy"]
    def objective(params):
        scores ={}
        ne=int(params['n_estimators'])
        md=int(params['max_depth'])
        crt=params['criterion']

        model = RandomForestClassifier(class_weight='balanced', random_state=100,
                               n_estimators=ne,max_depth=md, criterion=crt)

        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)

        a = classification_report(y_test, y_pred, output_dict=True)
        score = 1-a['1']['f1-score']
        return score

    def optimize(trial):
        params={'n_estimators':hp.choice('n_estimators',[100, 200, 300, 400,500,600,700,800,900]),
               'max_depth':hp.quniform("max_depth", 1, 15,1),
               "criterion": hp.choice("criterion", ["gini", "entropy"]),}

        best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=100)
        return best

    trial=Trials()
    best=optimize(trial)
    best['n_estimators']=n_estimators[best['n_estimators']]
    best['criterion']=criterion[best['criterion']]
    return best


# In[7]:


def logistic_hyperopt(x_train,y_train,x_test,y_test):
    
    # using hyperopt for logistic regression

    seed=100
    scores ={}
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    def objective(params):
        scores ={}
        c=params['c_value']
        s=params['solvers']
        p=params['penalty']
        model=LogisticRegression(penalty=p,C=c,solver=s,class_weight='balanced')
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)

        a = classification_report(y_test, y_pred, output_dict=True)
        score = 1-a['1']['f1-score']
        return score

    def optimize(trial):

        params={'c_value':hp.choice('c_value',[100, 10, 1.0, 0.1, 0.01]),
               'solvers':hp.choice('solvers', ['newton-cg', 'lbfgs', 'liblinear']),
               'penalty': hp.choice('penalty', ['l2'])}
        best=fmin(fn=objective,space=params,algo=tpe.suggest,trials=trial,max_evals=500,rstate=np.random.default_rng(seed))
        return best

    trial=Trials()
    best=optimize(trial)
    best['c_value']=c_values[best['c_value']]
    best['penalty']=penalty[best['penalty']]
    best['solvers']=solvers[best['solvers']]
    return best


# In[8]:


def hyperparameter_tuning(model_comparison,feature_selection_df,x_train,y_train,x_test,y_test):
    all_features=feature_selection_df
    feature_dict={}
    best_params_df=pd.DataFrame(columns=['algorithm_features','best_params'])
    
    feature_dict['lasso']=all_features[all_features['lasso_select_from_model']]['Feature'].to_list()
    feature_dict['logistic']=all_features[all_features['logistic_select_from_model']]['Feature'].to_list()
    feature_dict['lgbm']=all_features[all_features['lgbm_select_from_model']]['Feature'].to_list()
    feature_dict['svm']=all_features[all_features['svm_select_from_model']]['Feature'].to_list()
    feature_dict['mutual_info']=all_features[all_features['mutual_info']]['Feature'].to_list()
    feature_dict['kbest']=all_features[all_features['kbest']]['Feature'].to_list()
    feature_dict['rfe_logistic']=all_features[all_features['rfe_logistic']]['Feature'].to_list()
    feature_dict['rfe_lgbm']=all_features[all_features['rfe_lgbm']]['Feature'].to_list()
    feature_dict['rfe_svm']=all_features[all_features['rfe_svm']]['Feature'].to_list()
    feature_dict['total_7']=all_features[all_features['Total']>=7]['Feature'].to_list()
    feature_dict['total_6']=all_features[all_features['Total']>=6]['Feature'].to_list()
    feature_dict['total_5']=all_features[all_features['Total']>=5]['Feature'].to_list()

    top_algorithms=model_comparison[['algorithm','Features']][:3]

    for i in range(0,3):
        algorithm=top_algorithms.iloc[i]['algorithm']
        features=top_algorithms.iloc[i]['Features']
        print("algorithm",algorithm)
        print("features",features)
        x_train_new=x_train[feature_dict[features]]
        x_test_new=x_test[feature_dict[features]]

#         #Fix the Distribution
#         power = PowerTransformer(method='yeo-johnson', standardize=True)
#         # power = joblib.load("large-fleet-transformer.sav")
#         x_train_new.iloc[:,:] = power.fit_transform(x_train_new.loc[:,:])
#         x_test_new.iloc[:,:] = power.fit_transform(x_test_new.loc[:,:])


#         #Scaling
#         scaler = StandardScaler()
#         x_train_new.iloc[:,:] = scaler.fit_transform(x_train_new.loc[:,:])
#         x_test_new.iloc[:,:] = scaler.fit_transform(x_test_new.loc[:,:])


        if algorithm=='LGBMClassifier':        
            best=lgbm_hyperopt(x_train_new,y_train,x_test_new,y_test)
        elif algorithm=='LogisticRegression':
            best=logistic_hyperopt(x_train_new,y_train,x_test_new,y_test)
        elif algorithm=='XGBClassifier':
            best=xgb_hyperopt(x_train_new,y_train,x_test_new,y_test)
        else:
            best=randomforest_hyperopt(x_train_new,y_train,x_test_new,y_test)
        key=algorithm+'_'+features
        best_params_df=best_params_df.append({'algorithm_features':key,
                                    'best_params':best},ignore_index=True)
        print("best_params_df",best_params_df)
        
#     filename_hyperparameter_tuning="hyperparameter_tuning"+'_'+filename+".xlsx"
#     hyperparameter_tuning_path=os.path.join('data/expresscheck',filename_hyperparameter_tuning)
#     print("hyperparameter_tuning_path:",hyperparameter_tuning_path)
#     best_params_df.to_excel(hyperparameter_tuning_path)
    feature_dict_path=os.path.join(base_dir,"feature_dict_expresscheck","feature_dict_expresscheck.json")
    print("feature_dict_path:",feature_dict_path)
    with open(feature_dict_path, 'w') as fp:
        json.dump(feature_dict, fp)
    
    # make a copy
    today = date.today()
    archive_file_name = 'feature_dict_expresscheck-' + str(today) + '.json'
    feature_dict_path_archive = os.path.join(base_dir,"feature_dict_expresscheck_archive", archive_file_name)
    print("feature_dict_path:",feature_dict_path_archive)
    with open(feature_dict_path_archive, 'w') as fp:
        json.dump(feature_dict, fp)
        
    return best_params_df, feature_dict

    
    


# In[9]:


def lgbm_predictor(best_param,x_train,y_train,x_test,y_test):
    metrics={}
    lr=round(best_param['learning_rate'],2)
    md=int(round(best_param['max_depth']))
    mcs=int(round(best_param['min_child_samples']))
    nl=int(round(best_param['num_leaves']))
    ra=round(best_param['reg_alpha'],2)
    model=LGBMClassifier(objective='binary', is_unbalance=True, metric='roc_auc',random_state=100, 
                           learning_rate=lr, max_depth=md, min_child_samples=mcs, num_leaves=nl,reg_alpha=ra)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    metrics['accuracy']=accuracy_score(y_test, y_pred)
    metrics['precision']=precision_score(y_test, y_pred)
    metrics['recall']=recall_score(y_test, y_pred)
    metrics['f1_score']=f1_score(y_test, y_pred)
#     metrics['classification_report']=classification_report(y_test, y_pred)
    
    print('Accuracy score on testing set: ', accuracy_score(y_test, y_pred))
    print('Precision on testing set: ', precision_score(y_test, y_pred))
    print('Recall on testing set: ', recall_score(y_test, y_pred))
    print('f1_score on testing set: ', f1_score(y_test, y_pred))
#     print("classification report:",classification_report(y_test, y_pred))
    return metrics


# In[10]:


def xgb_predictor(best_param,x_train,y_train,x_test,y_test):
    metrics={}
    lr=round(best_param['learning_rate'],2)
    model=XGBClassifier(booster='gblinear', random_state=100,learning_rate=lr)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    metrics['accuracy']=accuracy_score(y_test, y_pred)
    metrics['precision']=precision_score(y_test, y_pred)
    metrics['recall']=recall_score(y_test, y_pred)
    metrics['f1_score']=f1_score(y_test, y_pred)
#     metrics['classification_report']=classification_report(y_test, y_pred)
    
    print('Accuracy score on testing set: ', accuracy_score(y_test, y_pred))
    print('Precision on testing set: ', precision_score(y_test, y_pred))
    print('Recall on testing set: ', recall_score(y_test, y_pred))
    print('f1_score on testing set: ', f1_score(y_test, y_pred))
#     print("classification report:",classification_report(y_test, y_pred))
    return metrics


# In[11]:


def random_predictor(best_param,x_train,y_train,x_test,y_test):
    metrics={}

    ne=int(best_param['n_estimators'])
    md=int(round(best_param['max_depth']))
    crt=best_param['criterion']

    model = RandomForestClassifier(class_weight='balanced', random_state=100,
                           n_estimators=ne,max_depth=md, criterion=crt)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    metrics['accuracy']=accuracy_score(y_test, y_pred)
    metrics['precision']=precision_score(y_test, y_pred)
    metrics['recall']=recall_score(y_test, y_pred)
    metrics['f1_score']=f1_score(y_test, y_pred)
#     metrics['classification_report']=classification_report(y_test, y_pred)
    
    print('Accuracy score on testing set: ', accuracy_score(y_test, y_pred))
    print('Precision on testing set: ', precision_score(y_test, y_pred))
    print('Recall on testing set: ', recall_score(y_test, y_pred))
    print('f1_score on testing set: ', f1_score(y_test, y_pred))
#     print("classification report:",classification_report(y_test, y_pred))
    return metrics


# In[12]:


def logistic_predictor(best_param,x_train,y_train,x_test,y_test):
    metrics={}

    c=best_param['c_value']
    s=best_param['solvers']
    p=best_param['penalty']
    model=LogisticRegression(penalty=p,C=c,solver=s,class_weight='balanced')
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    metrics['accuracy']=accuracy_score(y_test, y_pred)
    metrics['precision']=precision_score(y_test, y_pred)
    metrics['recall']=recall_score(y_test, y_pred)
    metrics['f1_score']=f1_score(y_test, y_pred)
#     metrics['classification_report']=classification_report(y_test, y_pred)
    
    print('Accuracy score on testing set: ', accuracy_score(y_test, y_pred))
    print('Precision on testing set: ', precision_score(y_test, y_pred))
    print('Recall on testing set: ', recall_score(y_test, y_pred))
    print('f1_score on testing set: ', f1_score(y_test, y_pred))
#     print("classification report:",classification_report(y_test, y_pred))
    return metrics


# In[13]:


def evaluation(best_params_df,feature_dict,x_train,y_train,x_test,y_test):
    
    for i in range(0,len(best_params_df)):
        algo_feat=best_params_df.iloc[i]['algorithm_features']
        best_param=best_params_df.iloc[i]['best_params']
        algorithm=algo_feat.split('_',1)[0]
        feature=algo_feat.split('_',1)[1]
        
        req_features=feature_dict[feature]
        x_train_new=x_train[req_features]
        x_test_new=x_test[req_features]
        
#         #Fix the Distribution
#         power = PowerTransformer(method='yeo-johnson', standardize=True)
#         # power = joblib.load("large-fleet-transformer.sav")
#         x_train_new.iloc[:,:] = power.fit_transform(x_train_new.loc[:,:])
#         x_test_new.iloc[:,:] = power.fit_transform(x_test_new.loc[:,:])


#         #Scaling
#         scaler = StandardScaler()
#         x_train_new.iloc[:,:] = scaler.fit_transform(x_train_new.loc[:,:])
#         x_test_new.iloc[:,:] = scaler.fit_transform(x_test_new.loc[:,:])
        
        if algorithm=='LGBMClassifier':        
            metrics=lgbm_predictor(best_param,x_train_new,y_train,x_test_new,y_test)
        elif algorithm=='LogisticRegression':
            metrics=logistic_predictor(best_param,x_train_new,y_train,x_test_new,y_test)
        elif algorithm=='XGBClassifier':
            metrics=xgb_predictor(best_param,x_train_new,y_train,x_test_new,y_test)
        else:
            metrics=random_predictor(best_param,x_train_new,y_train,x_test_new,y_test)
        if i==0:
            metrics_df=pd.DataFrame([metrics])
        else:
            metrics_df=metrics_df.append(metrics,ignore_index=True)
    
    best_df=pd.concat([best_params_df,metrics_df],axis=1)
    best_df=best_df.sort_values('f1_score',ascending= False)
    filename_hyperparameter_tuning="hyperparameter_tuning_expresscheck.csv"
    hyperparameter_tuning_path=os.path.join(base_dir,"hyperparameter_tuning_expresscheck",filename_hyperparameter_tuning)
    print("hyperparameter_tuning_path:",hyperparameter_tuning_path)
    best_df.to_csv(hyperparameter_tuning_path, header=True, index=False)
    
    # make a copy
    today = date.today()
    archive_file_name = 'hyperparameter_tuning_expresscheck-' + str(today) + '.csv'
    hyperparameter_tuning_path_archive = os.path.join(base_dir,"hyperparameter_tuning_expresscheck_archive", archive_file_name)
    print("hyperparameter_tuning_path:",hyperparameter_tuning_path_archive)
    best_df.to_csv(hyperparameter_tuning_path_archive, header=True, index=False)
    
    return best_df


# In[14]:


if __name__=='__main__':
    
    #Train data
    
    #Load full_df data
    base_dir = "/opt/ml/processing"
    full_df_path = f"{base_dir}/train/train.csv"
    full_df = pd.read_csv(full_df_path)
    
    # full_df = pd.read_csv("data/all_data_new.csv")
    full_df[full_df.select_dtypes(exclude='object').columns] = full_df[full_df.select_dtypes(exclude='object').columns].fillna(0.0)
    print('full feature data generated: {0} rows, {1} columns'.format(len(full_df), len(full_df.columns)))

    #Select only express-fleet customers
    full_df_express=full_df[full_df['customer_size'].isin(['others'])]

#     #Find max week in last_transaction_week
#     max_week_train=full_df_express['last_transaction_week'].max()
#     #Subtract 2 weeks from max_week
#     start = datetime.strptime(str(max_week_train) + '0', '%Y%W%w')
#     end = start - timedelta(weeks=2)
#     run_week_train=end.strftime('%Y%W')
#     print("run_week_train",run_week_train)

#     #Find count of attrited records
#     full_df['churn_flag']=np.where(full_df['last_transaction_week']>int(run_week_train),0,1)

    #Remove unwanted columns
    full_df_express.drop(['parentname','last_transaction_week',
     'last_transaction_month','customer_size','customer_segment','first_trx_month'],axis=1,inplace=True)

    # encoding with both only nominal (One Hot encoding)
    CATEGORICAL_COLS=[ 'state_code', 'channel', 'product_type_most_frequent',#'last_payment_method',
                      'num_of_trucks','billing_freq','credit_grade' ]
    full_df_express=pd.concat([full_df_express.drop(CATEGORICAL_COLS, axis=1), pd.get_dummies(full_df_express[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)], axis=1)
    print("shape of train data:",full_df_express.shape)

    # split to x and y
    y_train=full_df_express['churn_flag']
    full_df_express.drop(['churn_flag','credit_limit'],axis=1,inplace=True)
    x_train=full_df_express.copy()

    #Test data

    full_df_test_path = f"{base_dir}/test/test.csv"
    full_df_test = pd.read_csv(full_df_test_path)

    print('Predicting Attrition Probability of express check customers')
    #Getting only the express fleet customers
    full_df_ex_test=full_df_test[full_df_test['customer_size'].isin(['others'])]
    full_df_ex_test=full_df_ex_test.reset_index()
    full_df_ex_test_copy=full_df_ex_test.copy()
    print("full_df_ex_test shape:",full_df_ex_test.shape)

    # #Find max week in last_transaction_week
    # max_week_test=full_df_ex_test['last_transaction_week'].max()
    # #Subtract 2 weeks from max_week_test
    # start = datetime.strptime(str(max_week_test) + '0', '%Y%W%w')
    # end = start - timedelta(weeks=2)
    # run_week_test=end.strftime('%Y%W')
    # print("run_week_test",run_week_test)
    
    #Remove unwanted columns
    full_df_ex_test.drop(['parentname','last_transaction_week',
     'last_transaction_month','customer_size','customer_segment','first_trx_month'],axis=1,inplace=True)
    
    #encoding
    CATEGORICAL_COLS=[ 'state_code', 'num_of_trucks', 'billing_freq',
        'credit_grade', 'channel', 'product_type_most_frequent' #,'last_payment_method'
                     ]

    full_df_ex_test=pd.concat([full_df_ex_test.drop(CATEGORICAL_COLS, axis=1), pd.get_dummies(full_df_ex_test[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)], axis=1)

    print("full_df_ex_test shape:",full_df_ex_test.shape)
    y_test=full_df_ex_test['churn_flag']
    x_test=full_df_ex_test.copy()
    x_test.drop('churn_flag',axis=1,inplace=True)
    
#     #Fix the Distribution
#     power = PowerTransformer(method='yeo-johnson', standardize=True)
#     # power = joblib.load("large-fleet-transformer.sav")
#     x_train.iloc[:,:] = power.fit_transform(x_train.loc[:,:])
#     x_test.iloc[:,:] = power.fit_transform(x_test.loc[:,:])


#     #Scaling
#     scaler = StandardScaler()
#     x_train.iloc[:,:] = scaler.fit_transform(x_train.loc[:,:])
#     x_test.iloc[:,:] = scaler.fit_transform(x_test.loc[:,:])

    #Generate filename for all steps

    # today_date=date.today()
    #filename=str(today_date)
    
    #fix missing columns
    
    x_train_columns=set(x_train.columns)
    x_test_columns=set(x_test.columns)
    missing_columns=list(x_train_columns-x_test_columns)
    for col in missing_columns:
        x_test[col] = 0.0
    print('{0} missing columns were added: {1}'.format(len(missing_columns), missing_columns))
    assert all(c in x_test.columns for c in x_train_columns)
    
    # x_train_columns=set(x_train.columns)
    # missing_columns = []
    # for c in x_train_columns:
    #     if c not in x_test.columns.values:
    #         missing_columns.append(c)
    # for col in missing_columns:
    #     x_test[col] = 0.0
    # print('{0} missing columns were added: {1}'.format(len(missing_columns), missing_columns))
    # assert all(c in x_test.columns for c in x_train_columns)
    
    #Get Feature selection result    
    feature_selection_df=feature_selection(x_train,y_train)
    
    #Get Model selection result
    model_comparison = model_selection(feature_selection_df,x_train,y_train,x_test,y_test)
    
    #Get best hyperparameters
    best_params_df,feature_dict=hyperparameter_tuning(model_comparison,feature_selection_df,x_train,y_train,x_test,y_test)
    
    #Get best metrics
    best_df=evaluation(best_params_df,feature_dict,x_train,y_train,x_test,y_test)
    
    # check whehter the highest F1 score is above 0.5
    best_f1 = best_df['f1_score'].max()
    if best_f1 < 0.5:
        # raise Exception("Best f1_score is still below 0.5! Manual Investigation needed!")
        print("Best f1_score is still below 0.5! Manual Investigation needed!")

    print("best_df=",best_df)




