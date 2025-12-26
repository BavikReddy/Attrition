import json
import os
import ast
import sys
import traceback
import pandas as pd
import pickle
import joblib
from datetime import datetime, date, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

if __name__=='__main__':
    
    #Train data
    #Load full_df data
    base_dir = "/opt/ml/processing"
    full_df_path = f"{base_dir}/train/train.csv"
    full_df = pd.read_csv(full_df_path)

    #Select only largefleet customers
    # full_df_large=full_df[full_df['customer_size'].isin(['large','medium'])]
    full_df_large=full_df[full_df['customer_size'].isin(['large'])]
    
    # today_date=date.today()
    # filename_hyperparameter="hyperparameter_tuning_largefleet_"+str(today_date)+".csv"
    filename_hyperparameter="hyperparameter_tuning_largefleet.csv"
    #hyperparameter tuning df path
    hyperparameter_tuning_path=os.path.join(base_dir,'hyperparameter_tuning_largefleet',filename_hyperparameter)
    # hyperparameter_tuning_path = f"{base_dir}/hyperparameter_tuning_expresscheck/hyperparameter_tuning_expresscheck.csv"
    print("hyperparameter_tuning_path",hyperparameter_tuning_path)
    hyperparameter_df=pd.read_csv(hyperparameter_tuning_path)
    hyperparameter_df=hyperparameter_df.sort_values('f1_score',ascending=False)
    algorithm_features=hyperparameter_df.iloc[0]['algorithm_features']
    best_param=hyperparameter_df.iloc[0]['best_params']
    best_param = ast.literal_eval(best_param)
    algorithm,features=algorithm_features.split('_',1)

    #Load feature_dict json file
    feature_dict_path = f"{base_dir}/feature_dict_largefleet/feature_dict_largefleet.json"
    with open(feature_dict_path, 'r') as fp:
        feature_dict=json.load(fp)
    
    #Remove unwanted columns
    full_df_large.drop(['parentname','last_transaction_week',
     'last_transaction_month','customer_size','customer_segment','first_trx_month'],axis=1,inplace=True)

    # encoding with both only nominal (One Hot encoding)
    CATEGORICAL_COLS=[ 'state_code', 'channel', 'product_type_most_frequent', #'last_payment_method',
                      'num_of_trucks','billing_freq','credit_grade' ]
    full_df_large=pd.concat([full_df_large.drop(CATEGORICAL_COLS, axis=1), pd.get_dummies(full_df_large[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)], axis=1)
    print("shape of train data:",full_df_large.shape)
    
    # split to x and y
    y_train=full_df_large['churn_flag']   
    req_features=feature_dict[features]
    x_train=full_df_large[req_features]
    
    #Fix the Distribution
    power = PowerTransformer(method='yeo-johnson', standardize=True)
    x_train.iloc[:,:] = power.fit_transform(x_train.loc[:,:])

    #Scaling
    scaler = StandardScaler()
    x_train.iloc[:,:] = scaler.fit_transform(x_train.loc[:,:])
    
    if algorithm=='LGBMClassifier': 
        lr=round(best_param['learning_rate'],2)
        md=int(round(best_param['max_depth']))
        mcs=int(round(best_param['min_child_samples']))
        nl=int(round(best_param['num_leaves']))
        ra=round(best_param['reg_alpha'],2)
        model=LGBMClassifier(objective='binary', is_unbalance=True, metric='roc_auc',random_state=100, 
                               learning_rate=lr, max_depth=md, min_child_samples=mcs, num_leaves=nl,reg_alpha=ra)
        model.fit(x_train,y_train)       
        
    elif algorithm=='LogisticRegression':
        c=best_param['c_value']
        s=best_param['solvers']
        p=best_param['penalty']
        model=LogisticRegression(penalty=p,C=c,solver=s,class_weight='balanced')
        model.fit(x_train,y_train)

    elif algorithm=='XGBClassifier':
        lr=round(best_param['learning_rate'],2)
        model=XGBClassifier(booster='gblinear', random_state=100,learning_rate=lr)
        model.fit(x_train,y_train)
    else:
        ne=int(best_param['n_estimators'])
        md=int(round(best_param['max_depth']))
        crt=best_param['criterion']
        model = RandomForestClassifier(class_weight='balanced', random_state=100,
                               n_estimators=ne,max_depth=md, criterion=crt)
        model.fit(x_train,y_train)
    
    # save_path = os.path.join(base_dir,'largefleet_model', 'largefleet_model.sav')
    with open(os.path.join(base_dir,'largefleet_model', 'largefleet_model.pkl'), 'wb') as out:
        pickle.dump(model, out)
        
    # make a copy
    today = date.today()
    archive_file_name = 'largefleet_model-' + str(today) + '.pkl'
    
    with open(os.path.join(base_dir,'largefleet_model_archive', archive_file_name), 'wb') as out:
        pickle.dump(model, out)
    
    print("Training is completed")
    