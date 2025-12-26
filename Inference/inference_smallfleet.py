import os
import ast
import json
import tarfile
import shap
import sklearn
import lightgbm
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import joblib
import pickle
# from joblib import dump, load
from datetime import datetime, date, timedelta
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
import combining_data, utils,static_features, transaction_features, fee_features,sf_call_features, sf_survey_features


if __name__ == "__main__":
    
    #Load full_df data
    base_dir = "/opt/ml/processing"
    full_df_path = f"{base_dir}/inference/inference.csv"
    full_df = pd.read_csv(full_df_path)
    
    #Create smallfleet acct Model
     
    print('Predicting Attrition Probability of smallfleet customers')
    #Select only small-fleet customers
    full_df_small=full_df[full_df['customer_size'].isin(['small', 'medium'])]
    full_df_small=full_df_small.reset_index()
    full_df_small_copy=full_df_small.copy()

    # today_date=date.today()
    # filename_hyperparameter="hyperparameter_tuning_smallfleet_"+str(today_date)+".csv"
    filename_hyperparameter="hyperparameter_tuning_smallfleet.csv"
    #hyperparameter tuning df path
    hyperparameter_tuning_path=os.path.join(base_dir,'model-artifacts/smallfleet',filename_hyperparameter)
    # hyperparameter_tuning_path = f"{base_dir}/hyperparameter_tuning_expresscheck/hyperparameter_tuning_expresscheck.csv"
    print("hyperparameter_tuning_path",hyperparameter_tuning_path)
    hyperparameter_df=pd.read_csv(hyperparameter_tuning_path)
    hyperparameter_df=hyperparameter_df.sort_values('f1_score',ascending=False)
    algorithm_features=hyperparameter_df.iloc[0]['algorithm_features']
    best_param=hyperparameter_df.iloc[0]['best_params']
    best_param = ast.literal_eval(best_param)
    algorithm,features=algorithm_features.split('_',1)
    print("algorithm==",algorithm)
    print("features==",features)
    #Load feature_dict json file
    feature_dict_path = f"{base_dir}/model-artifacts/smallfleet/feature_dict_smallfleet.json"
    with open(feature_dict_path, 'r') as fp:
        feature_dict=json.load(fp)
    
    
    #encoding
    CATEGORICAL_COLS=[ 'state_code', 'num_of_trucks', 'billing_freq',
        'credit_grade', 'channel', 'product_type_most_frequent'#,'last_payment_method'
                     ]
    
    full_df_small=pd.concat([full_df_small.drop(CATEGORICAL_COLS, axis=1), pd.get_dummies(full_df_small[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)], axis=1)
   
    model_path = os.path.join(base_dir,"model-artifacts/smallfleet", "smallfleet_model.pkl")
    with open(model_path, "rb") as input_file:
         estimator_small= pickle.load(input_file)
    
    req_features=feature_dict[features]
    missing_columns = []
    for c in set(req_features):
        if c not in full_df_small.columns.values:
            missing_columns.append(c)
    for col in missing_columns:
        full_df_small[col] = 0.0
    print('{0} missing columns were added: {1}'.format(len(missing_columns), missing_columns))
    assert all(c in full_df_small.columns for c in req_features)

    y_test=full_df_small['churn_flag']
    x_test=full_df_small[req_features]
    
    #Fix the Distribution
    power = PowerTransformer(method='yeo-johnson', standardize=True)
    # power = joblib.load("express-fleet-transformer.sav")
    x_test.iloc[:,:] = power.fit_transform(x_test.loc[:,:])
    #Scaling
    scaler = StandardScaler()
    # scaler = joblib.load("express-fleet-scaler.sav")
    x_test.iloc[:,:] = scaler.fit_transform(x_test.loc[:,:])
    
    y_pred = estimator_small.predict(x_test)
    # print('Accuracy score on testing set: ', accuracy_score(y_test, y_pred))


    print('\n-----------------results smallfleet-----------------------\n')
    try:
        predicted_prob_new = estimator_small.predict_proba(x_test)
        predicted_class_new = estimator_small.predict(x_test)
        full_df_small_copy['attrition_prob'] = np.round(predicted_prob_new[:, 1], 4)
        full_df_small_copy['attrition_rank'] = full_df_small_copy.attrition_prob.apply(lambda x: int(x * 10))
        full_df_small_copy['churn_flag_predicted'] =  predicted_class_new
        full_df_small_copy.rename(columns={'churn_flag':'churn_flag_actual'},inplace=True)
    except NotFittedError as e:
        print(repr(e))
    print('done. {0} Smallfleet  customer scores predicted with the following distribution:\n{1}'.\
      format(full_df_small_copy.attrition_rank.count(), full_df_small_copy.attrition_rank.value_counts(dropna=False).sort_index()))

    categorical_dummy_variable=[i for i in features for sub in CATEGORICAL_COLS if sub in i ] 
    
    if algorithm in ('XGBClassifier','LogisticRegression'):
        explainer=shap.LinearExplainer(estimator_small,x_test[req_features],feature_perturbation="correlation_dependent")
        shap_values_small = explainer.shap_values(x_test[req_features])
        new_top_10_factor_df = utils.generate_explanations(shap_values_small, x_test[req_features], req_features,categorical_dummy_variable )
    else:
        explainer=shap.TreeExplainer(estimator_small, feature_perturbation='tree_path_dependent')
        shap_values_small = explainer.shap_values(x_test[req_features])
        new_top_10_factor_df = utils.generate_explanations_random_forest(shap_values_small, x_test[req_features], req_features,categorical_dummy_variable )
    
    
    print("Successfully created new_top_10_factor_df")
     #Merge the data together
    full_df_small_copy = full_df_small_copy[['parentname','first_trx_month','last_transaction_week','tenure','attrition_prob','attrition_rank',\
                                             'churn_flag_predicted','churn_flag_actual','credit_limit','due_days','customer_size']+CATEGORICAL_COLS]

    full_df_small = full_df_small_copy.merge(new_top_10_factor_df, left_index=True, right_index=True)

    full_df_small.to_csv(f"{base_dir}/small_df/small_df.csv", header=True, index=False)
    
    # save a copy
    today = date.today()
    archive_file_name = 'small_df-' + str(today) + '.csv'
    full_df_small.to_csv(f"{base_dir}/small_df_archive/{archive_file_name}", header=True, index=False)