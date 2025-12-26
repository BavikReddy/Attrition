"""Feature engineers the abalone dataset."""
import argparse
import logging
import os
import pathlib
import requests
import tempfile

import boto3
import pandas as pd
import sys
import numpy as np
from datetime import datetime, date, timedelta
# from sqlalchemy.engine import create_engine
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
from joblib import dump, load
# from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
# import shap
import warnings
warnings.filterwarnings('ignore')
# sys.path.append(os.path.dirname(__file__))
# from featureset import combining_data, utils,static_features, transaction_features, fee_features,sf_call_features, sf_survey_features

import combining_data, utils,static_features, transaction_features, fee_features,sf_call_features, sf_survey_features


if __name__ == "__main__":

#     files = list(filter(None,os.popen('ls /opt/ml/processing/input | grep chunk').read().split("\n")))
#     for file in files:
#         print(file)
    print("For the directory of the script being run",pathlib.Path(__file__).parent.resolve())
    print("For the current working directory",pathlib.Path().resolve())
    # params
    parser = argparse.ArgumentParser()

    # create dict for query arguments (to be passed to queries)
    DATA_MAP = {}
    
    
    base_dir = "/opt/ml/processing"
    
    #Download featureset
#     featureset_path = f"{base_dir}/input/featureset/combining_data.py"
#     s3 = boto3.resource('s3')

#     s3.Bucket('sagemaker-migration-hubble').download_file('code/featureset/combining_data.py', '/opt/ml/processing/input/code/combining_data.py')
#     import combining_data
    
    
    
    max_df_path = f"{base_dir}/input/max_week_df.csv"
    max_week_df = pd.read_csv(max_df_path)
    max_week = str(min(max_week_df.max_week))
    print("runweek:{0}".format(max_week))

    churn_cutoff_week = str(min(max_week_df.churn_cut_off_week))
    print("churn_cutoff_week:{0}".format(churn_cutoff_week))
    
     # add the optional arguments
     # RUN_WEEK SET TO 201952 FOR TESTING PURPOSE, FOR REALTIME USAGE SET default=max_week
    parser.add_argument('--RUN_WEEK', type=str, default=max_week, help='week number (yyyyww) to use for data loading - \
                            a time period of 52 weeks backwards from the RUN_WEEK will be retrieved, default - max week available in the data')
    parser.add_argument('--CHURN_CUTOFF_WEEK', type=str, default=churn_cutoff_week, help='smallest week number (yyyyww) when the customer must transact to be considered active, \
                            all customers with last transaction before will be considered churned, default 202000')
    parser.add_argument('--MIN_TRANSACTION_WEEKS', type=int, default=13, help='min required activity weeks in the 52-week time period, default 13')
    parser.add_argument('--ACTIVITY_WEEKS', type=int, default=52 ,help='time interval in weeks from which to take active customers for the prediction, default 52')
    parser.add_argument('--WEEK_OFFSET', type=int, default=0, help='param for hiding last X weeks (used only when preparing training data, set to 0 for prediction), default 0')
    parser.add_argument('--RESULTS_SCHEMA', type=str, default='user_reporting', help='DB schema for the results table, default user_reporting')
    parser.add_argument('--RESULTS_TABLE', type=str, default='hubble_prod_release_1', help='table name for the pre-prod testing')
    parser.add_argument('--VALIDATIONS_TABLE', type=str, default='hubble_validation_table_release_1', help='table name for the pre-prod testing')
    args = parser.parse_args()
    
    # adding the max week to data map (to be passed to queries)
    # DATA_MAP['current_week'] = args.RUN_WEEK
    # DATA_MAP['current_week'] = 202205
    today = date.today()
    print("Today's date:", today)
    year = today.isocalendar()[0]
    week = today.isocalendar()[1]
    today_week = int(str(year) + str(week))
    print("Today's week:", today_week)

    current_week = today - timedelta(weeks = 26)
    print("'Current date' for training set:", current_week)
    year = current_week.isocalendar()[0]
    week = current_week.isocalendar()[1]
    current_week = int(str(year) + str(week))
    print("'Current week' for training set:", current_week)
    DATA_MAP['current_week'] = current_week
    
    # churn cutoff week - the min week when the customer must transact to be considered active, all customers with last transaction before are considered churned
    DATA_MAP['churn_cutoff_week'] = args.CHURN_CUTOFF_WEEK
    # interval from which we take active customers
    DATA_MAP['activity_weeks'] = args.ACTIVITY_WEEKS
    # number of weeks to load from the RUN_WEEK backwards: since we need 52 weeks per customer, the min required interval is:
    DATA_MAP['history_weeks'] = args.ACTIVITY_WEEKS + 52 

    print('RUN_WEEK is {0}'.format(args.RUN_WEEK))
    
    #Load static and dynamic df
    static_path = f"{base_dir}/input/static_df.csv"
    static_df = pd.read_csv(static_path)

    assert not static_df.empty
    print('static data  loaded: {0} rows'.format(len(static_df)))

    dynamic_path = f"{base_dir}/input/dynamic_df.csv"
    dynamic_df = pd.read_csv(dynamic_path)
    assert not dynamic_df.empty
    print('dynamic data loaded: {0} rows'.format( len(dynamic_df)))

    # add SalesForce data
    sf_path = f"{base_dir}/input/sf_df.csv"
    sf_df = pd.read_csv(sf_path)
    print('salesforce call data loaded: {0} rows'.format( len(sf_df)))
    sf_df=sf_df.drop_duplicates(keep='first')

    surv_path = f"{base_dir}/input/surv_df.csv"
    surv_df = pd.read_csv(surv_path)
    print('salesforce surv data loaded: {0} rows'.format( len(surv_df)))
    surv_df=surv_df.drop_duplicates(keep='first')
    
    
#     print("For the directory of the script being run",pathlib.Path(__file__).parent.resolve())
#     print("For the current working directory",pathlib.Path().resolve())
        
#     #combining accts at parent level
#     sys.path.append(os.path.dirname(__file__))
#     from featureset import combining_data,utils,static_features,transaction_features,fee_features,sf_call_features,sf_survey_features

#     print("combining accts at parent level")
#     dynamic_df.fillna(0,inplace=True)
#     static_df,dynamic_df=combining_data.combining_accts_at_parentlevel(static_df,dynamic_df,DATA_MAP)
#     print("static_df=====",static_df)
       

    
#     sf_df.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
#     surv_df.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)

         #combining accts at parent level
    print("combining accts at parent level")
    dynamic_df.fillna(0,inplace=True)
    static_df,dynamic_df=combining_data.combining_accts_at_parentlevel(static_df,dynamic_df,DATA_MAP)

    print('preprocessing dynamic data ...')
    weekly_df = utils.preprocess_weekly_data(dynamic_df,static_df, args.MIN_TRANSACTION_WEEKS, args.WEEK_OFFSET, random_cutoff_week_for_active=False)
    weekly_df_copy=weekly_df.copy()    
    idx = weekly_df.groupby(['parentname'])['week'].transform(max) == weekly_df['week']
    last_w_df = weekly_df[idx][['parentname', 'week']]
    print('done preprocessing')

    #pull calender Month to get the last trx month
    calendar_path = f"{base_dir}/input/calendar_df.csv"
    calendar_df = pd.read_csv(calendar_path)
    last_w_df=last_w_df.merge(calendar_df,on='week',how='left')

    print('generating static features for platform(s) PROP and MC')
    static_feature_df = static_features.run_static_features(static_df, last_w_df)
    assert len(static_feature_df) == static_feature_df.parentname.nunique()
    print('static features generated: {0} rows'.format(len(static_feature_df)))

    print('generating dynamic (transaction/fee/flag) features')

    tran_df=transaction_features.run_transaction_features(weekly_df)
    print('transaction features generated: {0} rows'.format(len(tran_df)))
    fee_df = fee_features.run_fee_features(weekly_df)
    print('fee features generated: {0} rows'.format(len(fee_df)))

    fee_flag_df = fee_features.run_flag_fee_features(weekly_df)
    print('flag features generated: {0} rows'.format(len(fee_flag_df)))

    #Generating salesforce call features
    salesforce_call_df=sf_call_features.run_sf_call_features(sf_df)
    weekly_df=weekly_df_copy[[ 'parentname', 'week','week_rank', 'week_range_4', 'week_range_8', 'week_range_13',
        'week_range_26', 'week_range_52']]

    salesforce_call_df.rename(columns={'call_week':'week'},inplace=True)
    weekly_call_df=weekly_df.merge(salesforce_call_df,on=['parentname','week'],how='left')

    sf_call_df=sf_call_features.generate_sf_call_features(weekly_call_df)
    print('salesforce call features generated: {0} rows'.format(len(sf_call_df)))

     #Generating salesforce surv features
    salesforce_surv_df=sf_survey_features.run_sf_survey_features(surv_df)
    weekly_df=weekly_df_copy[[ 'parentname', 'week','week_rank', 'week_range_4', 'week_range_8', 'week_range_13',
        'week_range_26', 'week_range_52']]

    salesforce_surv_df.rename(columns={'case_week':'week'},inplace=True)
    weekly_surv_df=weekly_df.merge(salesforce_surv_df,on=['parentname','week'],how='left')
    sf_surv_df=sf_survey_features.generate_sf_surv_features(weekly_surv_df)
    print('salesforce call features generated: {0} rows'.format(len(sf_surv_df)))
    
    
    tran_df['parentname'] = tran_df['parentname'].astype(str)
    fee_df['parentname'] = fee_df['parentname'].astype(str)
    fee_flag_df['parentname'] = fee_flag_df['parentname'].astype(str)

    assert len(tran_df.loc[~tran_df.parentname.isin(static_feature_df.parentname)]) == 0
    assert len(fee_df.loc[~fee_df.parentname.isin(static_feature_df.parentname)]) == 0
    assert len(fee_flag_df.loc[~fee_flag_df.parentname.isin(static_feature_df.parentname)]) == 0

     # merge the data
    full_df = static_feature_df.merge(tran_df, left_on='parentname', right_on='parentname', how='inner')\
                                .merge(fee_df, left_on='parentname', right_on='parentname', how='inner')\
                                .merge(fee_flag_df, left_on='parentname', right_on='parentname', how='inner')\
                                .merge(sf_call_df, left_on='parentname', right_on='parentname', how='inner')\
                                .merge(sf_surv_df, left_on='parentname', right_on='parentname', how='inner')

     # fill NULLs (missing SalesForce data types between the different platforms may result in NULL values)
    full_df.fillna(0.0, inplace=True)
    print('full feature data generated: {0} rows, {1} columns'.format(len(full_df), len(full_df.columns)))
     
    full_df['churn_flag']=np.where(full_df['last_transaction_week']>int(args.CHURN_CUTOFF_WEEK),0,1)
    
    #load last_trx_cutoff_week
    last_trx_cutoff_week_path = f"{base_dir}/input/last_trx_cutoff_week_df.csv"
    last_trx_cutoff_week_df = pd.read_csv(last_trx_cutoff_week_path)
#     last_trx_cutoff_week_query = read_sql_query(path_to_query='queries/pull_last_trx_cutoff_week.sql')
#     last_trx_cutoff_week_df = pd.read_sql_query(sql=last_trx_cutoff_week_query.format(**DATA_MAP), con=engine)
    last_trx_cutoff_week=int(last_trx_cutoff_week_df.last_trx_cutooff_week[0])
    full_df=full_df[full_df['last_transaction_week']>last_trx_cutoff_week]
    print("full_df=====",full_df)
    
    #split into train and test
    # train_data, test_data = train_test_split(full_df, test_size=0.8, random_state=100)
    # train_data, test_data = np.split(full_df.sample(frac=1, random_state=1729), [int(0.8 * len(full_df))])
    # print("train_data column names==",list(train_data.columns.values))
    # print("train_data===",train_data)
    full_df.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
    # train_data.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    # test_data.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
    
    # shouldn't do more feature engineering here (e.g., power transform)
    # because we have not filtered based on fleet size yet.
    # train-test split is not needed either.
    
    # save a copy for archive
    today = date.today()
    archive_file_name = 'test-' + str(today) + '.csv'
    full_df.to_csv(f"{base_dir}/test_archive/{archive_file_name}", header=True, index=False)