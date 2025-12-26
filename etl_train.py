import pandas as pd
import numpy as np

from sqlalchemy.engine import create_engine
from db_connect import setup_db_engine, read_sql_query

import time
from datetime import datetime, date, timedelta

import argparse

import warnings
warnings.filterwarnings('ignore')

def main():
    # bucket = 'sagemaker-migration-hubble-analytics/'
    # prefix = 'data-train/'
    # s3_uri = 's3://' + bucket + prefix
    
    base_dir = "/opt/ml/processing/train/"

    ############################################################
    engine = setup_db_engine()
    start = time.time()

    max_week_query = read_sql_query(path_to_query='queries/max_week_query.sql')
    max_week_df = pd.read_sql_query(sql=max_week_query, con=engine)
    end = time.time()
    print('execution time: {0:.4f}s'.format(end - start))

    print('max_week_df query completed!')
    print('max_week_df shape: ', max_week_df.shape)

    file_name = 'max_week_df.csv'
    path = base_dir + file_name    
    max_week_df.to_csv(path, header=True, index=False)
    print('max_week_df has been saved to: ', path)

    ############################################################
    max_week = str(min(max_week_df.max_week))
    print("runweek:{0}".format(max_week))

    churn_cutoff_week = str(min(max_week_df.churn_cut_off_week))
    print("churn_cutoff_week:{0}".format(churn_cutoff_week))

    parser = argparse.ArgumentParser()

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
    args = parser.parse_args(args=[])

    # create dict for query arguments (to be passed to queries)
    DATA_MAP = {}

    # adding the max week to data map (to be passed to queries)
    # DATA_MAP['current_week'] = args.RUN_WEEK
    # DATA_MAP['current_week'] = 202152
    today = date.today()
    print("Today's date:", today)
    year = today.isocalendar()[0]
    # week = current_week.isocalendar()[1]
    week = today.strftime("%V")
    today_week = int(str(year) + str(week))
    print("Today's week:", today_week)

    current_week = today - timedelta(weeks = 34)
    print("'Current date' for training set:", current_week)
    year = current_week.isocalendar()[0]
    # week = current_week.isocalendar()[1]
    week = current_week.strftime("%V")
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

    ############################################################
    engine = setup_db_engine()
    start = time.time()
    static_df_query = read_sql_query(path_to_query='queries/static.sql')
    static_df = pd.read_sql_query(sql=static_df_query.format(**DATA_MAP), con=engine)
    end = time.time()
    print('execution time: {0:.4f}s'.format(end - start))

    print('static_df query completed!')
    print('static_df shape: ', static_df.shape)

    file_name = 'static_df.csv'
    path = base_dir + file_name    
    static_df.to_csv(path, header=True, index=False)
    print('static_df has been saved to: ', path)

    ############################################################
    engine = setup_db_engine()
    start = time.time()
    dynamic_df_query = read_sql_query(path_to_query='queries/dynamic.sql')
    dynamic_df = pd.read_sql_query(sql=dynamic_df_query.format(**DATA_MAP), con=engine)
    end = time.time()
    print('execution time: {0:.4f}s'.format(end - start))

    print('dynamic_df query completed!')
    print('dynamic_df shape: ', dynamic_df.shape)

    file_name = 'dynamic_df.csv'
    path = base_dir + file_name    
    dynamic_df.to_csv(path, header=True, index=False)
    print('dynamic_df has been saved to: ', path)

    ############################################################
    engine = setup_db_engine()
    start = time.time()
    salesforce_call_query = read_sql_query(path_to_query='queries/Saleforce_call.sql')
    sf_df = pd.read_sql_query(sql=salesforce_call_query.format(**DATA_MAP), con=engine)
    end = time.time()
    print('execution time: {0:.4f}s'.format(end - start))

    print('sf_df query completed!')
    print('sf_df shape: ', sf_df.shape)

    file_name = 'sf_df.csv'
    path = base_dir + file_name    
    sf_df.to_csv(path, header=True, index=False)
    print('sf_df has been saved to: ', path)

    ############################################################
    engine = setup_db_engine()
    start = time.time()
    salesforce_surv_query = read_sql_query(path_to_query='queries/Saleforce_survey.sql')
    surv_df = pd.read_sql_query(sql=salesforce_surv_query.format(**DATA_MAP), con=engine)
    end = time.time()
    print('execution time: {0:.4f}s'.format(end - start))

    print('surv_df query completed!')
    print('surv_df shape: ', surv_df.shape)

    file_name = 'surv_df.csv'
    path = base_dir + file_name    
    surv_df.to_csv(path, header=True, index=False)
    print('surv_df has been saved to: ', path)

    ############################################################
    engine = setup_db_engine()
    start = time.time()
    calendar_df_query = read_sql_query(path_to_query='queries/pull_calendar_month.sql')
    calendar_df = pd.read_sql_query(sql=calendar_df_query.format(**DATA_MAP), con=engine)
    end = time.time()
    print('execution time: {0:.4f}s'.format(end - start))

    print('calendar_df query completed!')
    print('calendar_df shape: ', calendar_df.shape)

    file_name = 'calendar_df.csv'
    path = base_dir + file_name    
    calendar_df.to_csv(path, header=True, index=False)
    print('calendar_df has been saved to: ', path)

    ############################################################
    engine = setup_db_engine()
    start = time.time()
    last_trx_cutoff_week_query = read_sql_query(path_to_query='queries/pull_last_trx_cutoff_week.sql')
    last_trx_cutoff_week_df = pd.read_sql_query(sql=last_trx_cutoff_week_query.format(**DATA_MAP), con=engine)
    end = time.time()
    print('execution time: {0:.4f}s'.format(end - start))

    print('last_trx_cutoff_week_df query completed!')
    print('last_trx_cutoff_week_df shape: ', last_trx_cutoff_week_df.shape)

    file_name = 'last_trx_cutoff_week_df.csv'
    path = base_dir + file_name    
    last_trx_cutoff_week_df.to_csv(path, header=True, index=False)
    print('last_trx_cutoff_week_df has been saved to: ', path)


if __name__ == "__main__":
    main()
