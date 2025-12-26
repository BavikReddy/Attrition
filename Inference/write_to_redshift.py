import pandas as pd
import numpy as np
from sqlalchemy.engine import create_engine
from db_connect import setup_db_engine, read_sql_query
import time
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

import os
import json


def main():
    
    base_dir = "/opt/ml/processing"

    ############################################################
    engine = setup_db_engine()
    start = time.time()

    df_express = pd.read_csv('/opt/ml/processing/express_df/express_df.csv')
    print('shape of df_express: ', df_express.shape)
    df_large = pd.read_csv('/opt/ml/processing/large_df/large_df.csv')
    print('shape_of_df_large: ', df_large.shape)
    df_small = pd.read_csv('/opt/ml/processing/small_df/small_df.csv')
    print('shape_of_df_small: ', df_small.shape)
    
    # combine all output together
    full_df_combined = pd.concat([df_small, df_large, df_express])

    full_df_combined['customer_size']=np.where(full_df_combined['customer_size']=='others','express-check',full_df_combined['customer_size'])
    full_df_combined['customer_size']=np.where(full_df_combined['customer_size']=='medium','small',full_df_combined['customer_size'])

    # get data map of prep data
    data_map_path = os.path.join(base_dir,'input/data_map.json')
    with open(data_map_path, "r") as fp:
        data_map = json.load(fp)
    current_week = str(data_map['current_week'])
    full_df_combined['run_week'] = current_week
    full_df_combined['run_date'] = datetime.now().date()
    
    full_df_combined.to_sql('hubble_prod_release_2_test', schema='user_revenue', con=engine, index=False, if_exists='append', chunksize=10000, method='multi')
    end = time.time()
    print('execution time: {0:.4f}s'.format(end - start))

    file_name = 'full_output_df.csv'
    path = base_dir + file_name    
    full_df_combined.to_csv(path, header=True, index=False)
    print('combined result has been saved to: ', path)

if __name__ == "__main__":
    main()
