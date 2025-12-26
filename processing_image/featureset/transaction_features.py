
import numpy as np
import pandas as pd
from featureset import utils


def run_transaction_features(weekly_df):

    # feature list
    FEATURES = ['total_gallons', 'total_fuel_trx', 'total_nonfuel_trx', 'total_fuel_spend', 'total_nonfuel_spend','total_num_trx',
                'total_comcheck_spend', 'total_vcap_spend',
       'total_ecashspend', 'ecash_gallons']
    weekly_df = weekly_df[['parentname', 'week', 'week_range_4', 'week_range_8', 'week_range_13', 'week_range_26','week_range_52'] + FEATURES].copy()
    # weekly_df.describe()

    # for each feature, get avg values
    for feature_name in FEATURES:
        weekly_df[feature_name + '_avg'] = utils.get_avg_per_active_week(weekly_df, feature_name,'week_range_52')  # no week range is equivalent to 1-52
        weekly_df[feature_name + '_avg4'] = utils.get_avg_per_active_week(weekly_df, feature_name, 'week_range_4')
        weekly_df[feature_name + '_avg8'] = utils.get_avg_per_active_week(weekly_df, feature_name, 'week_range_8')
        weekly_df[feature_name + '_avg13'] = utils.get_avg_per_active_week(weekly_df, feature_name, 'week_range_13')
        weekly_df[feature_name + '_avg26'] = utils.get_avg_per_active_week(weekly_df, feature_name, 'week_range_26')


    # share of active fuel weeks : num of active weeks / interval length
    weekly_df['num_active_fuel'] = weekly_df.loc[weekly_df['total_gallons'] > 0].groupby('parentname')['total_gallons'].transform('count')
    weekly_df['num_active_nonfuel'] = weekly_df.loc[weekly_df['total_nonfuel_spend'] != 0].groupby('parentname')['total_nonfuel_spend'].transform('count')
    weekly_df['num_active_trx'] = weekly_df.loc[weekly_df['total_num_trx'] != 0].groupby('parentname')['total_num_trx'].transform('count')
    weekly_df['num_active_comcheck'] = weekly_df.loc[weekly_df['total_comcheck_spend'] != 0].groupby('parentname')['total_comcheck_spend'].transform('count')
    weekly_df['num_active_vcap'] = weekly_df.loc[weekly_df['total_vcap_spend'] != 0].groupby('parentname')['total_vcap_spend'].transform('count')
    weekly_df['num_active_ecash'] = weekly_df.loc[weekly_df['total_ecashspend'] != 0].groupby('parentname')['total_ecashspend'].transform('count')
    weekly_df['week_cnt'] = weekly_df.groupby('parentname')['week'].transform('count')
    weekly_df['share_active_fuel52'] = weekly_df['num_active_fuel'] / weekly_df['week_cnt']
    weekly_df['share_active_nonfuel52'] = weekly_df['num_active_nonfuel'] / weekly_df['week_cnt']
    weekly_df['share_active_trx52'] = weekly_df['num_active_trx'] / weekly_df['week_cnt']
    weekly_df['share_active_comcheck_spend52']=weekly_df['num_active_comcheck'] / weekly_df['week_cnt']
    weekly_df['share_active_vcap_spend52']=weekly_df['num_active_vcap'] / weekly_df['week_cnt']
    weekly_df['share_active_ecash_spend52']=weekly_df['num_active_ecash'] / weekly_df['week_cnt']
    
    
    # same for each week range:
    for week_range in ['week_range_4','week_range_8', 'week_range_13', 'week_range_26']:
        weekly_df['share_active_fuel' + week_range.split('_')[-1]] = utils.get_share_of_active_weeks(weekly_df, week_range,'total_gallons' )
        weekly_df['share_active_nonfuel' + week_range.split('_')[-1]] = utils.get_share_of_active_weeks(weekly_df, week_range,'total_nonfuel_spend' )
        weekly_df['share_active_trx' + week_range.split('_')[-1]] = utils.get_share_of_active_weeks(weekly_df, week_range, 'total_num_trx')
        weekly_df['share_active_comcheck_spend' + week_range.split('_')[-1]] = utils.get_share_of_active_weeks(weekly_df, week_range, 'total_comcheck_spend')  
        weekly_df['share_active_vcap_spend' + week_range.split('_')[-1]] = utils.get_share_of_active_weeks(weekly_df, week_range, 'total_vcap_spend')  
        weekly_df['share_active_ecash_spend' + week_range.split('_')[-1]] = utils.get_share_of_active_weeks(weekly_df, week_range, 'total_ecashspend')
    
    # get 1-row-per-parent DF with the _avg values
    feature_df = weekly_df.sort_values(by=['parentname','week'],ascending=False)[['parentname'] + [f + '_avg' for f in FEATURES] + [f + '_avg'+i for f in FEATURES for i in ['4','8','13','26']]+\
                           ['share_active_' + i + j for i in ['fuel', 'nonfuel','trx','comcheck_spend','vcap_spend','ecash_spend'] for j in ['4','8', '13', '26', '52']]]\
                          .groupby('parentname', as_index=False).first()
    assert len(feature_df) == feature_df.parentname.nunique()

   

    
  
    # get quintile scores for feature avg values
    for f in FEATURES:
       feature_df[f+'_avg'].replace(0,np.nan,inplace=True)
       feature_df[f + '_score'] = pd.qcut(feature_df[f+'_avg'], 4, labels=False, duplicates='drop')
       feature_df[f + '_score']=feature_df[f + '_score']+1 
       feature_df[f + '_score'].replace(np.nan,0,inplace=True)
       feature_df[f + '_score']=feature_df[f + '_score'].astype('int')
       feature_df[f+'_avg'].replace(np.nan,0,inplace=True)
   
    # fill nulls with 0s (otherwise derived % change features will be NULL and end up filled with medians)

    feature_df.fillna(0.0, inplace=True)    
   
    
   # get % changes
    for f in FEATURES:
        tmp_df = weekly_df[['parentname', 'week_range_4','week_range_8', 'week_range_13', 'week_range_26',
                            f + '_avg4',f + '_avg8', f + '_avg13', f + '_avg26', f + '_avg']].drop_duplicates()
        rez_df = utils.get_pct_change(tmp_df, f)
#         print(tmp_df.parentname.nunique(),rez_df.parentname.nunique())
        feature_df = feature_df.merge(rez_df, on='parentname', how='outer')
    len(feature_df), feature_df.parentname.nunique()
    
#     # drop the _avg (we only need _score)
#     feature_df.drop([f + '_avg' for f in FEATURES], axis=1, inplace=True)

    # fill nulls with 1.0 for inf (a pct change from 0 to X) / -1.0 for -inf (from 0 to -X)
    feature_df.replace(np.inf, 1.0, inplace=True)
    feature_df.replace(-np.inf, -1.0, inplace=True)
    feature_df.fillna(0.0, inplace=True)
   

    return feature_df