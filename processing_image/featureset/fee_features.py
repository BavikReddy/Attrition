import numpy as np
import pandas as pd
from featureset import utils


def run_fee_features(weekly_df):
    
    weekly_df['unique_types_of_fees_applied']=weekly_df['num_of_late_fees_applied'].map(lambda x: 1 if x > 0 else 0)\
                                               +weekly_df['num_of_risk_fees_applied'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_optional_fees_applied'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_fleet_adv_applied'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_fleet_fees_applied'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_inac_fees_applied'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_admin_fees_applied'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_expr_fees_applied'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_eld_eqi_fees_applied'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_card_fees_applied'].map(lambda x: 1 if x > 0 else 0)
    
    weekly_df['unique_types_of_fees_refunded']=weekly_df['num_of_late_fees_refunded'].map(lambda x: 1 if x > 0 else 0)\
                                               +weekly_df['num_of_risk_fees_refunded'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_optional_fees_refunded'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_fleet_adv_refunded'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_inac_fees_refunded'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_admin_fees_refunded'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_expr_fees_refunded'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_eld_eqi_fees_refunded'].map(lambda x: 1 if x > 0 else 0)\
                                               + weekly_df['num_of_card_fees_refunded'].map(lambda x: 1 if x > 0 else 0)
    
    # fees to compute (cent per gallon)
    CENTS_PER_GALLON_FEATURES = ['late_interest_fee_amount', 'risk_based_pricing_amount', 'program_fee_amount', 'optional_program_amount',
                                 'other_fees_amount', 'total_discount']

  
    # for each cents-per-gallon feature, get avg values
    for feature_name in CENTS_PER_GALLON_FEATURES:
        # no week range is equivalent to 1-52
        weekly_df[feature_name + '_cents_p_gallon_avg'] = utils.get_cents_per_gallon(weekly_df, feature_name, 'week_range_52')
        weekly_df[feature_name + '_cents_p_gallon_avg4'] = utils.get_cents_per_gallon(weekly_df, feature_name, 'week_range_4')
        weekly_df[feature_name + '_cents_p_gallon_avg8'] = utils.get_cents_per_gallon(weekly_df, feature_name, 'week_range_8')
        weekly_df[feature_name + '_cents_p_gallon_avg13'] = utils.get_cents_per_gallon(weekly_df, feature_name, 'week_range_13')
        weekly_df[feature_name + '_cents_p_gallon_avg26'] = utils.get_cents_per_gallon(weekly_df, feature_name, 'week_range_26')


    # get cents-per-gallon across all fees for intervals 4/8 and full lifetime before (to be used for max cents-per-gallon flag)
    weekly_df['cents_p_gallon4'] = utils.get_overall_cents_per_gallon(weekly_df, 'week_range_4', 'CURRENT')
    weekly_df['cents_p_gallon_before4'] = utils.get_overall_cents_per_gallon(weekly_df, 'week_range_4', 'BEFORE')

    weekly_df['cents_p_gallon8'] = utils.get_overall_cents_per_gallon(weekly_df, 'week_range_8', 'CURRENT')
    weekly_df['cents_p_gallon_before8'] = utils.get_overall_cents_per_gallon(weekly_df, 'week_range_8', 'BEFORE')

    
    # get sys locks, trx declines, fees and credit limit usage for the full customer lifetime (weeks 1-52)
    weekly_df['unique_num_of_fees_avg'] = weekly_df.loc[(weekly_df['unique_types_of_fees_applied']>0)].groupby('parentname')['unique_types_of_fees_applied'].transform('mean')
                                          
    for week_range in ['week_range_4','week_range_8','week_range_13', 'week_range_26']:

        weekly_df['unique_num_of_fees_avg' + week_range.split('_')[-1]] = \
            weekly_df.loc[(weekly_df['unique_types_of_fees_applied']>0)&(weekly_df[week_range]=='range-1')].groupby(['parentname', week_range])['unique_types_of_fees_applied'].transform('mean')

      

 

    # Share of invoices with Late Fee / Risk-based Pricing Fee (_avg prefix is only for cosistency with other pct change feature names)
    for fee_name in ['risk_based_pricing_amount', 'late_interest_fee_amount','program_fee_amount', 'optional_program_amount',
                                 'other_fees_amount']:
        weekly_df[fee_name + '_share_avg'] = utils.get_share_of_invoices(weekly_df, fee_name, 'week_range_52')  # no week range is equivalent to 1-52
        weekly_df[fee_name + '_share_avg4'] = utils.get_share_of_invoices(weekly_df, fee_name, 'week_range_4')
        weekly_df[fee_name + '_share_avg8'] = utils.get_share_of_invoices(weekly_df, fee_name, 'week_range_8')
        weekly_df[fee_name + '_share_avg13'] = utils.get_share_of_invoices(weekly_df, fee_name, 'week_range_13')
        weekly_df[fee_name + '_share_avg26'] = utils.get_share_of_invoices(weekly_df, fee_name, 'week_range_26')
        


    # get 1-row-per-customer DF with the 1-52 range values
    FEATURE_LIST = [f + '_cents_p_gallon' for f in CENTS_PER_GALLON_FEATURES] + \
               [ 'unique_num_of_fees',
                'risk_based_pricing_amount_share', 'late_interest_fee_amount_share','program_fee_amount_share']
    FEES = ['risk_based_pricing_amount', 'late_interest_fee_amount','program_fee_amount', 'optional_program_amount',
                                 'other_fees_amount'] 
    
    feature_df = weekly_df.sort_values(by=['parentname','week'],ascending=False)[['parentname'] + [f + '_avg' for f in FEATURE_LIST] +\
                          [f+'_share_avg'+j for f in FEES for j in ['4', '8', '13', '26']] \
                           +['cents_p_gallon4', 'cents_p_gallon_before4',  'cents_p_gallon8', 'cents_p_gallon_before8']]\
                .groupby('parentname', as_index=False).first()
    assert len(feature_df) == feature_df.parentname.nunique()
    len(feature_df), feature_df.parentname.nunique()

    # get max cents-per-gallon flags
    feature_df.replace(np.inf, 0.0, inplace=True) # need to replace inf values (happens if a customer has no transactions for the time period)
    feature_df.replace(-np.inf, 0.0, inplace=True)
    feature_df['max_cents_p_gallon4'] = feature_df.apply(lambda x: 1 if x['cents_p_gallon4'] > x['cents_p_gallon_before4'] else 0, axis=1)
    feature_df['max_cents_p_gallon8'] = feature_df.apply(lambda x: 1 if x['cents_p_gallon8'] > x['cents_p_gallon_before8'] else 0, axis=1)



    
    # get quintile scores for 1-52 range feature values
    for f in FEATURE_LIST:
        feature_df[f+'_avg'].replace(0,np.nan,inplace=True)
        feature_df[f + '_score'] = pd.qcut(feature_df[f+'_avg'], 4, labels=False, duplicates='drop')
        feature_df[f + '_score']=feature_df[f + '_score']+1 
        feature_df[f + '_score'].replace(np.nan,0,inplace=True)
        feature_df[f + '_score']=feature_df[f + '_score'].astype('int')
        feature_df[f+'_avg'].replace(np.nan,0,inplace=True)

    feature_df.fillna(0.0, inplace=True)

    # get % changes
    for f in FEATURE_LIST:
        tmp_df = weekly_df[['parentname', 'week_range_4','week_range_8', 'week_range_13', 'week_range_26',
                            f + '_avg4',f + '_avg8', f + '_avg13', f + '_avg26', f + '_avg']].drop_duplicates()
        rez_df = utils.get_pct_change(tmp_df, f)
        feature_df = feature_df.merge(rez_df, on='parentname', how='outer')
    len(feature_df), feature_df.parentname.nunique()

#     # drop the 1-52 range featuers (we only need _score)
#     feature_df.drop([f + '_avg' for f in FEATURE_LIST] + ['cents_p_gallon4', 'cents_p_gallon_before4', 'cents_p_gallon8', 'cents_p_gallon_before8'], axis=1, inplace=True)


    # fill nulls with 1.0 for inf (a pct change from 0 to X) / -1.0 for -inf (from 0 to -X)
    feature_df.replace(np.inf, 1.0, inplace=True)
    feature_df.replace(-np.inf, -1.0, inplace=True)
 
    feature_df.fillna(0.0, inplace=True)

    return feature_df


def run_flag_fee_features(weekly_df):

    # fee list
    FEES = ['late_interest_fee', 'program_fee', 'optional_program', 'risk_based_pricing', 'other_fees']

    # get num of invoices/refunds for each fee for weeks 1-4, 1-8, 1-13
    for fee_name in FEES:
        weekly_df[fee_name + '_inv_cnt4'] = weekly_df.loc[(weekly_df[fee_name + '_amount'] != 0) & (weekly_df.week_range_4 == 'range-1')]\
                                                .groupby(['parentname'])[fee_name + '_amount'].transform('count')
        weekly_df[fee_name + '_ref_cnt4'] = weekly_df.loc[(weekly_df[fee_name + '_refund'] != 0) & (weekly_df.week_range_4 == 'range-1')]\
                                                .groupby(['parentname'])[fee_name + '_refund'].transform('count')

        weekly_df[fee_name + '_inv_cnt8'] = weekly_df.loc[(weekly_df[fee_name + '_amount'] != 0) & (weekly_df.week_range_4.isin(['range-1', 'range-2']))]\
                                                .groupby(['parentname'])[fee_name + '_amount'].transform('count')
        weekly_df[fee_name + '_ref_cnt8'] = weekly_df.loc[(weekly_df[fee_name + '_refund'] != 0) & (weekly_df.week_range_4.isin(['range-1', 'range-2']))]\
                                                .groupby(['parentname'])[fee_name + '_refund'].transform('count')

        weekly_df[fee_name + '_inv_cnt13'] = weekly_df.loc[(weekly_df[fee_name + '_amount'] != 0) & (weekly_df.week_range_13 == 'range-1')]\
                                                .groupby(['parentname'])[fee_name + '_amount'].transform('count')
        weekly_df[fee_name + '_ref_cnt13'] = weekly_df.loc[(weekly_df[fee_name + '_refund'] != 0) & (weekly_df.week_range_13 == 'range-1')]\
                                                .groupby(['parentname'])[fee_name + '_refund'].transform('count')
        
        weekly_df[fee_name + '_inv_cnt26'] = weekly_df.loc[(weekly_df[fee_name + '_amount'] != 0) & (weekly_df.week_range_26 == 'range-1')]\
                                                .groupby(['parentname'])[fee_name + '_amount'].transform('count')
        weekly_df[fee_name + '_ref_cnt26'] = weekly_df.loc[(weekly_df[fee_name + '_refund'] != 0) & (weekly_df.week_range_26 == 'range-1')]\
                                                .groupby(['parentname'])[fee_name + '_refund'].transform('count')
        
        weekly_df[fee_name + '_inv_cnt52'] = weekly_df.loc[(weekly_df[fee_name + '_amount'] != 0) & (weekly_df.week_range_26.isin(['range-1', 'range-2']))]\
                                                .groupby(['parentname'])[fee_name + '_amount'].transform('count')
        weekly_df[fee_name + '_ref_cnt52'] = weekly_df.loc[(weekly_df[fee_name + '_refund'] != 0) & (weekly_df.week_range_26.isin(['range-1', 'range-2']))]\
                                                .groupby(['parentname'])[fee_name + '_refund'].transform('count')

        # feature markers for checking if a fee is 1st during a given period - fee counts for full lifetime BEFORE 4/8 week interval
        weekly_df[fee_name + '_inv_cnt_before4'] = weekly_df.loc[(weekly_df[fee_name + '_amount'] != 0) & (weekly_df.week_range_4 != 'range-1')]\
                                                .groupby(['parentname'])[fee_name + '_amount'].transform('count')
        weekly_df[fee_name + '_inv_cnt_before8'] = weekly_df.loc[(weekly_df[fee_name + '_amount'] != 0) & pd.isnull(weekly_df.week_range_4)]\
                                                .groupby(['parentname'])[fee_name + '_amount'].transform('count')

        # get num of fees for second-last intervals of 4/8 weeks (to check for opt-outs)
        weekly_df[fee_name + '_inv_cnt_prev_period4'] = weekly_df.loc[(weekly_df[fee_name + '_amount'] != 0) & (weekly_df.week_range_4 == 'range-2')]\
                                                .groupby('parentname')[fee_name + '_amount'].transform('count')
        weekly_df[fee_name + '_inv_cnt_prev_period8'] = weekly_df.loc[(weekly_df[fee_name + '_amount'] != 0) & (weekly_df.week_range_8 == 'range-2')]\
                                                .groupby('parentname')[fee_name + '_amount'].transform('count')
    for fee_name in FEES:   
        
        weekly_df[fee_name + '_refund_dollar_ratio4'] = utils.get_refund_dollar_ratio(weekly_df, fee_name, 'week_range_4')
        weekly_df[fee_name + '_refund_dollar_ratio8'] = utils.get_refund_dollar_ratio(weekly_df, fee_name, 'week_range_8')
        weekly_df[fee_name + '_refund_dollar_ratio13'] = utils.get_refund_dollar_ratio(weekly_df, fee_name, 'week_range_13')
        weekly_df[fee_name + '_refund_dollar_ratio26'] = utils.get_refund_dollar_ratio(weekly_df, fee_name, 'week_range_26')
        weekly_df[fee_name + '_refund_dollar_ratio52'] = utils.get_refund_dollar_ratio(weekly_df, fee_name, 'week_range_52')
                                               
      
    # get due days min/max per week range (need for due days increased/decreased flag)
   
   

    # get 1-row-per-customer DF with all inv_cnt / ref_cnt values
    FEATURE_LIST = [f + '_inv_cnt' + str(i) for f in FEES for i in [4, 8, 13,26,52]] + [f + '_ref_cnt' + str(i) for f in FEES for i in [4, 8, 13,26,52]] + \
                   [f + '_inv_cnt_before' + str(i) for f in FEES for i in [4, 8]] + \
                   [f + '_inv_cnt_prev_period' + str(i) for f in FEES for i in [4, 8]]+\
                   [f + '_refund_dollar_ratio' + str(i) for f in FEES for i in [4, 8,13,26,52]]
   
    feature_df =  weekly_df.sort_values(by=['parentname','week'],ascending=False)[['parentname'] + FEATURE_LIST ].groupby('parentname', as_index=False).first()
    assert len(feature_df) == feature_df.parentname.nunique()
    assert weekly_df.parentname.nunique() == feature_df.parentname.nunique()
    len(feature_df), feature_df.parentname.nunique()

    for fee_name in FEES: 
       # get share of late/program/risk invoices refunded for weeks 1-8 & 1-13 (1-4 is too short)
       feature_df[fee_name + '_ref_incidence_ratio4'] = feature_df[fee_name + '_ref_cnt4'] / feature_df[fee_name + '_inv_cnt4']
       feature_df[fee_name + '_ref_incidence_ratio8'] = feature_df[fee_name + '_ref_cnt8'] / feature_df[fee_name + '_inv_cnt8']
       feature_df[fee_name + '_ref_incidence_ratio13'] = feature_df[fee_name + '_ref_cnt13'] / feature_df[fee_name + '_inv_cnt13']
       feature_df[fee_name + '_ref_incidence_ratio26'] = feature_df[fee_name + '_ref_cnt26'] / feature_df[fee_name + '_inv_cnt26']
       feature_df[fee_name + '_ref_incidence_ratio52'] = feature_df[fee_name + '_ref_cnt52'] / feature_df[fee_name + '_inv_cnt52']
   
   

    # get '1st fee' flags
    for fee_name in FEES:
        feature_df[fee_name + '_1st_4'] = feature_df.apply(lambda x: 1 if (x[fee_name + '_inv_cnt4'] >= 1) & pd.isnull(x[fee_name + '_inv_cnt_before4']) else 0, axis=1)
        feature_df[fee_name + '_1st_8'] = feature_df.apply(lambda x: 1 if (x[fee_name + '_inv_cnt8'] >= 1) & pd.isnull(x[fee_name + '_inv_cnt_before8']) else 0, axis=1)
    # '2nd fee' flag only for late fee
    feature_df['late_interest_fee_2nd_4'] = feature_df.apply(lambda x: 1 if (x['late_interest_fee_inv_cnt4'] >= 2) & pd.isnull(x['late_interest_fee_inv_cnt_before4']) else 0, axis=1)
    feature_df['late_interest_fee_2nd_8'] = feature_df.apply(lambda x: 1 if (x['late_interest_fee_inv_cnt8'] >= 2) & pd.isnull(x['late_interest_fee_inv_cnt_before8']) else 0, axis=1)

  

    # get opt-out flags:
    # an opt-out is True when a fee is absent in the current period of X weeks and present in the previous period of same length
    for fee_name in FEES:
        feature_df[fee_name + '_optout4'] = feature_df.apply(lambda x: 1 if pd.isnull(x[fee_name + '_inv_cnt4']) & (x[fee_name + '_inv_cnt_prev_period4'] > 0) else 0, axis=1)
        feature_df[fee_name + '_optout8'] = feature_df.apply(lambda x: 1 if pd.isnull(x[fee_name + '_inv_cnt8']) & (x[fee_name + '_inv_cnt_prev_period8'] > 0) else 0, axis=1)


  

    # drop the BEFORE / PREV_PERIOD fetures
    feature_df.drop([f + '_inv_cnt_before' + str(i) for f in FEES for i in [4, 8]] + [f + '_inv_cnt_prev_period' + str(i) for f in FEES for i in [4, 8]], axis=1, inplace=True)
    # drop the DUE DAYS tmp features
#     feature_df.drop(TMP_FEATURE_LIST, axis=1, inplace=True)
    assert len(feature_df) == feature_df.parentname.nunique()

    feature_df.replace(np.inf, 1.0, inplace=True)
    feature_df.replace(-np.inf, -1.0, inplace=True)
    feature_df.replace(-0.0, 0, inplace=True)
    feature_df.fillna(0.0, inplace=True)  

    return feature_df
