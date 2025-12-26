import numpy as np
import pandas as pd

def run_sf_call_features(sf_df):

    # call types not to be split into subtypes
    TYPES = ['account details and maintenance', 'card pin maintenance', 'crosssell', 'express check', 'fincen documentation', 'location site issues',
            'misdirected call', 'online system inquiry', 'payment support', 'return check', 'security call back', 'service account disruption',
            'tax inquiry', 'tech support', 'transactions', 'ts and cs acceptance', 'credit fraud inquiry', 'survey', 'close inquiry']
    # call types to be split into subtypes
    TYPES_TO_SPLIT = ['billing and payments', 'fees', 'rebates pricing savings']
    SUBTYPES = ['balance due request', 'check by phone', 'general reconciliation research', 'invoice copy request', 'lock protection declined',
                'lock protection enrolled', 'missing statement', 'other', 'partial payment posted', 'payment date request', 'payment did not post',
                'payment posted twice', 'payment status request', 'refund request', 'rege or escheatment requests', 'report request', 'accelerator rewards',
                'card fees', 'clean advantage fee', 'convenience network fee', 'fleetdash', 'fraud protection product', 'high risk', 'late fee finance charge',
                'level 2', 'membership', 'minimum program admin fee', 'out of network', 'over credit limit', 'reporting fee', 'convenience network',
                'rebate inquiry', 'universal pricing', 'volume discount', 'nsf', 'eft', 'collection']

    # keep subtype here but not use at this time

    sf_df[['type', 'subtype']] = sf_df[['type', 'subtype']].fillna('')
    sf_df[['type', 'subtype']] = sf_df[['type', 'subtype']].replace('\s+', ' ', regex=True)
    sf_df[['type', 'subtype']] = sf_df[['type', 'subtype']].replace('/', '', regex=True)
    sf_df[['type', 'subtype']] = sf_df[['type', 'subtype']].replace('’', '', regex=True)
    sf_df[['type', 'subtype']] = sf_df[['type', 'subtype']].replace(' +', ' ', regex=True)
    
    sf_df['type'] = sf_df['type'].str.lower()
    sf_df['subtype'] = sf_df['subtype'].str.lower()
#     sf_df['createddate'] = sf_df['createddate'].apply(lambda x: pd.to_datetime(x).date())

    len(sf_df), sf_df.customer_id.nunique()

    sf_df.head()

    sf_df['type'].replace('post call survey call back', 'survey', inplace=True)
    sf_df['type'].replace('post call survey followup', 'survey', inplace=True)
    sf_df['type'].replace('quarterly survey followup', 'survey', inplace=True)

    sf_df['type'].replace('locationsite issues', 'location site issues', inplace=True)
    sf_df['type'].replace('site', 'location site issues', inplace=True)

    sf_df['type'].replace('nextraq cross sell', 'crosssell', inplace=True)
    sf_df['type'].replace('ecs cross sell', 'crosssell', inplace=True)

    sf_df['type'].replace('cardpin maintenance', 'card pin maintenance', inplace=True)
    sf_df['type'].replace('card denial inquiry', 'card pin maintenance', inplace=True)
    sf_df['type'].replace('loststolen card', 'card pin maintenance', inplace=True)

    sf_df['type'].replace('billing & payments', 'billing and payments', inplace=True)
    sf_df['type'].replace('billing payments', 'billing and payments', inplace=True)
    sf_df['type'].replace('payment issue', 'billing and payments', inplace=True)
    sf_df['type'].replace('billing inquiry', 'billing and payments', inplace=True)
    sf_df['type'].replace('check by phone', 'billing and payments', inplace=True)

    sf_df.loc[sf_df.type.str.contains('eft'), 'subtype'] = 'eft'
    sf_df['type'].replace('eft new customer request', 'billing and payments', inplace=True)
    sf_df['type'].replace('eft setup in process', 'billing and payments', inplace=True)
    sf_df['type'].replace('eft change bank request', 'billing and payments', inplace=True)
    sf_df['type'].replace('eft status inquiry', 'billing and payments', inplace=True)
    sf_df['type'].replace('eft unenroll request', 'billing and payments', inplace=True)
    sf_df['type'].replace('eft un-enroll request', 'billing and payments', inplace=True)

    sf_df.loc[sf_df.type.str.contains('collection'), 'subtype'] = 'collection'
    sf_df['type'].replace('collections attempt', 'billing and payments', inplace=True)

    sf_df.loc[sf_df.type.str.contains('refund request'), 'subtype'] = 'refund request'
    sf_df['type'].replace('refund request', 'billing and payments', inplace=True)

    sf_df['type'].replace('serviceaccount disruption', 'service account disruption', inplace=True)
    sf_df['type'].replace('service interruption', 'service account disruption', inplace=True)
    sf_df['type'].replace('delinquency', 'service account disruption', inplace=True)
    sf_df['type'].replace('locked account', 'service account disruption', inplace=True)

    sf_df['type'].replace('credit inquiry', 'credit fraud inquiry', inplace=True)
    sf_df['type'].replace('fraud investigation detection', 'credit fraud inquiry', inplace=True)
    sf_df['type'].replace('fraud inquiry', 'credit fraud inquiry', inplace=True)

    sf_df['type'].replace('account details & maintenance', 'account details and maintenance', inplace=True)
    sf_df['type'].replace('account details maintenance', 'account details and maintenance', inplace=True)
    sf_df['type'].replace('account maintenance', 'account details and maintenance', inplace=True)
    sf_df['type'].replace('credit line increase', 'account details and maintenance', inplace=True)
    sf_df['type'].replace('terms and conditions request', 'account details and maintenance', inplace=True)
    sf_df['type'].replace('term changes', 'account details and maintenance', inplace=True)
    sf_df['type'].replace('term change', 'account details and maintenance', inplace=True)

    sf_df['type'].replace('disputed transaction', 'transactions', inplace=True)
    sf_df['type'].replace('transaction inquiry', 'transactions', inplace=True)
    sf_df['type'].replace('transaction', 'transactions', inplace=True)
    sf_df['type'].replace('proprietary transaction', 'transactions', inplace=True)
    sf_df['type'].replace('mastercard transaction', 'transactions', inplace=True)
    sf_df['type'].replace('authorization', 'transactions', inplace=True)
    sf_df['type'].replace('dispute status request', 'transactions', inplace=True)

    sf_df['type'].replace('ts and cs acceptance test', 'ts and cs acceptance', inplace=True)

    sf_df['type'].replace('pricing inquiry', 'rebates pricing savings', inplace=True)

    sf_df.loc[sf_df.type.str.contains('nsf'), 'subtype'] = 'nsf'
    sf_df['type'].replace('nsf', 'billing and payments', inplace=True)

    sf_df['type'].replace('close account', 'close inquiry', inplace=True)
    sf_df['type'].replace('after hours close inquiry', 'close inquiry', inplace=True)


    ## change two billing category into fee category
    sf_df.loc[sf_df.subtype == "late fee finance charge", 'type'] = "fees"
    sf_df.loc[sf_df.subtype == "high risk", 'type'] = "fees"

    ## status chagne flag
    sf_df['statuschangeflag'] = [1 if x >=3 else 0 for x in sf_df['statuschanges']]
    
    #sf_df=sf_df[['customer_id', 'createddate', 'call_week', 'type']].drop_duplicates()
    selected_types=['account details and maintenance', 'card pin maintenance','express check','tech support','billing and payments', 'fees','online system inquiry','transactions','close inquiry']
    
    # regroup categories
#     sf_df=sf_df[['parentname','call_week', 'type']].drop_duplicates()
    sf_df['type']=np.where(sf_df['type'].isin(selected_types),sf_df['type'],'Others')
    
    sf_df.drop(['customer_id','statuschanges', 'subtype','satisficationflag','lastmodifieddate','closeddate'],axis=1,inplace=True)
    
    ## aggregate at account and week level
    sf_type_df_1 = sf_df.loc[sf_df.type == 'account details and maintenance']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_1.rename(columns={'type':'account details and maintenance'},inplace=True)
    
    sf_type_df_2 = sf_df.loc[sf_df.type == 'card pin maintenance']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_2.rename(columns={'type':'card pin maintenance'},inplace=True)
    
    sf_type_df_3 = sf_df.loc[sf_df.type == 'express check']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_3.rename(columns={'type':'express check'},inplace=True)
    
    sf_type_df_4 = sf_df.loc[sf_df.type == 'tech support']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_4.rename(columns={'type':'tech support'},inplace=True)
    
    sf_type_df_5 = sf_df.loc[sf_df.type == 'billing and payments']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_5.rename(columns={'type':'billing and payments'},inplace=True)
    
    sf_type_df_6 = sf_df.loc[sf_df.type == 'fees']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_6.rename(columns={'type':'fees'},inplace=True)
    
    sf_type_df_7 = sf_df.loc[sf_df.type == 'online system inquiry']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_7.rename(columns={'type':'online system inquiry'},inplace=True)
    
    sf_type_df_8 = sf_df.loc[sf_df.type == 'transactions']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_8.rename(columns={'type':'transactions'},inplace=True)
    
    
    sf_type_df_9 = sf_df.loc[sf_df.type == 'close inquiry']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_9.rename(columns={'type':'close inquiry'},inplace=True)
    
    sf_type_df_10 = sf_df.loc[sf_df.type == 'Others']\
                        .groupby(['parentname', 'call_week'])['type'].count().reset_index()
    sf_type_df_10.rename(columns={'type':'Others'},inplace=True)
    
    sf_sat_ct_df = sf_df.groupby(['parentname', 'call_week'])['statuschangeflag'].count().reset_index()
    sf_sat_ct_df.rename(columns={'statuschangeflag':'statuschangeflag_count'},inplace=True)
    
    sf_sat_sum_df = sf_df.groupby(['parentname', 'call_week'])['statuschangeflag'].sum().reset_index()
    sf_sat_sum_df.rename(columns={'statuschangeflag':'statuschangeflag_sum'},inplace=True)
    
    sf_df=sf_df[['parentname','call_week']].drop_duplicates()
    
    sf_cb_1=sf_df.merge(sf_sat_ct_df, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_sat_sum_df, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_1, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_2, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_3, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_4, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_5, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_6, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_7, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_8, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_9, on=['parentname','call_week'], how='outer')
    sf_cb_1=sf_cb_1.merge(sf_type_df_10, on=['parentname','call_week'], how='outer')
    
    feature_df = sf_cb_1
    feature_df.fillna(0,inplace=True)
    
    return feature_df


def generate_sf_call_features(weekly_df):
    
    selected_types=['account details and maintenance', 'card pin maintenance','express check','tech support','billing and payments', 'fees','online system inquiry','transactions']
    
    for type_name in selected_types:
        weekly_df['freq_'+type_name+'_52']=get_call_case_count(weekly_df,'week_range_52',type_name)
        weekly_df['freq_'+type_name+'_8']=get_call_case_count(weekly_df,'week_range_8',type_name)
    
    weekly_df['status_change_lt3_ratio_52']=get_status_change_ratio(weekly_df,'week_range_52')
    weekly_df['status_change_lt3_ratio_26']=get_status_change_ratio(weekly_df,'week_range_26')
    weekly_df['status_change_lt3_ratio_8']=get_status_change_ratio(weekly_df,'week_range_8')

    feature_df=weekly_df.sort_values(by=['parentname','week'],ascending=False)[['parentname'] + ['freq_'+t+'_52'for t in selected_types] +\
               ['freq_'+t+'_8'for t in selected_types]+['status_change_lt3_ratio_52', 'status_change_lt3_ratio_26','status_change_lt3_ratio_8']].groupby('parentname', as_index=False).first()
    
    feature_df.replace(np.inf, 1.0, inplace=True)
    feature_df.replace(-np.inf, -1.0, inplace=True)
 
    feature_df.fillna(0.0, inplace=True)
    feature_df['status_change_lt3_ch_1-52_1-8']=feature_df['status_change_lt3_ratio_52']-feature_df['status_change_lt3_ratio_8']
    feature_df['status_change_lt3_ch_1-52_1-26']=feature_df['status_change_lt3_ratio_52']-feature_df['status_change_lt3_ratio_26']
    
    feature_df.fillna(0.0, inplace=True)
    return feature_df

def get_call_case_count(feature_df,groupby_clause,feature_name):
    df=feature_df[['parentname',groupby_clause,feature_name]].copy()
    df[feature_name + '_total_count'] = df.loc[df[groupby_clause]=='range-1'].groupby('parentname')[feature_name].transform('sum')
    df[feature_name + '_cnt'] = df.loc[(df[groupby_clause]=='range-1')&(df[feature_name] != 0)]\
        .groupby('parentname')[feature_name].transform('count')
    return  df[feature_name + '_total_count']/df[feature_name + '_cnt']

def get_status_change_ratio(feature_df,groupby_clause):
    df=feature_df[['parentname',groupby_clause,'statuschangeflag_count','statuschangeflag_sum']].copy()
    df['statuschangeflag_total_count'] = df.loc[df[groupby_clause]=='range-1'].groupby('parentname')['statuschangeflag_count'].transform('sum')
    #sum of statusflag change >=3
    df['statuschangeflag_cnt'] = df.loc[df[groupby_clause]=='range-1'].groupby('parentname')['statuschangeflag_sum'].transform('sum')
    
    return  df['statuschangeflag_cnt']/df['statuschangeflag_total_count']
        