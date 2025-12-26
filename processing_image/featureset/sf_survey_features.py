import numpy as np
import pandas as pd
from featureset import utils

## survey features generation
def run_sf_survey_features(sf_df):
    
    ### issue solved indicator
    sf_df['issuesolvedflag'] = [1 if x =='Absolutely' else 0 for x in sf_df['issuesolvedind']]
    sf_df['issuesolved_all_count']= sf_df.groupby(['parentname', 'case_week'])['issuesolvedflag'].transform('count')
    
    sf_df['issuesolved_solved_count'] = sf_df.groupby(['parentname', 'case_week'])['issuesolvedflag'].transform('sum')
    
    ## rating weekly average
    sf_df['avg_product_rating'] = sf_df.loc[sf_df['productrating']>0].groupby(['parentname', 'case_week'])['productrating'].transform('mean')
    sf_df['avg_agent_rating'] = sf_df.loc[sf_df['agentrating']>0].groupby(['parentname', 'case_week'])['agentrating'].transform('mean')
    
    sf_df['productrating_cnt'] = sf_df.loc[sf_df['productrating']>0].groupby(['parentname', 'case_week'])['productrating'].transform('count')
    sf_df['agentrating_cnt']= sf_df.loc[sf_df['productrating']>0].groupby(['parentname', 'case_week'])['agentrating'].transform('count')
    ## for the count, include non missing only?
    
    ## number of promoter/detractor
    sf_df['product_promoter_cnt'] = sf_df.loc[sf_df.product_nps == 'Promoter']\
                        .groupby(['parentname', 'case_week'])['product_nps'].transform('count')
    sf_df['product_detractor_cnt'] = sf_df.loc[sf_df.product_nps == 'Detractor']\
                        .groupby(['parentname', 'case_week'])['product_nps'].transform('count')
    sf_df['agent_promoter_cnt'] = sf_df.loc[sf_df.agent_nps == 'Promoter']\
                        .groupby(['parentname', 'case_week'])['agent_nps'].transform('count')
    sf_df['agent_detractor_cnt'] = sf_df.loc[sf_df.agent_nps == 'Detractor']\
                        .groupby(['parentname', 'case_week'])['agent_nps'].transform('count')
    sf_df['total_counts'] = sf_df.groupby(['parentname', 'case_week'])['case_week'].transform('count')

    sf_df.fillna(0,inplace=True)
    
    return sf_df


def generate_sf_surv_features(weekly_df):
    
    
    weekly_df['product_rating_avg_52']=weekly_df.loc[(weekly_df['avg_product_rating']>0)&(weekly_df['week_range_52']=='range-1')].groupby('parentname')['avg_product_rating'].transform('mean')
    weekly_df['product_rating_avg_8']=weekly_df.loc[(weekly_df['avg_product_rating']>0)&(weekly_df['week_range_8']=='range-1')].groupby('parentname')['avg_product_rating'].transform('mean')
    weekly_df['product_rating_avg_26']=weekly_df.loc[(weekly_df['avg_product_rating']>0)&(weekly_df['week_range_26']=='range-1')].groupby('parentname')['avg_product_rating'].transform('mean')

    weekly_df['agent_rating_avg_52']=weekly_df.loc[(weekly_df['avg_agent_rating']>0)&(weekly_df['week_range_52']=='range-1')].groupby('parentname')['avg_agent_rating'].transform('mean')
    weekly_df['agent_rating_avg_8']=weekly_df.loc[(weekly_df['avg_agent_rating']>0)&(weekly_df['week_range_8']=='range-1')].groupby('parentname')['avg_agent_rating'].transform('mean')
    weekly_df['agent_rating_avg_26']=weekly_df.loc[(weekly_df['avg_agent_rating']>0)&(weekly_df['week_range_26']=='range-1')].groupby('parentname')['avg_agent_rating'].transform('mean')

    features=['productrating_cnt','agentrating_cnt', 'product_promoter_cnt', 'product_detractor_cnt',
       'agent_promoter_cnt', 'agent_detractor_cnt']
    for feat in features:
        weekly_df[feat+'_ratio_52']=get_ratio(weekly_df,'week_range_52',feat,'total_counts')
        weekly_df[feat+'_ratio_26']=get_ratio(weekly_df,'week_range_26',feat,'total_counts')
        weekly_df[feat+'_ratio_8']=get_ratio(weekly_df,'week_range_8',feat,'total_counts')

    weekly_df['issue_solved_ratio_52']=get_ratio(weekly_df,'week_range_52','issuesolved_solved_count','issuesolved_all_count')
    weekly_df['issue_solved_ratio_26']=get_ratio(weekly_df,'week_range_26','issuesolved_solved_count','issuesolved_all_count')
    weekly_df['issue_solved_ratio_8']=get_ratio(weekly_df,'week_range_8','issuesolved_solved_count','issuesolved_all_count')

    feature_df=weekly_df.sort_values(by=['parentname','week'],ascending=False)[['parentname']+['product_rating_avg_52','product_rating_avg_8','product_rating_avg_26','agent_rating_avg_52','agent_rating_avg_8','agent_rating_avg_26']+\
                                                                               [f+'_ratio_'+i for f in features for i in ['52','26','8']]+['issue_solved_ratio_52','issue_solved_ratio_26','issue_solved_ratio_8']].groupby('parentname', as_index=False).first()
    feature_df.replace(np.inf, 1.0, inplace=True)
    feature_df.replace(-np.inf, -1.0, inplace=True)
 
    feature_df.fillna(0.0, inplace=True)
    for feat in features:
        feature_df[feat+'_ch_1-52_1-26']=feature_df[feat+'_ratio_52']-feature_df[feat+'_ratio_26']
        feature_df[feat+'_ch_1-52_1-8']=feature_df[feat+'_ratio_52']-feature_df[feat+'_ratio_8']
    
    feature_df['issue_solved'+'_ch_1-52_1-26']=feature_df['issue_solved_ratio_52']-feature_df['issue_solved_ratio_26']   
    feature_df['issue_solved'+'_ch_1-52_1-8']=feature_df['issue_solved_ratio_52']-feature_df['issue_solved_ratio_8']

    feature_df['product_rating_avg'+'_ch_1-52_1-8']=feature_df['product_rating_avg_52']-feature_df['product_rating_avg_8']
    feature_df['product_rating_avg'+'_ch_1-52_1-26']=feature_df['product_rating_avg_52']-feature_df['product_rating_avg_26']

    feature_df['agent_rating_avg'+'_ch_1-52_1-8']=feature_df['agent_rating_avg_52']-feature_df['agent_rating_avg_8']
    feature_df['agent_rating_avg'+'_ch_1-52_1-26']=feature_df['agent_rating_avg_52']-feature_df['agent_rating_avg_26']
    feature_df.fillna(0.0, inplace=True)
    
    return feature_df


def get_ratio(feature_df,groupby_clause,featurename,total_count_feature):
    df=feature_df[['parentname',groupby_clause,featurename,total_count_feature]].copy()
    df[featurename+'_total_count'] = df.loc[df[groupby_clause]=='range-1'].groupby('parentname')[total_count_feature].transform('sum')
    #sum of statusflag change >=3
    df[featurename+'_cnt'] = df.loc[df[groupby_clause]=='range-1'].groupby('parentname')[featurename].transform('sum')
    
    return  df[featurename+'_cnt']/df[featurename+'_total_count']