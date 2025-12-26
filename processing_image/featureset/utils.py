import pandas as pd
import numpy as np

def preprocess_weekly_data(weekly_df,static_df, min_transaction_weeks, week_offset, random_cutoff_week_for_active):
  
    # only leave customers with at least MIN_TRANSACTION_WEEKS active weeks
    active_weeks = weekly_df.loc[(weekly_df['total_gallons'] > 0) | (weekly_df['total_nonfuel_spend'] != 0)|\
                                      (weekly_df['total_ecashspend'] != 0) |(weekly_df['total_vcomcheckspend'] != 0) |\
                                      (weekly_df['total_comcheck_spend'] != 0)|(weekly_df['total_vcap_spend']!=0)]\
                                      .groupby('parentname').size()
    sufficient_history_custs = active_weeks.loc[active_weeks >= min_transaction_weeks].index
    len(sufficient_history_custs)
    weekly_df = weekly_df.loc[weekly_df.parentname.isin(sufficient_history_custs)]
#     print(len(weekly_df)), print(weekly_df.parentname.nunique())
    

    weekly_df = weekly_df.sort_values(by=['parentname', 'week']).reset_index()

    # week gets an object dtype when coming directly from a query, need to make sure it's int
    weekly_df['week'] = weekly_df['week'].astype(int)
    
#     # get min/max activity weeks, cut anything pre/post that
    weekly_df['min_lifetime_week'] = weekly_df['week'].where((weekly_df['total_gallons'] > 0) | \
                                                             (weekly_df['total_nonfuel_spend'] != 0)|\
                                                             (weekly_df['total_ecashspend'] != 0) |\
                                                             (weekly_df['total_vcomcheckspend'] != 0) |\
                                                             (weekly_df['total_comcheck_spend'] != 0)|\
                                                             (weekly_df['total_vcap_spend']!=0))\
                                                      .groupby(weekly_df['parentname'])\
                                                      .transform('min').astype(int)
    weekly_df['max_lifetime_week'] = weekly_df['week'].where((weekly_df['total_gallons'] > 0) | \
                                                             (weekly_df['total_nonfuel_spend'] != 0)|\
                                                             (weekly_df['total_ecashspend'] != 0) |\
                                                             (weekly_df['total_vcomcheckspend'] != 0) |\
                                                             (weekly_df['total_comcheck_spend'] != 0)|\
                                                             (weekly_df['total_vcap_spend']!=0))\
                                                      .groupby(weekly_df['parentname'])\
                                                      .transform('max').astype(int)

   
   

    weekly_df = weekly_df.loc[(weekly_df.week >= weekly_df.min_lifetime_week) & (weekly_df.week <= weekly_df.max_lifetime_week)]
    assert len(weekly_df.loc[pd.isnull(weekly_df['max_lifetime_week'])]) == 0
    assert len(weekly_df.loc[pd.isnull(weekly_df['min_lifetime_week'])]) == 0
   
    if random_cutoff_week_for_active:

        weekly_df['week_rank'] = weekly_df.groupby('parentname')['week'].rank(ascending=True)
        weekly_df['max_rank'] = weekly_df['week_rank'].groupby(weekly_df['parentname']).transform('max')

        np.random.seed(0)
        active_custs = static_df.loc[static_df.churn_flag == 'ACTIVE'].parentname.unique()
        tmp_df = weekly_df[['parentname', 'max_rank']].drop_duplicates()

        # having ranked weeks from 1 to max, assign a new max value to ACTIVE customers
        # the random value shouldn't be too close to 1 (important for newer customers) and also not too far back (for customers with long transaction history)
        tmp_df['max_rank_rand'] = tmp_df.apply(lambda x: np.random.randint(max(min_transaction_weeks-1, x['max_rank']-52), x['max_rank'])
                                                         if (x['parentname'] in active_custs) else x['max_rank'], axis=1)

        assert len(tmp_df) == tmp_df.parentname.nunique()
        weekly_df = weekly_df.merge(tmp_df[['parentname', 'max_rank_rand']], left_on='parentname', right_on='parentname', how='left')
        weekly_df.drop('max_rank', axis=1, inplace=True)
        weekly_df.rename(columns={'max_rank_rand': 'max_rank'}, inplace=True)

        weekly_df = weekly_df.loc[weekly_df.week_rank <= weekly_df.max_rank]

        weekly_df['max_lifetime_week'] = weekly_df['week'].groupby(weekly_df['parentname']).transform('max')


    assert len(weekly_df) == len(weekly_df.loc[(weekly_df.week >= weekly_df.min_lifetime_week) & (weekly_df.week <= weekly_df.max_lifetime_week)])
#     print(len(weekly_df)), print(weekly_df.parentname.nunique())

    # for each customer define non-overlapping 4 week ranges
    weekly_df['week_rank'] = weekly_df.groupby('parentname')['week'].rank(ascending=False)
    weekly_df['week_range_4'] = weekly_df.apply(get_4week_range, offset=week_offset, axis=1)
    weekly_df['week_range_8'] = weekly_df.apply(get_8week_range, offset=week_offset, axis=1)
    weekly_df['week_range_13'] = weekly_df.apply(get_13week_range, offset=week_offset, axis=1)
    weekly_df['week_range_26'] = weekly_df.apply(get_26week_range, offset=week_offset, axis=1)
    weekly_df['week_range_52'] = weekly_df.apply(get_52week_range, offset=week_offset, axis=1)

    # cut everything not in the 52-week range (minus the offset)
    weekly_df = weekly_df.loc[~pd.isnull(weekly_df.week_range_26)]  #.reset_index()
#     print(weekly_df.parentname.nunique())
    # check for min_transaction_weeks again (some customers could get less because of week_offset)
    active_weeks = weekly_df.groupby('parentname').size()
    sufficient_history_custs = active_weeks.loc[active_weeks >= min_transaction_weeks].index

    return weekly_df.loc[weekly_df.parentname.isin(sufficient_history_custs)].reset_index()
   

def get_4week_range(row, offset):
    '''
    label data rows with as time intervals of 4-week length
    '''
    if (row['week_rank'] >= 1 + offset) & (row['week_rank'] <= 4 + offset):
        return 'range-1'
    elif (row['week_rank'] > 4 + offset) & (row['week_rank'] <= 8 + offset):
        return 'range-2'
    else:
        return None


def get_8week_range(row, offset):
    '''
    label data rows with as time intervals of 8-week length
    '''
    if (row['week_rank'] >= 1 + offset) & (row['week_rank'] <= 8 + offset):
        return 'range-1'
    elif (row['week_rank'] > 8 + offset) & (row['week_rank'] <= 16 + offset):
        return 'range-2'
    else:
        return None


def get_13week_range(row, offset):
    '''
    label data rows with as time intervals of 13-week length (3 months)
    '''
    if (row['week_rank'] >= 1 + offset) & (row['week_rank'] <= 13 + offset):
        return 'range-1'
    elif (row['week_rank'] > 13 + offset) & (row['week_rank'] <= 26 + offset):
        return 'range-2'
    else:
        return None


def get_26week_range(row, offset):
    '''
    label data rows with as time intervals of 26-week length (0.5 year)
    '''
    if (row['week_rank'] >= 1 + offset) & (row['week_rank'] <= 26 + offset):
        return 'range-1'
    elif (row['week_rank'] > 26 + offset) & (row['week_rank'] <= 52 + offset):
        return 'range-2'
    else:
        return None
    
def get_52week_range(row, offset):
    '''
    label data rows with as time intervals of 26-week length (0.5 year)
    '''
    if (row['week_rank'] >=1 + offset) & (row['week_rank'] <= 52 + offset):
        return 'range-1'
    else:
        return None    

def get_share_of_active_weeks(weekly_df, week_range, feature):
    '''
    get share of active fuel/nonfuel weeks for a given time range
    '''
  
    df = weekly_df[['parentname', week_range, feature]].copy()
    df['num_active'] = df.loc[(df[week_range] == 'range-1') & (df[feature] != 0)].groupby('parentname')[feature].transform('count')
    df['week_cnt'] = df.loc[df[week_range] == 'range-1'].groupby('parentname')[feature].transform('count')
    return df['num_active'] / df['week_cnt']


def get_avg_per_active_week(weekly_df, feature_name, groupby_clause):
    '''
    get average metric values per active week for a given time range (included in groupby_clause)
    '''
    
    df = weekly_df[['parentname',groupby_clause,feature_name]].copy()
    df[feature_name + '_total'] = df.loc[df[groupby_clause]=='range-1'].groupby('parentname')[feature_name].transform('sum')
    df[feature_name + '_cnt'] = df.loc[(df[groupby_clause]=='range-1')&(df[feature_name] != 0)]\
        .groupby('parentname')[feature_name].transform('count')
    return df[feature_name + '_total'] / df[feature_name + '_cnt']

def get_cents_per_gallon(weekly_df, feature_name, groupby_clause):
    '''
    get cents per gallon for a given time range (included in groupby_clause)
    '''
    df = weekly_df[['parentname']+[groupby_clause] + [feature_name] + ['total_gallons']].copy()
    df[feature_name + '_total'] = df.loc[df[groupby_clause] == 'range-1'].groupby('parentname')[feature_name].transform('sum')
    df['gallons_total'] = df.loc[df[groupby_clause] == 'range-1'].groupby('parentname')['total_gallons'].transform('sum')
    return df[feature_name + '_total'] / df['gallons_total']


def get_overall_cents_per_gallon(weekly_df, range, current_or_before):
    '''
    get cents-per-gallon across all fees for the CURRENT week range OR for the full lifetime BEFORE the range
    '''
    df = weekly_df[['parentname', 'tot_fee_amount', 'total_gallons', range]].copy()

    if current_or_before == 'BEFORE':
        df['fees_total'] = df.loc[df[range] != 'range-1'].groupby('parentname')['tot_fee_amount'].transform('sum')
        df['gallons_total'] = df.loc[df[range] != 'range-1'].groupby('parentname')['total_gallons'].transform('sum')
    else:
        df['fees_total'] = df.loc[df[range] == 'range-1'].groupby('parentname')['tot_fee_amount'].transform('sum')
        df['gallons_total'] = df.loc[df[range] == 'range-1'].groupby('parentname')['total_gallons'].transform('sum')

    return df['fees_total'] / df['gallons_total']


def get_share_of_invoices(weekly_df, fee_name, groupby_clause):
    '''
    get the share of a given fee invoices (out of all fee invoices) for a given time range (included in groupby_clause)
    '''
    df = weekly_df[['parentname']+[groupby_clause] + [fee_name] + ['unique_types_of_fees_applied']].copy()
   
    df[fee_name + '_cnt'] = df.loc[(df[fee_name] != 0)&(df[groupby_clause] == 'range-1')].groupby('parentname')[fee_name].transform('count')
    df['week_cnt'] = df.loc[df[groupby_clause] == 'range-1'].groupby('parentname')[fee_name].transform('count')
    return df[fee_name + '_cnt'] / df['week_cnt']

def get_refund_dollar_ratio(weekly_df, fee_name, groupby_clause):
    '''
    get refund_dollar ratio for a given time range (included in groupby_clause)
    '''
    df = weekly_df[['parentname']+[groupby_clause] + [fee_name+'_amount'] +[fee_name+'_refund']].copy()
   
    df[fee_name + '_amount_total'] = df.loc[(df[groupby_clause] == 'range-1')].groupby('parentname')[fee_name+'_amount'].transform('sum')
    df[fee_name + '_refund_total'] = df.loc[(df[groupby_clause] == 'range-1')].groupby('parentname')[fee_name+'_refund'].transform('sum')
    return  (df[fee_name + '_refund_total']/df[fee_name + '_amount_total'])*-1


def get_pct_change(df, feature):
    '''
    get % change features between four pairs of time intervals: weeks 1-4 vs 5-8, 1-4 vs 1-13, 1-4 vs 1-26, 1-4 vs 1-52
    '''
 
    # percentage change across different columns
    range_4_averages = df.loc[~pd.isnull(df[feature + '_avg4'])].drop_duplicates(subset=['parentname', 'week_range_4', feature + '_avg4'])\
        .pivot(index='parentname', columns='week_range_4', values=feature + '_avg4')
    if range_4_averages.empty:
            range_4_averages = df.drop_duplicates(subset=['parentname', 'week_range_4', feature + '_avg4'])\
             .pivot(index='parentname', columns='week_range_4', values=feature + '_avg4')
           
    #     print(range_4_averages)
    range_8_averages = df.loc[~pd.isnull(df[feature + '_avg8'])].drop_duplicates(subset=['parentname', 'week_range_8', feature + '_avg8'])\
        .pivot(index='parentname', columns='week_range_8', values=feature + '_avg8')
    if range_8_averages.empty:
            range_8_averages = df.drop_duplicates(subset=['parentname', 'week_range_8', feature + '_avg8'])\
             .pivot(index='parentname', columns='week_range_8', values=feature + '_avg8')
        
    #     print(range_4_averages)
    range_13_averages = df.loc[~pd.isnull(df[feature + '_avg13'])].drop_duplicates(subset=['parentname', 'week_range_13', feature + '_avg13'])\
        .pivot(index='parentname', columns='week_range_13', values=feature + '_avg13')
    if range_13_averages.empty:
            range_13_averages = df.drop_duplicates(subset=['parentname', 'week_range_13', feature + '_avg13'])\
             .pivot(index='parentname', columns='week_range_13', values=feature + '_avg13')
    
    
    range_26_averages = df.loc[~pd.isnull(df[feature + '_avg26'])].drop_duplicates(subset=['parentname', 'week_range_26', feature + '_avg26'])\
        .pivot(index='parentname', columns='week_range_26', values=feature + '_avg26')
    if range_26_averages.empty:
            range_26_averages = df.drop_duplicates(subset=['parentname', 'week_range_26', feature + '_avg26'])\
             .pivot(index='parentname', columns='week_range_26', values=feature + '_avg26')

#     print(range_26_averages)
    range_4_averages.rename(columns={'range-1': 'w1-4'}, inplace=True)
    range_8_averages.rename(columns={'range-1': 'w1-8'}, inplace=True)
    range_13_averages.rename(columns={'range-1': 'w1-13'}, inplace=True)
    range_26_averages.rename(columns={'range-1': 'w1-26'}, inplace=True)
    
    range_52_averages= df.loc[~pd.isnull(df[feature + '_avg'])][['parentname', feature + '_avg']].drop_duplicates()
    if range_52_averages.empty:
        range_52_averages= df[['parentname', feature + '_avg']].drop_duplicates()
    cross_changes = range_52_averages\
        .merge(range_4_averages, left_on='parentname', right_index=True, how='outer')\
        .merge(range_8_averages, left_on='parentname', right_index=True, how='outer')\
        .merge(range_13_averages, left_on='parentname', right_index=True, how='outer')\
        .merge(range_26_averages, left_on='parentname', right_index=True, how='outer')
   
    cross_changes.fillna(0,inplace=True)
    cross_changes[feature + '_ch_1-8_1-4'] = (cross_changes['w1-4'] - cross_changes['w1-8']) / cross_changes['w1-8']
    cross_changes[feature + '_ch_1-13_1-4'] = (cross_changes['w1-4'] - cross_changes['w1-13']) / cross_changes['w1-13']
    cross_changes[feature + '_ch_1-26_1-4'] = (cross_changes['w1-4'] - cross_changes['w1-26']) / cross_changes['w1-26']
    cross_changes[feature + '_ch_1-52_1-4'] = (cross_changes['w1-4'] - cross_changes[feature + '_avg']) / cross_changes[feature + '_avg']

    # full featureset
    fulldf = cross_changes[['parentname',feature + '_ch_1-8_1-4' ,feature + '_ch_1-13_1-4', feature + '_ch_1-26_1-4', feature + '_ch_1-52_1-4']]
        

    return fulldf


def get_precision_at_n(y_true, y_score, n):

    # the label of the positive (target) class
    pos_label = 1
    n_pos = np.sum(y_true == pos_label)

    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:n])
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, n) such that the best achievable score is always 1.0.
    return float(n_relevant) / min(n_pos, n)

def generate_explanations(shap_values, prediction_df, model_columns, static_features):

    shap_df = pd.DataFrame(shap_values).reset_index()
    shap_flat_df = shap_df.melt(id_vars='index')
    shap_flat_df['variable'] = shap_flat_df['variable'].apply(lambda x: model_columns[x])
    shap_flat_df.rename(columns={'index': 'customer_index', 'variable': 'feature'}, inplace=True)
    
    # group by customer_id, sort by value desc, get top N
    shap_flat_df['value_abs'] = shap_flat_df['value'].apply(lambda x: abs(x))
    shap_flat_df['value_abs_sum'] = shap_flat_df.groupby('customer_index').value_abs.transform('sum')
    shap_flat_df['impact'] = shap_flat_df['value'] / shap_flat_df['value_abs_sum']
    # remove STATIC_FEATURES
    shap_flat_df = shap_flat_df.loc[~shap_flat_df.feature.isin(static_features)]
    top_factors_df = shap_flat_df.sort_values('value_abs', ascending = False).groupby('customer_index').head(10)

    # add original feature values from prediction_df
    prediction_df_copy = prediction_df.reset_index()
    prediction_df_copy.rename(columns={'index': 'customer_index'}, inplace=True)
#     print(prediction_df_copy)
    original_feature_val_df = prediction_df_copy.melt(id_vars='customer_index', value_vars=model_columns)
    original_feature_val_df.rename(columns={'value': 'original_value'}, inplace=True)
    top_factors_df = top_factors_df.merge(original_feature_val_df, left_on=['customer_index', 'feature'], right_on=['customer_index', 'variable'], how='left')

    # convert values to string
    top_factors_df['impact'] = top_factors_df['impact'].apply(lambda x: str(np.round(x, 2)))
    top_factors_df['original_value'] = top_factors_df['original_value'].apply(lambda x: str(np.round(x, 3)))

    top_factors_df['factor'] = top_factors_df.apply(lambda x:
                                '{\'feature\': \'' + x['feature'] + '\', \'impact\': ' + x['impact'] + ', \'value\': ' + x['original_value'] + '}',
                                axis=1)
    # get one row per customer
    return top_factors_df.groupby('customer_index').agg(
                            factor_1 = ('factor', lambda x: x.iloc[0]),
                            factor_2 = ('factor', lambda x: x.iloc[1]),
                            factor_3 = ('factor', lambda x: x.iloc[2]),
                            factor_4 = ('factor', lambda x: x.iloc[3]),
                            factor_5 = ('factor', lambda x: x.iloc[4]),
                            factor_6 = ('factor', lambda x: x.iloc[5]),
                            factor_7 = ('factor', lambda x: x.iloc[6]),
                            factor_8 = ('factor', lambda x: x.iloc[7]),
                            factor_9 = ('factor', lambda x: x.iloc[8]),
                            factor_10 = ('factor', lambda x: x.iloc[9]),   
                                )


def generate_explanations_random_forest(shap_values, prediction_df, model_columns, static_features):

    

    shap_df = pd.DataFrame(shap_values[1]).reset_index()
    shap_flat_df = shap_df.melt(id_vars='index')
    shap_flat_df['variable'] = shap_flat_df['variable'].apply(lambda x: model_columns[x])
    shap_flat_df.rename(columns={'index': 'customer_index', 'variable': 'feature'}, inplace=True)

    # group by customer_id, sort by value desc, get top N
    shap_flat_df['value_abs'] = shap_flat_df['value'].apply(lambda x: abs(x))
    shap_flat_df['value_abs_sum'] = shap_flat_df.groupby('customer_index').value_abs.transform('sum')
    shap_flat_df['impact'] = shap_flat_df['value'] / shap_flat_df['value_abs_sum']
    # remove STATIC_FEATURES
    shap_flat_df = shap_flat_df.loc[~shap_flat_df.feature.isin(static_features)]
    top_factors_df = shap_flat_df.sort_values('value_abs', ascending = False).groupby('customer_index').head(25)

    # add original feature values from prediction_df
    prediction_df_copy = prediction_df.reset_index()
    prediction_df_copy.rename(columns={'index': 'customer_index'}, inplace=True)
    
    original_feature_val_df = prediction_df_copy.melt(id_vars='customer_index', value_vars=model_columns)
    original_feature_val_df.rename(columns={'value': 'original_value'}, inplace=True)
    top_factors_df = top_factors_df.merge(original_feature_val_df, left_on=['customer_index', 'feature'], right_on=['customer_index', 'variable'], how='left')

    # convert values to string
    top_factors_df['impact'] = top_factors_df['impact'].apply(lambda x: str(np.round(x, 2)))
    top_factors_df['original_value'] = top_factors_df['original_value'].apply(lambda x: str(np.round(x, 3)))

    top_factors_df['factor'] = top_factors_df.apply(lambda x:
                                '{\'feature\': \'' + x['feature'] + '\', \'impact\': ' + x['impact'] + ', \'value\': ' + x['original_value'] + '}',
                                axis=1)
    
    top_10_factor_df = top_factors_df.groupby('customer_index').agg(
                            factor_1 = ('feature', lambda x: x.iloc[0]),
                            factor_2 = ('feature', lambda x: x.iloc[1]),
                            factor_3 = ('feature', lambda x: x.iloc[2]),
                            factor_4 = ('feature', lambda x: x.iloc[3]),
                            factor_5 = ('feature', lambda x: x.iloc[4]),

                            factor_6 = ('feature', lambda x: x.iloc[5]),
                            factor_7 = ('feature', lambda x: x.iloc[6]),
                            factor_8 = ('feature', lambda x: x.iloc[7]),
                            factor_9 = ('feature', lambda x: x.iloc[8]),
                            factor_10 = ('feature', lambda x: x.iloc[9])
                            )
    return  top_10_factor_df

def create_validation_table_metric(df):
       
      
        validation_df=pd.DataFrame(columns=['Last_Transaction_Weeks','Weeks','#Actual_churners','#Predicted_churners','Percentage_Prediction'])
   
        validation_df=validation_df.append({'Last_Transaction_Weeks':'<202050','Weeks':'Past 8 Weeks',\
                                        '#Actual_churners':df[df['last_transaction_week']<202050]['parentname'].count(),\
                                        '#Predicted_churners':df[df['last_transaction_week']<202050]['churn_flag_predicted'].sum()\
                                        },ignore_index=True)
    
        validation_df=validation_df.append({'Last_Transaction_Weeks':'202050-202101','Weeks':'4 Weeks',\
                                         '#Actual_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202101)]['parentname'].count(),\
                                         '#Predicted_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202101)]['churn_flag_predicted'].sum()\
                                         },ignore_index=True)
        validation_df=validation_df.append({'Last_Transaction_Weeks':'202050-202105','Weeks':'8 Weeks',\
                                         '#Actual_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202105)]['parentname'].count(),\
                                         '#Predicted_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202105)]['churn_flag_predicted'].sum()\
                                         },ignore_index=True)
        validation_df=validation_df.append({'Last_Transaction_Weeks':'202050-202110','Weeks':'13 Weeks',\
                                         '#Actual_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202110)]['parentname'].count(),\
                                         '#Predicted_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202110)]['churn_flag_predicted'].sum()\
                                         },ignore_index=True)
        validation_df=validation_df.append({'Last_Transaction_Weeks':'202050-202120','Weeks':'23 Weeks',\
                                         '#Actual_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202120)]['parentname'].count(),\
                                         '#Predicted_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202120)]['churn_flag_predicted'].sum()\
                                         },ignore_index=True)
            
        validation_df=validation_df.append({'Last_Transaction_Weeks':'202050-202130','Weeks':'33 Weeks',\
                                         '#Actual_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202130)]['parentname'].count(),\
                                         '#Predicted_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202130)]['churn_flag_predicted'].sum()\
                                         },ignore_index=True)    
        validation_df=validation_df.append({'Last_Transaction_Weeks':'202050-202150','Weeks':'52 Weeks',\
                                         '#Actual_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202150)]['parentname'].count(),\
                                         '#Predicted_churners':df[(df['last_transaction_week']>=202050)&(df['last_transaction_week']<=202150)]['churn_flag_predicted'].sum()\
                                         },ignore_index=True)
        validation_df['Percentage_Prediction']=(validation_df['#Predicted_churners']/validation_df['#Actual_churners'])*100
        
        
        return validation_df