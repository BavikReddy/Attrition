import pandas as pd
import numpy as np

def combining_accts_at_parentlevel(static_df,dynamic_df,DATA_MAP):
    static_df=static_df[~static_df['parentname'].isnull()]
    static_df.drop_duplicates(subset='customer_id',keep='first',inplace=True)
    static_df['parentname_count']=static_df.groupby('parentname')['customer_id'].transform('count')

    dynamic_df=dynamic_df[~dynamic_df['customer_id'].isnull()]
    dynamic_df=dynamic_df.groupby(['customer_id','week']).sum().reset_index()   
    
   # Separate one parent accts and multiple parent accts
    static_df_one_parent_acct=static_df[static_df['parentname_count']==1]
    print("Total number of one parent accts:",format(len(static_df_one_parent_acct)))

    static_df_multiple_parent_acct=static_df[static_df['parentname_count']>1]
    print("Total number of multiple parent accts:",format(len(static_df_multiple_parent_acct)))

    # static_df_multiple_parent_acct['last_transaction_week']=static_df_multiple_parent_acct['last_transaction_week'].astype('int64')
    # static_df_multiple_parent_acct['last_transaction_month']=static_df_multiple_parent_acct['last_transaction_month'].astype('int64')
    static_df_multiple_parent_acct['last_transaction_week_original']=static_df_multiple_parent_acct['last_transaction_week']

    static_df_multiple_parent_acct['net_volume']=static_df_multiple_parent_acct.groupby('parentname')['volume'].transform('max')
    static_df_multiple_parent_acct['net_spend']=static_df_multiple_parent_acct.groupby('parentname')['total_spend'].transform('max')
    static_df_multiple_parent_acct['net_trx']=static_df_multiple_parent_acct.groupby('parentname')['total_num_trx'].transform('max')
    static_df_multiple_parent_acct['credit_limit']=static_df_multiple_parent_acct.groupby('parentname')['credit_limit'].transform('sum')
    static_df_multiple_parent_acct['first_trx_month']=static_df_multiple_parent_acct.groupby('parentname')['first_trx_month'].transform('min')
    static_df_multiple_parent_acct['last_transaction_week']=static_df_multiple_parent_acct.groupby('parentname')['last_transaction_week'].transform('max')
    static_df_multiple_parent_acct['total_producttype_spend']=static_df_multiple_parent_acct.groupby('parentname')['total_producttype_spend'].transform('max')


    #Get high volume customers
    static_df_multiple_parent_acct=static_df_multiple_parent_acct[static_df_multiple_parent_acct['volume']==static_df_multiple_parent_acct['net_volume']]
    #Get duplicate parent name  have zero volume or same volume
    static_df_multiple_parent_acct_zero_volume=static_df_multiple_parent_acct[static_df_multiple_parent_acct['parentname'].duplicated(keep=False)]
    # #Remove zero volume or same volume accts from multiple_parent accts
    static_df_multiple_parent_acct=static_df_multiple_parent_acct[~static_df_multiple_parent_acct['customer_id']\
                                                              .isin(list(static_df_multiple_parent_acct_zero_volume.customer_id))]



    #For parent name has zero volume , consider total_trx to manage the duplicates
    static_df_multiple_parent_acct_zero_volume=static_df_multiple_parent_acct_zero_volume[static_df_multiple_parent_acct_zero_volume['total_num_trx']==static_df_multiple_parent_acct_zero_volume['net_trx']]
    #Get duplicate parent name or parent names have zero trx or same trx
    static_df_multiple_parent_acct_zero_trx=static_df_multiple_parent_acct_zero_volume[static_df_multiple_parent_acct_zero_volume['parentname'].duplicated(keep=False)]
    #Remove zero trx accts from zero volume accts
    static_df_multiple_parent_acct_zero_volume=static_df_multiple_parent_acct_zero_volume[~static_df_multiple_parent_acct_zero_volume['customer_id']\
                                                              .isin(list(static_df_multiple_parent_acct_zero_trx.customer_id))]

    # for parent name has zero trx,consider product type spend to manage the duplicates 
    static_df_multiple_parent_acct_zero_trx=static_df_multiple_parent_acct_zero_trx[static_df_multiple_parent_acct_zero_trx['total_producttype_spend']==static_df_multiple_parent_acct_zero_trx['total_producttype_spend']]
    #Get duplicate parent name have zero product spend or same product spend
    static_df_multiple_parent_acct_zero_product_spend=static_df_multiple_parent_acct_zero_trx[static_df_multiple_parent_acct_zero_trx['parentname'].duplicated(keep=False)]
    # Remove zero product spend accts from zero trx accts
    static_df_multiple_parent_acct_zero_trx=static_df_multiple_parent_acct_zero_trx[~static_df_multiple_parent_acct_zero_trx['customer_id']\
                                                              .isin(list(static_df_multiple_parent_acct_zero_product_spend.customer_id))]


    # for parent name has zero product spend or same product spend,consider orginal last_trx_week  to manage the duplicates 
    static_df_multiple_parent_acct_zero_product_spend=static_df_multiple_parent_acct_zero_product_spend[static_df_multiple_parent_acct_zero_product_spend['last_transaction_week_original']==static_df_multiple_parent_acct_zero_product_spend['last_transaction_week']]
    #Get duplicate parent name have same last_trx_week
    static_df_multiple_parent_acct_same_last_trx_week=static_df_multiple_parent_acct_zero_product_spend[static_df_multiple_parent_acct_zero_product_spend['parentname'].duplicated(keep=False)]
    # Remove accts have same last trx week
    static_df_multiple_parent_acct_zero_product_spend=static_df_multiple_parent_acct_zero_product_spend[~static_df_multiple_parent_acct_zero_product_spend['customer_id']\
                                                              .isin(list(static_df_multiple_parent_acct_same_last_trx_week.customer_id))]


    # #Remove
    static_df_multiple_parent_acct_same_last_trx_week.drop_duplicates(subset='parentname',keep='first',inplace=True)

    static_df_multiple_parent_acct_combined=pd.concat([static_df_multiple_parent_acct,static_df_multiple_parent_acct_zero_volume,\
                                                  static_df_multiple_parent_acct_zero_trx,static_df_multiple_parent_acct_zero_product_spend,\
                                                  static_df_multiple_parent_acct_same_last_trx_week ],sort=False)




    static_df_multiple_parent_acct_combined.drop_duplicates(subset='parentname',keep='first',inplace=True)

   #combine one parent accts and multiple accts together
    static_df_combined=pd.concat([static_df_one_parent_acct,static_df_multiple_parent_acct_combined],sort=False)
    assert len(static_df_combined['parentname'].unique())==len(static_df['parentname'].unique())

   #Calculate churn flag
    static_df_combined['last_transaction_week']=static_df_combined['last_transaction_week'].astype('int64')
    static_df_combined['churn_flag']=np.where(static_df_combined['last_transaction_week']>int(DATA_MAP['churn_cutoff_week']),'ACTIVE','CHURNER')
    print("Total num of parentname in static data:",format(len(static_df_combined.parentname.unique())))


   #Calculating the sum of dynamic features based on parent name

    dynamic_df=dynamic_df.merge(static_df[['customer_id','parentname']],on='customer_id',how='inner')
    dynamic_df=dynamic_df[~dynamic_df['parentname'].isnull()]
    dynamic_df_combined=dynamic_df.groupby(['parentname','week']).sum().reset_index()
    assert len(dynamic_df_combined['parentname'].unique())==len(static_df_combined['parentname'].unique())
    print("Total num of parentname in dynamic data:",format(len(dynamic_df_combined.parentname.unique())))
    assert len(dynamic_df_combined['parentname'].unique())==len(static_df_combined['parentname'].unique())
    
    return static_df_combined,dynamic_df_combined
