import pandas as pd
import numpy as np

def run_static_features(static_df, last_w_df):
     
    static_features=['parentname','first_trx_month', 'last_transaction_week',
                   'last_transaction_month', 'state_code', 'num_of_trucks', 'credit_limit',
                    'billing_freq', 'due_days','credit_grade','tenure','channel','product_type_most_frequent',
                    'churn_flag',  
                    'customer_size','customer_segment','last_payment_method']
    
    
    # num of trucks
    static_df['num_of_trucks'].fillna('unknown',inplace=True)
    static_df['num_of_trucks']= static_df['num_of_trucks'].str.lower()
    static_df['num_of_trucks']= static_df['num_of_trucks'].astype('str')
    truck_types=['b.1-11 trucks','c.12-20 trucks','d.21-50 trucks','e.51-100 trucks','f.101-250 trucks',
                  'g.251-500 trucks','h.over 500 trucks','unknown']
    static_df['num_of_trucks']=static_df['num_of_trucks'].apply(lambda x: x if x in  truck_types else 'others')
    
    
    #Customer size
    selected_small_trucks=['b.1-11 trucks','c.12-20 trucks','d.21-50 trucks','e.51-100 trucks']
    selected_medium_trucks=['f.101-250 trucks','g.251-500 trucks']
    selected_large_trucks=['h.over 500 trucks']
    static_df['customer_size']=np.where(static_df['num_of_trucks'].isin(selected_small_trucks),'small',
                                 np.where(static_df['num_of_trucks'].isin(selected_medium_trucks),'medium',
                                   np.where(static_df['num_of_trucks'].isin(selected_large_trucks),'large','others')))
                               
    
    #Remove accts they have more than 100 trucks
    #selected_truck_flag=['a.express check account','b.1-11 trucks','c.12-20 trucks','d.21-50 trucks','e.51-100 trucks','others']
    #static_df=static_df.loc[static_df['num_of_trucks'].isin(selected_truck_flag)]
#     assert len(static_df) == static_df.parentname.nunique()
  
    
    
    # State code: make 10 categories - top-10 + OTHER
    static_df['state_code'].fillna('UNK',inplace=True)
    top_9_states = ['CA','TX','FL','OH','IL','GA','MO','NC','TN','PA','UNK']
    static_df['state_code']= static_df['state_code'].str.upper()
    static_df['state_code']= static_df['state_code'].astype('str')
    static_df['state_code'] = static_df['state_code'].apply(lambda x: x if x in top_9_states else 'OTHERS')
    
    # Channel
    static_df['channel_1'].replace('Unk',np.nan,inplace=True)
    static_df['channel']=static_df['channel_1']
    static_df['channel']=np.where(static_df['channel'].isnull(),static_df['channel_2'],static_df['channel'])
    static_df['channel'].fillna('unknown',inplace=True)
    static_df['channel']=static_df['channel'].astype('str')
    static_df['channel']=static_df['channel'].str.lower()
    static_df['channel']=np.where(static_df['channel'].str.contains('inbound'),'inbound',\
                             np.where(static_df['channel'].str.contains('outbound'),'outbound',\
                             np.where(static_df['channel'].str.contains('field'),'field',\
                             np.where(static_df['channel'].str.contains('telesales'),'outbound',\
                             np.where(static_df['channel'].str.contains('regional'),'strategic/AM',\
                             np.where(static_df['channel'].str.contains('national'),'strategic/AM',\
                             np.where(static_df['channel'].str.contains('strategic'),'strategic/AM',\
                             np.where(static_df['channel'].str.contains('account mgmt'),'strategic/AM',\
                                      np.where(static_df['channel'].str.contains('unknown'),'unknown','others')))))))))
   
    #billing freq
    static_df['billing_freq'].fillna('unknown',inplace=True)
    static_df['billing_freq']=static_df['billing_freq'].str.lower()
    static_df['billing_freq']=static_df['billing_freq'].astype ('str')
    billing_freq_types=['semi-monthly','daily','weekly','monthly','unknown']
    static_df['billing_freq']=static_df['billing_freq'].apply(lambda x: x if x  in billing_freq_types  else 'others')
   
    # Product information (First Most freq in last 3 months)
    static_df['product_information_1'].fillna('others',inplace=True)
    static_df['product_information_2'].fillna('others',inplace=True)
    static_df['product_information_3'].fillna('others',inplace=True)
    static_df['product_information_4'].fillna('others',inplace=True)
    static_df['product_information_5'].fillna('others',inplace=True)
    static_df['product_information_6'].fillna('others',inplace=True)

    
    static_df['product_type_combined']=static_df[['product_information_1','product_information_2',
       'product_information_3', 'product_information_4',
       'product_information_5', 'product_information_6']].values.tolist()
    
    static_df['product_type'] = static_df.apply(product_type_filter, axis=1) 
    
    ECASH_ONLY=['ECASH']
    ECASH_COMCHECK=['ECASH_EXPRESS CHECK']
    ECASH_COMCHECK_FUEL=['ECASH_EXPRESS CHECK_FUEL','ECASH_EXPRESS CHECK_FUEL_MASTERCARD','ECASH_EXPRESS CHECK_FUEL_PROPRIETARY','ECASH_EXPRESS CHECK_PROPRIETARY']
    ECASH_FUEL=['ECASH_FUEL','ECASH_FUEL_MASTERCARD','ECASH_FUEL_PROPRIETARY','ECASH_MASTERCARD','ECASH_PROPRIETARY']
    COMCHECK_ONLY=['EXPRESS CHECK']
    COMCHECK_FUEL=['EXPRESS CHECK_FUEL','EXPRESS CHECK_FUEL_MASTERCARD','EXPRESS CHECK_FUEL_MASTERCARD_PROPRIETARY','EXPRESS CHECK_MASTERCARD','EXPRESS CHECK_FUEL_PROPRIETARY','EXPRESS CHECK_PROPRIETARY']
    FUEL_ONLY=['FUEL','FUEL_MASTERCARD','FUEL_MASTERCARD_PROPRIETARY','FUEL_PROPRIETARY','MASTERCARD_PROPRIETARY','PROPRIETARY','MASTERCARD']
   
    static_df['product_type']=static_df['product_type'].str.strip()
    static_df['product_type_most_frequent']=np.where(static_df['product_type'].isin(ECASH_ONLY),'ECASH_ONLY',
                                       np.where(static_df['product_type'].isin(ECASH_COMCHECK),'ECASH_COMCHECK',
                                          np.where(static_df['product_type'].isin(ECASH_COMCHECK_FUEL),'ECASH_COMCHECK_FUEL',
                                              np.where(static_df['product_type'].isin(ECASH_FUEL),'ECASH_FUEL',
                                                  np.where(static_df['product_type'].isin(COMCHECK_ONLY),'COMCHECK_ONLY',
                                                        np.where(static_df['product_type'].isin(COMCHECK_FUEL),'COMCHECK_FUEL',
                                                            np.where(static_df['product_type'].isin(FUEL_ONLY),'FUEL_ONLY', 'OTHERS')))))))

   
    
    #first transaction month
    static_df['first_trx_month'].fillna(static_df['first_trx_month'].median(),inplace=True)
    
    
    #Credit limit
    static_df['credit_limit'].fillna(static_df['credit_limit'].median(),inplace=True)
    static_df['credit_limit'].replace(0,static_df['credit_limit'].median(),inplace=True)
    static_df['credit_limit']=static_df['credit_limit'].astype('int64')
   
   
    #Num of days to pay
    static_df['due_days'].fillna(static_df['due_days'].median(), inplace=True)
    static_df['due_days']=static_df['due_days'].astype('int64')
    

    bins=[-1,300,579,669,739,799,np.inf]
    labels_name=['Worse','Poor','Fair','Good','Very-Good','Excellent']
    static_df['credit_grade'] = pd.cut(static_df['credit_score'], bins, labels=labels_name)   
    static_df['credit_grade']= static_df['credit_grade'].astype('str')
    static_df['credit_grade'].replace('nan','UNK',inplace=True)    
    

    # merge the full DF (inner join since last_w_df will have customers active at least MIN_TRANSACTION_WEEKS in 2019)
    static_df = static_df.merge(last_w_df[['parentname', 'yearmonth', 'week']], left_on='parentname', right_on='parentname', how='inner')
    assert len(static_df) == static_df.parentname.nunique()

    
    
    #Calculate tenure
    static_df['yearmonth']=static_df['yearmonth'].astype('int64')
    static_df['tenure']=((pd.to_datetime(static_df['yearmonth'],format='%Y%m')-                   pd.to_datetime(static_df['first_trx_month'],format='%Y%m'))/np.timedelta64(1, 'D'))/365
    static_df['tenure']=np.round(static_df['tenure'],2)
    #Create Customer segment
    static_df['customer_segment']=np.where(static_df['tenure']<=1,'NEW','EXISTING')
    
    
    
    
     # churn flag
    static_df['churn_flag'] = static_df['churn_flag'].apply(lambda x: 0 if x == 'ACTIVE' else 1)
#     print(static_df.churn_flag.value_counts(dropna=False))
    
    
    static_df['last_payment_method'].fillna('unknown',inplace=True)
    last_payment_types=['CDN/ACH', 'Check', 'EBP/FIS', 'Wire', 'CustACH',
       'WesternUnion','unknown']
    static_df['last_payment_method']=static_df['last_payment_method'].apply(lambda x: x if x in  last_payment_types else 'unknown')
    
  
    
    
    return static_df[static_features]


def product_type_filter(row):
  
        row=list(filter(('others').__ne__, row['product_type_combined']))
        row=list(filter(('NON-CORE').__ne__, row))  
        row=list(set(row))
        row=sorted(row)
        row='_'.join(row)
        return row

