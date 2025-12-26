----Pull active ctsomers in MC and PROP------------------------
WITH ACTIVE_CUSTS_UNIQUE AS
(
   SELECT 
              TRIM(dt.ACCTCODE) as ACCTCODE
   FROM 
              prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
               LEFT JOIN  prod_share.edl_repl.repl_nat_newlob CO 
               ON  TRIM(dt.ACCTCODE) = CO.ACCTCODE
             LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',dt.TXNDATE)
    WHERE CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{activity_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
              AND  CO.FINALLOB = 'NAT'
               
   GROUP BY dt.ACCTCODE
  
  UNION 
    
  SELECT F.FMDL_CUST_ACCOUNT_CODE AS ACCTCODE
    FROM  prod_share.edl_repl.repl_fmlog F
    LEFT  JOIN prod_share.edl_repl.repl_nat_newlob CO 
                       ON FMDL_CUST_ACCOUNT_CODE = CO.ACCTCODE
    LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',F.FMDL_rk_transaction_date )
    WHERE CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{activity_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
              AND FMDL_SC_SRVC_CNTR_CODE NOT IN ('MC901','MC902','MC903','MC904','MC905')
             AND  CO.FINALLOB = 'NAT'
    GROUP BY  F.FMDL_CUST_ACCOUNT_CODE
),
-------------Pull  Cal month corresponding to the current week----------------------------
PULL_CAL_MONTH AS(
select CONCAT(CAL.YEAR, LPAD(CAL.FM_WEEK, 2, '0')) AS CURRENT_WEEK,
       CONCAT(CAL.YEAR, LPAD(CAL.FM_MONTH, 2, '0')) AS CURRENT_MONTH

from prod_share.edl_repl.REPL_MART_CALENDAR CAL WHERE CAL.WEEK_NO='{current_week}'
),
-----------------------Active customers in MC-----------------------------------
ACTIVE_MC AS
(
 SELECT 
              TRIM(dt.ACCTCODE) as ACCTCODE,
               SUM( ISNULL(dt.Diesel_Gallons,0) + ISNULL(dt.Gas_Gallons,0))  AS VOLUME,
               SUM(dt.SPEND) AS TOTAL_SPEND,
               SUM (dt.TXNS) AS TOTAL_NUM_TRX
               
   FROM 
              prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
              LEFT  JOIN  prod_share.edl_repl.repl_nat_newlob CO 
                    ON  TRIM(dt.ACCTCODE) = CO.ACCTCODE
              LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',DT.TXNDATE )
             WHERE CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{activity_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
                        AND  CO.FINALLOB = 'NAT'
   GROUP BY dt.ACCTCODE

),
--------------Active customers in PROP-------------------------
ACTIVE_PROP AS 
(
SELECT  
                   F.FMDL_CUST_ACCOUNT_CODE AS ACCTCODE,
                   SUM((NVL(F.FMDL_TR_OTHER_GALLONS, 0) + NVL(F.FMDL_TOTAL_GALLONS, 0))) AS VOLUME,
                   SUM( NVL(F.FMDL_TR_TRACTOR_COST,0)
	+ NVL(F.FMDL_TR_TRAILER_COST,0)
	+ (NVL(F.FMDL_TR_OTHER_COST ,0)
	+ NVL(F.FMDL_TR_OIL_COST,0)
	+ NVL(F.FMDL_TR_CASH_ADVANCE_AMOUNT,0) 
	+ NVL(F.FMDL_TR_PRODUCT_AMOUNT_1,0) 
	+ NVL(F.FMDL_TR_PRODUCT_AMOUNT_2,0) 
	+ NVL(F.FMDL_TR_PRODUCT_AMOUNT_3,0)))  as TOTAL_SPEND,
                   SUM(CASE WHEN  (NVL(F.FMDL_TF_TRACTOR_FUEL,0) + NVL(F.FMDL_TF_TRAILER_FUEL,0)+NVL(F.FMDL_TF_OTHER_FUEL,0)+
             NVL(F.FMDL_TF_PRODUCT_1,0) +NVL(F.FMDL_TF_PRODUCT_2,0) 
                +NVL(F.FMDL_TF_PRODUCT_3,0) + NVL(F.FMDL_TF_OIL,0)+
                NVL(F.FMDL_TF_CASH,0)) >0 THEN 1 ELSE 0 END )AS TOTAL_NUM_TRX
     
     FROM  prod_share.edl_repl.repl_fmlog F
    LEFT  JOIN  prod_share.edl_repl.repl_nat_newlob CO 
                       ON FMDL_CUST_ACCOUNT_CODE = CO.ACCTCODE
    LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',F.FMDL_rk_transaction_date )
    WHERE CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{activity_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
                  AND FMDL_SC_SRVC_CNTR_CODE NOT IN ('MC901','MC902','MC903','MC904','MC905')
                AND CO.FINALLOB = 'NAT' 
    GROUP BY  F.FMDL_CUST_ACCOUNT_CODE
), 
-----------------------------Combine fuel information of MC and Prop-------------------------
FUEL_STATS AS 
(
 SELECT ACCTCODE,
 SUM(VOLUME) AS VOLUME,
 SUM(TOTAL_SPEND) AS TOTAL_SPEND,
 SUM(TOTAL_NUM_TRX) AS TOTAL_NUM_TRX
 
 FROM (
    SELECT * FROM ACTIVE_MC
    UNION ALL 
    SELECT * FROM ACTIVE_PROP
 )
 GROUP BY ACCTCODE
 ORDER BY ACCTCODE
) ,
LAST_TRX AS 
(
  SELECT ACCTCODE, MAX(LAST_TRANSACTION_WEEK) AS LAST_TRANSACTION_WEEK,
          MAX(LAST_TRANSACTION_MONTH) AS LAST_TRANSACTION_MONTH ,
          MAX(TRIM_LAST_TRX_MONTH) AS TRIM_LAST_TRX_MONTH
  FROM(
       SELECT 
              TRIM(dt.ACCTCODE) as ACCTCODE,
               MAX(CAL.WEEK_NO)::int AS LAST_TRANSACTION_WEEK,
                MAX(CONCAT(CAL.YEAR, LPAD(CAL.FM_MONTH, 2, '0')))::int AS LAST_TRANSACTION_MONTH,
               CASE WHEN MAX(CONCAT(CAL.YEAR, LPAD(CAL.FM_MONTH, 2, '0')))>MAX(PCAL.CURRENT_MONTH) THEN MAX(PCAL.CURRENT_MONTH) ELSE MAX(CONCAT(CAL.YEAR, LPAD(CAL.FM_MONTH, 2, '0'))) END AS TRIM_LAST_TRX_MONTH          
                
        FROM 
              prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
              LEFT  JOIN  prod_share.edl_repl.repl_nat_newlob CO 
              ON  TRIM(dt.ACCTCODE) = CO.ACCTCODE
             LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',DT.TXNDATE )
             CROSS JOIN PULL_CAL_MONTH PCAL
   
         WHERE   
             CO.FINALLOB = 'NAT'
   GROUP BY dt.ACCTCODE

   UNION ALL 

   SELECT  
         F.FMDL_CUST_ACCOUNT_CODE AS ACCTCODE,
       MAX(CAL.WEEK_NO)::int AS LAST_TRANSACTION_WEEK,
       MAX(CONCAT(CAL.YEAR, LPAD(CAL.FM_MONTH, 2, '0')))::int AS LAST_TRANSACTION_MONTH,
    CASE WHEN MAX(CONCAT(CAL.YEAR, LPAD(CAL.FM_MONTH, 2, '0')))>MAX(PCAL.CURRENT_MONTH) THEN MAX(PCAL.CURRENT_MONTH) ELSE MAX(CONCAT(CAL.YEAR, LPAD(CAL.FM_MONTH, 2, '0'))) END AS TRIM_LAST_TRX_MONTH            
            
     FROM  prod_share.edl_repl.repl_fmlog F
    LEFT  JOIN  prod_share.edl_repl.repl_nat_newlob CO 
                       ON FMDL_CUST_ACCOUNT_CODE = CO.ACCTCODE
   LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',F.FMDL_rk_transaction_date  )
     CROSS JOIN PULL_CAL_MONTH PCAL                  
   
      WHERE  FMDL_SC_SRVC_CNTR_CODE NOT IN ('MC901','MC902','MC903','MC904','MC905')
     AND CO.FINALLOB = 'NAT' 
    GROUP BY  F.FMDL_CUST_ACCOUNT_CODE
  )
  GROUP BY ACCTCODE
) ,
NAT_REVENUE AS
(
    SELECT NAT.ACCTCODE,
           NAT.TRUCKSFLAG  AS NUM_OF_TRUCKS,
           NAT.parentfirsttxnflag AS FIRST_TRX_MONTH,
           NAT.parentname AS PARENTNAME
    FROM  prod_share.edl_repl.repl_nat_account_attributes NAT
    INNER JOIN ACTIVE_CUSTS_UNIQUE CUSTS ON CUSTS.ACCTCODE=NAT.ACCTCODE
),
CREDIT_LIMIT AS
(
    SELECT 
                XK501_ACCT_CODE AS ACCTCODE,
                MAX( XK501_CREDIT_LIMIT) AS CREDIT_LIMIT
     FROM
                prod_share.edl_repl.repl_geac_501_record
    WHERE XK501_CREDIT_LIMIT>0 
    GROUP BY XK501_ACCT_CODE

),
BILL_FREQ AS(
    
    SELECT ACCTCODE,billing_freq
    FROM(
           SELECT
                 substring(CUST_ACCOUNT_NUMBER  from 10 for 5) AS ACCTCODE,
                 DA.billing_cycle AS billing_freq,
                 ROW_NUMBER() OVER ( PARTITION BY substring(CUST_ACCOUNT_NUMBER  from 10 for 5) ORDER BY MODIFIED_DATE DESC) RNUM
           FROM
                 prod_share.edl_repl.repl_ccex_customer CU
          LEFT JOIN prod_share.edl_repl.repl_ccex_v_udf_data DA ON DA.CUST_SYSTEM_CODE=CU.CUST_SYSTEM_CODE 
          INNER JOIN ACTIVE_CUSTS_UNIQUE CUST ON CUST.ACCTCODE=substring(CUST_ACCOUNT_NUMBER  from 10 for 5)
          WHERE   DA.billing_cycle <> 'NULL'  AND  DA.billing_cycle<>'N/A'
          )WHERE RNUM=1        

),
DUE_DAYS AS(
   SELECT ACCTCODE,due_days
   FROM(
           SELECT
                 substring(CUST_ACCOUNT_NUMBER  from 10 for 5)  AS ACCTCODE,
                 DA.terms_customer AS due_days,
                 ROW_NUMBER() OVER ( PARTITION BY substring(CUST_ACCOUNT_NUMBER  from 10 for 5)  ORDER BY MODIFIED_DATE DESC) RNUM
           FROM
                prod_share.edl_repl.repl_ccex_customer CU
           LEFT JOIN prod_share.edl_repl.repl_ccex_v_udf_data DA ON DA.CUST_SYSTEM_CODE=CU.CUST_SYSTEM_CODE 
           INNER JOIN ACTIVE_CUSTS_UNIQUE CUST ON CUST.ACCTCODE=substring(CUST_ACCOUNT_NUMBER  from 10 for 5) 
           WHERE    DA.terms_customer <>'NULL' and DA.terms_customer <>'N/A'
           )WHERE RNUM=1       

),
CUST_STATE AS 
(
   SELECT 
              AcctCode AS ACCTCODE,
              ST AS CUST_STATE 
   FROM  prod_share.edl_repl.repl_fin1_tbl_dailyaccountsdownload
),
CREDIT_SCORE AS
(
  SELECT APP.AccountCode AS ACCTCODE,
         APP.CBRScore AS CREDIT_SCORE,
         APP.CBRScore_Original,APP.FICO_RefreshScore,APP.FICO_RefreshDate 
  FROM  prod_share.edl_repl.repl_fin1_tbl_smallfleetaccounts APP
),


PRODUCT_INFORMATION_SUMMARY AS
(
     SELECT ACCTCODE,PRODUCTTYPE,value_frequency ,
     ROW_NUMBER() OVER ( PARTITION BY ACCTCODE ORDER BY value_frequency DESC) RNUM
     FROM(
                   SELECT  tbl.ACCTCODE, tbl.PRODUCTTYPE,
                   COUNT(tbl.PRODUCTTYPE) value_frequency 
                   FROM prod_share.edl_repl.repl_nat_tbl_summarized_revenue tbl
                   CROSS JOIN PULL_CAL_MONTH CAL
                  WHERE  tbl.YEARMONTH::VARCHAR BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(CAL.CURRENT_MONTH ,'YYYYMM'),-12),'YYYYMM')  AND CAL.CURRENT_MONTH::VARCHAR
                  GROUP BY ACCTCODE,PRODUCTTYPE 
                 )
         
),
PRODUCT_INFORMATION_1 AS
(
      SELECT ACCTCODE ,PRODUCTTYPE AS PRODUCTTYPE_1
      FROM PRODUCT_INFORMATION_SUMMARY
      WHERE RNUM=1
      
),
PRODUCT_INFORMATION_2 AS
(
      SELECT ACCTCODE ,PRODUCTTYPE AS PRODUCTTYPE_2
      FROM PRODUCT_INFORMATION_SUMMARY
      WHERE RNUM=2
      
),
PRODUCT_INFORMATION_3 AS
(
      SELECT ACCTCODE ,PRODUCTTYPE AS PRODUCTTYPE_3
      FROM PRODUCT_INFORMATION_SUMMARY
      WHERE RNUM=3
      
),
PRODUCT_INFORMATION_4 AS
(
      SELECT ACCTCODE ,PRODUCTTYPE AS PRODUCTTYPE_4
      FROM PRODUCT_INFORMATION_SUMMARY
      WHERE RNUM=4
      
),
PRODUCT_INFORMATION_5 AS
(
      SELECT ACCTCODE ,PRODUCTTYPE AS PRODUCTTYPE_5
      FROM PRODUCT_INFORMATION_SUMMARY
      WHERE RNUM=5
      
),
PRODUCT_INFORMATION_6 AS
(
      SELECT ACCTCODE ,PRODUCTTYPE AS PRODUCTTYPE_6
      FROM PRODUCT_INFORMATION_SUMMARY
      WHERE RNUM=6
      
),
PRODUCT_INFORMATION_SPEND AS (
    SELECT ACCTCODE,
           COALESCE( SUM(ECASHSPEND),0) AS TOTAL_ECASHSPEND,
           COALESCE(SUM(EXPRESSCHECK),0) AS TOTAL_EXPRESSCHECKSPEND,
           COALESCE(SUM(MASTERCARD),0) AS TOTAL_MASTERCARDSPEND,
          COALESCE(SUM( PROPRIETARY),0) AS TOTAL_PROPRIETARYSPEND,
          COALESCE(SUM(FUEL),0) AS TOTAL_FUELSPEND,
          COALESCE(SUM(NON_CORE),0) AS TOTAL_NON_CORESPEND
          
   FROM(
      SELECT  NAT.ACCTCODE, NAT.YEARMONTH,
      CASE WHEN NAT.PRODUCTTYPE='ECASH' THEN NAT.NETSPEND  END AS ECASHSPEND,
      CASE WHEN NAT.PRODUCTTYPE='EXPRESS CHECK' THEN NAT.NETSPEND  END AS EXPRESSCHECK,
      CASE WHEN NAT.PRODUCTTYPE='PROPRIETARY' THEN NAT.NETSPEND  END AS PROPRIETARY,
      CASE WHEN NAT.PRODUCTTYPE='FUEL' THEN NAT.NETSPEND  END AS FUEL,
      CASE WHEN NAT.PRODUCTTYPE='NON-CORE' THEN NAT.NETSPEND  END AS NON_CORE,
      CASE WHEN NAT.PRODUCTTYPE='MASTERCARD' THEN NAT.NETSPEND  END AS MASTERCARD
      
      FROM prod_share.edl_repl.repl_nat_tbl_summarized_revenue NAT 
     LEFT JOIN LAST_TRX LA ON LA.ACCTCODE=NAT.ACCTCODE
     CROSS JOIN PULL_CAL_MONTH CAL
      WHERE NAT.YEARMONTH::VARCHAR BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(CAL.CURRENT_MONTH ,'YYYYMM'),-12),'YYYYMM') AND CAL.CURRENT_MONTH::VARCHAR  
    --WHERE  NAT.YEARMONTH::VARCHAR BETWEEN TO_CHAR(ADD_MONTHS(TO_DATE(LA.TRIM_LAST_TRX_MONTH ,'YYYYMM'),-12),'YYYYMM')  AND LA.TRIM_LAST_TRX_MONTH ::VARCHAR
 
        )  
        GROUP BY ACCTCODE
        ORDER BY ACCTCODE
),
CHANNEL_1 AS(

SELECT ACCTCODE,CHANNEL
    FROM(
      SELECT
                 account_id AS ACCTCODE,
                 channel AS CHANNEL,psr_date,
                 ROW_NUMBER() OVER ( PARTITION BY account_id ORDER BY psr_date DESC) RNUM
           FROM user_global_sales.vw_all_deals
           WHERE channel not in ('Telesales 1-11','Telesales 12-50') 
                 
          
          )WHERE RNUM=1   
),

CHANNEL_2 AS(

SELECT ACCTCODE,CHANNEL
    FROM(
         SELECT
                 acct AS ACCTCODE,
                 flt_team AS CHANNEL,psr_day,
                 ROW_NUMBER() OVER ( PARTITION BY acct ORDER BY psr_day DESC) RNUM
           FROM prod_share.edl_repl.repl_mart_sfdc_nat_all_deals1
                 
                 
          
          )WHERE RNUM=1   
),
LAST_PAYMENT AS(
    SELECT ACCTCODE,LAST_PAYMENT_METHOD
    FROM(
             SELECT substring(RK508_ACCOUNT_NO  from 10 for 5) AS ACCTCODE,
                        CASE

                         WHEN LEFT([RK508_REM_NO],1) = 'A' THEN 'CustACH'

                        WHEN LEFT([RK508_REM_NO],1) = 'D' THEN 'EBP/FIS'

                        WHEN LEFT([RK508_REM_NO],1) = 'Q' THEN 'WesternUnion'

                       WHEN LEFT([RK508_REM_NO],1) = 'W' THEN 'Wire'

                       WHEN SUBSTRING([RK508_REM_NO],3,3) = 'ACH' THEN 'CDN/ACH'

                       ELSE 'Check'

                      End  AS LAST_PAYMENT_METHOD,

                      ROW_NUMBER() OVER ( PARTITION BY substring(RK508_ACCOUNT_NO  from 10 for 5) ORDER BY [RK508_REM_DT] DESC) RNUM
                       FROM prod_share.edl_mdl.vw_fin1_geac_rk508b_closed_items
                       INNER JOIN ACTIVE_CUSTS_UNIQUE CUST ON CUST.ACCTCODE=substring(  RK508_ACCOUNT_NO from 10 for 5) 
          )WHERE RNUM=1


)


SELECT  
            TRIM(CUST.ACCTCODE) AS customer_id,
            TR.PARENTNAME AS PARENTNAME,
            trx.VOLUME AS VOLUME,
            trx.TOTAL_SPEND AS TOTAL_SPEND,
            trx.TOTAL_NUM_TRX AS TOTAL_NUM_TRX,
            
            TR.FIRST_TRX_MONTH AS first_trx_month,
            LAST.LAST_TRANSACTION_WEEK AS last_transaction_week,
            LAST.LAST_TRANSACTION_MONTH AS last_transaction_month,
            ST.CUST_STATE AS state_code,
            TR.NUM_OF_TRUCKS AS num_of_trucks,
            CL.CREDIT_LIMIT AS credit_limit,
            BF.billing_freq AS billing_freq,
            DY.due_days AS due_days,
           CH1.CHANNEL AS CHANNEL_1,
           CH2.CHANNEL AS CHANNEL_2,
           CS.CREDIT_SCORE,
            PI1.PRODUCTTYPE_1 AS product_information_1,
            PI2.PRODUCTTYPE_2 AS product_information_2,
            PI3.PRODUCTTYPE_3 AS product_information_3,
            PI4.PRODUCTTYPE_4 AS product_information_4,
            PI5.PRODUCTTYPE_5 AS product_information_5,
            PI6.PRODUCTTYPE_6 AS product_information_6,
               NVL(TOTAL_ECASHSPEND,0)+
               NVL(TOTAL_EXPRESSCHECKSPEND,0)+
               NVL(TOTAL_MASTERCARDSPEND,0) +
                NVL(TOTAL_PROPRIETARYSPEND,0)+
                NVL(TOTAL_FUELSPEND,0)+
                NVL(TOTAL_NON_CORESPEND,0) AS TOTAL_PRODUCTTYPE_SPEND,
             CASE WHEN last_transaction_week>'{churn_cutoff_week}' THEN 'ACTIVE' ELSE 'CHURNER' END AS CHURN_FLAG,
            LPAY.LAST_PAYMENT_METHOD as last_payment_method 
FROM  ACTIVE_CUSTS_UNIQUE CUST 
LEFT JOIN FUEL_STATS TRX ON CUST.ACCTCODE=TRX.ACCTCODE
LEFT JOIN CUST_STATE ST ON ST.ACCTCODE=CUST.ACCTCODE
LEFT JOIN NAT_REVENUE TR ON TR.ACCTCODE=CUST.ACCTCODE
LEFT JOIN CREDIT_LIMIT CL ON CL.ACCTCODE=CUST.ACCTCODE
LEFT JOIN BILL_FREQ BF ON BF.ACCTCODE=CUST.ACCTCODE
LEFT JOIN DUE_DAYS DY ON DY.ACCTCODE=CUST.ACCTCODE
LEFT JOIN CREDIT_SCORE CS ON CS.ACCTCODE=CUST.ACCTCODE
LEFT JOIN PRODUCT_INFORMATION_1 PI1 ON PI1.ACCTCODE=CUST.ACCTCODE
LEFT JOIN PRODUCT_INFORMATION_2 PI2 ON PI2.ACCTCODE=CUST.ACCTCODE
LEFT JOIN PRODUCT_INFORMATION_3 PI3 ON PI3.ACCTCODE=CUST.ACCTCODE
LEFT JOIN PRODUCT_INFORMATION_4 PI4 ON PI4.ACCTCODE=CUST.ACCTCODE
LEFT JOIN PRODUCT_INFORMATION_5 PI5 ON PI5.ACCTCODE=CUST.ACCTCODE
LEFT JOIN PRODUCT_INFORMATION_6 PI6 ON PI6.ACCTCODE=CUST.ACCTCODE
LEFT JOIN LAST_TRX LAST ON LAST.ACCTCODE=CUST.ACCTCODE
LEFT JOIN PRODUCT_INFORMATION_SPEND PS ON PS.ACCTCODE=CUST.ACCTCODE
LEFT JOIN CHANNEL_1 CH1 ON CH1.ACCTCODE=CUST.ACCTCODE
LEFT JOIN CHANNEL_2 CH2 ON CH2.ACCTCODE=CUST.ACCTCODE
LEFT JOIN LAST_PAYMENT LPAY ON LPAY.ACCTCODE=CUST.ACCTCODE
WHERE TRIM(CUST.ACCTCODE) IS NOT NULL AND TR.PARENTNAME IS NOT NULL
ORDER BY CUST.ACCTCODE
