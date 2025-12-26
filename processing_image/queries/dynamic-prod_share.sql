--------------------Unique customers-------------------
WITH ACTIVE_CUSTS_UNIQUE AS
(
   SELECT 
              TRIM(dt.ACCTCODE) as ACCTCODE
   FROM 
              prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
   LEFT  JOIN  prod_share.edl_repl.repl_nat_newlob CO 
               ON  TRIM(dt.ACCTCODE) = CO.ACCTCODE
    LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',DT.TXNDATE)
    WHERE 
                 CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{activity_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
                AND  CO.FINALLOB = 'NAT'
    GROUP BY dt.ACCTCODE
  
  UNION 
    
  SELECT 
               F.FMDL_CUST_ACCOUNT_CODE AS ACCTCODE
  FROM  prod_share.edl_repl.repl_fmlog F
  LEFT  JOIN prod_share.edl_repl.repl_nat_newlob CO 
                       ON FMDL_CUST_ACCOUNT_CODE = CO.ACCTCODE
  LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL 
                      ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',F.FMDL_rk_transaction_date)
  WHERE CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{activity_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
               AND FMDL_SC_SRVC_CNTR_CODE NOT IN ('MC901','MC902','MC903','MC904','MC905')
              AND  CO.FINALLOB = 'NAT'
  GROUP BY  F.FMDL_CUST_ACCOUNT_CODE
),

WEEKS_CUST_MAP AS
 (
   SELECT
	CAL.WEEK_NO AS YEARWEEK,
	CONCAT(CAL.YEAR, LPAD(CAL.FM_MONTH, 2, '0')) AS MONTH,
	ACCTCODE
	FROM prod_share.edl_REPL.REPL_MART_CALENDAR CAL
	CROSS JOIN (SELECT DISTINCT ACCTCODE FROM ACTIVE_CUSTS_UNIQUE) t
	WHERE CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
	ORDER BY 1
),
----------------------------------------MC customers Fuel transactions-----------------------------------------------
dynamic_trx_mc as(
    SELECT 
              TRIM(dt.ACCTCODE) as ACCTCODE,
              CAL.WEEK_NO as YEARWEEK,
               ISNULL(Diesel_Gallons,0) + ISNULL(Gas_Gallons,0) as TOTAL_GALLONS,
              CASE WHEN Diesel_Gallons>0 THEN [SPEND] 
	               WHEN Gas_Gallons>0 THEN [SPEND] 
                                      ELSE NULL END AS FUEL_SPEND,
        
             CASE WHEN Diesel_Gallons>0 THEN NULL
			      WHEN  Gas_Gallons>0  THEN NULL 
	            ELSE [SPEND] END AS NONFUEL_SPEND,
             
           SPEND AS TOTAL_SPEND,
           CASE WHEN Diesel_Gallons>0 THEN TXNS 
	     WHEN Gas_Gallons>0 THEN TXNS
                          ELSE 0 END AS FUEL_TRX,

            CASE WHEN Diesel_Gallons>0 THEN 0
			     WHEN  Gas_Gallons>0  THEN 0 
                                    ELSE TXNS END AS NONFUEL_TRX,

           ISNULL(RebateAmt_Mainframe,0)
		+ ISNULL(RebateAmt_SQL,0)
		+ ISNULL(RebateAmt_Other,0)
		+ ISNULL(RebateAmt_Variance,0)
		+ ISNULL(DiscountAmt_Mainframe,0) AS TOTAL_DISCOUNT

       
   FROM 
             prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
    LEFT  JOIN prod_share.edl_repl.repl_nat_newlob CO 
             ON  TRIM(dt.ACCTCODE) = CO.ACCTCODE
    LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',dt.TXNDATE )
    WHERE 
                CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer                            
                AND  CO.finallob = 'NAT'
               AND dt.BILLINGTYPE = 'MasterCard'
              AND dt.BINN NOT in ('556736','555622','556766') -- OnRoad 556736 vComcheck 555622 VCAP 556766
 
	
),
-----------------------------------------------------Prop Customers Fuel transactions------------------------------------
dynamic_trx_prop AS
(
  SELECT   
             F.FMDL_CUST_ACCOUNT_CODE as acctcode, 
            CAL.WEEK_NO as YEARWEEK,   
          
              
            NVL(F.FMDL_TR_OTHER_GALLONS, 0) + NVL(F.FMDL_TOTAL_GALLONS, 0) as TOTAL_GALLONS,

           (NVL(F.FMDL_TR_TRACTOR_COST,0) + NVL(F.FMDL_TR_TRAILER_COST,0)+NVL(F.FMDL_TR_OTHER_COST,0)) as FUEL_SPEND,
        
               NVL(F.FMDL_TR_PRODUCT_AMOUNT_1,0) +NVL(F.FMDL_TR_PRODUCT_AMOUNT_2,0) +NVL(F.FMDL_TR_PRODUCT_AMOUNT_3,0) + NVL(F.FMDL_TR_OIL_COST,0)
                   + NVL  (F.FMDL_TR_CASH_ADVANCE_AMOUNT,0)   as NONFUEL_SPEND,

              -- vv Start TOTALSPEND
	NVL(F.FMDL_TR_TRACTOR_COST,0)
	+ NVL(F.FMDL_TR_TRAILER_COST,0)
	+ (NVL(F.FMDL_TR_OTHER_COST ,0)
	+ NVL(F.FMDL_TR_OIL_COST,0)
	+ NVL(F.FMDL_TR_CASH_ADVANCE_AMOUNT,0) 
	+ NVL(F.FMDL_TR_PRODUCT_AMOUNT_1,0) 
	+ NVL(F.FMDL_TR_PRODUCT_AMOUNT_2,0) 
	+ NVL(F.FMDL_TR_PRODUCT_AMOUNT_3,0))as TOTAL_SPEND,         
                -- ^^ End TOTALSPEND
           --FUEL AND NON FUEL TRANSACTIONS
               CASE WHEN (NVL(F.FMDL_TF_TRACTOR_FUEL,0) + NVL(F.FMDL_TF_TRAILER_FUEL,0)+NVL(F.FMDL_TF_OTHER_FUEL,0))>0 THEN 1 ELSE 0 END AS FUEL_TRX ,
            CASE WHEN (NVL(F.FMDL_TF_PRODUCT_1,0) +NVL(F.FMDL_TF_PRODUCT_2,0) 
                +NVL(F.FMDL_TF_PRODUCT_3,0) + NVL(F.FMDL_TF_OIL,0)+
                NVL(F.FMDL_TF_CASH,0))>0 THEN 1 ELSE 0 END AS NONFUEL_TRX,		
          
             
                -- ^^ End TOTALSPEND
                 NVL(F.FMDL_SC_RD_SC_SELECT_DISCOUNT,0) as TOTAL_DISCOUNT

  FROM                   
	prod_share.edl_repl.repl_fmlog F     
  LEFT  JOIN  prod_share.edl_repl.repl_nat_newlob CO 
                       ON FMDL_CUST_ACCOUNT_CODE = CO.ACCTCODE             
  LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',F.FMDL_rk_transaction_date)
  WHERE 
                 CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
                  AND  CO.FINALLOB = 'NAT'
                  AND FMDL_SC_SRVC_CNTR_CODE NOT IN ('MC901','MC902','MC903','MC904','MC905')
                
  ORDER BY 1,3     

),
------------------------------------Combine fuel transaction details of  MC and Prop---------------------------------------
fuel_stats AS
(
  SELECT 
       ACCTCODE,YEARWEEK,
      COALESCE(SUM(TOTAL_GALLONS),0) AS TOTAL_GALLONS,
      COALESCE(SUM(FUEL_TRX),0) AS TOTAL_FUEL_TRX,
      COALESCE(SUM(NONFUEL_TRX),0) AS TOTAL_NONFUEL_TRX,
      COALESCE(SUM((NVL(FUEL_TRX,0)+NVL(NONFUEL_TRX,0))),0) AS TOTAL_NUM_TRX,
      COALESCE(SUM(FUEL_SPEND),0) AS TOTAL_FUEL_SPEND,
      COALESCE(SUM(NONFUEL_SPEND),0) AS TOTAL_NONFUEL_SPEND,
      COALESCE(SUM(TOTAL_SPEND),0) AS TOTAL_SPEND,
      COALESCE(SUM(TOTAL_DISCOUNT),0) AS TOTAL_DISCOUNT
                                                                
  FROM (select * from dynamic_trx_mc 
           UNION ALL 
        select * from dynamic_trx_prop)
  GROUP BY ACCTCODE,YEARWEEK
  ORDER BY ACCTCODE,YEARWEEK
),
---------------------------------------------------Calculate EXPRESS CHECK FEE-----------------
EXPRESS_CHECK AS (
   SELECT 
              TRIM(dt.ACCTCODE) AS ACCTCODE,
               CAL.WEEK_NO as YEARWEEK,
               'EXPRESS CHECK' AS PRODUCTTYPE
               ,'EXPRESS CHECK FEE' AS TRANSACTIONTYPE,
                COALESCE(SUM(dt.[Spend]),0) as TOTAL_EXPRESSCHECKFEE
   FROM
	prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
                 INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=dt.ACCTCODE
                  LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',dt.TXNDATE )
  WHERE 
                CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
               AND dt.BILLINGTYPE  IN  ('Express Check')
   GROUP BY
	 CAL.WEEK_NO
	,TRIM(dt.ACCTCODE)
),
------------------------------------------Calculate Vcomcheck interchange fee-------------------------------
VCOMCHECK AS(
     SELECT 
	'EXPRESS CHECK' as PRODUCTTYPE
	,CAL.WEEK_NO as YEARWEEK
	,TRIM(dt.ACCTCODE) as ACCTCODE
	,'vComcheck INTERCHANGE FEE' as TRANSACTIONTYPE
	,COALESCE(SUM(dt.[Spend]),0) as TOTAL_EXPRESSCHECK_VCOMFEE
	
  FROM 
	prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
                 INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=dt.ACCTCODE
                  LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',dt.TXNDATE )
    WHERE 
                 CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer   
                AND dt.BILLINGTYPE = 'MasterCard'
                 AND dt.BINN IN ('555622')
GROUP BY
	CAL.WEEK_NO
	,TRIM(dt.ACCTCODE)


),
-------------------------Calculate VCAP------------------------------------------
VCAP AS(
SELECT 
	'EXPRESS CHECK' as PRODUCTTYPE
	,CAL.WEEK_NO as YEARWEEK
	,TRIM(dt.ACCTCODE) as ACCTCODE
	,'VCAP' as TRANSACTIONTYPE
	,COALESCE(SUM(dt.[Spend]),0) as TOTAL_EXPRESSCHECK_VCAP
FROM 
	prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
	 INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=dt.ACCTCODE
                   LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',dt.TXNDATE )
WHERE         
                 CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  

	AND dt.BILLINGTYPE = 'MasterCard'
		
	AND dt.BINN IN ('556766') -- VCAP 556766 
GROUP BY
	CAL.WEEK_NO
	,TRIM(dt.ACCTCODE)


),
--------------------Calculate ECASH SPEND-----------------------------------
ECASH_SPEND AS (
   SELECT 
              TRIM(dt.ACCTCODE) as ACCTCODE,
              CAL.WEEK_NO as YEARWEEK,
              'ECASH' as PRODUCTTYPE,
              'ONROAD INTERCHANGE FEE' as TRANSACTIONTYPE,
              COALESCE(SUM(dt.[Spend]),0) as TOTAL_ECASHSPEND,
             SUM(ISNULL(dt.Diesel_Gallons,0) + ISNULL(dt.Gas_Gallons,0)) as ECASH_GALLONS
   FROM 
	prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
                   INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=dt.ACCTCODE
                 LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',dt.TXNDATE )
  WHERE 
                 CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
	AND dt.BINN IN ('556736') -- OnRoad BINN
	AND dt.BILLINGTYPE IN ('MasterCard')
   GROUP BY
	CAL.WEEK_NO
	,TRIM(dt.ACCTCODE)
),
----FLEET TRX FEE------
fleet_trx_fee AS(
 SELECT   
             F.FMDL_CUST_ACCOUNT_CODE as acctcode, 
           CAL.WEEK_NO as YEARWEEK, 
          
           SUM( NVl(F.FMDL_RD_FUEL_RATE,0) + NVL(F.FMDL_RD_CASH_ADVANCE_RATE,0) + NVL(F.FMDL_RD_COMP_HANDLING_CHG,0)) as fleet_transaction_fee, 
            SUM(CASE WHEN   (NVL(F.FMDL_RD_FUEL_RATE,0) + NVL(F.FMDL_RD_CASH_ADVANCE_RATE,0) + NVL(F.FMDL_RD_COMP_HANDLING_CHG,0)) > 0 THEN 
                          1 ELSE 0 END) AS num_of_fleet_fees_applied
  FROM                   
	prod_share.edl_repl.repl_fmlog F                  
	INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=F.FMDL_CUST_ACCOUNT_CODE
                  LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK', F.FMDL_rk_transaction_date)
 WHERE CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
	
 GROUP BY
     FMDL_CUST_ACCOUNT_CODE,CAL.WEEK_NO        
   ORDER BY 1,2     

),
--LATE FEE
late_interest_fee AS(

  SELECT 
              dt.ACCTCODE,
              dt.YEARWEEK,
            COALESCE( SUM(CASE WHEN  dt.FEE_AMT< 0 THEN -dt.FEE_AMT  ELSE 0 END),0) AS late_interest_fee,
            COALESCE(SUM(CASE WHEN dt.FEE_AMT > 0 THEN -dt.FEE_AMT ELSE 0 END),0) AS late_interest_fee_refund,
            COUNT(CASE WHEN  dt.FEE_AMT< 0 THEN dt.FEE_TYPE  ELSE NULL END)AS num_of_late_fees_applied,
            COUNT(CASE WHEN  dt.FEE_AMT> 0 THEN dt.FEE_TYPE  ELSE NULL END)AS num_of_late_fees_refunded
  FROM(
             SELECT
                         (RIGHT(SJT_AR_ACCT_NBR, 5)) as ACCTCODE ,   
	       CAL.WEEK_NO as YEARWEEK,
                         STR.SJT_BUCKET_15_GL_AMT as FEE_AMT,
                         CASE 
	            WHEN lpad(STR.SJT_BUCKET_15_GL_NO::VARCHAR,5,'0') IN ('00055') THEN 'LATE FEE'
	            WHEN lpad(STR.SJT_BUCKET_15_GL_NO::VARCHAR,5,'0') IN ('00054') THEN 'LATE FEE INTEREST'
                        END  AS FEE_TYPE 
               FROM   
	       prod_share.edl_repl.repl_sj_slsdetl_trans_rec  STR 
                         INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=(RIGHT(SJT_AR_ACCT_NBR, 5))
                         LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',STR.SJT_LOAD_DATE  )
              WHERE 
                            CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
                           AND lpad(STR.SJT_BUCKET_15_GL_NO::VARCHAR,5,'0') IN ('00055','00054')  
	     
         )dt
  GROUP BY
                  dt.ACCTCODE,
                 dt.YEARWEEK
  ORDER BY  
               dt.ACCTCODE,
               dt.YEARWEEK  
 ),
--RISK BASED PRICING FEE
 risk_based_pricing_fee AS(

  SELECT
             (RIGHT(SJT_AR_ACCT_NBR, 5)) as ACCTCODE,
             CAL.WEEK_NO AS YEARWEEK,
             COALESCE(SUM(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN -SJT_CUST_REV_GL_AMT  ELSE 0 END),0) AS risk_based_pricing_fee,
             COALESCE(SUM(CASE WHEN SJT_CUST_REV_GL_AMT > 0 THEN -SJT_CUST_REV_GL_AMT ELSE 0 END),0) AS risk_based_pricing_refund,
            COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_risk_fees_applied,
            COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT> 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_risk_fees_refunded
  FROM
            prod_share.edl_repl.repl_sj_slsdetl_trans_rec STR
           INNER JOIN ACTIVE_CUSTS_UNIQUE   CUST ON CUST.ACCTCODE=(RIGHT(SJT_AR_ACCT_NBR, 5))
            LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',STR.SJT_TRAN_DATE)
                     
  WHERE		
	(
	  sjt_product_code  =  '00500'
	 AND SJT_SUB_PRODUCT_CODE1  =  '90001'
	 AND SJT_SUB_PRODUCT_CODE2 = '00068'
	)	

    AND  CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer    

	
   	
 GROUP BY
	(RIGHT(SJT_AR_ACCT_NBR, 5)),
                CAL.WEEK_NO
	
 ORDER BY
	ACCTCODE,
                 YEARWEEK
    
 ),

--Optional program fee/Tire program fee
optional_program_fee AS
(
  SELECT
	(RIGHT(SJT_AR_ACCT_NBR, 5)) as ACCTCODE,
	CAL.WEEK_NO AS YEARWEEK,
	COALESCE(SUM(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN -SJT_CUST_REV_GL_AMT  ELSE 0 END),0) AS optional_program_fee,
                  COALESCE(SUM(CASE WHEN SJT_CUST_REV_GL_AMT > 0 THEN -SJT_CUST_REV_GL_AMT ELSE 0 END),0) AS optional_program_refund,
                 COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_optional_fees_applied,
                 COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT> 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_optional_fees_refunded
		
  FROM
	prod_share.edl_repl.repl_sj_slsdetl_trans_rec STR
              
                 INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=RIGHT(STR.SJT_AR_ACCT_NBR,5)
                  LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',STR.SJT_TRAN_DATE)
 WHERE		
	(
	sjt_product_code  =  '00500'
	AND SJT_SUB_PRODUCT_CODE1  =  '90001'
	AND SJT_SUB_PRODUCT_CODE2 = '00067'
	)	
   AND  CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
 
 GROUP BY
	(RIGHT(SJT_AR_ACCT_NBR, 5)),
               CAL.WEEK_NO
		
 ORDER BY
	ACCTCODE,
                YEARWEEK
),
--Proximity Fee
proximity_fee AS(

  SELECT
             (RIGHT(SJT_AR_ACCT_NBR, 5)) as ACCTCODE,
            CAL.WEEK_NO AS YEARWEEK,
            COALESCE(SUM(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN -SJT_CUST_REV_GL_AMT  ELSE 0 END),0) AS proximity_fee,
            COALESCE(SUM(CASE WHEN SJT_CUST_REV_GL_AMT > 0 THEN -SJT_CUST_REV_GL_AMT ELSE 0 END),0) AS proximity_fee_refund,
            COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_prox_fees_applied,
            COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT> 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_prox_fees_refunded
		
  FROM
          prod_share.edl_repl.repl_sj_slsdetl_trans_rec STR
         
         INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=RIGHT(STR.SJT_AR_ACCT_NBR,5)
         LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',STR.SJT_TRAN_DATE)
  WHERE		
	(
	sjt_product_code  =  '00500'
	AND SJT_SUB_PRODUCT_CODE1  =  '90001'
	AND SJT_SUB_PRODUCT_CODE2 = '00074'
	)
                 AND  CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
 
 GROUP BY
	(RIGHT(SJT_AR_ACCT_NBR, 5)),
                CAL.WEEK_NO
		
  ORDER BY
	ACCTCODE,
                 YEARWEEK
),
--FLEET ADVANCE FEE
fleet_advance_fee AS
(

   SELECT
	(RIGHT(SJT_AR_ACCT_NBR, 5)) as ACCTCODE,
	CAL.WEEK_NO AS YEARWEEK,
                COALESCE( SUM(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN -SJT_CUST_REV_GL_AMT  ELSE 0 END),0) AS fleet_advance_fee,
                 COALESCE(SUM(CASE WHEN SJT_CUST_REV_GL_AMT > 0 THEN -SJT_CUST_REV_GL_AMT ELSE 0 END),0) AS fleet_advance_refund,
                 COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_fleet_adv_applied,
                COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT> 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_fleet_adv_refunded
		
  FROM
	prod_share.edl_repl.repl_sj_slsdetl_trans_rec STR
                  INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=(RIGHT(SJT_AR_ACCT_NBR, 5))
                  LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',STR.SJT_TRAN_DATE)
  WHERE		
	STR.sjt_product_code IN ('00175')		
    AND  CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
    GROUP BY
	(RIGHT(SJT_AR_ACCT_NBR, 5)),
                 CAL.WEEK_NO
    ORDER BY
	ACCTCODE,
                 YEARWEEK
) ,
--Express Codes Fee
express_code_fee AS (
   SELECT
             trim(CUSTOMERACCOUNTCODE) as ACCTCODE, 
             CAL.WEEK_NO as YEARWEEK,
             COALESCE(SUM(CASE WHEN  FeeAmount> 0 THEN FeeAmount  ELSE 0 END),0) AS express_fee,
             COALESCE(SUM(CASE WHEN FeeAmount < 0 THEN FeeAmount ELSE 0 END),0) AS express_fee_refund,
             COUNT(CASE WHEN  FeeAmount> 0 THEN FeeAmount  ELSE NULL END)AS num_of_expr_fees_applied,
            COUNT(CASE WHEN  FeeAmount< 0 THEN FeeAmount  ELSE NULL END)AS num_of_expr_fees_refunded
   FROM  
            prod_share.edl_repl.repl_all_settlement_alllocations
         
           INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=trim(CUSTOMERACCOUNTCODE)
            LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',InvoiceDate)
   WHERE
             CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
        
            AND trim(FEETYPE)='X'
   GROUP BY
	trim(CUSTOMERACCOUNTCODE),
                 CAL.WEEK_NO
  ORDER BY
               ACCTCODE,
              YEARWEEK
	
 ) , 

----	
--Inactivity Fee
inactivity_fee AS(

    SELECT
                 (RIGHT(SJT_AR_ACCT_NBR, 5)) as ACCTCODE,
                 CAL.WEEK_NO AS YEARWEEK,
                 COALESCE(SUM(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN -SJT_CUST_REV_GL_AMT  ELSE 0 END),0) AS inactivity_fee,
                 COALESCE(SUM(CASE WHEN SJT_CUST_REV_GL_AMT > 0 THEN -SJT_CUST_REV_GL_AMT ELSE 0 END),0) AS inactivity_fee_refund,
                 COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_inac_fees_applied,
                 COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT> 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_inac_fees_refunded
		
   FROM
	 prod_share.edl_repl.repl_sj_slsdetl_trans_rec 
                  INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=RIGHT(SJT_AR_ACCT_NBR,5)
                 LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',SJT_TRAN_DATE)
   WHERE
	(
	sjt_product_code  =  '00200'
	AND SJT_SUB_PRODUCT_CODE1  =  '90005'
	)
	AND  CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
            
    GROUP BY
	(RIGHT(SJT_AR_ACCT_NBR, 5)),
                  CAL.WEEK_NO
    ORDER BY
	ACCTCODE,
                 YEARWEEK
 ),
 ----
card_fee AS(

     SELECT
	(RIGHT(SJT_AR_ACCT_NBR, 5)) as ACCTCODE,
	CAL.WEEK_NO AS YEARWEEK,
	COALESCE(SUM(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN -SJT_CUST_REV_GL_AMT  ELSE 0 END),0) AS card_fee,
                 COALESCE(SUM(CASE WHEN SJT_CUST_REV_GL_AMT > 0 THEN -SJT_CUST_REV_GL_AMT ELSE 0 END),0) AS card_fee_refund,
                 COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_card_fees_applied,
                 COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT> 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_card_fees_refunded
       FROM
	prod_share.edl_repl.repl_sj_slsdetl_trans_rec
        INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=RIGHT(SJT_AR_ACCT_NBR,5)
        LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',SJT_TRAN_DATE)  
      WHERE		
	(
	sjt_product_code  =  '00500'
	AND SJT_SUB_PRODUCT_CODE1  =  '90001'
	AND SJT_SUB_PRODUCT_CODE2 = '00070'
	)
            
	AND  CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
      GROUP BY
	(RIGHT(SJT_AR_ACCT_NBR, 5)),
                 CAL.WEEK_NO
      ORDER BY
	ACCTCODE,
                 YEARWEEK
),
--comcheck Admin fee
comcheck_admin_fee AS(
    SELECT
                (RIGHT(SJT_AR_ACCT_NBR, 5)) as ACCTCODE,
                CAL.WEEK_NO AS YEARWEEK,
                COALESCE(SUM(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN -SJT_CUST_REV_GL_AMT  ELSE 0 END),0) AS check_admin_fee,
                COALESCE(SUM(CASE WHEN SJT_CUST_REV_GL_AMT > 0 THEN -SJT_CUST_REV_GL_AMT ELSE 0 END),0) AS check_admin_refund,
               COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT< 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_admin_fees_applied,
               COUNT(CASE WHEN  SJT_CUST_REV_GL_AMT> 0 THEN SJT_CUST_REV_GL_AMT  ELSE NULL END)AS num_of_admin_fees_refunded
   FROM
	 prod_share.edl_repl.repl_sj_slsdetl_trans_rec 
                 INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=RIGHT(SJT_AR_ACCT_NBR,5)
                  LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',SJT_TRAN_DATE)  
   WHERE
	(
	sjt_product_code IN ('00630')
	AND SJT_SUB_PRODUCT_CODE1 IN ('90001')
	AND SJT_SUB_PRODUCT_CODE2 IN ('00004')
	)
                AND  CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
	
   GROUP BY
	(RIGHT(SJT_AR_ACCT_NBR, 5)),
                 CAL.WEEK_NO
		
   ORDER BY
	ACCTCODE,
                 YEARWEEK
 ),
--Non Core ELD Subscription and EQUIPMENT
eld_equipment_fee AS (
    SELECT    
               dt.AcctCode as ACCTCODE,
               dt.YEARWEEK AS YEARWEEK,
               COALESCE(SUM(CASE WHEN  dt.FEE_AMT< 0 THEN -dt.FEE_AMT  ELSE 0 END),0) AS eld_equi_fee,
               COALESCE(SUM(CASE WHEN dt.FEE_AMT > 0 THEN -dt.FEE_AMT ELSE 0 END),0) AS eld_equi_fee_refund,
               COUNT(CASE WHEN  dt.FEE_AMT< 0 THEN dt.FEE_TYPE  ELSE NULL END)AS num_of_eld_eqi_fees_applied,
              COUNT(CASE WHEN  dt.FEE_AMT> 0 THEN dt.FEE_TYPE  ELSE NULL END)AS num_of_eld_eqi_fees_refunded
    FROM  (
                SELECT 
                           (RIGHT(SJT_AR_ACCT_NBR, 5)) as AcctCode,
                            CAL.WEEK_NO AS YEARWEEK,
                            STR.SJT_CUST_REV_GL_AMT AS FEE_AMT,
                           CASE WHEN SP.PROD_DESCRIPTION ='ELD EQUIPMENT' THEN  'ELD EQUIPMENT FEE'  ELSE  'ELD SUBSCR FEE' END AS FEE_TYPE
               FROM   prod_share.edl_repl.repl_sj_slsdetl_trans_rec STR    
               INNER JOIN prod_share.edl_repl.repl_sj_products SP
                       ON ( STR.SJT_PRODUCT_CODE = SP.PROD_PRODUCT_CODE  )    
                      AND  ( STR.SJT_SUB_PRODUCT_CODE1 = SP.PROD_SUB_PRODUCT_1  )    
                      AND  ( STR.SJT_SUB_PRODUCT_CODE2 = SP.PROD_SUB_PRODUCT_2  )     
                INNER JOIN ACTIVE_CUSTS_UNIQUE  CUST ON CUST.ACCTCODE=   (RIGHT(SJT_AR_ACCT_NBR, 5))
                LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',SJT_TRAN_DATE)  
               WHERE
                    CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
                   AND SP.PROD_DESCRIPTION IN ('ELD EQUIPMENT','ELD SUBSCR FEE')            
           )dt
GROUP BY    
	dt.ACCTCODE,
                 dt.YEARWEEK
ORDER BY
               ACCTCODE,
               YEARWEEK
 ) ,
pull_parent_name AS
(
 SELECT * FROM prod_share.edl_repl.repl_nat_account_attributes
)
SELECT 
           WM.ACCTCODE AS CUSTOMER_ID,
          WM.YEARWEEK AS WEEK,
           
            trx.TOTAL_GALLONS AS TOTAL_GALLONS,
            trx.TOTAL_FUEL_TRX  AS TOTAL_FUEL_TRX,
          trx.TOTAL_NONFUEL_TRX AS TOTAL_NONFUEL_TRX,
          trx.TOTAL_NUM_TRX AS TOTAL_NUM_TRX,
          trx.TOTAL_FUEL_SPEND AS TOTAL_FUEL_SPEND,
         trx.TOTAL_NONFUEL_SPEND AS TOTAL_NONFUEL_SPEND,
         trx.TOTAL_SPEND AS TOTAL_SPEND,
        trx.TOTAL_DISCOUNT AS TOTAL_DISCOUNT,
            
            EC.TOTAL_EXPRESSCHECKFEE AS TOTAL_COMCHECK_SPEND,
            VC.TOTAL_EXPRESSCHECK_VCOMFEE AS TOTAL_VCOMCHECKSPEND,
            VCP.TOTAL_EXPRESSCHECK_VCAP AS TOTAL_VCAP_SPEND,
            ES.TOTAL_ECASHSPEND AS TOTAL_ECASHSPEND,
            ES.ECASH_GALLONS AS ECASH_GALLONS,
            
            laf.late_interest_fee AS late_interest_fee_amount,
         laf.late_interest_fee_refund AS late_interest_fee_refund,
         rf. risk_based_pricing_fee AS risk_based_pricing_amount,
         rf.risk_based_pricing_refund AS risk_based_pricing_refund ,
         opf. optional_program_fee AS optional_program_amount ,
         opf.optional_program_refund AS optional_program_refund,
         pf.proximity_fee AS proximity_fee_amount,
         pf.proximity_fee_refund AS proximity_fee_refund,
--        trx.fleet_transaction_fee,
--        cf.card_fee,
--        cf.card_fee_refund,
        ff.fleet_advance_fee AS fleet_advance_fee_amount ,
      ff.fleet_advance_refund AS fleet_advance_refund,
--       inf.inactivity_fee,  inf.inactivity_fee_refund,cof.check_admin_fee,cof.check_admin_refund,ef.express_fee,ef.express_fee_refund,elf.eld_equi_fee,elf.eld_equi_fee_refund,

         NVL(ff.fleet_advance_fee,0)+NVL(cf.card_fee,0)+NVL(frxf.fleet_transaction_fee,0) AS program_fee_amount,
        NVL(ff.fleet_advance_refund,0)+NVL(cf.card_fee_refund,0) AS program_fee_refund,
        NVL( inf.inactivity_fee,0)+NVL(cof.check_admin_fee,0)+ NVL(ef.express_fee,0)+NVL(elf.eld_equi_fee,0) AS other_fees_amount,
        NVL( inf.inactivity_fee_refund,0)+NVL(cof.check_admin_refund,0)+ NVL(ef.express_fee_refund,0)+NVL(elf.eld_equi_fee_refund,0) AS other_fees_refund,

        NVL(laf.late_interest_fee,0)+NVL(rf. risk_based_pricing_fee,0)+ NVL(opf. optional_program_fee,0)+NVL(pf.proximity_fee,0)
           +NVL(ff.fleet_advance_fee,0)+NVL(cf.card_fee,0)+NVL(frxf.fleet_transaction_fee,0)
           +NVL( inf.inactivity_fee,0)+NVL(cof.check_admin_fee,0)+ NVL(ef.express_fee,0)+NVL(elf.eld_equi_fee,0) AS tot_fee_amount,

        NVL(laf.late_interest_fee_refund,0)+NVL(rf. risk_based_pricing_refund,0)+ NVL(opf. optional_program_refund,0)+NVL(pf.proximity_fee_refund,0)
           +NVL(ff.fleet_advance_refund,0)+NVL(cf.card_fee_refund,0)
           +NVL( inf.inactivity_fee_refund,0)+NVL(cof.check_admin_refund,0)+ NVL(ef.express_fee_refund,0)+NVL(elf.eld_equi_fee_refund,0)  AS tot_fee_refund,   

      laf.num_of_late_fees_applied,laf.num_of_late_fees_refunded,rf.num_of_risk_fees_applied,rf.num_of_risk_fees_refunded,opf.num_of_optional_fees_applied,opf.num_of_optional_fees_refunded,
      pf.num_of_prox_fees_applied,pf.num_of_prox_fees_refunded,ff.num_of_fleet_adv_applied,ff.num_of_fleet_adv_refunded,cf.num_of_card_fees_applied,cf.num_of_card_fees_refunded,
       frxf .num_of_fleet_fees_applied,inf.num_of_inac_fees_applied,inf.num_of_inac_fees_refunded,cof.num_of_admin_fees_applied,cof.num_of_admin_fees_refunded,
     ef.num_of_expr_fees_applied,ef.num_of_expr_fees_refunded,elf. num_of_eld_eqi_fees_applied,elf. num_of_eld_eqi_fees_refunded,
          
          --laf.num_of_late_fees_applied,rf.num_of_risk_fees_applied,opf.num_of_optional_fees_applied,pf.num_of_prox_fees_applied,
          --laf.num_of_late_fees_refunded,rf.num_of_risk_fees_refunded,opf.num_of_optional_fees_refunded,pf.num_of_prox_fees_refunded,
          
        
           NVL(ff.num_of_fleet_adv_applied,0)
           +NVL(cf.num_of_card_fees_applied,0)
           +NVL(frxf.num_of_fleet_fees_applied,0)
           AS num_of_program_fees_applied,
           
           NVL(ff.num_of_fleet_adv_refunded,0)
           +NVL(cf.num_of_card_fees_refunded,0)
           AS num_of_program_fees_refunded,

          
          NVL(inf.num_of_inac_fees_applied,0)
          +NVL(cof.num_of_admin_fees_applied,0)
          +NVL(ef.num_of_expr_fees_applied,0)
          +NVL(elf. num_of_eld_eqi_fees_applied,0)
                AS num_of_other_fees_applied,

          NVL(inf.num_of_inac_fees_refunded,0)
          +NVL(cof.num_of_admin_fees_refunded,0)
          +NVL(ef.num_of_expr_fees_refunded,0)
          +NVL(elf. num_of_eld_eqi_fees_refunded,0)
                AS num_of_other_fees_refunded 
        

      -- CASE WHEN num_of_late_fees_applied >0 THEN 1 ELSE 0 END 
      --    +CASE WHEN rf.num_of_risk_fees_applied>0 THEN 1 ELSE 0 END 
      --    +CASE WHEN opf.num_of_optional_fees_applied>0 THEN 1 ELSE 0 END
     --     +CASE WHEN pf.num_of_prox_fees_applied>0 THEN 1 ELSE 0 END
      --    +CASE WHEN ff.num_of_fleet_adv_applied>0 THEN 1 ELSE 0 END
       --   +CASE WHEN cf.num_of_card_fees_applied>0 THEN 1 ELSE 0 END
        --  +CASE WHEN frxf.num_of_fleet_fees_applied>0 THEN 1 ELSE 0 END
      --    +CASE WHEN inf.num_of_inac_fees_applied>0 THEN 1 ELSE 0 END
      --    +CASE WHEN cof.num_of_admin_fees_applied>0 THEN 1 ELSE 0 END
      --    +CASE WHEN ef.num_of_expr_fees_applied>0 THEN 1 ELSE 0 END
       --   +CASE WHEN elf. num_of_eld_eqi_fees_applied>0 THEN 1 ELSE 0 END
         --     AS unique_types_of_fees_applied,
         
       --  CASE WHEN num_of_late_fees_refunded >0 THEN 1 ELSE 0 END 
       --   +CASE WHEN rf.num_of_risk_fees_refunded>0 THEN 1 ELSE 0 END 
       --   +CASE WHEN opf.num_of_optional_fees_refunded>0 THEN 1 ELSE 0 END
      --    +CASE WHEN pf.num_of_prox_fees_refunded>0 THEN 1 ELSE 0 END
      --    +CASE WHEN ff.num_of_fleet_adv_refunded>0 THEN 1 ELSE 0 END
       --   +CASE WHEN cf.num_of_card_fees_refunded>0 THEN 1 ELSE 0 END
       --   +CASE WHEN inf.num_of_inac_fees_refunded>0 THEN 1 ELSE 0 END
       --   +CASE WHEN cof.num_of_admin_fees_refunded>0 THEN 1 ELSE 0 END
      --   +CASE WHEN ef.num_of_expr_fees_refunded>0 THEN 1 ELSE 0 END
      --    +CASE WHEN elf. num_of_eld_eqi_fees_refunded>0 THEN 1 ELSE 0 END
         --     AS unique_types_of_fees_refunded
        
FROM WEEKS_CUST_MAP WM 
 LEFT  JOIN fuel_stats trx ON  WM.ACCTCODE=trx.AcctCode AND  WM.YEARWEEK=trx.YEARWEEK
 LEFT  JOIN fleet_trx_fee frxf ON WM.ACCTCODE=frxf.AcctCode AND  WM.YEARWEEK=frxf.YEARWEEK
 LEFT  JOIN late_interest_fee laf ON WM.ACCTCODE=laf.AcctCode AND  WM.YEARWEEK=laf.YEARWEEK
 LEFT  JOIN risk_based_pricing_fee rf ON  WM.ACCTCODE=rf.AcctCode AND WM.YEARWEEK=rf.YEARWEEK
 LEFT  JOIN  optional_program_fee opf ON  WM.ACCTCODE=opf.AcctCode AND WM.YEARWEEK=opf.YEARWEEK
 LEFT  JOIN  proximity_fee pf ON  WM.ACCTCODE=pf.AcctCode AND WM.YEARWEEK=pf.YEARWEEK
 LEFT  JOIN fleet_advance_fee  ff ON WM.ACCTCODE=ff.AcctCode AND WM.YEARWEEK=ff.YEARWEEK
 LEFT  JOIN  card_fee  cf ON WM.ACCTCODE=cf.AcctCode AND WM.YEARWEEK=cf.YEARWEEK
 LEFT  JOIN  inactivity_fee  inf ON WM.ACCTCODE=inf.AcctCode AND WM.YEARWEEK=inf.YEARWEEK
 LEFT  JOIN  comcheck_admin_fee   cof ON WM.ACCTCODE=cof.AcctCode AND WM.YEARWEEK=cof.YEARWEEK
 LEFT  JOIN  express_code_fee   ef ON WM.ACCTCODE=ef.AcctCode AND WM.YEARWEEK=ef.YEARWEEK
 LEFT  JOIN eld_equipment_fee elf  ON WM.ACCTCODE=elf.AcctCode AND WM.YEARWEEK=elf.YEARWEEK
 LEFT  JOIN EXPRESS_CHECK EC ON EC.ACCTCODE=WM.ACCTCODE AND EC.YEARWEEK=WM.YEARWEEK
 LEFT  JOIN VCOMCHECK VC ON VC.ACCTCODE=WM.ACCTCODE AND VC.YEARWEEK=WM.YEARWEEK
 LEFT JOIN VCAP VCP ON VCP.ACCTCODE=WM.ACCTCODE AND VCP.YEARWEEK=WM.YEARWEEK
 LEFT  JOIN ECASH_SPEND ES ON ES.ACCTCODE=WM.ACCTCODE AND ES.YEARWEEK=WM.YEARWEEK
WHERE WM.ACCTCODE IS NOT NULL

                     
 ORDER BY WM.ACCTCODE,WM.YEARWEEK

