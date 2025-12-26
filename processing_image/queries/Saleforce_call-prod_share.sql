WITH ACTIVE_CUSTS AS
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
pull_parent_name AS
(
 SELECT * FROM prod_share.edl_repl.repl_nat_account_attributes
),
call_cases AS (
  select
		 
		  sf_acc.account_code_from_account__c as customer_id,
                                     pa.parentname,
		  to_date(cas.createddate, 'YYYY-MM-DD') AS createddate,
		  cas.lastmodifieddate,
		  cas.closeddate,
          type AS type,
		  sub_type__c AS subtype,
      cas.count_status_changes__c AS statuschanges,
      cas.customer_satisfied_with_result__c  AS satisficationflag,
		  resolution__c AS resolution,
          CAL.WEEK_NO as call_week 
          
      
  from prod_share.edl_repl.repl_sf_gold_case cas
  inner join (select distinct s.accountid,s.account_code_from_account__c from prod_share.edl_repl.repl_sf_gold_contact s) sf_acc ON sf_acc.accountid = cas.accountid
  inner join active_custs cust on cust.ACCTCODE=sf_acc.account_code_from_account__c
  inner join pull_parent_name  pa on pa.ACCTCODE=sf_acc.account_code_from_account__c
  left join prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',cas.createddate)
   where CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  

  and sf_acc.account_code_from_account__c IS NOT NULL
  and pa.parentname IS NOT NULL
),
remove_duplicate_call_cases as(
select * from (
    select customer_id,parentname,call_week,type,subtype,statuschanges,satisficationflag,resolution,createddate,lastmodifieddate,
		  closeddate,   
    ROW_NUMBER() OVER(PARTITION BY customer_id, 
                                          parentname, 
                                          call_week,type,subtype,statuschanges,satisficationflag,resolution,createddate,
		  lastmodifieddate,
		  closeddate                            
		  
           ORDER BY call_week) AS DuplicateCount
    from call_cases
) where DuplicateCount=1
)

SELECT * FROM remove_duplicate_call_cases