WITH ACTIVE_CUSTS AS
(
  SELECT 
              TRIM(dt.ACCTCODE) as ACCTCODE
   FROM 
              edl.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009 dt
   LEFT  JOIN  edl.edl_repl.repl_nat_newlob CO 
               ON  TRIM(dt.ACCTCODE) = CO.ACCTCODE
    LEFT JOIN edl.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',DT.TXNDATE)
    WHERE 
                 CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{activity_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
                AND  CO.FINALLOB = 'NAT'
    GROUP BY dt.ACCTCODE
  
  UNION 
    
  SELECT 
               F.FMDL_CUST_ACCOUNT_CODE AS ACCTCODE
  FROM  edl.edl_repl.repl_fmlog F
  LEFT  JOIN edl.edl_repl.repl_nat_newlob CO 
                       ON FMDL_CUST_ACCOUNT_CODE = CO.ACCTCODE
  LEFT JOIN edl.edl_repl.REPL_MART_CALENDAR CAL 
                      ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',F.FMDL_rk_transaction_date)
  WHERE CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{activity_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer
               AND FMDL_SC_SRVC_CNTR_CODE NOT IN ('MC901','MC902','MC903','MC904','MC905')
              AND  CO.FINALLOB = 'NAT'
  GROUP BY  F.FMDL_CUST_ACCOUNT_CODE
),
pull_parent_name AS
(
 SELECT * FROM edl.edl_repl.repl_nat_account_attributes
),
survey AS (
    select
		  sf_acc.account_code_from_account__c as customer_id,
      pa.parentname as parentname,
      surv.issue_solved_indicator_text__c AS issuesolvedind,
      surv.product_rating_number__c AS productrating,
      surv.rep_rating_number__c AS agentrating,
		  surv.product_nps_rating__c AS product_nps,
		  surv.rep_nps_rating__c AS agent_nps,
		 NVL( to_date(surv.survey_response_date__c, 'YYYY-MM-DD'),to_date(surv.createddate,'YYYY-MM-DD')) AS case_response_dt,
        CAL.week_no AS case_week 
         
    from edl.edl_repl.repl_sf_gold_post_call_survey surv
    inner join (select distinct s.accountid,s.account_code_from_account__c from edl.edl_repl.repl_sf_gold_contact s) sf_acc ON surv.account__c = sf_acc.accountid 
    inner join active_custs CUST ON CUST.ACCTCODE=sf_acc.account_code_from_account__c
    left join pull_parent_name pa on pa.ACCTCODE=sf_acc.account_code_from_account__c
    left join edl.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',NVL(surv.survey_response_date__c,surv.createddate))
   where CAL.WEEK_NO BETWEEN to_char(to_date('{current_week}', 'yyyyww') - interval '{history_weeks} weeks', 'yyyyww')::integer AND '{current_week}'::integer  
    and sf_acc.account_code_from_account__c is not null
    and CAL.WEEK_NO>='201925'
   and parentname is not null
),
remove_duplicate_survey as(
select * from (
    select customer_id,parentname,case_week,  
                                         issuesolvedind,
                                          productrating,
                                          agentrating,
		      product_nps,
		      agent_nps,
		     case_response_dt    , 
    ROW_NUMBER() OVER(PARTITION BY customer_id, 
                                          parentname, 
                                          case_week,  
                                         issuesolvedind,
                                          productrating,
                                          agentrating,
		      product_nps,
		      agent_nps,
		     case_response_dt     
		  
           ORDER BY case_week) AS DuplicateCount
    from survey
) where DuplicateCount=1
)
select * from remove_duplicate_survey