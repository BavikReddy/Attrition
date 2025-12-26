SELECT max(CAL.WEEK_NO)::int AS max_week ,to_char(to_date(max(CAL.WEEK_NO)::int,'yyyyww')- interval '2 weeks','yyyyww')::int as churn_cut_off_week
FROM prod_share.edl_repl.repl_fmlog 
LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',FMDL_rk_transaction_date)

UNION ALL
SELECT  max(CAL.WEEK_NO)::int AS max_week ,to_char(to_date(max(CAL.WEEK_NO)::int,'yyyyww')- interval '2 weeks','yyyyww')::int as churn_cut_off_week 
FROM  prod_share.edl_repl.repl_fin1_tbl_modeling_by_day_2007_2009
LEFT JOIN prod_share.edl_repl.REPL_MART_CALENDAR CAL ON CAL.WEEK_STARTS = DATE_TRUNC('WEEK',TXNDATE)
