SELECT
         CONCAT(YEAR, LPAD(FM_MONTH, 2, '0')) AS YEARMONTH,
         week_no as week
FROM prod_share.edl_repl.REPL_MART_CALENDAR WHERE week_no::VARCHAR BETWEEN  TO_CHAR(to_date('{current_week}','YYYYWW')- interval '{history_weeks} weeks','YYYYWW') AND '{current_week}'
