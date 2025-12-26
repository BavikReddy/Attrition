CREATE TABLE IF NOT EXISTS {result_schema}.{validation_table}
(
 Last_Transaction_Weeks varchar(500),
Weeks varchar(500),
#Actual_churners INT4,
#Predicted_churners INT4,
Percentage_Prediction float4,
customer_size varchar(500),
run_week varchar(30),
  run_date TIMESTAMP
);
DELETE FROM {result_schema}.{validation_table}
WHERE run_date = '{run_date}' and run_week = '{run_week}'
