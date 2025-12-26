CREATE TABLE IF NOT EXISTS {result_schema}.{result_table}
(
  parentname varchar(2000) NOT NULL,
  first_trx_month INT4,  
 last_transaction_week INT4,
 state_code varchar(500),
 num_of_trucks  varchar(500),
   credit_limit    INT4,
  billing_freq varchar(500),
 due_days  INT4,
 credit_grade varchar(500),
tenure FLOAT4,
channel varchar(500),
 product_type_most_frequent varchar(500),
 last_payment_method varchar(500),
 churn_flag_actual   INT4,
churn_flag_predicted INT4,
attrition_prob FLOAT4,
attrition_rank INT4,
factor_1 text,
factor_2 text,
factor_3 text,
factor_4 text,
 factor_5 text,
factor_6 text,
factor_7 text,
factor_8 text,
factor_9 text,
factor_10 text,
customer_size varchar(30),
 run_week varchar(30),
  run_date TIMESTAMP



);
DELETE FROM {result_schema}.{result_table}
WHERE run_date = '{run_date}' and run_week = '{run_week}'

