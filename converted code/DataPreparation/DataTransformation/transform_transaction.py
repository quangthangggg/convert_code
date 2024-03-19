# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
import sys
from azureml.core import Run
import utilities as ut

# %%
# Notebook specific imports
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col, year, month, lit, to_date, countDistinct, last
from pyspark.sql.functions import  sum as pyspark_sum
from joblib import Parallel, delayed
import multiprocessing as mp

# %%
print('Transform the transaction data.')

# %%
# Đọc dữ liệu từ raw_data_transaction và mapping_data_translation vào DataFrame
spark = SparkSession.builder.getOrCreate()
run = Run.get_context()

# Read data from Azure Machine Learning Datasets into DataFrames
tbl = spark.read.format("delta").load(run.input_datasets['raw_data_transaction'])
tbl_translation = spark.read.format("delta").load(run.input_datasets['mapping_data_translation'])
tbl_demo = spark.read.format("delta").load(run.input_datasets['transformed_demographics_data'])

# %%
def parse_args():
    parser = argparse.ArgumentParser('transform_transaction')
    parser.add_argument('--output_data_path', type=str, help='transformed transaction table directory')
    parser.add_argument('--njobs', type=str, help='number of parallel threads to be run for applyParallel function')

    return parser.parse_args()

args = parse_args()
run_id = run.id
output_file_name = 'transformed_transaction_data'
output_path = f"{args.output_data_path}"#/{run_id}
print(f'Arguments: {args.__dict__}')

# %%
def fun_keep_vnd_currency(df, cus_id_col, currency_col, currency_value):
    '''This function removes rows which have made transaction in other than VND currency. 
    For Ex: if currency_value is USD then we will remove the data for that customer.'''

    # Select rows with only VND currencies
    tbl = df.filter(df[currency_col]==currency_value)
    # Select the customers which have VND currency transaction
    lst = df.select(cus_id_col).distinct()
    # Filter the customers not from list above
    tbl1 = df.join(lst, df[cus_id_col] == lst[cus_id_col], "left_anti")
    tbl1.persist()
    print('Number of customers made transactions in others currencies: ', tbl1.select(cus_id_col).distinct().count())
    
    return tbl

# %%
def fun_transaction_rollup_monthly(df):
    ''' Rolling up weekly transaction data to monthly. Summing all the values in aggregation. 
    For column HASHED_AC, we take number of accounts in rollup.'''

    # Columns that need not be rolled up
    col_object = ['YEAR', 'MONTH', 'WEEK', 'HASHED_CIF', 'HASHED_AC', 'CCY']

    # Defining dictionary with column and aggregation
    col_num = [pyspark_sum(col(i)).alias(i) for i in df.columns if i not in col_object]
    dict_col_agg = {'HASHED_AC': countDistinct('HASHED_AC').alias('No_of_Accounts')}  # HASHED_AC aggregation

    # Aggregating the other columns based on sum of the month
    df_monthly = df.groupBy('YEAR', 'MONTH', 'HASHED_CIF').agg(*col_num, dict_col_agg)
    df_monthly = df_monthly.withColumnRenamed('No_of_Accounts', 'No_of_Accounts')  # Rename the aggregated column
    df_monthly.persist()
    print(f'Number of records in Transformed Transactions Table: {df_monthly.count()}')
    print(f'Number of features in Transformed Transactions Table: {len(df_monthly.columns)}')
    return df_monthly


# %%
def fun_impute_missing_cusid(df_cust: DataFrame, args_lst: list):
    '''Imputing customer id with null data for the months in which they have not made any transactions. 
    This function will take 1 customer data at a time.
    Creating new customer flag, this flag will tell in which month and week the customer cif was created.'''
    
    # Creating flag cif_missing_flag which is 1 when the customer has no data for a particular month
    # If the customer is new then new_customer_flag is 1 and this flag is 0
    df_cust = df_cust.withColumn('cif_missing_flag', lit(0))
    
    # Get min & max date from transaction data
    min_date, max_date = args_lst[0], args_lst[1]

    # Add the customer dummy data for that HASHED_CIF, YEAR & MONTH,
    # We add all the other columns after these 3 and at the end we add the cif_missing_flag column
    for year in range(min_date.year, max_date.year + 1):
        # Get list of months for the year
        year_min_date = max(min_date, dt.date(year, 1, 1))
        year_max_date = min(max_date, dt.date(year, 12, 1))
        
        # Get list of months which are missing from data for customer
        lst_dummy_month = [m for m in range(year_min_date.month, year_max_date.month) if m not in df_cust.select('MONTH').distinct().rdd.flatMap(lambda x: x).collect()]
        
        # Create DataFrame of dummy data for missing months
        dummy_data = spark.createDataFrame([(year, month, df_cust.first()['HASHED_CIF'], *[0] * (len(df_cust.columns) - 4), 1) for month in lst_dummy_month], schema=df_cust.schema)
        
        # Union dummy data with existing customer data
        df_cust = df_cust.union(dummy_data)
            
    # Sorting the values based on date
    df_cust = df_cust.orderBy('YEAR', 'MONTH')

    # For PRE_CLS_BAL & CCY columns we will populate the previous values in dummy month rows
    df_cust = df_cust.withColumn('PRE_CLS_BAL', last('PRE_CLS_BAL', True).over(Window.partitionBy('HASHED_CIF').orderBy('YEAR', 'MONTH').rowsBetween(-sys.maxsize, 0)))
    #df_cust = df_cust.withColumn('CCY', last('CCY', True).over(Window.partitionBy('HASHED_CIF').orderBy('YEAR', 'MONTH').rowsBetween(-sys.maxsize, 0)))

    # Making cid_not_exist_flag = 1, where the customer does not exist with the bank
    df_cust = df_cust.withColumn('cid_not_exist_flag', last('new_customer_flag', True).over(Window.partitionBy('HASHED_CIF').orderBy('YEAR', 'MONTH').rowsBetween(-sys.maxsize, 0)))
    df_cust = df_cust.withColumn('cid_not_exist_flag', last('cid_not_exist_flag', True).over(Window.partitionBy('HASHED_CIF').orderBy('YEAR', 'MONTH').rowsBetween(0, sys.maxsize)))
    
    # Remove rows where customer did not exist
    df_cust = df_cust.filter(~((col('cid_not_exist_flag') == 1) & (col('new_customer_flag') == 0)))
    
    # Drop cid_not_exist_flag column
    df_cust = df_cust.drop('cid_not_exist_flag')
    
    # Filling all other values with 0
    df_cust = df_cust.fillna(0).orderBy('YEAR', 'MONTH')

    return df_cust

# %%
# Re-arranging transaction columns
order_trans_cols = ['YEAR', 'MONTH', 'WEEK', 'HASHED_CIF', 'HASHED_AC', 
                    'NO_TRANSATION', 'NO_CREDIT', 'NO_DEBIT', 
                    'NO_ATM', 'NO_SMB', 'NO_FUND_TRANSFER', 'NO_DEPOSIT', 'NO_WITHDRAW', 
                    'NO_INTEREST_PAYMENT', 'NO_FEE_TRANSACTION', 'NO_TRANSATION_AUTO',
                    'CCY',
                    'PRE_CLS_BAL', 'AMT_CREDIT', 'AMT_DEBIT', 
                    'AMOUNT_TRANSACTION', 'AMOUNT_TRANSACTION_AUTO']

# %%
start_time = time.time()

# Convert vietnamese column names or vietnamese values to english
tbl = ut.fun_convert_vnt_to_eng(tbl_translation, tbl)

# Dropping duplicate rows
tbl = ut.fun_drop_duplicates(tbl, 'Transaction Data')

# Re-arranging transaction columns
df_trans = tbl[order_trans_cols]

# Getting column and missing percentage information for columns with missing values
print('Initial Missing Report:')
df_missing = ut.fun_get_missing_perc(df_trans)

# Remove customers with transaction other than VND
df_trans = fun_keep_vnd_currency(df_trans, 'HASHED_CIF', 'CCY', 'VND')

# Getting column and missing percentage information for columns with missing values
print('Before Rollup to monthly Missing Report:')
df_missing = ut.fun_get_missing_perc(df_trans)
print('Unique years in data:', df_trans['YEAR'].unique())
print('Unique months in data:', df_trans['MONTH'].unique())

# Aggregating weekly transaction data to monthly
# Roll up transaction data to monthly level
df_trans_monthly = fun_transaction_rollup_monthly(df_trans)

# Creating date column by taking 1st date of the month
df_trans_monthly = df_trans_monthly.withColumn('date', to_date(col('YEAR').cast('string') + lit('_') + col('MONTH').cast('string') + lit('_01'), 'yyyy_MM_dd'))

# Get date range in transaction data
min_date = df_trans_monthly.select('date').agg({'date': 'min'}).collect()[0][0]
max_date = df_trans_monthly.select('date').agg({'date': 'max'}).collect()[0][0]

# Merge transaction with demographics for creating new customer flag
df_trans_demo_monthly = df_trans_monthly.join(tbl_demo.select('HASHED_CIF', 'CIF_CREATE'), on='HASHED_CIF', how='left')

# Creating new customer Flag using cif_create date
df_trans_demo_monthly = df_trans_demo_monthly.withColumn('new_customer_flag', 
                                                         (year('date') == year('CIF_CREATE')) & 
                                                         (month('date') == month('CIF_CREATE')))
df_trans_demo_monthly = df_trans_demo_monthly.withColumn('new_customer_flag', df_trans_demo_monthly['new_customer_flag'].cast('integer'))

# Impute customer data with null if not exist for any month, for new customer it will not impute with dummy data, 
# Also, columns pre_cls_bal will be imputed with previous value
# Parallel function
df_trans_monthly_imputed = ut.applyParallel(
    dfGrouped=df_trans_demo_monthly.groupby('HASHED_CIF'), 
    func=fun_impute_missing_cusid, 
    args_lst=[min_date, max_date],
    njobs = int(args.njobs)
)

# Non-parallel function - only for testing
# df_trans_monthly_imputed = df_trans_monthly_demo.groupby('HASHED_CIF').apply(fun_impute_missing_cusid, 
#                                                                              ([min_date, max_date]))
# df_trans_monthly_imputed.reset_index(drop=True, inplace=True)

# Dropping columns which are not required
# Select columns you want to keep
columns_to_keep = [col for col in df_trans_monthly_imputed.columns if col not in ['date', 'CIF_CREATE']]

# Create new DataFrame with selected columns
df_trans_monthly_imputed = df_trans_monthly_imputed.select(columns_to_keep)

# Final table output i.e. transaction table monthly information
tbl = df_trans_monthly_imputed

# Dropping duplicate rows
tbl = ut.fun_drop_duplicates(tbl, 'Transaction Data')

end_time = time.time()

print(f'Time to run the code to transform transaction data: {time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))}')

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)


