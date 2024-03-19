# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col
import pyspark.sql.functions as F

from azureml.core import Run
import utilities as ut

# %%
print('Transform the account data.')

# %%
run = Run.get_context()
spark = SparkSession.builder.getOrCreate()
tbl = spark.read.format('delta').load(run.input_datasets['raw_data_account'])
tbl_translation = spark.read.format('delta').load(run.input_datasets['mapping_data_translation'])

# %%
def parse_args():
    parser = argparse.ArgumentParser('transform_account')
    parser.add_argument('--output_data_path', type=str, help='transformed account table directory')

    return parser.parse_args()
 
args = parse_args()
run_id = run.id
output_file_name = 'transformed_account_data'
output_path = f"{args.output_data_path}"#/{run_id}
print(f'Arguments: {args.__dict__}')

# %%
def fun_remove_anomaly(df):
    ''' This function removes rows which are anomalies.
     For Ex: DATE_OF_STATUS is older than AC_CREATE means Date of Status is older than Date of Account created,
      then we remove those customers with that condition. 
      Here are the list of anomalies:
        - Anomaly 1: Date of Status is older than Date of Account created
        - Anomaly 2: Single Account with different AC_CREATE dates.
        - Anomaly 3: Single Account mapped to multiple customers
        - Anomaly 4: Removing rows with multiple product id
        - Anomaly 5: Account active date is latest as compared to account closed date
        - Anomaly 6: Account active date is not present.'''

    # Anomaly 1: Date of Status is older than Date of Account created
    print('Anomaly 1: Date of Status is older than Date of Account created')
    print('-'*50)
    df_ar = df.filter(col('DATE_OF_STATUS') >= col('AC_CREATE'))
    df_ar = df_ar.orderBy('HASHED_CIF', 'HASHED_AC', 'DATE_OF_STATUS')
    # Print the result of how many rows have been removed
    print(f'Number of unique Customers in Account Life Cycle Table: {df_ar.select("HASHED_CIF").distinct().count()}', 
          f', dropped {df.select("HASHED_CIF").distinct().count() - df_ar.select("HASHED_CIF").distinct().count()} customers')
    print(f'Number of unique Accounts in Account Life Cycle Table: {df_ar.select("HASHED_AC").distinct().count()}', 
          f', dropped {df.select("HASHED_AC").distinct().count() - df_ar.select("HASHED_AC").distinct().count()} accounts')
    print(f'Number of records in Account Life Cycle Table: {df_ar.count()}', 
          f', dropped {df.count() - df_ar.count()} records')
    print('-'*70)

    # Anomaly 2: Single Account with different AC_CREATE dates.
    print('Anomaly 2: Single Account with different AC_CREATE dates')
    print('-'*50)
    window_spec = Window.partitionBy('HASHED_CIF', 'HASHED_AC', 'STATUS_DESCRIPTION', 'STATUS_ID', 'PRODUCT_ID').orderBy('AC_CREATE')
    df_ar1 = df_ar.orderBy('HASHED_CIF', 'HASHED_AC', 'AC_CREATE').withColumn('row_number', F.row_number().over(window_spec))
    df_ar2 = df_ar1.filter(col('row_number') == 1)
    # Print the result of how many rows have been removed
    print(f'Before drop data anomalies: {df_ar1.count()}')
    rows_before = df_ar1.count()
    df_ar2 = df_ar1.filter(col('row_number') == 1)
    rows_after = df_ar2.count()
    print(f'After drop data anomalies: {rows_after}')
    print(f'Number of rows dropped: {rows_before - rows_after}')
    print('-'*70)

    # Anomaly 3: Single Account mapped to multiple customers
    print('Anomaly 3: Single Account mapped to multiple customers')
    print('-'*50)
    # Count how many customers have the same account
    tbl = df_ar2.groupBy('HASHED_AC').agg(F.countDistinct('HASHED_CIF').alias('count_HASHED_CIF'))
    # Filter the account has more than 1 customer
    tbl = tbl.filter(col('count_HASHED_CIF') > 1)
    # Filter list of accounts have more than 1 customer
    tbl2 = df_ar2.join(tbl, 'HASHED_AC').orderBy('HASHED_AC', 'DATE_OF_STATUS', ascending=False)
    # Remove the list of accounts have more than 1 customer
    df_ar3 = df_ar2.join(tbl2, 'HASHED_AC', 'left_anti')
    # Remove the list of customers have the same account
    df_ar3 = df_ar3.join(tbl2, 'HASHED_CIF', 'left_anti')
    print(f'Before drop data anomalies: {df_ar2.count()}')
    print(f'After drop data anomalies: {df_ar3.count()}')
    print('-'*50)
    # Print the result
    print(f'Number of rows dropped: {df_ar2.count() - df_ar3.count()}')
    print(f'Number of Customers dropped: {df_ar2.select("HASHED_CIF").distinct().count() - df_ar3.select("HASHED_CIF").distinct().count()}')
    print(f'Number of Accounts dropped: {df_ar2.select("HASHED_AC").distinct().count() - df_ar3.select("HASHED_AC").distinct().count()}')
    print('-'*70)

    # Anomaly 4: Removing rows with multiple product id
    print('Anomaly 4: Removing rows with multiple product id')
    print('-'*50)
    df_ar4 = df_ar3.dropDuplicates(['HASHED_CIF', 'HASHED_AC', 'STATUS_DESCRIPTION', 'DATE_OF_STATUS'])
    print(f'Number of rows dropped: {df_ar3.count() - df_ar4.count()}')
    print(f'Number of Customers dropped: {df_ar3.select("HASHED_CIF").distinct().count() - df_ar4.select("HASHED_CIF").distinct().count()}')
    print(f'Number of Accounts dropped: {df_ar3.select("HASHED_AC").distinct().count() - df_ar4.select("HASHED_AC").distinct().count()}')
    print('-'*70)
    # Anomaly 5: Account active date is latest as compared to account closed date
    print('Anomaly 5: Account active date is latest as compared to account closed date')
    print('-'*50)
    tbl_ar = df_ar4.groupBy('HASHED_CIF', 'HASHED_AC').pivot('STATUS_DESCRIPTION').agg(F.first('DATE_OF_STATUS')).orderBy('HASHED_CIF', 'Active')
    tbl_ar = tbl_ar.filter(tbl_ar['Active'].isNotNull() & tbl_ar['Closed'].isNotNull())
    e = tbl_ar.filter(col('Active') > col('Closed'))
    print(f'Number of Unique Customers have Account active date is latest: {e.select("HASHED_CIF").distinct().count()}')
    print(f'Number of Unique Accounts have active date is latest: {e.select("HASHED_AC").distinct().count()}')
    print('-'*50)
    cols_4 = ['HASHED_CIF', 'PRODUCT_ID', 'AC_CREATE', 'HASHED_AC']
    df_ar5 = df_ar4.join(tbl_ar.select(cols_4), ['HASHED_CIF', 'HASHED_AC', 'AC_CREATE'], 'inner')
    df_ar5 = df_ar5.dropDuplicates(['HASHED_CIF', 'HASHED_AC', 'AC_CREATE'])
    print(f'Number of rows dropped: {df_ar4.count() - df_ar5.count()}')
    print(f'Number of Customers dropped: {df_ar4.select("HASHED_CIF").distinct().count() - df_ar5.select("HASHED_CIF").distinct().count()}')
    print(f'Number of Accounts dropped: {df_ar4.select("HASHED_AC").distinct().count() - df_ar5.select("HASHED_AC").distinct().count()}')
    print('-'*70)

    # Anomaly 6: Account active date is not present
    print('Anomaly 6: Account active date is not present')
    print('-'*50)
    df_ar6 = df_ar5.withColumn('Active', 
                                F.when(col('Active').isNull() & col('Closed').isNotNull(), 
                                       F.date_add(col('AC_CREATE'), 1)).otherwise(col('Active')))
    print('Replace Null Active date with Account create date + 1')
    abc = df_ar6.filter(col('Active').isNull())
    print(f'Number of Unique Customers where Active date is Null: {abc.select("HASHED_CIF").distinct().count()}')
    print(f'Number of Unique Accounts where Active date is Null: {abc.select("HASHED_AC").distinct().count()}')

    df_ar7 = df_ar6.fillna({'Active': F.date_add(col('AC_CREATE'), 1)})
    print(f'Number of rows dropped: {df_ar6.count() - df_ar7.count()}')
    print(f'Number of Customers dropped: {df_ar6.select("HASHED_CIF").distinct().count() - df_ar7.select("HASHED_CIF").distinct().count()}')
    print(f'Number of Accounts dropped: {df_ar6.select("HASHED_AC").distinct().count() - df_ar7.select("HASHED_AC").distinct().count()}')
    print('-'*70)
    
    # Account table final
    df_ar8 = df_ar7
    print(f'Number of unique Customers in Account Table: {df.select("HASHED_CIF").distinct().count()}')
    print(f'Number of unique Accounts in Account Table: {df.select("HASHED_AC").distinct().count()}')
    print(f'Number of unique Customers in Account Table Remaining: {df_ar8.select("HASHED_CIF").distinct().count()}')
    print(f'Number of unique Accounts in Account Table Remaining: {df_ar8.select("HASHED_AC").distinct().count()}')
    print(f'Number of unique Customers in Account Table Removed: {df.select("HASHED_CIF").distinct().count() - df_ar8.select("HASHED_CIF").distinct().count()}')
    print(f'Number of unique Accounts in Account Table Removed: {df.select("HASHED_AC").distinct().count() - df_ar8.select("HASHED_AC").distinct().count()}')
    print(f'Number of rows dropped: {df.count() - df_ar8.count()}')
    print('-'*70)

    # Final Transform table: the account information
    tbl_final = df_ar8.groupBy('HASHED_CIF').count()
    lst = [f'num_of_{i}' for i in tbl_final.columns if i not in ['HASHED_CIF', 'HASHED_AC']]
    lst = ['HASHED_CIF', 'HASHED_AC'] + lst
    tbl_final = tbl_final.toDF(*lst)
    tbl_final = tbl_final.withColumnRenamed('HASHED_AC', 'No_of_Acc_Held')

    return tbl_final

# %%
# Convert vietnamese column names or vietnamese values to english
tbl = ut.fun_convert_vnt_to_eng(tbl_translation, tbl)

# Dropping duplicate rows
tbl = ut.fun_drop_duplicates(tbl, 'Account Life Cycle Data')

# Converting date string columns to datetime format
tbl = tbl.withColumn('DATE_OF_STATUS', F.to_date(tbl['DATE_OF_STATUS'], 'dd-MMM-yy'))

# Chuyển đổi cột AC_CREATE sang định dạng ngày tháng
tbl = tbl.withColumn('AC_CREATE', F.to_date(tbl['AC_CREATE'], 'yyyy-MM-dd'))

# Re-arranging account columns
acc_cols = ['HASHED_CIF', 'HASHED_AC', 'DATE_OF_STATUS', 'STATUS_DESCRIPTION', 'AC_CREATE', 'STATUS_ID', 'PRODUCT_ID']
tbl = tbl[acc_cols]
# Final table output i.e. account table information
tbl = fun_remove_anomaly(tbl)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)


