# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace

from azureml.core import Run
import utilities as ut

# %%
# Notebook specific imports
import re

# %%
print('Merge Transaction Demographics Account Feature Engineered data.')

# %%
run = Run.get_context()
spark = SparkSession.builder.getOrCreate()
tbl_feat_eng = spark.read.format('delta').load(run.input_datasets['feature_eng_transaction_data'])
tbl_transaction= spark.read.format('delta').load(run.input_datasets['transformed_transaction_data'])
tbl_demographics = spark.read.format('delta').load(run.input_datasets['transformed_demographics_data'])
tbl_account = spark.read.format('delta').load(run.input_datasets['transformed_account_data'])

# %%
def parse_args():
    parser = argparse.ArgumentParser('merged_transformed_data')
    parser.add_argument('--output_data_path', type=str, 
                        help='merged Transaction Demographics Account Feature Engineered table directory')
    parser.add_argument('--observation_month_number', type=str, help='Merging data for only observation month')
    parser.add_argument('--observation_year', type=str, help='Merging data for only observation month & year')

    return parser.parse_args()
 
args = parse_args()
run_id = run.id
output_file_name = 'merged_transformed_data'
output_path = f"{args.output_data_path}"#/{run_id}
print(f'Arguments: {args.__dict__}')

# %%
def fun_merge_table(tbl_trans, tbl_acc, tbl_demo, observation_month_number, observation_year):
    ## Merging Transaction & Account
    print('Transaction columns', tbl_trans.columns)
    print('Account columns', tbl_acc.columns)
    print('Demographic columns', tbl_demo.columns)
    
    # Merging Transaction and Account tables
    df_merge1 = tbl_trans.join(tbl_acc.select(['HASHED_CIF', 'No_of_Acc_Held']), 'HASHED_CIF', 'left')
    
    # Remove customers where account information is null
    remove_lst = [row['HASHED_CIF'] for row in df_merge1.filter(col('No_of_Acc_Held').isNull()).select('HASHED_CIF').distinct().collect()]
    df_merge2 = df_merge1.filter(~col('HASHED_CIF').isin(remove_lst))

    ## Merging with Demographics
    df_merge3 = df_merge2.join(tbl_demo.drop('AVG_INCOME'), 'HASHED_CIF', 'left')

    # Merging only with observation month data with Feature Engineered Columns
    df_merge4 = df_merge3.filter((col('YEAR') == observation_year) & (col('MONTH') == observation_month_number))

    return df_merge4

# %%
def fun_replace_special_chars(col_name):
    '''Replace special characters in a string with underscore.'''
    return regexp_replace(col_name, '[^a-zA-Z0-9 \n\.]', '_')

# %%
drop_columns = [
    'PROVINCE', 'PROVINCE_REGION', 'OCCUPATION_GROUP', 'OCCUPATION', 'AGE_GROUP', 'CUS_AGE', 
    'MARITAL_GROUP', 'MARTIAL_STATUS', 'COUNTRY_GROUP', 'COUNTRY', 'CUSTOMER_SEGMENT', 'GENDER', 
    'BRANCH_ID','num_of_AC_CREATE'
]

# %%
# Merge the tables
tbl = fun_merge_table(tbl_transaction, tbl_account, tbl_demographics, int(args.observation_month_number), int(args.observation_year))

# replace special characters in column names with underscore
tbl.columns = [fun_replace_special_chars(col_name) for col_name in tbl.columns]
tbl.columns = [i.replace(' ', '_') for i in tbl.columns]

tbl = tbl.drop(*drop_columns)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)


