#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime as dt
import time
import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from azureml.core import Run
import utilities as ut


# In[ ]:


print('Merge with churn flag data.')


# In[ ]:


run = Run.get_context()
spark = SparkSession.builder.getOrCreate()
tbl_churn = spark.read.format('delta').load(run.input_datasets['created_churn_flag_data'])
tbl_merged = spark.read.format('delta').load(run.input_datasets['merged_transformed_data'])


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser('merged_churn_flag')
    parser.add_argument('--output_data_path', type=str, help='merged with churn flag table directory')
    parser.add_argument('--observation_month_number', type=str, help='Merging data for only observation month')
    parser.add_argument('--observation_year', type=str, help='Merging data for only observation month & year')

    return parser.parse_args()
 
args = parse_args()
run_id = run.id
output_file_name = 'merged_with_churn_flag_data'
output_path = f"{args.output_data_path}"#/{run_id}
print(f'Arguments: {args.__dict__}')


# In[ ]:


def fun_merge_table(tbl_merged, tbl_churn, observation_month_number, observation_year):
    # Lựa chọn các dòng cho tháng và năm quan sát
    tbl_churn = tbl_churn.filter((col('YEAR') == observation_year) & (col('MONTH') == observation_month_number))
    
    # Thực hiện join giữa hai DataFrame
    df_merge = tbl_merged.join(tbl_churn, ['HASHED_CIF', 'YEAR', 'MONTH'], 'left_outer')
    
    return df_merge


# In[ ]:


drop_columns = [
    'NO_TRANSATION_n1m', 'NO_TRANSATION_n2m', 'NO_TRANSATION_n1m_n2m',
]


# In[ ]:


tbl = fun_merge_table(tbl_merged, tbl_churn, int(args.observation_month_number), int(args.observation_year))
tbl = tbl.drop(*drop_columns)


# In[ ]:


# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)

