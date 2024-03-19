#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime as dt
import time
import argparse
import os
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col
from azureml.core import Run
import utilities as ut


# In[ ]:


print('Sample demographics, account & transaction data.')


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser('sample_data')
    parser.add_argument('--output_data_path_demo', type=str, help='sampled demographics table directory')
    parser.add_argument('--output_data_path_trx', type=str, help='sampled transaction table directory')
    parser.add_argument('--output_data_path_acc', type=str, help='sampled account table directory')
    parser.add_argument('--num_sample_customer_ids', type=int, help='number of customers to be sampled')
    parser.add_argument('--reuse_sample_flag', type=str, help='reuse the previously created sample')

    return parser.parse_args()

args = parse_args()
output_file_name_demo = 'sample_demographic_data'
output_file_name_trx = 'sample_transaction_data'
output_file_name_acc = 'sample_account_data'
output_path_demo = f"{args.output_data_path_demo}"
output_path_trx = f"{args.output_data_path_trx}"
output_path_acc = f"{args.output_data_path_acc}"
print(f'Arguments: {args.__dict__}')


# In[ ]:

spark = SparkSession.builder.getOrCreate()
run = Run.get_context()

if args.reuse_sample_flag=='False':
    tbl_demo = spark.read.format("delta").load(run.input_datasets['raw_data_demographics'])
    tbl_trx = spark.read.format("delta").load(run.input_datasets['raw_data_transaction'])
    tbl_acc = spark.read.format("delta").load(run.input_datasets['raw_data_account'])
else:
    tbl_demo = run.input_datasets['raw_data_demographics']
    tbl_trx = run.input_datasets['raw_data_transaction']
    tbl_acc = run.input_datasets['raw_data_account']


# In[ ]:
if args.reuse_sample_flag == 'False':
    # Lấy danh sách khách hàng mẫu
    sample_cus_lst = tbl_trx.select('HASHED_CIF').distinct().limit(args.num_sample_customer_ids)
    
    # Lọc dữ liệu theo danh sách khách hàng mẫu
    output_tbl_demo = tbl_demo.join(sample_cus_lst, on='HASHED_CIF', how='inner')
    output_tbl_trx = tbl_trx.join(sample_cus_lst, on='HASHED_CIF', how='inner')
    output_tbl_acc = tbl_acc.join(sample_cus_lst, on='HASHED_CIF', how='inner')


# In[ ]:


if args.reuse_sample_flag=='False':
    # Getting column and missing percentage information for columns with missing values
    df_missing = ut.fun_get_missing_perc(output_tbl_demo)

    # Writing the output to file
    ut.fun_write_file(output_tbl_demo, output_path_demo, output_file_name_demo, run=run, csv=False)


# In[ ]:


if args.reuse_sample_flag=='False':
    # Getting column and missing percentage information for columns with missing values
    df_missing = ut.fun_get_missing_perc(output_tbl_trx)

    # Writing the output to file
    ut.fun_write_file(output_tbl_trx, output_path_trx, output_file_name_trx, run=run, csv=False)


# In[ ]:


if args.reuse_sample_flag=='False':
    # Getting column and missing percentage information for columns with missing values
    df_missing = ut.fun_get_missing_perc(output_tbl_acc)

    # Writing the output to file
    ut.fun_write_file(output_tbl_acc, output_path_acc, output_file_name_acc, run=run, csv=False)

