#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import datetime as dt
import time
import argparse
import os
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


run = Run.get_context()

if args.reuse_sample_flag=='False':
    tbl_demo = run.input_datasets['raw_data_demographics'].to_pandas_dataframe()
    tbl_trx = run.input_datasets['raw_data_transaction'].to_pandas_dataframe()
    tbl_acc = run.input_datasets['raw_data_account'].to_pandas_dataframe()
else:
    tbl_demo = run.input_datasets['raw_data_demographics']
    tbl_trx = run.input_datasets['raw_data_transaction']
    tbl_acc = run.input_datasets['raw_data_account']


# In[ ]:


if args.reuse_sample_flag=='False':
    sample_cus_lst = tbl_trx['HASHED_CIF'].unique()[:args.num_sample_customer_ids]
    output_tbl_demo = tbl_demo[tbl_demo['HASHED_CIF'].isin(sample_cus_lst)]
    output_tbl_trx = tbl_trx[tbl_trx['HASHED_CIF'].isin(sample_cus_lst)]
    output_tbl_acc = tbl_acc[tbl_acc['HASHED_CIF'].isin(sample_cus_lst)]


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

