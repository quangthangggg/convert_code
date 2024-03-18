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


# Notebook specific imports
from joblib import Parallel, delayed
import multiprocessing as mp


# In[ ]:


print('Create the churn flag for model training & testing.')


# In[ ]:


run = Run.get_context()
tbl_transaction = run.input_datasets['transformed_transaction_data'].to_pandas_dataframe()


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser('created_churn_flag')
    parser.add_argument('--output_data_path', type=str, help='created churn flag table directory')
    parser.add_argument('--njobs', type=str, help='number of parallel threads to be run for applyParallel function')

    return parser.parse_args()
 
args = parse_args()
run_id = run.id
output_file_name = 'created_churn_flag_data'
output_path = f"{args.output_data_path}"#/{run_id}
print(f'Arguments: {args.__dict__}')


# In[ ]:


def fun_create_churn_flag(df_cust, args_lst=[]):
    ''' Function to create churn flag based on the definition:
        if there is no transaction in next 2 months, then churn flag is 1 else 0.'''

    # Select column based on which churn flag is defined
    cols_to_shift = ['NO_TRANSATION']
    
    # Create column for sum of number of transactions in next 1 month
    df_cust[[i+'_n1m' for i in cols_to_shift]] = df_cust[cols_to_shift].shift(-1).fillna(0)
    
    # Create column for with sum of number of transactions in next 2nd month
    df_cust[[i+'_n2m' for i in cols_to_shift]] = df_cust[cols_to_shift].shift(-2).fillna(0)
    
    # Check if the sum of number of transactions for next 2 months is greater then 0 then non-churn else churn
    df_cust['NO_TRANSATION_n1m_n2m'] = df_cust['NO_TRANSATION_n1m'] + df_cust['NO_TRANSATION_n2m']
    df_cust['churn_flag'] = df_cust['NO_TRANSATION_n1m_n2m'].apply(lambda x: 1 if x == 0.0 else 0)
    
    return df_cust


# In[ ]:


print('Columns from transformed transaction:', tbl_transaction.columns)
tbl_transaction = tbl_transaction[['HASHED_CIF', 'YEAR', 'MONTH', 'NO_TRANSATION']]

## Creating churn flag
# Normal Function - only for testing
# df_trans_monthly = tbl_transaction.groupby('HASHED_CIF').apply(fun_create_churn_flag)
# df_trans_monthly_imputed.reset_index(drop=True, inplace=True)

# Parallel Function
df_trans_monthly_churn = ut.applyParallel(
    tbl_transaction.groupby('HASHED_CIF'), 
    fun_create_churn_flag,
    njobs = int(args.njobs)
    )
churn_cols = ['HASHED_CIF', 'YEAR', 'MONTH', 'NO_TRANSATION_n1m', 'NO_TRANSATION_n2m', 'NO_TRANSATION_n1m_n2m', 'churn_flag']
tbl = df_trans_monthly_churn[churn_cols]


# In[ ]:


# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)

