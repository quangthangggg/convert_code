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


print('Merge with churn flag data.')


# In[ ]:


run = Run.get_context()
tbl_churn = run.input_datasets['created_churn_flag_data'].to_pandas_dataframe()
tbl_merged = run.input_datasets['merged_transformed_data'].to_pandas_dataframe()


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

    # selecting only rows for observation month & year
    tbl_churn = tbl_churn[(tbl_churn['YEAR']==observation_year) & (tbl_churn['MONTH']==observation_month_number)]
    df_merge = pd.merge(tbl_merged, tbl_churn, on = ['HASHED_CIF', 'YEAR', 'MONTH'], suffixes=[None, '_x'], how='left')

    return df_merge


# In[ ]:


drop_columns = [
    'NO_TRANSATION_n1m', 'NO_TRANSATION_n2m', 'NO_TRANSATION_n1m_n2m',
]


# In[ ]:


tbl = fun_merge_table(tbl_merged, tbl_churn, int(args.observation_month_number), int(args.observation_year))
tbl = tbl.drop(columns=drop_columns)


# In[ ]:


# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)

