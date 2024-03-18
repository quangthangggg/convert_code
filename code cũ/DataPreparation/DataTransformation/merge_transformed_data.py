# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
from azureml.core import Run
import utilities as ut

# %%
# Notebook specific imports
import re

# %%
print('Merge Transaction Demographics Account Feature Engineered data.')

# %%
run = Run.get_context()
tbl_feat_eng = run.input_datasets['feature_eng_transaction_data'].to_pandas_dataframe()
tbl_transaction= run.input_datasets['transformed_transaction_data'].to_pandas_dataframe()
tbl_demographics = run.input_datasets['transformed_demographics_data'].to_pandas_dataframe()
tbl_account = run.input_datasets['transformed_account_data'].to_pandas_dataframe()

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
    df_merge1 = pd.merge(tbl_trans, tbl_acc, on = ['HASHED_CIF'], suffixes=[None, '_x'], how='left')
    
    remove_lst = df_merge1[df_merge1['No_of_Acc_Held'].isna()].HASHED_CIF.unique()
    # Remove the customers where information from account is null
    df_merge2 = df_merge1[~df_merge1['HASHED_CIF'].isin(remove_lst)]

    ## Merging with Demographics
    df_merge3 = pd.merge(df_merge2, tbl_demo, on = ['HASHED_CIF'],suffixes=[None, '_x'], how='left')
    df_merge3.drop(columns=['AVG_INCOME'],inplace=True)

    # Merging only with observation month data with Feature Engineered Columns
    df_merge3 = df_merge3[(df_merge3['YEAR']==observation_year) & (df_merge3['MONTH']==observation_month_number)]
    df_merge4 = pd.merge(df_merge3, tbl_feat_eng, on = ['HASHED_CIF'], suffixes=[None, '_x'], how='left')

    return df_merge4

# %%
def fun_replace_special_chars(col_name):
    '''Replace special characters in a string with underscore.'''
    col_name = re.sub('[^a-zA-Z0-9 \n\.]', '_', col_name)
    return col_name

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

tbl = tbl.drop(columns=drop_columns)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)


