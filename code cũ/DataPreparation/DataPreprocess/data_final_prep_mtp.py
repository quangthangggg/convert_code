# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
from azureml.core import Run
import utilities as ut

# %%
print('Finalizing the prepared data for Model Training Pipeline (mtp).')

# %%
run = Run.get_context()
tbl_input_train = run.input_datasets['train_data'].to_pandas_dataframe()
tbl_input_test = run.input_datasets['test_data'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('data_final_prep')
    parser.add_argument('--output_data_path_train', type=str, help='train data table directory')
    parser.add_argument('--output_data_path_test', type=str, help='test data table directory')
    parser.add_argument('--output_data_path_id_train', type=str, help='HASHED_CIF train data table directory')
    parser.add_argument('--output_data_path_id_test', type=str, help='HASHED_CIF test data table directory')
    # parser.add_argument('--step_output_column_list', type=str, help='List of columns selected for modeling in pipeline table directory')
    # parser.add_argument('--step_output_column_list_latest', type=str, help='List of columns selected for modeling in common table directory')

    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')
run_id = run.id
output_file_name_train = 'final_data_train'
output_file_name_test = 'final_data_test'
output_path_train = f"{args.output_data_path_train}"
output_path_test = f"{args.output_data_path_test}"

output_file_name_id_train = 'final_data_id_train'
output_file_name_id_test = 'final_data_id_test'
output_path_id_train = f"{args.output_data_path_id_train}"
output_path_id_test = f"{args.output_data_path_id_test}"

# output_file_name_column_list = 'final_data_column_list'
# output_file_name_column_list_latest = 'final_data_column_list_latest'
# output_path_column_list = f"{args.step_output_column_list}"
# output_path_column_list_latest = f"{args.step_output_column_list_latest}"

# %%
# Reset index
tbl_output_train = tbl_input_train.reset_index(drop=True)
tbl_output_test = tbl_input_test.reset_index(drop=True)

# %%
# Storing HASHED_CIF for merging after model prediction
id_output_train = tbl_output_train[['HASHED_CIF']]
id_output_test = tbl_output_test[['HASHED_CIF']]

# %%
# Features to be dropped if present from dataset at this step
drop_features = ['HASHED_CIF']

tbl_output_train = tbl_output_train.drop(columns=drop_features,  errors='ignore')
tbl_output_test = tbl_output_test.drop(columns=drop_features,  errors='ignore')

# %%
# tbl_output_column_list = pd.DataFrame(tbl_output_train.columns, columns=['selected_column_names'])
# tbl_output_column_list_latest = pd.DataFrame(tbl_output_train.columns, columns=['selected_column_names'])

# %%
# For train set
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl_output_train)

# Writing the output to file
ut.fun_write_file(tbl_output_train, output_path_train, output_file_name_train, run=run, csv=False)

# %%
# For test set
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl_output_test)

# Writing the output to file
ut.fun_write_file(tbl_output_test, output_path_test, output_file_name_test, run=run, csv=False)

# %%
# For train set HASHED_CIF id
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(id_output_train)

# Writing the output to file
ut.fun_write_file(id_output_train, output_path_id_train, output_file_name_id_train, run=run, csv=False)

# %%
# For test set HASHED_CIF id
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(id_output_test)

# Writing the output to file
ut.fun_write_file(id_output_test, output_path_id_test, output_file_name_id_test, run=run, csv=False)
