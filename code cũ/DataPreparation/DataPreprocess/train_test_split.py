# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
from azureml.core import Run
import utilities as ut

# %%
# # Notebook specific imports
from sklearn.model_selection import train_test_split

# %%
print('Spliting the data into Train and Test Set.')

# %%
run = Run.get_context()
tbl_input = run.input_datasets['merged_with_churn_flag_data'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('outlier_treatment')
    parser.add_argument('--output_data_path_train', type=str, help='train data table directory')
    parser.add_argument('--output_data_path_test', type=str, help='test data table directory')
    parser.add_argument('--test_size', type=float, help='percentage of test data size')
    parser.add_argument('--random_state', type=int, help='use everytime same random_state to get same results')

    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')
run_id = run.id
output_file_name_train = 'train_data'
output_file_name_test = 'test_data'
output_path_train = f"{args.output_data_path_train}"
output_path_test = f"{args.output_data_path_test}"

# %%
output_split_train, output_split_test = train_test_split(tbl_input, test_size=args.test_size, random_state=args.random_state)
output_split_train.reset_index(inplace=True, drop=True)
output_split_test.reset_index(inplace=True, drop=True)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_split_train)

# Writing the output to file
ut.fun_write_file(output_split_train, output_path_train, output_file_name_train, run=run, csv=False)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_split_test)

# Writing the output to file
ut.fun_write_file(output_split_test, output_path_test, output_file_name_test, run=run, csv=False)


