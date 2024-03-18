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
import numpy as np
from sklearn.preprocessing import StandardScaler

# %%
print('Normalization of Merged data.')

# %%
run = Run.get_context()
tbl_input_train = run.input_datasets['train_data'].to_pandas_dataframe()
tbl_input_test = run.input_datasets['test_data'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('data_normalization')
    parser.add_argument('--output_data_path_train', type=str, help='train data table directory')
    parser.add_argument('--output_data_path_test', type=str, help='test data table directory')
    parser.add_argument('--normalization_method', type=str, help='methond of the data normalization')
    #parser.add_argument('--validation_flag', type=str, help='Train-Test data or validation data type')

    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')
run_id = run.id
output_file_name_train = 'normalized_data_train'
output_file_name_test = 'normalized_data_test'
output_path_train = f"{args.output_data_path_train}"
output_path_test = f"{args.output_data_path_test}"

# %%
# Features to be dropped if present from dataset at this step
drop_features = []

# Features which will not be included in this step if present 
ignore_features = ['HASHED_CIF']

# Features to keep isrrespective of the selection score - as per business these are important to be kept
keep_features = [
    'NO_CREDIT', 'NO_DEBIT', 'NO_TRANSATION_AUTO', 'PRE_CLS_BAL',
    'AMT_CREDIT', 'AMT_DEBIT', 'AMOUNT_TRANSACTION', 'AMOUNT_TRANSACTION_AUTO'
]

target_feature = 'churn_flag'

# %%
def fun_normalization_data(tbl, ignore_features, drop_features, target_feature, scale=False, is_train=True):
    
    # Set index as hashed_cif
    df = tbl.set_index('HASHED_CIF')
    
    # Remove columns
    df = df.loc[:, ~df.columns.isin(ignore_features+drop_features)]

    # Split into X, Y
    y_df = df[target_feature]
    X_df = df.drop([target_feature], axis=1)

    if is_train:
        # transform the training data column
        # define scaling method object
        scale = StandardScaler()
        X_df_stand = scale.fit_transform(X_df)
        scaled_df = pd.DataFrame(X_df_stand, columns=X_df.columns, index=X_df.index)
    else:
        # transform the testing data column
        X_df_stand = scale.transform(X_df)
        scaled_df = pd.DataFrame(X_df_stand, columns=X_df.columns, index=X_df.index)
    
    # Add the column churn flag
    final_scaled_df = pd.merge(scaled_df, y_df.reset_index(), on='HASHED_CIF', how='inner')
    # Reset index HASHED_CIF
    #final_scaled_df = final_scaled_df.reset_index()
    
    return final_scaled_df, scale

# %%
if args.normalization_method == 'standard':
    print('Normalization based on Standard Scalar.')
    # We train the normalization model on training set and then pass that model to test set
    normalize_tbl_train, model = fun_normalization_data(
        tbl_input_train, ignore_features, drop_features, target_feature, scale=False, is_train=True
    )
    normalize_tbl_test, model = fun_normalization_data(
        tbl_input_test, ignore_features, drop_features, target_feature, scale=model, is_train=False
    )
else:
    print(f'No normalization method selected or {args.normalization_method} method not defined yet.')
    normalize_tbl_train = tbl_input_train
    normalize_tbl_test = tbl_input_test

# Select the features based on the test
tbl_output_train = normalize_tbl_train
tbl_output_test = normalize_tbl_test

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


