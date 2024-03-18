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
from tqdm import tqdm as tqdm

# %%
print('Multicollinearity Treatment of Merged data.')

# %%
run = Run.get_context()
tbl_input_train = run.input_datasets['train_data'].to_pandas_dataframe()
tbl_input_test = run.input_datasets['test_data'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('multicollinearity_treatment')
    parser.add_argument('--output_data_path_train', type=str, help='train data table directory')
    parser.add_argument('--output_data_path_test', type=str, help='test data table directory')
    parser.add_argument('--multiCollinearity_threshold', type=float, help='percentage of outliers to be removed')
    parser.add_argument('--multiCollinearity_method', type=str, help='methond of the multicollinearity treatment')
    #parser.add_argument('--validation_flag', type=str, help='Train-Test data or validation data type')

    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')
run_id = run.id
output_file_name_train = 'multicollinearity_treated_data_train'
output_file_name_test = 'multicollinearity_treated_data_test'
output_path_train = f"{args.output_data_path_train}"
output_path_test = f"{args.output_data_path_test}"

# %%
# Features to be dropped if present from dataset at this step
drop_features = [
    'YEAR','MONTH','CIF_CREATE'
]

# Features which will not be included in this step if present 
ignore_features = ['HASHED_CIF']

# Features to keep isrrespective of the selection score - as per business these are important to be kept
keep_features = [
    'NO_CREDIT', 'NO_DEBIT', 'NO_TRANSATION_AUTO', 'PRE_CLS_BAL',
    'AMT_CREDIT', 'AMT_DEBIT', 'AMOUNT_TRANSACTION', 'AMOUNT_TRANSACTION_AUTO'
]

target_feature = 'churn_flag'

# %%
def fun_multicollinearity_treatment(df, drop_features, ignore_features, target_feature, corr_value):
    '''Anh: Add function definition'''

    # set index as hased_cif
    df = df.set_index('HASHED_CIF')

    # drop features from drop_features & ignore_features list
    df1 = df.loc[:, ~df.columns.isin(drop_features+ignore_features)]
    
    # Split the set into feature set and target set
    X_cols = df1.columns.tolist()
    X_cols.remove(target_feature)
    X = df1[X_cols].copy()
    Y = df1[target_feature].copy()
    
    t0 = time.time()
    # Generating the correlation matrix
    corr = X.corr()    

    # Select upper triangle of correlation matrix
    corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    data = []
    # We compare the correlation between features and remove one of two features that have a correlation higher than corr_value
    # Creating list of boolean corresponding to the column list - False means column not selected, True means column is selected
    columns = np.full((corr.shape[0],), True, dtype=bool)
    for i in tqdm(range(corr.shape[0])):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= corr_value:
                if columns[j]:
                    # calculating features correlation with the target
                    col1_corr_with_target = X[X.columns[i]].corr(Y)
                    col2_corr_with_target = X[X.columns[j]].corr(Y)
                    data.append((X.columns[i], X.columns[j],
                             corr.iloc[i,j], col1_corr_with_target, col2_corr_with_target))
                    # Selecting one column from 2 correlated columns based on there correlation with target variable
                    if col1_corr_with_target > col2_corr_with_target:
                        columns[j] = False
                    else: 
                        columns[i] = False
    # Passing the boolean list to filter out the columns
    selected_columns = X.columns[columns]
    print('Number of columns selected based on correlation value >= ', corr_value, ' are: ', 
          len(selected_columns), ' out of: ', len(X.columns))

    t1 = time.time()
    total = t1-t0
    print('Time to run the code: ', total)

    # combine and get unique feature list
    print('type', type(selected_columns), type(ignore_features), type(keep_features))
    print('Selected features based on multicollinearity test are:', selected_columns)
    selected_columns_lst = list(ignore_features) + list(keep_features) + list(selected_columns)
    selected_columns_lst.append(target_feature)
    selected_columns_lst = list(set(selected_columns_lst))
    
    return selected_columns_lst

# %%
# Calling the multicollinearity function
if args.multiCollinearity_method == 'pearson':
    print('Multicollinearity Test based on Pearson Correlation.')
    # Use train set to find columns based on multicollinearity treatment
    selected_columns_lst = fun_multicollinearity_treatment(
        tbl_input_train, drop_features, ignore_features, target_feature, args.multiCollinearity_threshold
    )
else:
    print(f'No multicollinearity test method selected or {args.multiCollinearity_method} method not defined yet.')
    selected_columns_lst = tbl_input_train.columns

# Select the features based on the test
tbl_output_train = tbl_input_train[selected_columns_lst]
tbl_output_test = tbl_input_test[selected_columns_lst]

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


