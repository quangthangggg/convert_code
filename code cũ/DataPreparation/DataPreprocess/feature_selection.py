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
import lightgbm as lgb
import numpy as np

# %%
print('Feature Selection from Merged data.')

# %%
run = Run.get_context()
tbl_input_train = run.input_datasets['train_data'].to_pandas_dataframe()
tbl_input_test = run.input_datasets['test_data'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('feature_selection')
    parser.add_argument('--output_data_path_train', type=str, help='train data table directory')
    parser.add_argument('--output_data_path_test', type=str, help='test data table directory')
    parser.add_argument('--featureSelection_threshold', type=float, help='threshold above which features to be selected')
    parser.add_argument('--featureSelection_percentage', type=float, help='percentage of features to be selected')
    parser.add_argument('--featureSelection_method', type=str, help='methond of the feature selection')
    parser.add_argument('--random_state', type=int, help='random state for consistent outputs')
    #parser.add_argument('--validation_flag', type=str, help='Train-Test data or validation data type')
    parser.add_argument('--njobs', type=int, help='number of parallel threads to be run for applyParallel function')

    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')
run_id = run.id
output_file_name_train = 'featureSelected_data_train'
output_file_name_test = 'featureSelected_data_test'
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
def fun_feature_selection(
        df, drop_features, ignore_features, target_feature, 
        njobs, feature_importance_threshold, select_percentage_features, random_state
    ):
    '''This function is to select important features based on Light Gradient Boosting model'''

    # set index as hased_cif
    df = df.set_index('HASHED_CIF')

    # drop features from drop_features & ignore_features list
    df1 = df.loc[:, ~df.columns.isin(drop_features+ignore_features)]
    
    # Split the set into feature set and target set
    X_cols = df1.columns.tolist()
    X_cols.remove(target_feature)
    X = df1[X_cols].copy()
    Y = df1[target_feature].copy()
    
    # Train model LGBM
    model = lgb.LGBMClassifier(n_estimators=100,learning_rate=0.09,random_state=random_state,n_jobs=njobs)
    # Fit model
    model.fit(X,Y)
    # get Feature importances
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, X.columns), reverse=True), 
                               columns=['Value','Feature'])
    percentiles = np.arange(0, 1, 0.05)
    print(feature_imp['Value'].describe(percentiles=percentiles))
    
    # Select features using threshold
    selected_features_df = feature_imp[feature_imp['Value']>feature_importance_threshold]
    # print(f'Min Value: {feature_imp['Value'].min()}, Max Value: {feature_imp['Value'].max()}')
    print(f'Features selected based on threshold {feature_importance_threshold}: {len(selected_features_df)}')

    # Select top features using percentage
    selected_features_df = selected_features_df[:int(selected_features_df.shape[0]*select_percentage_features)]
    print(f'Features selected based on percentage {select_percentage_features*100}%: {len(selected_features_df)}')
    
    # Print out the top 100 features
    print(selected_features_df.head(50))
    
    # Take all the selected features, ignore features and keep features
    selected_features = selected_features_df.Feature

    # combine and get unique feature list
    print('type', type(selected_features), type(ignore_features), type(keep_features))
    print('Selected features based on feature selection test are:', selected_features)
    selected_columns_lst = list(ignore_features) + list(keep_features) + list(selected_features)
    selected_columns_lst.append(target_feature)
    selected_columns_lst = list(set(selected_columns_lst))
    
    return selected_columns_lst

# %%
# Calling the multicollinearity function
if args.featureSelection_method == 'lightGBM':
    print('Feature Selection based on Light Gradient Boosting Algorithm.')
    # Use train set to find columns based on multicollinearity treatment
    selected_columns_lst = fun_feature_selection(
        tbl_input_train, drop_features, ignore_features, target_feature, 
        args.njobs, args.featureSelection_threshold, args.featureSelection_percentage, args.random_state
    )
else:
    print(f'No feature selection method selected or {args.featureSelection_method} method not defined yet.')
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


