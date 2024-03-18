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
from sklearn.ensemble import IsolationForest
import numpy as np
from pathlib import Path
import joblib

# %%
print('Outlier Treatment of Merged data.')

# %%
run = Run.get_context()
tbl_input_train = run.input_datasets['train_data'].to_pandas_dataframe()
# tbl_input_test = run.input_datasets['test_data'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('outlier_treatment')
    parser.add_argument('--output_data_path_train', type=str, help='train data table directory')
    #parser.add_argument('--output_data_path_test', type=str, help='test data table directory')
    parser.add_argument('--outliers_threshold', type=float, help='percentage of outliers to be removed')
    parser.add_argument('--outliers_method', type=str, help='methond of the outlier treatment')
    #parser.add_argument('--validation_flag', type=str, help='Train-Test data or validation data type')
    parser.add_argument('--njobs', type=int, help='number of parallel threads to be run for applyParallel function')

    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')
run_id = run.id
output_file_name_train = 'outlier_treated_data_train'
# output_file_name_test = 'output_data_path_test'
output_path_train = f"{args.output_data_path_train}"
# output_path_test = f"{args.output_data_path_test}"

# %%
# Features which will not be included while checking for outliers
ignore_features = [
    'HASHED_CIF', 'YEAR','MONTH','CIF_CREATE',
    'L3M_CCARD_FLAG', 'L3M_LOAN_FLAG', 'L3M_FD_FLAG',
    'L6M_CCARD_FLAG', 'L6M_LOAN_FLAG', 'L6M_FD_FLAG', 
    'churn_flag'
]

# Feature Engineered columns to ignore
ignore_features_feat_eng = [col for col in tbl_input_train.columns if any(s in col for s in ['sum', 'max', 'min', 'mean', 'std'])]
ignore_features = ignore_features + ignore_features_feat_eng

# %%
def fun_outlier_treatment(df, ignore_features, njobs, threshold):

    '''ADD FUNCTION DEFINITION'''
    
    # Model pickle filename
    # file_path = '../pickles/'
    # filename = 'outlier_treatment.pkl'
    # Create directory if not exist
    # Path(file_path).mkdir(parents=True, exist_ok=True)
    # print('Current Directory', os.getcwd())
    # if not (file_path is None):
    #     os.makedirs(file_path, exist_ok=True)
    #     print('Created directory', file_path)

    df1 = df.set_index('HASHED_CIF')
    df2 = df1.loc[:, ~df1.columns.isin(ignore_features)].copy()

    # Define and Fit Model for train set
    random_state = np.random.RandomState(42)
    model=IsolationForest(n_estimators=100,max_samples='auto',
                        contamination=float(threshold),random_state=random_state,
                        n_jobs=njobs)
    model.fit(df2)
    
    # ---
    df2['scores'] = model.decision_function(df2)

    # ---
    anomaly=df2.loc[df2['scores']<0]
    anomaly_index=list(anomaly.index)

    # Subseting the dataframe by removing the anomalies
    df_remove_anomaly = df[~df.HASHED_CIF.isin(anomaly_index)]
    
    return df_remove_anomaly

# %%
# For train set
tbl_output_train = fun_outlier_treatment(
    tbl_input_train, ignore_features, args.njobs, args.outliers_threshold
)  

# %%
# For train set
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl_output_train)

# Writing the output to file
ut.fun_write_file(tbl_output_train, output_path_train, output_file_name_train, run=run, csv=False)


