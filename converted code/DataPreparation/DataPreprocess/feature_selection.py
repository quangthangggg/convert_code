# %%
# import pandas as pd
import datetime as dt
import time
import argparse
import os
# from azureml.core import Run
import utilities as ut

# %%
# Notebook specific imports
# import lightgbm as lgb
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
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
        feature_importance_threshold, select_percentage_features, random_state
    ):
    '''This function is to select important features based on Gradient Boosted Trees model'''

    # Drop features from drop_features & ignore_features list
    df1 = df.select([col for col in df.columns if col not in drop_features+ignore_features])
    
    # Split the set into feature set and target set
    assembler = VectorAssembler(inputCols=[col for col in df1.columns if col != target_feature], outputCol='features')
    df_assembled = assembler.transform(df1)
    
    # Train model GBTClassifier
    gbt = GBTClassifier(maxIter=100, maxDepth=5, seed=random_state)
    model = gbt.fit(df_assembled)
    
    # Get Feature importances
    feature_importances = model.featureImportances.toArray()
    features = [col for col in df1.columns if col != target_feature]
    feature_imp = [(importance, feature) for importance, feature in zip(feature_importances, features)]
    feature_imp_sorted = sorted(feature_imp, key=lambda x: x[0], reverse=True)
    spark = SparkSession.builder.getOrCreate()
    feature_imp_df = spark.createDataFrame(feature_imp_sorted, ['Value', 'Feature'])

    # Select features using threshold
    selected_features_df = feature_imp_df.filter(F.col('Value') > feature_importance_threshold)
    print(f'Features selected based on threshold {feature_importance_threshold}: {selected_features_df.count()}')

    # Select top features using percentage
    selected_features_df = selected_features_df.limit(int(selected_features_df.count() * select_percentage_features))
    print(f'Features selected based on percentage {select_percentage_features*100}%: {selected_features_df.count()}')
    
    # Print out the top 100 features
    selected_features_df.show(50, False)
    
    # Take all the selected features, ignore features
    selected_features = selected_features_df.select('Feature').rdd.flatMap(lambda x: x).collect()

    # Combine and get unique feature list
    selected_columns_lst = ignore_features + selected_features + [target_feature]
    selected_columns_lst = list(set(selected_columns_lst))
    
    return selected_columns_lst
# %%
# Calling the multicollinearity function
if args.featureSelection_method == 'GBM':
    print('Feature Selection based on Gradient Boosting Algorithm.')
    # Use train set to find columns based on multicollinearity treatment
    selected_columns_lst = fun_feature_selection(
        tbl_input_train, drop_features, ignore_features, target_feature, 
        args.njobs, args.featureSelection_threshold, args.featureSelection_percentage, args.random_state
    )
else:
    print(f'No feature selection method selected or {args.featureSelection_method} method not defined yet. If you are using "lightGBM" method, please use "GBM" instead.')
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


