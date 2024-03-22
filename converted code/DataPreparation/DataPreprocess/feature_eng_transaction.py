# %%
# import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, month, year
import datetime as dt
import time
import argparse
import os
from azureml.core import Run
import utilities as ut

# %%
# Notebook specific imports
from tqdm.notebook import tqdm
tqdm.pandas()
import calendar

# %%
print('Creating Feature Engineering columns from transaction data.')

# %%
run = Run.get_context()
spark = SparkSession.builder.getOrCreate()
tbl = spark.read.format('delta').load(run.input_datasets['transformed_transaction_data'])

# %%
def parse_args():
    parser = argparse.ArgumentParser('feature_eng_transaction')
    parser.add_argument('--output_data_path', type=str, help='Feature Engineered transaction table directory')
    parser.add_argument("--observation_month_number", type=str, help="historical features to be created for the observation month")
    parser.add_argument("--observation_year", type=str, help="historical features to be created for the observation month & year")
    parser.add_argument("--historical_months", type=str, help="number of historical months to be considered for historical features")

    return parser.parse_args()
 
args = parse_args()
run_id = run.id
output_file_name = 'feature_eng_transaction_data'
output_path = f"{args.output_data_path}"#/{run_id}
print(f'Arguments: {args.__dict__}')

# %%
# Define dictionary with column:aggregation pairs
# Define what feature engineering needs to be done on which column
dict_feature_eng =  {
    'NO_TRANSATION': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_TRANSATION_AUTO': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_DEPOSIT': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_ATM': ['sum', 'max', 'min', 'mean', 'std'],
    'AMOUNT_TRANSACTION': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_INTEREST_PAYMENT': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_CREDIT': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_DEBIT': ['sum', 'max', 'min', 'mean', 'std'],
    'No_of_Accounts': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_WITHDRAW': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_FEE_TRANSACTION': ['sum', 'max', 'min', 'mean', 'std'],
    'PRE_CLS_BAL': ['sum', 'max', 'min', 'mean', 'std'],
    'AMT_DEBIT': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_SMB': ['sum', 'max', 'min', 'mean', 'std'],
    'AMT_CREDIT': ['sum', 'max', 'min', 'mean', 'std'],
    'AMOUNT_TRANSACTION_AUTO': ['sum', 'max', 'min', 'mean', 'std'],
    'NO_FUND_TRANSFER': ['sum', 'max', 'min', 'mean', 'std']
}

# %%
def fun_feature_eng(df: DataFrame, month_number: int, observation_year: int, historical_months: int = 6) -> DataFrame:
    # For loop to go from last 1 month to last 6 months hence range(1, 7)
    for i in range(1, historical_months + 1):
        start_date = f"{observation_year}-{month_number:02d}-01" - pd.DateOffset(months=i-1)
        end_date = f"{observation_year}-{month_number:02d}-01"
        
        # Filter the df to get just the data for the last i month in the same year
        filtered_df = df.filter((col('DATE') >= start_date) & (col('DATE') <= end_date))
        
        # Perform necessary aggregations
        result = filtered_df.groupBy('HASHED_CIF').agg(dict_feature_eng)
        
        # Renaming the columns
        result = result.toDF(*[f'{col[0]}_l{i}m_{col[1]}' for col in result.columns])
        
        # On the first iteration of the for loop the empty dataframe is assigned to result
        if i == 1:
            final_result = result
        # On all other iterations result is merged with final_result on HASHED_CIF
        else:
            final_result = final_result.join(result, 'HASHED_CIF', 'outer')

    return final_result

# %%
# Columns on which feature engineering will be done
base_cols = ['YEAR', 'MONTH', 'HASHED_CIF', 'No_of_Accounts',
          'NO_TRANSATION', 'NO_CREDIT', 'NO_DEBIT', 
          'NO_ATM', 'NO_SMB', 'NO_FUND_TRANSFER', 'NO_DEPOSIT', 'NO_WITHDRAW', 
          'NO_INTEREST_PAYMENT', 'NO_FEE_TRANSACTION', 'NO_TRANSATION_AUTO',
          'PRE_CLS_BAL', 'AMT_CREDIT', 'AMT_DEBIT', 
          'AMOUNT_TRANSACTION', 'AMOUNT_TRANSACTION_AUTO']

# %%
df = tbl[base_cols]
print(f'Number of unique Customers in Merged Table: {df["HASHED_CIF"].nunique()}')
print(f'Number of records in Merged Table: {df.shape}')

df_feat_eng = fun_feature_eng(df, int(args.observation_month_number), int(args.observation_year), int(args.historical_months))
tbl = df_feat_eng

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)


