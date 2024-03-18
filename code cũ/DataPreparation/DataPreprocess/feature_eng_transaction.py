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
from tqdm.notebook import tqdm
tqdm.pandas()
import calendar

# %%
print('Creating Feature Engineering columns from transaction data.')

# %%
run = Run.get_context()
tbl = run.input_datasets['transformed_transaction_data'].to_pandas_dataframe()

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
def fun_feature_eng(df, month_number, observation_year, historical_months=6):
    # Mapping of month names to their corresponding numbers
    # month_map = {
    #     'January': 1, 'February': 2, 'March': 3, 'April': 4,
    #     'May': 5, 'June': 6, 'July': 7, 'August': 8,
    #     'September': 9, 'October': 10, 'November': 11, 'December': 12
    # }
    # # Get the numeric representation of the input month
    # month_number = month_map[month_name] 
    
    # For loop to go from last 1 month to last 6 months hence range(1, 7)
    for i in tqdm(range(1, historical_months+1), 
                  desc=f'Creating features for month: {month_number}, for year: {observation_year}, for last {historical_months} months.'):
        
        start_date = pd.Timestamp(f"{observation_year}-{month_number}-01") - pd.DateOffset(months=i-1)
        end_date = pd.Timestamp(f"{observation_year}-{month_number}-01")
        df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH']].assign(day=1))
        list_of_months = pd.date_range(start_date, end_date, freq='MS').strftime("%Y_%m").tolist()
        print(f'Reading Data for Year and Months last {i} months:', list_of_months)
        # Filter the df to get just the data for the last i month in the same year
        # filtered_df = df[(df['MONTH'] > month_number - i) & (df['MONTH'] <= month_number) & (df['YEAR'] == observation_year)]
        filtered_df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
        
        # Filter based on HASHED_CIF to get data per customer, then perform calculations such as sum, mean, max etc
        result = filtered_df.groupby(['HASHED_CIF']).agg(dict_feature_eng)
        result.fillna(0, inplace=True)

        # Renaming the columns in the format colname_lxm_aggfunc eg: NO_TRANSATION_l3m_sum
        result.columns = [f'{col[0]}_l{i}m_{col[1]}' for col in result.columns.values]
        
        # Reset index and sort based on HASHED_CIF
        result = result.reset_index().sort_values('HASHED_CIF')
        
        # On the first iteration of the for loop the empty dataframe is assigned to result
        if i == 1:
            final_result = result
        # On all other iterations result is merged with final_result on HASHED_CIF
        else:
            final_result = pd.merge(final_result, result, on=['HASHED_CIF'], how='outer')

    # # Creating Observation_month column to represent month that data is relative to
    # final_result.loc[:,'observation_month'] = calendar.month_abbr[month_number]
    
    # # Making Month column the first column of the df
    # desired_order = ['observation_month'] + [col for col in final_result.columns if col != 'observation_month']
    # final_result = final_result.reindex(columns=desired_order)
    
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


