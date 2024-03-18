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
from joblib import Parallel, delayed
import multiprocessing as mp

# %%
print('Transform the transaction data.')

# %%
run = Run.get_context()
tbl = run.input_datasets['raw_data_transaction'].to_pandas_dataframe()
tbl_translation = run.input_datasets['mapping_data_translation'].to_pandas_dataframe()
tbl_demo = run.input_datasets['transformed_demographics_data'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('transform_transaction')
    parser.add_argument('--output_data_path', type=str, help='transformed transaction table directory')
    parser.add_argument('--njobs', type=str, help='number of parallel threads to be run for applyParallel function')

    return parser.parse_args()

args = parse_args()
run_id = run.id
output_file_name = 'transformed_transaction_data'
output_path = f"{args.output_data_path}"#/{run_id}
print(f'Arguments: {args.__dict__}')

# %%
def fun_keep_vnd_currency(df, cus_id_col, currency_col, currency_value):
    '''This function removes rows which have made transaction in other than VND currency. 
    For Ex: if currency_value is USD then we will remove the data for that customer.'''

    # Select rows with only VND currencies
    tbl = df[df[currency_col]==currency_value]
    # Select the customers which have VND currency transaction
    lst = tbl[cus_id_col].unique()
    # Filter the customers not from list above
    tbl1 = df[~df[cus_id_col].isin(lst)].copy()
    print('Number of customers made transactions in others currencies: ', tbl1[cus_id_col].nunique())
    
    return tbl

# %%
def fun_transaction_rollup_monthly(df):
    ''' Rolling up weekly transaction data to monthly. Summing all the values in aggregation. 
    For column HASHED_AC, we take number of accounts in rollup.'''

    # Columns that need not be rolled up
    col_object = ['YEAR', 'MONTH', 'WEEK', 'HASHED_CIF', 'HASHED_AC', 'CCY']

    # Defining dictionary with column and aggregation
    col_num = [{i:'sum'} for i in df.columns if i not in col_object]
    dict_col_agg = {}
    for d in col_num:
        for k, v in d.items():
            dict_col_agg.setdefault(k,v)
    dict_col_agg['HASHED_AC'] = 'nunique'

    # Aggregating the other columns based on sum of the month
    df_monthly = df.groupby(['YEAR', 'MONTH', 'HASHED_CIF']).agg(dict_col_agg)
    df_monthly.rename(columns={'HASHED_AC':'No_of_Accounts'},inplace=True)
    df_monthly = df_monthly.reset_index()
    print(f'Number of records in Transformed Transactions Table: {df_monthly.shape[0]}')
    print(f'Number of features in Transformed Transactions Table: {df_monthly.shape[1]}')
    return df_monthly

# %%
def fun_impute_missing_cusid(df_cust, args_lst):
    '''Imputing customer id with null data for the months in which they have not made any transactions. 
    This function will take 1 customer data at a time.
    Creating new customer flag, this flag will tell in which month and week the customer cif was created.'''
    
    # Creating flag cif_missing_flag which is 1 when the customer has no data for a particular month
    # If the customer is new then new_customer_flag is 1 and this flag is 0
    df_cust['cif_missing_flag'] = 0
    
    # Get min & max date from transaction data
    min_date = args_lst[0]
    max_date = args_lst[1]        

    # Add the customer dummy data for that HASHED_CIF, YEAR & MONTH,
    # We add all the other columns after these 3 and at the end we add the cif_missing_flag column
    for year in range(min_date.year, max_date.year+1):
        # Get list of months for the year
        if min_date.year == max_date.year:
            year_max_date = max_date
            year_min_date = min_date
        else:
            year_max_date = min(max_date, dt.date(year, 12, 1))
            year_min_date = max(min_date, dt.date(year, 1, 1))
        
        # Get list of months which are missing from data for customer
        lst_dummy_month = [m for m in range(year_min_date.month, 
                                            year_max_date.month) if m not in df_cust['MONTH'].unique()]
        
        # Add the customer dummy data for that HASHED_CIF, YEAR & MONTH,
        # We add all the other columns after these 3 and at the end we add the cif_missing_flag column
        for dummy_month in lst_dummy_month:
            df_cust.loc[len(df_cust)] = [year, 
                                         dummy_month, 
                                         df_cust['HASHED_CIF'].iloc[0],] + [0]*(len(df_cust.columns)-4) + [1]
            
    # Sorting the values based on date
    df_cust = df_cust.sort_values(['YEAR', 'MONTH'])

    # For PRE_CLS_BAL & CCY columns we will populate the previous values in dummy month rows
    df_cust['PRE_CLS_BAL'] = df_cust['PRE_CLS_BAL'].fillna(method='ffill')
    #df_cust['CCY'] = df_cust['CCY'].fillna(method='ffill').fillna(method='bfill')

    # Making cid_not_exist_flag = 1, where the customer does not exist with the bank
    df_cust['cid_not_exist_flag'] = 0
    df_cust['cid_not_exist_flag'] = df_cust['new_customer_flag'].replace(to_replace=0, method='bfill')
    
    # Remove rows where customer did not exist
    df_cust = df_cust[~((df_cust['cid_not_exist_flag']==1) & (df_cust['new_customer_flag']==0))]
    df_cust = df_cust.drop(columns=['cid_not_exist_flag'])
    
    # Filling all other values with 0
    df_cust = df_cust.fillna(0).sort_values(['YEAR', 'MONTH'])

    return df_cust

# %%
# Re-arranging transaction columns
order_trans_cols = ['YEAR', 'MONTH', 'WEEK', 'HASHED_CIF', 'HASHED_AC', 
                    'NO_TRANSATION', 'NO_CREDIT', 'NO_DEBIT', 
                    'NO_ATM', 'NO_SMB', 'NO_FUND_TRANSFER', 'NO_DEPOSIT', 'NO_WITHDRAW', 
                    'NO_INTEREST_PAYMENT', 'NO_FEE_TRANSACTION', 'NO_TRANSATION_AUTO',
                    'CCY',
                    'PRE_CLS_BAL', 'AMT_CREDIT', 'AMT_DEBIT', 
                    'AMOUNT_TRANSACTION', 'AMOUNT_TRANSACTION_AUTO']

# %%
start_time = time.time()

# Convert vietnamese column names or vietnamese values to english
tbl = ut.fun_convert_vnt_to_eng(tbl_translation, tbl)

# Dropping duplicate rows
tbl = ut.fun_drop_duplicates(tbl, 'Transaction Data')

# Re-arranging transaction columns
df_trans = tbl[order_trans_cols]

# Getting column and missing percentage information for columns with missing values
print('Initial Missing Report:')
df_missing = ut.fun_get_missing_perc(df_trans)

# Remove customers with transaction other than VND
df_trans = fun_keep_vnd_currency(df_trans, 'HASHED_CIF', 'CCY', 'VND')

# Getting column and missing percentage information for columns with missing values
print('Before Rollup to monthly Missing Report:')
df_missing = ut.fun_get_missing_perc(df_trans)
print('Unique years in data:', df_trans['YEAR'].unique())
print('Unique months in data:', df_trans['MONTH'].unique())

# Aggregating weekly transaction data to monthly
df_trans_monthly = fun_transaction_rollup_monthly(df_trans)

# Creating date column by taking 1st date of the month
df_trans_monthly['date'] = pd.to_datetime(df_trans_monthly['YEAR'].astype(str) + '_' + df_trans_monthly['MONTH'].astype(str) + '_01', 
                                  format='%Y_%m_%d')
# Get date range in transaction data
min_date = df_trans_monthly['date'].min().date()
max_date = df_trans_monthly['date'].max().date()

# Merge transaction with demographics for creating new customer flag
df_trans_demo_monthly = pd.merge(df_trans_monthly, tbl_demo[['HASHED_CIF', 'CIF_CREATE']], how='left', on = 'HASHED_CIF')

# Creating new customer Flag using cif_create date
df_trans_demo_monthly['new_customer_flag'] = (df_trans_demo_monthly['YEAR'] == df_trans_demo_monthly['CIF_CREATE'].dt.year) \
                            & (df_trans_demo_monthly['MONTH'] == df_trans_demo_monthly['CIF_CREATE'].dt.month)
df_trans_demo_monthly['new_customer_flag'] = df_trans_demo_monthly['new_customer_flag'].astype(int)

# Impute customer data with null if not exist for any month, for new customer it will not impute with dummy data, 
# Also, columns pre_cls_bal will be imputed with previous value
# Parallel function
df_trans_monthly_imputed = ut.applyParallel(
    dfGrouped=df_trans_demo_monthly.groupby('HASHED_CIF'), 
    func=fun_impute_missing_cusid, 
    args_lst=[min_date, max_date],
    njobs = int(args.njobs)
)

# Non-parallel function - only for testing
# df_trans_monthly_imputed = df_trans_monthly_demo.groupby('HASHED_CIF').apply(fun_impute_missing_cusid, 
#                                                                              ([min_date, max_date]))
# df_trans_monthly_imputed.reset_index(drop=True, inplace=True)

# Dropping columns which are not required
df_trans_monthly_imputed.drop(columns=['date', 'CIF_CREATE'], inplace=True)

# Final table output i.e. transaction table monthly information
tbl = df_trans_monthly_imputed

# Dropping duplicate rows
tbl = ut.fun_drop_duplicates(tbl, 'Transaction Data')

end_time = time.time()

print(f'Time to run the code to transform transaction data: {time.strftime("%H:%M:%S", time.gmtime(end_time-start_time))}')

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)


