# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
from azureml.core import Run
import utilities as ut

# %%
print('Transform the account data.')

# %%
run = Run.get_context()
tbl = run.input_datasets['raw_data_account'].to_pandas_dataframe()
tbl_translation = run.input_datasets['mapping_data_translation'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('transform_account')
    parser.add_argument('--output_data_path', type=str, help='transformed account table directory')

    return parser.parse_args()
 
args = parse_args()
run_id = run.id
output_file_name = 'transformed_account_data'
output_path = f"{args.output_data_path}"#/{run_id}
print(f'Arguments: {args.__dict__}')

# %%
def fun_remove_anomaly(df):
    ''' This function removes rows which are anomalies.
     For Ex: DATE_OF_STATUS is older than AC_CREATE means Date of Status is older than Date of Account created,
      then we remove those customers with that condition. 
      Here are the list of anomalies:
        - Anomaly 1: Date of Status is older than Date of Account created
        - Anomaly 2: Single Account with different AC_CREATE dates.
        - Anomaly 3: Single Account mapped to multiple customers
        - Anomaly 4: Removing rows with multiple product id
        - Anomaly 5: Account active date is lastest as compared to account closed date
        - Anomaly 6: Account active date is not present.'''


    # Anomaly 1: Date of Status is older than Date of Account created
    print('Anomaly 1: Date of Status is older than Date of Account created')
    print('-'*50)
    df_ar = df[df['DATE_OF_STATUS'] >= df['AC_CREATE']]
    df_ar = df_ar.sort_values(['HASHED_CIF', 'HASHED_AC', 'DATE_OF_STATUS'])
    # Print the result of how many rows have been removed
    print(f'Number of unique Customers in Account Life Cycle Table: {df_ar["HASHED_CIF"].nunique()}', 
      f', dropped {df["HASHED_CIF"].nunique() - df_ar["HASHED_CIF"].nunique()} customers')
    print(f'Number of unique Accounts in Account Life Cycle Table: {df_ar["HASHED_AC"].nunique()}', 
      f', dropped {df["HASHED_AC"].nunique() - df_ar["HASHED_AC"].nunique()} accounts')
    print(f'Number of records in Account Life Cycle Table: {df_ar.shape[0]}', 
      f', dropped {df.shape[0] - df_ar.shape[0]} records')
    print('-'*70)

    # Anomaly 2: Single Account with different AC_CREATE dates.
    print('Anomaly 2: Single Account with different AC_CREATE dates')
    print('-'*50)
    df_ar1 = df_ar.sort_values(['HASHED_CIF', 'HASHED_AC', 'AC_CREATE'])
    df_ar2 = df_ar1.drop_duplicates(['HASHED_CIF', 'HASHED_AC', 'STATUS_DESCRIPTION', 'STATUS_ID', 'PRODUCT_ID'], keep='last')
    # Print the result of how many rows have been removed
    print('Before drop data anomalies: ',df_ar1.shape)
    rows_before = df_ar1.shape[0]
    df_ar2 = df_ar1.drop_duplicates(['HASHED_CIF', 'HASHED_AC', 'STATUS_DESCRIPTION', 'STATUS_ID', 'PRODUCT_ID'], keep='last')
    rows_after = df_ar2.shape[0]
    print('After drop data anomalies: ',df_ar2.shape)
    print(f'Number of rows dropped: {rows_before - rows_after}')
    print('-'*70)

    # Anomaly 3: Single Account mapped to multiple customers
    print('Anomaly 3: Single Account mapped to multiple customers')
    print('-'*50)
    # Count how many customers have same account
    tbl = df_ar2.groupby('HASHED_AC').agg({'HASHED_CIF':'nunique'}).reset_index()
    # Filter the account has more than 1 customer
    tbl = tbl[tbl['HASHED_CIF'] > 1]
    # Filter list of accounts have more than 1 customer
    tbl2 = df_ar2[df_ar2['HASHED_AC'].isin(tbl['HASHED_AC'])].sort_values(['HASHED_AC','DATE_OF_STATUS'], ascending=False)
    # Remove the list of accounts have more than 1 customer
    df_ar3 = df_ar2[~df_ar2['HASHED_AC'].isin(tbl2['HASHED_AC'])]
    # Remove the list of customers have same account
    df_ar3 = df_ar3[~df_ar3['HASHED_CIF'].isin(tbl2['HASHED_CIF'])]
    print('Before drop data anomalies: ',df_ar2.shape)
    print('After drop data anomalies: ',df_ar3.shape)
    print('-'*50)
    # Print the result
    print(f'Number of rows dropped: {df_ar2.shape[0] - df_ar3.shape[0]}')
    print(f'Number of Customers dropped: {df_ar2["HASHED_CIF"].nunique() - df_ar3["HASHED_CIF"].nunique()}')
    print(f'Number of Accounts dropped: {df_ar2["HASHED_AC"].nunique() - df_ar3["HASHED_AC"].nunique()}')
    print('-'*70)

    # Anomaly 4: Removing rows with multiple product id
    print('Anomaly 4: Removing rows with multiple product id')
    print('-'*50)
    df_ar4 = df_ar3.drop_duplicates(['HASHED_CIF','HASHED_AC', 'STATUS_DESCRIPTION', 'DATE_OF_STATUS'], keep='last')
    print(f'Number of rows dropped: {df_ar3.shape[0] - df_ar4.shape[0]}')
    print(f'Number of Customers dropped: {df_ar3["HASHED_CIF"].nunique() - df_ar4["HASHED_CIF"].nunique()}')
    print(f'Number of Accounts dropped: {df_ar3["HASHED_AC"].nunique() - df_ar4["HASHED_AC"].nunique()}')
    print('-'*70)

    # Anomaly 5: Account active date is lastest as compared to account closed date
    print('Anomaly 5: Account active date is lastest as compared to account closed date')
    print('-'*50)
    # 
    tbl_ar = df_ar4.pivot(index=['HASHED_CIF','HASHED_AC'],columns='STATUS_DESCRIPTION',values='DATE_OF_STATUS').reset_index()
    tbl_ar = tbl_ar.sort_values(['HASHED_CIF', 'Active'])
    tbl_ar = pd.merge(df_ar4[['HASHED_CIF', 'HASHED_AC', 'AC_CREATE']], tbl_ar)
    tbl_ar = tbl_ar.drop_duplicates(keep='last')
    e = tbl_ar[tbl_ar['Active'] > tbl_ar['Closed']]
    print('Number of Unique Customers have Account active date is lastest:', e['HASHED_CIF'].nunique())
    print('Number of Unique Accounts have active date is lastest:', e['HASHED_AC'].nunique())
    print('-'*50)
    cols_4 = ['HASHED_CIF', 'PRODUCT_ID', 'AC_CREATE', 'HASHED_AC']
    df_ar5 = pd.merge(tbl_ar,df_ar4[cols_4],on=['HASHED_CIF','HASHED_AC','AC_CREATE'],how='inner')
    df_ar5.drop_duplicates(['HASHED_CIF','HASHED_AC','AC_CREATE'],inplace=True)
    print(f'Number of rows dropped: {df_ar4.shape[0] - df_ar5.shape[0]}')
    print(f'Number of Customers dropped: {df_ar4["HASHED_CIF"].nunique() - df_ar5["HASHED_CIF"].nunique()}')
    print(f'Number of Accounts dropped: {df_ar4["HASHED_AC"].nunique() - df_ar5["HASHED_AC"].nunique()}')
    print('-'*70)

    # Anomaly 6: Account active date is not present
    print('Anomaly 6: Account active date is not present')
    print('-'*50)
    df_ar6 = df_ar5.copy()
    df_ar6.loc[(df_ar6['Active'].isna()) & (df_ar6['Closed'].notnull()),'Active'] = df_ar6['AC_CREATE']+pd.DateOffset(1)

    # Replace Null Active date with Account create date + 1
    print('Replace Null Active date with Account create date + 1')
    
    abc = df_ar6[df_ar6['Active'].isna()]
    print('Number of Unique Customers where Active date is Null:', abc['HASHED_CIF'].nunique())
    print('Number of Unique Accounts where Active date is Null:', abc['HASHED_AC'].nunique())

    df_ar7 = df_ar6.copy()
    df_ar7.loc[(df_ar7['Active'].isna()),'Active'] = df_ar7['AC_CREATE']+pd.DateOffset(1)
    print(f'Number of rows dropped: {df_ar6.shape[0] - df_ar7.shape[0]}')
    print(f'Number of Customers dropped: {df_ar6["HASHED_CIF"].nunique() - df_ar7["HASHED_CIF"].nunique()}')
    print(f'Number of Accounts dropped: {df_ar6["HASHED_AC"].nunique() - df_ar7["HASHED_AC"].nunique()}')
    print('-'*70)
    
    # Account table final
    df_ar8 = df_ar7
    print(f'Number of unique Customers in Account Table: {df["HASHED_CIF"].nunique()}')
    print(f'Number of unique Accounts in Account Table: {df["HASHED_AC"].nunique()}')
    print(f'Number of unique Customers in Account Table Remaining: {df_ar8["HASHED_CIF"].nunique()}')
    print(f'Number of unique Accounts in Account Table Remaining: {df_ar8["HASHED_AC"].nunique()}')
    print(f'Number of unique Customers in Account Table Removed: {df["HASHED_CIF"].nunique() - df_ar8["HASHED_CIF"].nunique()}')
    print(f'Number of unique Accounts in Account Table Removed: {df["HASHED_AC"].nunique() - df_ar8["HASHED_AC"].nunique()}')
    print(f'Number of rows dropped: {df.shape[0] - df_ar8.shape[0]}')
    print('-'*70)

    # Final Transform table: the account information
    tbl_final = df_ar8.groupby(['HASHED_CIF']).count().reset_index()

    lst = [f'num_of_{i}' for i in tbl_final.columns if i not in ['HASHED_CIF','HASHED_AC']]
    lst = ['HASHED_CIF','HASHED_AC'] + lst
    tbl_final.columns = lst
    tbl_final.rename(columns={'HASHED_AC':'No_of_Acc_Held'},inplace=True)

    return tbl_final

# %%
# Convert vietnamese column names or vietnamese values to english
tbl = ut.fun_convert_vnt_to_eng(tbl_translation, tbl)

# Dropping duplicate rows
tbl = ut.fun_drop_duplicates(tbl, 'Account Life Cycle Data')

# Converting date string columns to datetime format
tbl['DATE_OF_STATUS'] = pd.to_datetime(tbl['DATE_OF_STATUS'], format='%d-%b-%y')
tbl['AC_CREATE'] = pd.to_datetime(tbl['AC_CREATE'], format='%Y-%m-%d')

# Re-arranging account columns
acc_cols = ['HASHED_CIF', 'HASHED_AC', 'DATE_OF_STATUS', 'STATUS_DESCRIPTION', 'AC_CREATE', 'STATUS_ID', 'PRODUCT_ID']
tbl = tbl[acc_cols]
# Final table output i.e. account table information
tbl = fun_remove_anomaly(tbl)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)


