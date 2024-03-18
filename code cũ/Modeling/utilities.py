# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
from azureml.core import Run

# %%
# Notebook specific imports
from joblib import Parallel, delayed
import multiprocessing as mp

# %%
def fun_drop_duplicates(df, table_name):
    '''Dropping duplicate rows in the data.'''
    rows_before = df.shape[0]
    df.drop_duplicates(inplace=True,ignore_index=True)
    rows_after = df.shape[0]
    print(f'Number of duplicate rows dropped: {rows_before-rows_after} from table {table_name}')
    print(f'Percentage of duplicate rows dropped: {round(100*(rows_before-rows_after)/rows_before, 2)} % from table {table_name}')
    print('-'*30)
    return df

# %%
def fun_convert_vnt_to_eng(df_translation, df):

    # convert csv to dict
    vn_en_dict = dict(zip(df_translation.VN, df_translation.EN))

    # translates columns if needed
    df.rename(columns=vn_en_dict, inplace=True)
    
    # translates rows if needed
    ## transaction table need no conversion
    if 'AMOUNT_TRANSACTION' not in df.columns:
        df.replace(vn_en_dict, inplace=True)
        print('Conversion Complete')
        
    return df

# %%
def fun_get_missing_perc(df):
    '''Function provides information on the columns with missing data.'''
    df_missing = df.isna().sum().reset_index()
    df_missing = df_missing[df_missing[0] > 0]#.drop(columns=['level_0'])
    df_missing['percentage_missing'] = round(100*df_missing[0]/df.shape[0], 2)
    df_missing.columns = ['column_name', 'missing_value', 'percentage_missing']
    print('Percentage of missing data in Columns: ')
    print(df_missing)
    return df_missing

# %%
def applyParallel(dfGrouped, func, args_lst=[], njobs=mp.cpu_count()-1):
    ''' Run the grouped data in parallel. '''
    retLst = Parallel(n_jobs = njobs, verbose = 2)(delayed(func)(group, args_lst) for name, group in dfGrouped)
    return pd.concat(retLst)

# %%
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

# %%
def fun_get_flag_distribution(tbl, column_name, output_file):
    '''Get value counts of flag with percentage.'''
    df1 = tbl[column_name].value_counts().rename_axis(column_name).reset_index(name='actualCount')
    df2 = tbl[column_name].value_counts(normalize=True).mul(100).round(1)\
            .rename_axis(column_name).reset_index(name='percentage')
    df3 = pd.merge(df1, df2, on=column_name)
    df3['percentage'] = df3['percentage'].astype(str) + ' %'
    df3['abbrCount'] = df3['actualCount'].apply(lambda x: human_format(x))
    df3['file_name'] = output_file
    df3 = df3[['file_name', column_name, 'abbrCount', 'percentage', 'actualCount']]
    return df3

# %%
def fun_write_file(tbl, output_path, output_file, run, csv=False):
    print('-'*30)
    print(f'Number of rows and columns in the file {output_file} are: {tbl.shape[0]} & {tbl.shape[1]}')
 
    # for logging as metric
    tbl_col_names = ', '.join(tbl.columns)
    columns = ['file_name', 'TotalColumns', 'TotalRowsAbbr', 'TotalRows', 'ColumnNames']
    list_of_values = [output_file, tbl.shape[1], human_format(tbl.shape[0]), tbl.shape[0], tbl_col_names]
    df_info = pd.DataFrame([list_of_values], columns=columns)

    # run.log('TotalRowsAbbr', human_format(tbl.shape[0]))
    # run.log('TotalRows', tbl.shape[0])
    # run.log('TotalColumns', tbl.shape[1])
    # cols = ', '.join(tbl.columns)
    # run.log('Columns', cols)
    print('Columns in data:', tbl.columns)
    if 'HASHED_CIF' in tbl.columns:
        print(f'Number of unique customer ids in the file are: {tbl["HASHED_CIF"].nunique()}')
        # run.log('NumofCustomersAbbr', human_format(tbl["HASHED_CIF"].nunique()))
        # run.log('NumofCustomers', tbl["HASHED_CIF"].nunique())
        df_info['NumofCustomersAbbr'] = human_format(tbl["HASHED_CIF"].nunique())
        df_info['NumofCustomers'] = tbl["HASHED_CIF"].nunique()
    run.log_table('Basic_Data_Information', df_info.to_dict('list'))

    if 'churn_flag' in tbl.columns:
        print(f'Target distribution in file: {tbl["churn_flag"].value_counts()}')
        df_flag = fun_get_flag_distribution(tbl, 'churn_flag', output_file)
        run.log_table("Churn_Flag_Distribution", df_flag.to_dict('list'))
    if 'churn_flag_predicted' in tbl.columns:
        print(f'Target distribution in file: {tbl["churn_flag_predicted"].value_counts()}')
        df_flag = fun_get_flag_distribution(tbl, 'churn_flag_predicted', output_file)
        run.log_table("Predicted_Churn_Flag_Distribution", df_flag.to_dict('list'))
    if not (output_path is None):
        os.makedirs(output_path, exist_ok=True)
        print("File %s created" % output_path)
        if csv:
            output_path = f'{output_path}/{output_file}.csv' #+ "/processed.parquet"
            write_df = tbl.to_csv(output_path, index=False)
        else: 
            output_path = f'{output_path}/{output_file}.parquet' #+ "/processed.parquet"
            write_df = tbl.to_parquet(output_path)
    else:
        print("-"*50)
        print("File %s already created" % output_path)
        print("-"*50)


