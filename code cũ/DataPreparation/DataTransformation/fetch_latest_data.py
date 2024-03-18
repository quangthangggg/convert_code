# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
# from azureml.core import Run
import utilities as ut

# %%
from azureml.core import Datastore, Dataset, Workspace
from azureml.fsspec import AzureMachineLearningFileSystem
from azureml.core.run import Run, _OfflineRun

# %%
print('Sample demographics, account & transaction data.')

# %%
def parse_args():
    parser = argparse.ArgumentParser('sample_data')
    parser.add_argument('--output_data_path_demo', type=str, help='sampled demographics table directory')
    parser.add_argument('--output_data_path_trx', type=str, help='sampled transaction table directory')
    parser.add_argument('--output_data_path_acc', type=str, help='sampled account table directory')
    parser.add_argument('--observation_year', type=int, help='year for which the data needs to be fetched')
    parser.add_argument('--observation_month_number', type=int, help='month for which the data needs to be fetched')
    parser.add_argument('--historical_months', type=int, help='months for which the historical data needs to be fetched')
    parser.add_argument('--input_raw_data_path', type=str, help='path in the blob store for raw files')
    parser.add_argument('--new_data_flag', type=str, help='flag is True => run the pipeline for new data')
    parser.add_argument('--monitoring_flag', type=str, help='flag is True => monitor model performance and fetch next 2 months data')
    parser.add_argument('--latest_data_ndp_flag', type=str, help='flag is True => run the pipeline for latest month new data')
    parser.add_argument('--ws_details', type=str, help='workspace details')

    return parser.parse_args()

args = parse_args()
output_file_name_demo = 'demographic_data'
output_file_name_trx = 'transaction_data'
output_file_name_acc = 'account_data'
output_path_demo = f"{args.output_data_path_demo}"
output_path_trx = f"{args.output_data_path_trx}"
output_path_acc = f"{args.output_data_path_acc}"
print(f'Arguments: {args.__dict__}')

# %%
run = Run.get_context()
ws = Workspace.from_config() if type(run) == _OfflineRun else run.experiment.workspace
blob_data_store = ws.get_default_datastore()

print('type(ws):', type(ws))
print('ws:', ws)

print('type(blob_data_store):', type(blob_data_store))
print('blob_data_store:', blob_data_store)

# %%
input_raw_data_path = args.input_raw_data_path
observation_year = args.observation_year
observation_month_number = args.observation_month_number
historical_months = args.historical_months
new_data_flag = args.new_data_flag
monitoring_flag = args.monitoring_flag
latest_data_ndp_flag = args.latest_data_ndp_flag
ws_details = args.ws_details

# %%
# Run the NDP pipeline for the latest data if latest_data_ndp = 'True'
if latest_data_ndp_flag == "True":
    # Get datastore blob path for input raw files
    datastore_name = eval(str(blob_data_store))['name']
    path_on_datastore = f'{input_raw_data_path}/transaction_data/'

    # ws_details = ws.get_details()['id']
    full_path = f'azureml:/{ws_details}/datastores/{datastore_name}/paths/{path_on_datastore}'
    fs = AzureMachineLearningFileSystem(full_path)
    list_of_folders = fs.ls()
    # only list folders with underscore in the name like 2022_04
    year_month_file_list = [i.split('/')[-2] for i in fs.ls() if '_' in i.split('/')[-2]]
    year_month_file_list = [i for i in year_month_file_list if '_' in i]
    year_month_file_list = [i for i in year_month_file_list if len(i) == 7]
    year_month_file_list = [i for i in year_month_file_list if i.startswith('20')]
    print(f'List of folders in {input_raw_data_path}:', year_month_file_list)

    # Get latest data observation_year & observation_month_number
    date_format = "%Y_%m"
    latest_year_month_file = max([dt.datetime.strptime(year_month, date_format) for year_month in year_month_file_list]).strftime(date_format)
    print(f'Latest folder is:', latest_year_month_file)
    observation_year, observation_month_number = latest_year_month_file.split('_')

# %%
def fun_get_csv_path(dataAssetName, observation_year, observation_month_number, historical_months, 
                     blob_data_store, input_raw_data_path, new_data_flag, monitoring_flag):
    ''' Function to get list of monthly files for transaction data.
        Get file paths for historical months from observation month for historical data calculations.
        Get file paths for next 2 months from observation month for churn flag creation.
        Return: List of all file paths to be read for model training.'''
    # Previous x months of data is needed for historical feature creation (here x = historical_months)
    start_date = pd.Timestamp(f"{observation_year}-{observation_month_number}-01") - pd.DateOffset(months=historical_months-1)
    # Next 2 months of data is required for churn flag creation
    end_date = pd.Timestamp(f"{observation_year}-{observation_month_number}-01") + pd.DateOffset(months=2)
    if new_data_flag == 'True' and monitoring_flag == 'False':
        # For new data we do not need next 2 months of data (next 2 months of data is required for churn flag creation)
        end_date = pd.Timestamp(f"{observation_year}-{observation_month_number}-01")
    elif new_data_flag == 'True' and monitoring_flag == 'True':
        # For new data we do not need next 2 months of data (next 2 months of data is required for churn flag creation)
        end_date = pd.Timestamp(f"{observation_year}-{observation_month_number}-01") + pd.DateOffset(months=2)
    list_of_months = pd.date_range(start_date, end_date, freq='MS').strftime("%Y_%m").tolist()
    print('Reading Data for Year and Months:', list_of_months)
    csv_path = []
    for year_month in list_of_months:
        year, month = year_month.split('_')
        csv_path.append((blob_data_store, f"{input_raw_data_path}/{dataAssetName}/{year}_{month}"))
    return csv_path

# %%
# Read demographics data
csv_path = [(blob_data_store, f"{input_raw_data_path}/demographic_data")]
print('Demographic data csv_path:', csv_path)
demographic_data = Dataset.Tabular.from_delimited_files(path=csv_path)
output_tbl_demo = demographic_data.to_pandas_dataframe()

# %%
# Read account data
csv_path = [(blob_data_store, f"{input_raw_data_path}/account_data")]
print('Account data csv_path:', csv_path)
account_data = Dataset.Tabular.from_delimited_files(path=csv_path)
output_tbl_acc = account_data.to_pandas_dataframe()

# %%
# Read transaction data
csv_path = fun_get_csv_path(
    'transaction_data', observation_year, observation_month_number, historical_months, 
    blob_data_store, input_raw_data_path, new_data_flag, monitoring_flag
)
print('Transaction data csv_path:', csv_path)
transaction_data = Dataset.Tabular.from_delimited_files(path=csv_path)
output_tbl_trx = transaction_data.to_pandas_dataframe()

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_tbl_demo)

# Writing the output to file
ut.fun_write_file(output_tbl_demo, output_path_demo, output_file_name_demo, run=run, csv=False)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_tbl_trx)

# Writing the output to file
ut.fun_write_file(output_tbl_trx, output_path_trx, output_file_name_trx, run=run, csv=False)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_tbl_acc)

# Writing the output to file
ut.fun_write_file(output_tbl_acc, output_path_acc, output_file_name_acc, run=run, csv=False)


