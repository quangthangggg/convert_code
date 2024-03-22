# %%
# import pandas as pd
import datetime as dt
import time
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
import os
from azureml.core import Run
import utilities as ut

# %%
print('Finalizing the prepared data for New Data Pipeline (ndp).')

# %%
run = Run.get_context()
spark = SparkSession.builder.getOrCreate()
tbl_input_new_data = spark.read.format('delta').load(run.input_datasets['new_data'])
# tbl_input_selected_columns = run.input_datasets['model_column_list'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('data_final_prep_new_data')
    parser.add_argument('--output_data_path_new_data', type=str, help='new data table directory')
    parser.add_argument('--output_data_path_id_new_data', type=str, help='HASHED_CIF new data table directory')
    parser.add_argument('--monitoring_flag', type=str, help='flag is True => monitor model performance and add churn_flag into final data')
    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')
run_id = run.id
output_file_name_new_data = 'final_data_new_data'
output_path_new_data = f"{args.output_data_path_new_data}"

output_file_name_id_new_data = 'final_data_id_new_data'
output_path_id_new_data = f"{args.output_data_path_id_new_data}"

# %%
# Tạo một cột index duy nhất
tbl_output_new_data = tbl_input_new_data.withColumn("index", monotonically_increasing_id())

# Loại bỏ cột index cũ nếu cần
# tbl_output_new_data = tbl_output_new_data.drop("index_old")

# %%
# Storing HASHED_CIF for merging after model prediction
monitoring_flag = args.monitoring_flag

if monitoring_flag == 'True':
    id_output_new_data = tbl_output_new_data.select("HASHED_CIF", "churn_flag")
else:
    id_output_new_data = tbl_output_new_data.select("HASHED_CIF")

# Các đặc trưng cần bị loại bỏ nếu tồn tại trong tập dữ liệu ở bước này
drop_features = ['HASHED_CIF']

# Loại bỏ các đặc trưng đã chọn
tbl_output_new_data = tbl_output_new_data.drop(*drop_features)

# %%
# # use same columns as selected during the model training process
# keep_columns = tbl_input_selected_columns['selected_column_names'].tolist()
# tbl_output_new_data = tbl_output_new_data[keep_columns]
print('-'*50)
print('Subset of columns selected based on training set, will be filtered out as part of Prediction_Results-New_Data Step.')
print('-'*50)

# %%
# For new_data set
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl_output_new_data)

# Writing the output to file
ut.fun_write_file(tbl_output_new_data, output_path_new_data, output_file_name_new_data, run=run, csv=False)

# %%
# For new_data set HASHED_CIF id
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(id_output_new_data)

# Writing the output to file
ut.fun_write_file(id_output_new_data, output_path_id_new_data, output_file_name_id_new_data, run=run, csv=False)


