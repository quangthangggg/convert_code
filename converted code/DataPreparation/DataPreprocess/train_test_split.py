# %%
# import pandas as pd
import datetime as dt
import time
import argparse
import os
from pyspark.sql.functions import rand, monotonically_increasing_id
# from azureml.core import Run
import utilities as ut

# %%
# # Notebook specific imports
from sklearn.model_selection import train_test_split

# %%
print('Spliting the data into Train and Test Set.')

# %%
run = Run.get_context()
tbl_input = run.input_datasets['merged_with_churn_flag_data'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('outlier_treatment')
    parser.add_argument('--output_data_path_train', type=str, help='train data table directory')
    parser.add_argument('--output_data_path_test', type=str, help='test data table directory')
    parser.add_argument('--test_size', type=float, help='percentage of test data size')
    parser.add_argument('--random_state', type=int, help='use everytime same random_state to get same results')

    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')
run_id = run.id
output_file_name_train = 'train_data'
output_file_name_test = 'test_data'
output_path_train = f"{args.output_data_path_train}"
output_path_test = f"{args.output_data_path_test}"

# %%

# Thực hiện chia tách dữ liệu thành tập huấn luyện và tập kiểm tra
train_ratio = 1.0 - args.test_size
train_df, test_df = tbl_input.randomSplit([train_ratio, args.test_size], seed=args.random_state)

# Đặt lại chỉ số của tập huấn luyện
train_df = train_df.withColumn("index", monotonically_increasing_id())
output_split_train = train_df.drop("index")

# Đặt lại chỉ số của tập kiểm tra
test_df = test_df.withColumn("index", monotonically_increasing_id())
output_split_test = test_df.drop("index")


# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_split_train)

# Writing the output to file
ut.fun_write_file(output_split_train, output_path_train, output_file_name_train, run=run, csv=False)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_split_test)

# Writing the output to file
ut.fun_write_file(output_split_test, output_path_test, output_file_name_test, run=run, csv=False)


