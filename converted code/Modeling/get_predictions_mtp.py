import datetime as dt
import time
import argparse
import os
from azureml.core import Run
import h2o
from pysparkling.ml import H2OAutoML
from ai.h2o.sparkling import H2OContext
from pyspark.sql import SparkSession
# import pandas as pd
import utilities as ut 

# %%
print('Get predictions from the trained model for train & test data sets.')

# %%
run = Run.get_context()
run_id = run.id
print('Run id:', run_id)

# Get the current context's workspace..
spark = SparkSession.builder.getOrCreate()

# Kết nối tới Azure ML Workspace
ws = run.experiment.workspace
print('Workspace:', ws)

tbl_train = spark.read.format('delta').load(run.input_datasets['train_data'])
tbl_test = spark.read.format('delta').load(run.input_datasets['test_data'])
tbl_id_train = spark.read.format('delta').load(run.input_datasets['train_data_id'])
tbl_id_test = spark.read.format('delta').load(run.input_datasets['test_data_id'])

# %%
def parse_args():
    parser = argparse.ArgumentParser('get_model_predictions')
    parser.add_argument('--output_data_path_test', type=str, help='model predictions test data table directory')
    parser.add_argument('--output_data_path_train', type=str, help='model predictions train data table directory')
    parser.add_argument('--experiment_name', type=str, help='experiment name')
    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')

output_file_name_train = 'model_predictions_train'
output_file_name_test = 'model_predictions_test'
output_path_train = f"{args.output_data_path_train}"
output_path_test = f"{args.output_data_path_test}"

# Load experiment
experiment = ws.experiments[args.experiment_name]
print('Experiment Name:', args.experiment_name)

# Lấy tất cả các chạy của thí nghiệm
runs = experiment.get_runs()

# Lấy chạy đầu tiên
current_run = next(runs)

# Khởi tạo H2O Context
h2o_context = H2OContext.getOrCreate()

# Chuyển đổi dữ liệu pandas thành H2OFrame
tbl_train_h2o = h2o_context.asH2OFrame(tbl_train)
tbl_test_h2o = h2o_context.asH2OFrame(tbl_test)
tbl_id_train_h2o = h2o_context.asH2OFrame(tbl_id_train)
tbl_id_test_h2o = h2o_context.asH2OFrame(tbl_id_test)

# Tách cột target từ tập huấn luyện
X_train_h2o = tbl_train_h2o.drop("churn_flag")
y_train_h2o = tbl_train_h2o["churn_flag"]

# Tách cột target từ tập kiểm tra
X_test_h2o = tbl_test_h2o.drop("churn_flag")
y_test_h2o = tbl_test_h2o["churn_flag"]

# Define và huấn luyện mô hình AutoML
aml = H2OAutoML(max_runtime_secs=3600)  # 1 giờ
aml.train(x=X_train_h2o.names, y="churn_flag", training_frame=tbl_train_h2o)

# Dự đoán cho tập huấn luyện
predictions_train_h2o = aml.leader.predict(tbl_train_h2o)
predictions_train_df = h2o_context.asSparkFrame(predictions_train_h2o)

# Lưu dự đoán cho tập huấn luyện
output_tbl_train = tbl_id_train_h2o.asSparkFrame().join(predictions_train_df, on="index", how="inner")

# Dự đoán cho tập kiểm tra
predictions_test_h2o = aml.leader.predict(tbl_test_h2o)
predictions_test_df = h2o_context.asSparkFrame(predictions_test_h2o)

# Lưu dự đoán cho tập kiểm tra
output_tbl_test = tbl_id_test_h2o.asSparkFrame().join(predictions_test_df, on="index", how="inner")


# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_tbl_train)

# Writing the output to file
ut.fun_write_file(output_tbl_train, output_path_train, output_file_name_train, run=run, csv=True)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_tbl_test)

# Writing the output to file
ut.fun_write_file(output_tbl_test, output_path_test, output_file_name_test, run=run, csv=True)

