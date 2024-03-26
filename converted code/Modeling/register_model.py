# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
from azureml.core import Run
import utilities as ut
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import json

# %%
from azureml.core import Experiment, Dataset, Workspace
from azureml.train.automl.run import AutoMLRun
from azureml.core.model import Model, Dataset
from azureml.core.run import Run, _OfflineRun
import json

# %%
print('Select best model and Register the trained model based on train & test data sets.')

# %%
run = Run.get_context()
run_id = run.id
print('Run id:', run_id)

# Get the current context's workspace..
ws = Workspace.from_config() if type(run) == _OfflineRun else run.experiment.workspace

# # Get Datastore
blob_data_store = ws.get_default_datastore()
datastore_name = eval(str(blob_data_store))['name']

# %%
def parse_args():
    parser = argparse.ArgumentParser('register_model')
    parser.add_argument('--output_data_path_test', type=str, help='model predictions test data table directory')
    parser.add_argument('--output_data_path_train', type=str, help='model predictions train data table directory')
    parser.add_argument("--model_name", required=True, help='best model as per azure automl')
    parser.add_argument("--model_path", required=True, help='best model path as per azure automl')
    parser.add_argument("--model_tags", required=True, help='best model tags')
    parser.add_argument("--observation_year", required=True, help='observation_year for which model is trained')
    parser.add_argument("--observation_month_number", required=True, help='observation_month for which model is trained')
    parser.add_argument("--historical_months", required=True, help='historical_months for which model is trained')
    parser.add_argument("--latest_data_ndp_flag",required=True, help='latest_data_ndp_flag false for this step as it is part of mtp')
    parser.add_argument("--user_selected_metric",required=True, help='user_selected_metric to select best model')
    parser.add_argument('--experiment_name', type=str, help='experiment name')
    parser.add_argument('--ws_details', type=str, help='workspace detail')

    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')
print('model_path type:', type(args.model_path))
print('Get current working directory:', os.getcwd())

user_selected_metric = args.user_selected_metric
ws_details = args.ws_details

# %%
tbl_train = run.input_datasets['train_data'].to_pandas_dataframe()
tbl_train_columns = ';'.join(list(tbl_train.columns))
model_properties = {
    'train_columns' : tbl_train_columns
}

model_tags = eval(args.model_tags)
model_tags['observation_year'] = args.observation_year
model_tags['observation_month_number'] = args.observation_month_number
model_tags['model_name'] = args.model_name
model_tags['historical_months'] = args.historical_months
model_tags['latest_data_ndp_flag'] = args.latest_data_ndp_flag
print('Model Tags:', model_tags)

# %%
def fun_get_model_metrics(current_pipeline_run, best_run):
    # Khởi tạo SparkSession
    spark = SparkSession.builder \
        .appName("Get Model Metrics") \
        .getOrCreate()
    
    # Download metrics_output từ current_pipeline_run
    metrics_output_port = current_pipeline_run.get_pipeline_output('metrics_output')
    metrics_output_port.download('.', show_progress=True)

    # Đọc nội dung của tập tin metrics_output
    metrics_filename = metrics_output_port._path_on_datastore
    with open(metrics_filename, 'r') as f:
        metrics_output_result = f.read()
    
    # Deserialize metrics_output từ JSON
    deserialized_metrics_output = json.loads(metrics_output_result)
    
    # Tạo DataFrame từ deserialized_metrics_output
    df = spark.createDataFrame(deserialized_metrics_output)
    
    # Chọn cột best_run.id từ DataFrame
    df_best_run = df.select(col(best_run.id).alias('metrics'))
    
    # Chuyển DataFrame thành dictionary
    dict_metrics = df_best_run.rdd.map(lambda row: row.asDict()).collect()
    
    return dict_metrics

# %%
# Get current experiment
experiment = Experiment(ws, args.experiment_name)
print('Experiment Name:', args.experiment_name)
lst_pipeline_runs = list(experiment.get_runs())

# Get current pipeline run
# current_pipeline_run_id = 'd8d0a0fc-1d29-4ee8-b692-509713c82267'
# from azureml.pipeline.core import PipelineRun
# current_pipeline_run = PipelineRun(experiment, current_pipeline_run_id)

current_pipeline_run = lst_pipeline_runs[0]
print('Details of current_pipeline_run:', current_pipeline_run)
# Get model trainig step run
automl_step_run = current_pipeline_run.find_step_run('Model_Training_Testing')[0]
print('Details of automl_step_run:', automl_step_run)
# Convert normal run to AutoMLRun object
automl_step_run = AutoMLRun(experiment, automl_step_run.id)

# %%
# Get best model from this run as per primary metric
best_run, model = automl_step_run.get_output()
model_path_on_datastore = best_run.properties['model_data_location'].split('artifact/')[1].split('/model.pkl')[0]
model_path = f'azureml:/{ws_details}/datastores/{datastore_name}/paths/{model_path_on_datastore}'
dict_metrics = fun_get_model_metrics(current_pipeline_run, best_run)
model_properties['best_model_as_per_primary_metric'] = dict_metrics
model_properties['best_model_as_per_primary_metric']['run_id'] = best_run.id
model_properties['best_model_as_per_primary_metric']['model_run_id'] = best_run.id
model_properties['best_model_as_per_primary_metric']['model_path'] = model_path
print(f'Model path for Model selected based on primary_metric:', model_path)

# %%
# Register model based on either primary_metric or user_selected_metric
# If user_selected_metric is not null then we select the best model based on value of user_selected_metric.
# If user_selected_metric is '' then best model is registered as per selection from automlstep based on primary metric.

if user_selected_metric != '':
    # Register best model from this run based on user_selected_metric
    best_run, model = automl_step_run.get_output(metric=user_selected_metric)
    model_path_on_datastore = best_run.properties['model_data_location'].split('artifact/')[1].split('/model.pkl')[0]
    model_path = f'azureml:/{ws_details}/datastores/{datastore_name}/paths/{model_path_on_datastore}'
    dict_metrics = fun_get_model_metrics(current_pipeline_run, best_run)
    model_properties['best_model_as_per_user_selected_metric'] = dict_metrics
    model_properties['best_model_as_per_user_selected_metric']['run_id'] = best_run.id
    model_properties['best_model_as_per_user_selected_metric']['model_path'] = model_path
    print(f'Model path for Model selected based on user_selected_metric {user_selected_metric}:', model_path)
    
    model = best_run.register_model(
        model_name=args.model_name,
        model_path='outputs/model.pkl',
        properties=model_properties,
        tags=model_tags,
    )
else:
    # Register best model from this run based on primary_metric
    model_path=args.model_path
    print('Model path for Model selected based on primary_metric:', model_path)

    model = Model.register(
        workspace=ws,
        model_path=args.model_path,
        model_name=args.model_name,
        properties=model_properties,
        tags=model_tags,
    )

# %%

print('Model Path - best model:', args.model_path)
print('Model Path - model to be used:', model_path)

# %%
run.log("Trained_Model_Name", model.name)
run.log("Trained_Model_Version", model.version)
run.log("Trained_Model_Columns", tbl_train_columns)
run.log("user_selected_metric", user_selected_metric)

print("Registered version {0} of model {1}".format(model.version, model.name))


