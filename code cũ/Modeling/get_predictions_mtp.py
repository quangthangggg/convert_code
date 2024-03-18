# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
from azureml.core import Run
import utilities as ut

# %%
from azureml.core import Run, Model, Experiment, Dataset
from azureml.pipeline.core import PipelineRun
from azureml.train.automl.run import AutoMLRun
import joblib
from sklearn.metrics import confusion_matrix

# %%
print('Get predictions from the trained model for train & test data sets.')

# %%
run = Run.get_context()
run_id = run.id
print('Run id:', run_id)

# Get the current context's workspace..
ws = run.experiment.workspace
print('Workspace:', ws)

# %%
tbl_train = run.input_datasets['train_data'].to_pandas_dataframe()
tbl_test = run.input_datasets['test_data'].to_pandas_dataframe()
tbl_id_train = run.input_datasets['train_data_id'].to_pandas_dataframe()
tbl_id_test = run.input_datasets['test_data_id'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('get_model_predictions')
    parser.add_argument('--output_data_path_test', type=str, help='model predictions test data table directory')
    parser.add_argument('--output_data_path_train', type=str, help='model predictions train data table directory')
    # parser.add_argument('--model_name', help='name of the model to be used for prediction')
    # parser.add_argument('--model_path', help='path of the model to be used for prediction')
    parser.add_argument('--experiment_name', type=str, help='experiment name')
    # parser.add_argument('--step_output_column_list', type=str, help='List of columns selected for modeling in pipeline table directory')
    # parser.add_argument('--step_output_column_list_latest', type=str, help='List of columns selected for modeling in common table directory')


    return parser.parse_args()

args = parse_args()
print(f'Arguments: {args.__dict__}')

output_file_name_train = 'model_predictions_train'
output_file_name_test = 'model_predictions_test'
output_path_train = f"{args.output_data_path_train}"
output_path_test = f"{args.output_data_path_test}"

# output_file_name_column_list = 'final_data_column_list'
# output_file_name_column_list_latest = 'final_data_column_list_latest'
# output_path_column_list = f"{args.step_output_column_list}"
# output_path_column_list_latest = f"{args.step_output_column_list_latest}"

# %%
experiment = Experiment(ws, args.experiment_name)
print('Experiment Name:', args.experiment_name)
lst_pipeline_runs = list(experiment.get_runs())
# pipeline_run = PipelineRun(experiment, run_id)

# Get current pipeline run
current_pipeline_run = lst_pipeline_runs[0]
print('Details of current_pipeline_run:', current_pipeline_run)
# Get model trainig step run
automl_step_run = current_pipeline_run.find_step_run('Model_Training_Testing')[0]
print('Details of automl_step_run:', automl_step_run)
# Convert normal run to AutoMLRun object
automl_step_run = AutoMLRun(experiment, automl_step_run.id)
# Fetch model trained from this run
best_run, model = automl_step_run.get_output()

# %%
# # Print out model name and path
# model_name = args.model_name
# model_path = args.model_path
# print(f"model_name : {args.model_name}")
# print(f"model_path: {args.model_path}")

# # ..in order to be able to retrieve a model from the repository..
# model_ws = Model(ws, model_name) #, version=11
# print('Model version is:', model_ws.version)

# # ..which we'll then download locally..
# pickled_model_name = model_ws.download(exist_ok = True)

# # ..and deserialize
# model = joblib.load(pickled_model_name)
# print("Model information: ",model)

# %%
# # Save the list of columns used for modeling - this will be used for new data pipeline
# tbl_output_column_list = pd.DataFrame(tbl_train.columns, columns=['selected_column_names'])
# tbl_output_column_list_latest = pd.DataFrame(tbl_train.columns, columns=['selected_column_names'])

# %%
# Seperate Target from Train Set
y_train = tbl_train['churn_flag']
X_train = tbl_train.drop(columns=['churn_flag'])

# Seperate Target from Test Set
y_test = tbl_test['churn_flag']
X_test = tbl_test.drop(columns=['churn_flag'])

# %%
# Model predict - Train
y_pred_train = model.predict(X_train)
tbl_id_train['churn_flag'] = tbl_train['churn_flag']
tbl_id_train['churn_flag_predicted'] = y_pred_train
prediction_of_probability = model.predict_proba(X_train)
tbl_id_train['0_predicted_proba'] = prediction_of_probability[:,0] 
tbl_id_train['1_predicted_proba'] = prediction_of_probability[:,1] 

# The predictions are stored in the `predictions` output path
# so that AML can find them and pass them to other steps
output_tbl_train = pd.merge(tbl_id_train, X_train, left_index=True, right_index=True, how='inner')

# %%
# Get test run info from autoML
test_run = next(best_run.get_children(type='automl.model_test'))
test_run.wait_for_completion(show_output=False, wait_post_processing=True)

# Get test metrics
test_run_metrics = test_run.get_metrics()
# for name, value in test_run_metrics.items():
#     print(f"{name}: {value}")

# Get test predictions as a Dataset
test_run_details = test_run.get_details()
dataset_id = test_run_details['outputDatasets'][0]['identifier']['savedId']
test_run_predictions = Dataset.get_by_id(ws, dataset_id)
tbl_test = test_run_predictions.to_pandas_dataframe()

# The predictions are stored in the `predictions` output path
# so that AML can find them and pass them to other steps
output_tbl_test = pd.merge(tbl_id_test, tbl_test, left_index=True, right_index=True, how='inner')
output_tbl_test.columns = [i.split('_orig')[0] for i in output_tbl_test.columns]

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

