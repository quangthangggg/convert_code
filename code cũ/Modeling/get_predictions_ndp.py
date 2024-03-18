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
print('Get predictions from the trained model for New Data set.')

# %%
run = Run.get_context()
run_id = run.id
print('Run id:', run_id)

# Get the current context's workspace..
ws = run.experiment.workspace
print('Workspace:', ws)

# %%
tbl_new_data = run.input_datasets['new_data'].to_pandas_dataframe()
tbl_id_new_data = run.input_datasets['new_data_id'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('get_model_predictions')
    parser.add_argument('--output_data_path_new_data', type=str, help='model predictions new data table directory')
    parser.add_argument('--model_name', help='name of the model to be used for prediction')
    # parser.add_argument('--model_path', help='path of the model to be used for prediction')
    parser.add_argument('--experiment_name', type=str, help='experiment name')
    parser.add_argument('--model_version', type=int, help='trained model_version')

    return parser.parse_args()

args = parse_args()
output_file_name_new_data = 'model_predictions_new_data'
output_path_new_data = f"{args.output_data_path_new_data}"
print(f'Arguments: {args.__dict__}')

# %%
# Print out model name and path
model_name = args.model_name
# model_path = args.model_path
print(f"model_name : {args.model_name}")
# print(f"model_path: {args.model_path}")

# ..in order to be able to retrieve a model from the repository..
if args.model_version == -1:
    model_ws = Model(ws, model_name)
else:
    model_ws = Model(ws, model_name, version=args.model_version) #, version=11

run.log('ModelVersion', model_ws.version)
run.log('ModelName', model_name)

# ..which we'll then download locally..
pickled_model_name = model_ws.download(exist_ok = True)

# ..and deserialize
model = joblib.load(pickled_model_name)
print("Model information: ",model)

# %%
# Filter columns as per train set
print('model_ws.properties:', model_ws.properties)
tbl_train_columns = model_ws.properties['train_columns']
tbl_train_columns = tbl_train_columns.split(';')
drop_columns = ['HASHED_CIF', 'churn_flag']
tbl_train_columns = [i for i in tbl_train_columns if i not in drop_columns]
print('Number of Selected columns based on train set:', len(tbl_train_columns))
print('Selected columns based on train set:', tbl_train_columns)
tbl_new_data = tbl_new_data[tbl_train_columns]
print('Number of Columns from new data set to be passed to the model:', len(tbl_new_data.columns))
print('Columns from new data set to be passed to the model:', tbl_new_data.columns)

# %%
# Model predict - New Data
y_pred_new_data = model.predict(tbl_new_data)
tbl_id_new_data['churn_flag_predicted'] = y_pred_new_data
prediction_of_probability = model.predict_proba(tbl_new_data)
tbl_id_new_data['0_predicted_proba'] = prediction_of_probability[:,0] 
tbl_id_new_data['1_predicted_proba'] = prediction_of_probability[:,1] 

# The predictions are stored in the `predictions` output path
# so that AML can find them and pass them to other steps
output_tbl_new_data = pd.merge(tbl_id_new_data, tbl_new_data, left_index=True, right_index=True, how='inner')

# %%
# df1 = tbl_id_new_data['churn_flag_predicted'].value_counts().rename_axis('churn_flag_predicted').reset_index(name='counts')
# df2 = tbl_id_new_data['churn_flag_predicted'].value_counts(normalize=True).mul(100).round(1)\
#         .rename_axis('churn_flag_predicted').reset_index(name='percentage')
# df2['counts'] = df2['counts'].astype(str) + ' %'
# df3 = pd.merge(df1, df2, on='churn_flag_predicted')
# df3['percentage'] = df3['percentage'].astype(str) + ' %'
# run.log_table("Predicted_Churn_Flag_Distribution", df3.to_dict('list'))

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(output_tbl_new_data)

# Writing the output to file
ut.fun_write_file(output_tbl_new_data, output_path_new_data, output_file_name_new_data, run=run, csv=True)


