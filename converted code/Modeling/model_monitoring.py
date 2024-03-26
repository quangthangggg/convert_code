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
from pyspark.sql import SparkSession
from pyspark.sql import Row
import joblib
from azureml.core.run import Run, _OfflineRun

# %%
# from sklearn.metrics import (
#     accuracy_score,
#     roc_auc_score,
#     average_precision_score,
#     balanced_accuracy_score,
#     f1_score,
#     recall_score,
#     precision_score
# )
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)
accuracy_score = MulticlassClassificationEvaluator(metricName="accuracy")
roc_auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")
pr_auc_evaluator = BinaryClassificationEvaluator(metricName="areaUnderPR")
f1_score = MulticlassClassificationEvaluator(metricName="f1")
recall_score = MulticlassClassificationEvaluator(metricName="recall")
precision_evaluator = MulticlassClassificationEvaluator(metricName="precision")
# %%
print('Get predictions from the trained model for New Data set.')

# %%
run = Run.get_context()
run_id = run.id
print('Run id:', run_id)

# Get the current context's workspace..
ws = Workspace.from_config() if type(run) == _OfflineRun else run.experiment.workspace
print('Workspace:', ws)

# %%
spark = SparkSession.builder.getOrCreate()
tbl_model_pred_data = spark.read.format('delta').load(run.input_datasets['model_pred_data'])

# %%
# tbl_new_data = run.input_datasets['new_data'].to_pandas_dataframe()
# tbl_id_new_data = run.input_datasets['new_data_id'].to_pandas_dataframe()

# %%
def parse_args():
    parser = argparse.ArgumentParser('get_model_predictions')
    # parser.add_argument('--output_data_path_new_data', type=str, help='model predictions new data table directory')
    parser.add_argument('--model_name', help='name of the model to be used for prediction')
    # parser.add_argument('--model_path', help='path of the model to be used for prediction')
    parser.add_argument('--experiment_name', type=str, help='experiment name')
    parser.add_argument('--model_version', type=int, help='trained model_version')
    parser.add_argument('--user_selected_metric', type=str, help='user_selected_metric to select best model')

    return parser.parse_args()

args = parse_args()
# output_file_name_new_data = 'model_predictions_new_data'
# output_path_new_data = f"{args.output_data_path_new_data}"
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
# Get the true_labels, predicted_labels and predicted_probabilities result
true_labels = tbl_model_pred_data['churn_flag']
predicted_labels = tbl_model_pred_data['churn_flag_predicted']
predicted_probabilities = tbl_model_pred_data['1_predicted_proba']

# %%
print(f'True Labels Type: {type(true_labels)}')
print(f'{true_labels.head()}')
print(f'Predicted Labels Type: {type(predicted_labels)}')
print(f'{predicted_labels.head()}')
print(f'Predicted Probabilities Type: {type(predicted_probabilities)}')
print(f'{predicted_probabilities.head()}')

# %%
# Compute accuracy
accuracy = accuracy_score.evaluate(true_labels, predicted_labels)

# Compute average precision scores
# ap_macro = average_precision_score(true_labels, predicted_probabilities, pos_label=1)

# Compute F1 scores
f1_macro = f1_score.evaluate(true_labels, predicted_labels)

# Compute normalized macro recall based on AzureML formula
recall_score_cal = recall_score.evaluate(true_labels, predicted_labels)
normalized_macro_recall = (recall_score_cal - 0.5) / (1 - 0.5)

# %%
# Create the NDP score dictionary and dataframe
ndp_score_dict = {
    'accuracy': accuracy,
    # 'average_precision_score_macro': ap_macro,
    'f1_score_macro': f1_macro,
    'norm_macro_recall': normalized_macro_recall
}

ndp_score_dict_rdd = spark.sparkContext.parallelize(list(ndp_score_dict.items()))
ndp_score_data = ndp_score_dict_rdd.map(lambda x: Row(metrics=x[0], ndp_score=x[1])).toDF()

# %%
from ast import literal_eval
if args.user_selected_metric == '':
    orig_model_metrics_dict = literal_eval(model_ws.properties['best_model_as_per_primary_metric'])['metrics']
else:
    orig_model_metrics_dict = literal_eval(model_ws.properties['best_model_as_per_user_selected_metric'])['metrics']


# %%
# Create the original model metrics score dataframe
orig_model_metrics_dict_rdd = spark.sparkContext.parallelize(list(orig_model_metrics_dict.items()))
orig_model_score_data = orig_model_metrics_dict_rdd.map(lambda x: Row(metrics=x[0], original_model_score=x[1])).toDF()
# %%
# Merge original model score and NDP score to monitor model's performance
monitoring_score_data = ndp_score_data.merge(orig_model_score_data)
monitoring_score_data['% diff'] = monitoring_score_data['ndp_score']/monitoring_score_data['original_model_score'] - 1

# %%
run.log_table("ModelMonitoring", monitoring_score_data.to_dict('list'))


