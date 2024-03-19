# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_date, udf

from azureml.core import Run
import utilities as ut

# %%
print('Transform the demographics data.')

# %%
run = Run.get_context()
spark = SparkSession.builder.getOrCreate()
tbl = spark.read.format('delta').load(run.input_datasets['raw_data_demographics'])
tbl_translation = spark.read.format('delta').load(run.input_datasets['mapping_data_translation'])
tbl_region = spark.read.format('delta').load(run.input_datasets['mapping_data_region'])
tbl_occupation = spark.read.format('delta').load(run.input_datasets['mapping_data_occupation'])

# %%
def parse_args():
    parser = argparse.ArgumentParser('transform_demographics')
    parser.add_argument('--output_data_path', type=str, help='transformed demographics table directory')

    return parser.parse_args()

args = parse_args()
run_id = run.id
output_file_name = 'transformed_demographics_data'
output_path = f"{args.output_data_path}"#/{run_id}
print(f'Arguments: {args.__dict__}')

# %%
def feature_binning(df, df_convert, new_column_name, column_convert):
    ''' This function grouping values to reduce the high cardinality.
     For Ex: PROVINCE with 64 unique values will be grouped into 3 regions: SOUTH, NORTH, and CENTRAL.'''

    # Transform the dataframe into a dictionary
    tbl_convert = dict(zip(df_convert.select('ORIGINAL').rdd.flatMap(lambda x: x).collect(),
                            df_convert.select('REVISED').rdd.flatMap(lambda x: x).collect()))

    # Create a new column using when-otherwise statement
    df = df.withColumn(new_column_name,
                       when(col(column_convert).isin(tbl_convert.keys()), col(column_convert))
                       .otherwise('UNDEFINED'))

    # Mapping new column with new variables
    df = df.replace(tbl_convert, subset=new_column_name)

    # Fill NULL values with 'UNDEFINED'
    df = df.fillna('UNDEFINED', subset=[new_column_name, column_convert])

    return df
# %%
def cus_age_group(x):
    ''' This function is to group customer age into 5 groups: Boomers, Gen X, Millennials, Gen Z and Others.
     For Ex: group of customers who born between 1981 to 1996 will group into Millennials. '''
    
    if 1946 <= x <= 1964:
        return "boomers"
    elif 1965 <= x <= 1980:
        return "genX"
    elif 1981 <= x <= 1996:
        return "millennials"
    elif 1997 <= x <= 2005:
        return "genZ"
    else:
        return "Others"

# %%
def fun_feature_encode_demographics(df, categorical_features):
    stages = []

    # StringIndexer cho các biến phân loại
    for col_name in categorical_features:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_index")
        stages.append(indexer)

    # OneHotEncoder cho các biến đã được mã hóa bằng StringIndexer
    for col_name in categorical_features:
        encoder = OneHotEncoder(inputCol=f"{col_name}_index", outputCol=f"{col_name}_encoded")
        stages.append(encoder)

    # Tạo pipeline
    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(df)
    df_encoded = pipeline_model.transform(df)

    # Lựa chọn các cột đã mã hóa và ghép vào DataFrame gốc
    encoded_columns = [f"{col_name}_encoded" for col_name in categorical_features]
    df_encoded = df_encoded.select([col(col_name) for col_name in encoded_columns] + df.columns)

    return df_encoded

# %%
# Convert vietnamese column names or vietnamese values to english
tbl = ut.fun_convert_vnt_to_eng(tbl_translation, tbl)

# Dropping duplicate rows
tbl = ut.fun_drop_duplicates(tbl, 'Demographics Data')

# Converting date string columns to datetime format
tbl = tbl.withColumn('CIF_CREATE', to_date(tbl['CIF_CREATE'], 'dd-MMM-yy'))

## Feature Engineering
# Province column
tbl = feature_binning(tbl, tbl_region, 'PROVINCE_REGION', 'PROVINCE')

# Occupation column
tbl = feature_binning(tbl, tbl_occupation, 'OCCUPATION_GROUP', 'OCCUPATION')

# Apply the function Age group
udf_cus_age = udf(lambda x:cus_age_group(x), StringType())
tbl = tbl.withColumn('AGE_GROUP', udf_cus_age(tbl['CUS_AGE']))

# Mapping groups. For Ex: Marital status takes only 3 groups: MARRIED, SINGLE and OTHER.
# tbl['MARITAL_GROUP'] = tbl['MARTIAL_STATUS'].apply(lambda x : 'OTHER' if x != ['MARRIED','SINGLE'] else x)
tbl = tbl.withColumn('MARITAL_GROUP', when(col('MARTIAL_STATUS').isin(['MARRIED', 'SINGLE']), col('MARTIAL_STATUS')).otherwise('OTHER'))

# Mapping groups for 'COUNTRY'
tbl = tbl.withColumn('COUNTRY_GROUP', when(col('COUNTRY') == 'VIETNAM', 'VIETNAM').otherwise('OTHER'))

# One Hot encoding
categorical_features = ['CUSTOMER_SEGMENT','PROVINCE_REGION','MARITAL_GROUP',
                        'COUNTRY_GROUP','OCCUPATION_GROUP','GENDER','AGE_GROUP']
tbl = fun_feature_encode_demographics(tbl,categorical_features)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)


