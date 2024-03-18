# %%
import pandas as pd
import datetime as dt
import time
import argparse
import os
from azureml.core import Run
import utilities as ut

# %%
print('Transform the demographics data.')

# %%
run = Run.get_context()
tbl = run.input_datasets['raw_data_demographics'].to_pandas_dataframe()
tbl_translation = run.input_datasets['mapping_data_translation'].to_pandas_dataframe()
tbl_region = run.input_datasets['mapping_data_region'].to_pandas_dataframe()
tbl_occupation = run.input_datasets['mapping_data_occupation'].to_pandas_dataframe()

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
    ''' This function grouping values to reducr the high cardinality.
     For Ex: PROVINCE with 64 unique values will group into 3 regions: SOUTH, NORTH and CENTRAL.'''

    # Transform the dataframe into dictionary
    tbl_convert = dict(zip(df_convert.ORIGINAL, df_convert.REVISED))

    # Create new column
    df[new_column_name] = df[column_convert]

    # Mapping new column with new variables
    df[new_column_name].replace(tbl_convert, inplace=True)

    # If there exists NULL, will fill with UNDEFINED.
    df[new_column_name].fillna(value='UNDEFINED', inplace=True)

    # If there exists NULL, will fill with UNDEFINED.
    df[column_convert].fillna(value='UNDEFINED', inplace=True)
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
    df_encoding = pd.get_dummies(df,columns=categorical_features, drop_first=False,dtype=int)
    df_merge = pd.concat([df_encoding, df[categorical_features]], axis=1)
    return df_merge

# %%
# Convert vietnamese column names or vietnamese values to english
tbl = ut.fun_convert_vnt_to_eng(tbl_translation, tbl)

# Dropping duplicate rows
tbl = ut.fun_drop_duplicates(tbl, 'Demographics Data')

# Converting date string columns to datetime format
tbl['CIF_CREATE'] = pd.to_datetime(tbl['CIF_CREATE'], format='%d-%b-%y')

## Feature Engineering
# Province column
tbl = feature_binning(tbl, tbl_region, 'PROVINCE_REGION', 'PROVINCE')

# Occupation column
tbl = feature_binning(tbl, tbl_occupation, 'OCCUPATION_GROUP', 'OCCUPATION')

# Apply the function Age group
tbl['AGE_GROUP'] = tbl['CUS_AGE'].apply(cus_age_group)

# Mapping groups. For Ex: Marital status takes only 3 groups: MARRIED, SINGLE and OTHER.
# tbl['MARITAL_GROUP'] = tbl['MARTIAL_STATUS'].apply(lambda x : 'OTHER' if x != ['MARRIED','SINGLE'] else x)
tbl['MARITAL_GROUP'] = tbl['MARTIAL_STATUS'].apply(lambda x : x if x in ['MARRIED','SINGLE'] else 'OTHER')

# Mapping groups. For Ex: Country takes only 2 groups: VIETNAM and OTHER.
tbl['COUNTRY_GROUP'] = tbl['COUNTRY'].apply(lambda x : 'OTHER' if x != 'VIETNAM' else x)

# One Hot encoding
categorical_features = ['CUSTOMER_SEGMENT','PROVINCE_REGION','MARITAL_GROUP',
                        'COUNTRY_GROUP','OCCUPATION_GROUP','GENDER','AGE_GROUP']
tbl = fun_feature_encode_demographics(tbl,categorical_features)

# %%
# Getting column and missing percentage information for columns with missing values
df_missing = ut.fun_get_missing_perc(tbl)

# Writing the output to file
ut.fun_write_file(tbl, output_path, output_file_name, run=run, csv=False)


