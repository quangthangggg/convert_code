# %%
# import pandas as pd
import datetime as dt
import time
import argparse
import os
from azureml.core import Run
from pyspark.sql.functions import monotonically_increasing_id, translate, col, round, count, lit
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StringType
# %%
# Notebook specific imports
from joblib import Parallel, delayed
import multiprocessing as mp

# %%
def fun_drop_duplicates(df, table_name):
    '''Dropping duplicate rows in the data.'''
    rows_before = df.count()
    # Thêm cột id để phân biệt các hàng
    df = df.withColumn("id", monotonically_increasing_id())
    # Drop các hàng trùng lặp dựa trên cột id
    df = df.dropDuplicates(["id"])
    rows_after = df.count()
    print(f'Number of duplicate rows dropped: {rows_before - rows_after} from table {table_name}')
    print(f'Percentage of duplicate rows dropped: {round(100 * (rows_before - rows_after) / rows_before, 2)} % from table {table_name}')
    print('-'*30)
    # Xóa cột id đã thêm
    df = df.drop("id")
    return df

# %%
def fun_convert_vnt_to_eng(df_translation, df):

    # Convert DataFrame to dictionary
    vn_en_dict = dict(zip(df_translation.VN, df_translation.EN))

    # Translate column names if needed
    for col in df.columns:
        if col in vn_en_dict:
            df = df.withColumnRenamed(col, vn_en_dict[col])
    
    # Translate rows if needed
    ## transaction table need no conversion
    if 'AMOUNT_TRANSACTION' not in df.columns:
        for col in df.columns:
            if col in vn_en_dict:
                df = df.withColumn(col, translate(col, vn_en_dict[col]))
        print('Conversion Complete')
        
    return df

# %%
def fun_get_missing_perc(df):
    '''Function provides information on the columns with missing data.'''
    # Calculate the count of missing values for each column
    df_missing = df.select([col(c).alias('column_name'), col(c).isNull().cast('int').alias('missing_value') for c in df.columns]) \
                   .groupBy('column_name').sum('missing_value').orderBy('column_name') \
                   .withColumn('percentage_missing', round(col('sum(missing_value)') / df.count() * 100, 2))
    
    # Filter out columns with no missing values
    df_missing = df_missing.filter(col('sum(missing_value)') > 0)
    
    print('Percentage of missing data in Columns: ')
    df_missing.show(truncate=False)
    
    return df_missing

# %%
def applyParallel(dfGrouped: DataFrame, func, args_lst=[], njobs=None) -> DataFrame:
    ''' Run the grouped data in parallel. '''
    if njobs is None:
        njobs = dfGrouped.rdd.getNumPartitions()  # Number of partitions in DataFrame
    
    # Create an empty DataFrame with the same schema as dfGrouped
    schema = dfGrouped.schema
    result_df = dfGrouped.sql_ctx.createDataFrame([], schema)

    # Apply the function to each group and union the results
    for name, group in dfGrouped.rdd.groupByKey():
        result_group = func(group.toDF(), args_lst)  # Convert RDD to DataFrame
        result_df = result_df.union(result_group)

    return result_df

# %%
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

# %%
def fun_get_flag_distribution(tbl, column_name, output_file):
    '''Get value counts of flag with percentage.'''
    # Calculate actual count
    df1 = tbl.groupBy(column_name).agg(count("*").alias("actualCount"))
    
    # Calculate percentage
    total_count = tbl.count()
    df2 = tbl.groupBy(column_name).agg((count("*") / total_count * 100).alias("percentage"))
    
    # Round percentage to one decimal place
    df2 = df2.withColumn("percentage", round(df2["percentage"], 1))
    
    # Convert percentage to string and append '%'
    df2 = df2.withColumn("percentage", df2["percentage"].cast(StringType()) + " %")
    
    # Join actual count and percentage
    df3 = df1.join(df2, on=column_name)
    
    # Format actual count
    df3 = df3.withColumn("abbrCount", df3["actualCount"].cast(StringType()))
    
    # Add output file name
    df3 = df3.withColumn("file_name", lit(output_file))
    
    # Reorder columns
    df3 = df3.select("file_name", column_name, "abbrCount", "percentage", "actualCount")
    
    return df3

# %%

def fun_write_file(tbl, output_path, output_file, run, csv=False):
    print('-'*30)
    print(f'Number of rows and columns in the file {output_file} are: {tbl.count()} & {len(tbl.columns)}')

    # for logging as metric
    tbl_col_names = ', '.join(tbl.columns)
    columns = ['file_name', 'TotalColumns', 'TotalRowsAbbr', 'TotalRows', 'ColumnNames']
    list_of_values = [output_file, len(tbl.columns), human_format(tbl.count()), tbl.count(), tbl_col_names]
    spark = SparkSession.builder.getOrCreate()
    df_info = spark.createDataFrame([list_of_values], columns=columns)

    print('Columns in data:', tbl.columns)
    if 'HASHED_CIF' in tbl.columns:
        num_unique_customers = tbl.select('HASHED_CIF').distinct().count()
        print(f'Number of unique customer ids in the file are: {num_unique_customers}')
        df_info = df_info.withColumn('NumofCustomersAbbr', lit(human_format(num_unique_customers)))
        df_info = df_info.withColumn('NumofCustomers', num_unique_customers)
    
    run.log_table('Basic_Data_Information', df_info.toPandas().to_dict('list'))

    if 'churn_flag' in tbl.columns:
        print(f'Target distribution in file: {tbl.groupby("churn_flag").count().collect()}')
        df_flag = fun_get_flag_distribution(tbl, 'churn_flag', output_file)
        run.log_table("Churn_Flag_Distribution", df_flag.toPandas().to_dict('list'))
    if 'churn_flag_predicted' in tbl.columns:
        print(f'Target distribution in file: {tbl.groupby("churn_flag_predicted").count().collect()}')
        df_flag = fun_get_flag_distribution(tbl, 'churn_flag_predicted', output_file)
        run.log_table("Predicted_Churn_Flag_Distribution", df_flag.toPandas().to_dict('list'))

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        print("File %s created" % output_path)
        if csv:
            output_path = f'{output_path}/{output_file}.csv'
            tbl.write.csv(output_path, mode="overwrite", header=True)
        else: 
            output_path = f'{output_path}/{output_file}.parquet'
            tbl.write.parquet(output_path, mode="overwrite")
    else:
        print("-"*50)
        print("File %s already created" % output_path)
        print("-"*50)


