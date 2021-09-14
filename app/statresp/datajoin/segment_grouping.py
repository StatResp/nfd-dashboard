from .grouping_query import get_group_segments_query
from pyspark.sql import SparkSession, DataFrame

VIEW_NAME = "predictionEngine"


def group_segments(group_name: str, df: DataFrame, spark: SparkSession):
    df.createOrReplaceTempView(VIEW_NAME)
    grouping_query = get_group_segments_query(group_name, VIEW_NAME)
    df = spark.sql(grouping_query)
    return df

