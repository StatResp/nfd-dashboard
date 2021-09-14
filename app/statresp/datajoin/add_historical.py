from .historical_query import get_historical_query
from pyspark.sql import SparkSession, DataFrame


VIEW_NAME = "predictionEngine"


def add_historical_data(window_size: int, df: DataFrame, spark: SparkSession):
    df.createOrReplaceTempView(VIEW_NAME)

    historical_query = get_historical_query(window_size, VIEW_NAME)
    df = spark.sql(historical_query)
    mean_incidents = (
        df.select("xdsegid", "count_incidents")
        .groupBy("xdsegid")
        .avg("count_incidents")
    )
    df = df.join(mean_incidents, ["xdsegid"])
    df = df.withColumnRenamed(
        "avg(count_incidents)", "mean_incidents_over_all_windows"
    )

    return df
