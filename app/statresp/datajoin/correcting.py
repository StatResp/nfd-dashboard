from pyspark.sql import DataFrame
from pyspark.sql.functions import col, dayofweek
from pyspark.sql.types import BooleanType


def correcting(drop_missing: bool, response_name: str, df: DataFrame):
    """
    This function adds a response column, corrects typing,
    and optionally drops missing values
    """
    df = (
        df.withColumn(response_name, col("count_incidents") > 0)
        .withColumn("is_weekend", col("is_weekend").cast(BooleanType()))
        .withColumn("day_of_week", dayofweek("time_local"))
    )
    if drop_missing is True:
        df = df.dropna(how="any")
    return df

