from pyspark.sql import SparkSession, DataFrame
from math import ceil


def drop_alpha(
    alpha: float, pivot_column: str, df: DataFrame, spark: SparkSession
) -> DataFrame:
    """
    Drops the bottom 1-alpha percentile of segments or groups based on count incidents
    """
    values_sorted_by_count_incidents = (
        df.select(pivot_column, "count_incidents")
        .groupBy(pivot_column)
        .sum("count_incidents")
        .sort("sum(count_incidents)", ascending=False)
    )
    number_unique_values = (
        values_sorted_by_count_incidents.select(pivot_column).distinct().count()
    )
    count_values_selected = ceil(number_unique_values * (1 - alpha))
    top_values_names = values_sorted_by_count_incidents.select(
        pivot_column
    ).limit(count_values_selected)
    df = df.join(other=top_values_names, on=[pivot_column], how="inner")
    return df

    pass
