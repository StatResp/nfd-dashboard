from pyathena import connect
import pandas as pd
from cloud.upload_tables import *

s3_region_name = "us-east-1"
s3_staging_directory = "s3://athena-query-virginia/Unsaved/"

conn = connect(
    s3_staging_dir=s3_staging_directory, region_name=s3_region_name
)

db_name = "trafficdb"
base_df_table_name = "base_df"
incident_table_name = "tdot_incidents"
weather_table_name = "weatherbit_weather"
segment_table_name = "inrix_segment"

base_df_location = "s3://alltrafficdatatn/prediction_engine/base_df_linux/"
incident_location = "s3://alltrafficdatatn/prediction_engine/incident/"
weather_location = "s3://alltrafficdatatn/prediction_engine/weather/"
segment_location = "s3://alltrafficdatatn/prediction_engine/inrix/"

add_segment_command = get_create_segment_table_command(db_name,
                                                       segment_table_name,
                                                       segment_location) 
repair_segment_command = get_repair_table_command(db_name, segment_table_name)

add_base_df_command = get_create_base_df_table_command(db_name,
                                                       base_df_table_name,
                                                       base_df_location) 
repair_base_df_command = get_repair_table_command(db_name, base_df_table_name)
add_incident_command = get_create_incident_table_command(db_name,
                                                       incident_table_name,
                                                       incident_location) 
repair_incident_command = get_repair_table_command(db_name, incident_table_name)
add_weather_command = get_create_weather_table_command(db_name,
                                                       weather_table_name,
                                                       weather_location) 
repair_weather_command = get_repair_table_command(db_name, weather_table_name)

pd.read_sql(add_base_df_command, conn)
pd.read_sql(repair_base_df_command, conn)
pd.read_sql(add_incident_command, conn)
pd.read_sql(repair_incident_command, conn)
pd.read_sql(add_weather_command, conn)
pd.read_sql(repair_weather_command, conn)
pd.read_sql(add_segment_command, conn)
pd.read_sql(repair_segment_command, conn)

