def get_repair_table_command(db_name, table_name):
    return(f"MSCK REPAIR TABLE {db_name}.{table_name};")

def get_create_base_df_table_command(db_name: str,
                                     table_name: str,
                                     s3_location: str) -> str:
    return(f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {db_name}.{table_name} (
      `time_local` timestamp,
      `xdsegid` double
    ) PARTITIONED BY (
      `year` int,
      `month` int
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
    WITH SERDEPROPERTIES (
      'serialization.format' = '1'
    ) LOCATION '{s3_location}'
    TBLPROPERTIES ('has_encrypted_data'='false');
    """)

def get_create_incident_table_command(db_name: str,
                                      table_name: str,
                                      s3_location: str) -> str:
    return(f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {db_name}.{table_name} (
      `log_mle` double,
      `county_incident` string,
      `route` string,
      `gps_coordinate_latitude` double,
      `gps_coordinate_longitude` double,
      `type_of_crash` string,
      `total_killed` int,
      `total_injured` int,
      `total_incapcitating_injuries` int,
      `total_other_injuries` int,
      `total_vehicles` int,
      `hazmat_involved` string,
      `hit_and_run` string,
      `id_original` string,
      `on_highway_type` string,
      `rail_crossing_identifier` string,
      `relation_to_first_junction` string,
      `relation_to_first_rdway` string,
      `school_bus_related` string,
      `updated_by` string,
      `updated_on` string,
      `urban_or_rural` string,
      `special_case` string,
      `case_number` int,
      `crash_location` string,
      `construction_zone_location` string,
      `intersection_type` string,
      `first_harmful_event` string,
      `manner_of_first_collision` string,
      `weather_conditions` string,
      `light_conditions` string,
      `locate_type` string,
      `time_local` string,
      `time_utc` string,
      `day` int,
      `day_of_week` int,
      `weekend_or_not` int,
      `hour` int,
      `window` int,
      `geometry` string,
      `geometry_string` string,
      `incident_id` int,
      `dist_to_seg` double,
      `xdsegid` double,
      `grouped_xdsegid` double,
      `frc` double
    ) PARTITIONED BY (
      year int,
      month int 
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
    WITH SERDEPROPERTIES (
      'serialization.format' = '1'
    ) LOCATION '{s3_location}'
    TBLPROPERTIES ('has_encrypted_data'='false');
    """)

# def get_create_weather_table_command(db_name: str,
#                                       table_name: str,
#                                       s3_location: str) -> str:
#     return(f"""
#     CREATE EXTERNAL TABLE IF NOT EXISTS {db_name}.{table_name} (
#       `rh` double,
#       `wind_spd` double,
#       `timestamp_utc` timestamp,
#       `vis` double,
#       `slp` double,
#       `pod` string,
#       `dni` double,
#       `elev_angle` double,
#       `pres`  double,
#       `h_angle` double,
#       `dewpt` double,
#       `snow` double,
#       `uv`  double,
#       `solar_rad` double,
#       `wind_dir` double,
#       `ghi` double,
#       `dhi` double,
#       `timestamp_local` timestamp,
#       `app_temp` double,
#       `azimuth` double,
#       `datetime` string,
#       `temp` double,
#       `precip` double,
#       `clouds` int,
#       `ts` int,
#       `weather.icon` string,
#       `weather.code` int,
#       `weather.description` string,
#       `gps_coordinate_latitude` double,
#       `gps_coordinate_longitude` double,
#       `spatial_id` string,
#       `station_id` string
#       ) PARTITIONED BY (
#           year int,
#           month int
#       )
#       ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
#       WITH SERDEPROPERTIES (
#         'serialization.format' = '1'
#       ) LOCATION '{s3_location}'
#       TBLPROPERTIES ('has_encrypted_data'='false');
#     """)

def get_create_weather_table_command(db_name: str,
                                      table_name: str,
                                      s3_location: str) -> str:
    return(f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {db_name}.{table_name} (
      `temp` double,
      `wind_spd` double,
      `vis` double,
      `precip` double,
      `snow` double,
      `station_id` string,
      `time_local` timestamp,
      `time_utc` timestamp,
      `day` int,
      `hour` int,
      `window` int
      ) PARTITIONED BY (
          year int,
          month int
      )
      ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
      WITH SERDEPROPERTIES (
        'serialization.format' = '1'
      ) LOCATION '{s3_location}'
      TBLPROPERTIES ('has_encrypted_data'='false');
    """)

def get_create_segment_table_command(db_name: str,
                                     table_name: str,
                                     s3_location: str) -> str:
    return(f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {db_name}.{table_name} (
        xdsegid double,
        previousxd double,
        nextxdsegi double,
        frc int,
        county_inrix string,
        miles double,
        lanes double,
        sliproad boolean,
        startlat double,
        startlong double,
        endlat double,
        endlong double,
        bearing string,
        xdgroup double, 
        roadnumber string,
        roadname string, 
        roadlist string, 
        `beg` string, 
        `end` string, 
        center string, 
        geometry string,
        geometry_highres string, 
        geometry_original_string string,
        geometry_highres_string string,
        osmwayids string,
        osmwaydirections string,
        waystartoffset_m double,
        wayendoffset_m double,
        waystartoffset_percent double,
        wayendoffset_percent double,
        isf_length double, 
        isf_length_min double,
        length_m double,
        nearest_weather_station string,
        lon string, 
        lat string,
        elevation string,  
        geometry3d string, 
        slope double, 
        slope_median double, 
        ends_ele_diff double,
        max_ele_diff double,
        slopes string, 
        grouped_xdsegid int,
        grouped_xdsegid_id int,
        grouped_xdsegid_id_miles double
      ) PARTITIONED BY (
        countyid int
      )
      ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
      WITH SERDEPROPERTIES (
        'serialization.format' = '1'
      ) LOCATION '{s3_location}'
      TBLPROPERTIES ('has_encrypted_data'='false');
    """)
