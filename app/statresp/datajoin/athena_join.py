from pyathena import connect
import pandas as pd


def get_sql_number_range_from_str(x: str, y: str) -> str:
    if x == y:
        return f"({int(x)})"
    else:
        delta = list(range(int(x), int(y) + 1))
        return str(delta).replace("[", "(").replace("]", ")")


def query_athena(
    start_date: str,
    end_date: str,
    window_size: int,
    group_name: str,
    s3_region_name,
    s3_staging_directory: str,
    s3_save_location: str,
    new_athena_table_name: str,
    segmentwindow_table_name: str,
    traffic_table_name: str,
    segment_table_name: str,
    weather_table_name: str,
    tdot_incident_table_name: str,
):
    start_year: str = start_date.split("-")[0]
    end_year: str = end_date.split("-")[0]
    year_range: str = get_sql_number_range_from_str(start_year, end_year)
    conn = connect(
        s3_staging_dir=s3_staging_directory, region_name=s3_region_name
    )
    query = f"""
    CREATE TABLE {new_athena_table_name}
    WITH (
        format='PARQUET',
        partitioned_by = ARRAY['year','month'],
        external_location='{s3_save_location}'
    ) AS (
        SELECT gamma.xdsegid
            , gamma.{group_name}
            , gamma.time_local
            , gamma.day
            , gamma.window
            , gamma.is_weekend
            , gamma.speed_min
            , gamma.speed_mean
            , gamma.speed_max
            , gamma.average_speed_min
            , gamma.average_speed_mean
            , gamma.average_speed_max
            , gamma.reference_speed_min
            , gamma.reference_speed_mean
            , gamma.reference_speed_max
            , gamma.congestion_min
            , gamma.congestion_mean
            , gamma.congestion_max
            , gamma.frc
            , gamma.county
            , gamma.nearest_weather_station
            , gamma.miles
            , gamma.lanes
            , gamma.isf_length
            , gamma.isf_length_min
            , gamma.length_m
            , gamma.slope
            , gamma.slope_median
            , gamma.ends_ele_diff
            , gamma.max_ele_diff
            , gamma.temp_min
            , gamma.temp_mean
            , gamma.temp_max
            , gamma.wind_spd_min
            , gamma.wind_spd_mean
            , gamma.wind_spd_max
            , gamma.vis_min
            , gamma.vis_mean
            , gamma.vis_max
            , gamma.precip_min
            , gamma.precip_mean
            , gamma.precip_max
            , gamma.time_utc
            , COALESCE(incident.total_killed,0) AS total_killed
            , COALESCE(incident.total_injured,0) AS total_injured
            , COALESCE(incident.total_incapcitating_injuries,0) AS total_incapcitating_injuries
            , COALESCE(incident.total_other_injuries,0) AS total_other_injuries
            , COALESCE(incident.total_vehicles,0) AS total_vehicles
            , COALESCE(incident.count_incidents,0) AS count_incidents
            , gamma.year
            , gamma.month
        FROM (
            SELECT beta.*
                , weather.temp_min
                , weather.temp_mean
                , weather.temp_max
                , weather.wind_spd_min
                , weather.wind_spd_mean
                , weather.wind_spd_max
                , weather.vis_min
                , weather.vis_mean
                , weather.vis_max
                , weather.precip_min
                , weather.precip_mean
                , weather.precip_max
                , weather.time_utc
            FROM (
                SELECT alpha.*
                    , {segment_table_name}.frc
                    , {segment_table_name}.county_inrix AS county
                    , {segment_table_name}.nearest_weather_station
                    , {segment_table_name}.miles
                    , {segment_table_name}.lanes
                    , {segment_table_name}.isf_length
                    , {segment_table_name}.isf_length_min
                    , {segment_table_name}.length_m
                    , {segment_table_name}.slope
                    , {segment_table_name}.slope_median
                    , {segment_table_name}.ends_ele_diff
                    , {segment_table_name}.max_ele_diff
                    , {segment_table_name}.{group_name}
                FROM (
                    SELECT segment_window.time_local
                        , segment_window.year
                        , segment_window.month
                        , segment_window.day
                        , segment_window.window
                        , segment_window.xdsegid
                        , CASE
                            WHEN day_of_week(time_local) = 6 THEN 1
                            WHEN day_of_week(time_local) = 7 THEN 1
                            ELSE 0
                          END is_weekend
                        , traffic.speed_min
                        , traffic.speed_mean
                        , traffic.speed_max
                        , traffic.average_speed_min
                        , traffic.average_speed_mean
                        , traffic.average_speed_max
                        , traffic.reference_speed_min
                        , traffic.reference_speed_mean
                        , traffic.reference_speed_max
                        , traffic.congestion_min
                        , traffic.congestion_mean
                        , traffic.congestion_max
                    FROM (
                        SELECT {segmentwindow_table_name}.time_local AS time_local
                            , year({segmentwindow_table_name}.time_local) AS year
                            , month({segmentwindow_table_name}.time_local) AS month
                            , day({segmentwindow_table_name}.time_local) AS day
                            , hour({segmentwindow_table_name}.time_local) / 4 AS window
                            , {segmentwindow_table_name}.xdsegid
                        FROM {segmentwindow_table_name}
                        WHERE EXTRACT (YEAR FROM DATE({segmentwindow_table_name}.time_local)) IN {year_range}
                            AND DATE(time_local) >= DATE('{start_date}')
                            AND DATE(time_local) <= DATE('{end_date}')
                    ) as segment_window
                    LEFT JOIN ( 
                        SELECT xdsegid
                            , ARBITRARY(countyid) as countyid
                            , EXTRACT (YEAR FROM DATE(measurement_tstamp)) AS year
                            , EXTRACT (MONTH FROM DATE(measurement_tstamp)) AS month
                            , EXTRACT (DAY FROM DATE(measurement_tstamp)) AS day
                            , EXTRACT (HOUR FROM measurement_tstamp) / {window_size} AS window
                            , MIN(DATE_FORMAT(measurement_tstamp,'%Y-%m-%d %H:%i:%s')) AS date
                            , MIN(speed) AS speed_min
                            , AVG(speed) AS speed_mean
                            , MAX(speed) AS speed_max
                            , MIN(average_speed) AS average_speed_min
                            , AVG(average_speed) AS average_speed_mean
                            , MAX(average_speed) AS average_speed_max
                            , MIN(reference_speed) AS reference_speed_min
                            , AVG(reference_speed) AS reference_speed_mean
                            , MAX(reference_speed) AS reference_speed_max
                            , MIN(congestion) AS congestion_min
                            , AVG(congestion) AS congestion_mean
                            , MAX(congestion) AS congestion_max
                        FROM (
                            SELECT xd_id AS xdsegid
                                , countyid
                                , measurement_tstamp
                                , speed
                                , average_speed
                                , reference_speed
                                , CASE
                                    WHEN (reference_speed - speed)/reference_speed >= 0 THEN (reference_speed - speed)/reference_speed
                                    WHEN (reference_speed - speed)/reference_speed < 0 THEN 0
                                  END congestion
                            FROM {traffic_table_name}
                            WHERE year IN {year_range}
                                AND DATE(measurement_tstamp) >= DATE('{start_date}')
                                AND DATE(measurement_tstamp) <= DATE('{end_date}')
                            ) AS inrixtraffic
                        GROUP BY xdsegid
                            , EXTRACT (DAY FROM DATE(measurement_tstamp))
                            , EXTRACT (MONTH FROM DATE(measurement_tstamp))
                            , EXTRACT (HOUR FROM measurement_tstamp) / {window_size} 
                            , EXTRACT (YEAR FROM DATE(measurement_tstamp))
                    ) AS traffic
                    ON segment_window.year = traffic.year
                        AND segment_window.month = traffic.month
                        AND segment_window.day = traffic.day
                        AND segment_window.window = traffic.window
                        AND segment_window.xdsegid = traffic.xdsegid
                    WHERE segment_window.year IN {year_range}
                        AND DATE(time_local) >= DATE('{start_date}')
                        AND DATE(time_local) <= DATE('{end_date}')
                ) AS alpha
                LEFT JOIN {segment_table_name}
                ON alpha.xdsegid={segment_table_name}.xdsegid
            ) AS beta
            LEFT JOIN (
                SELECT station_id
                    , MIN(DATE_FORMAT(time_utc,'%Y-%m-%d %H:%i:%s')) AS time_utc
                    , EXTRACT (YEAR FROM time_local) AS year
                    , EXTRACT (MONTH FROM time_local) AS month
                    , EXTRACT (DAY FROM time_local) AS day
                    , EXTRACT (HOUR FROM time_local) / {window_size} AS window
                    , MIN(temp) AS temp_min
                    , AVG(temp) AS temp_mean
                    , MAX(temp) AS temp_max
                    , MIN(wind_spd) AS wind_spd_min
                    , AVG(wind_spd) AS wind_spd_mean
                    , MAX(wind_spd) AS wind_spd_max
                    , MIN(vis) AS vis_min
                    , AVG(vis) AS vis_mean
                    , MAX(vis) AS vis_max
                    , MIN(precip) AS precip_min
                    , AVG(precip) AS precip_mean
                    , MAX(precip) AS precip_max
                FROM {weather_table_name}
                WHERE year IN {year_range}
                    AND DATE(time_local) >= DATE('{start_date}')
                    AND DATE(time_local) <= DATE('{end_date}')
                GROUP BY station_id
                    , EXTRACT (DAY FROM time_local)
                    , EXTRACT (MONTH FROM time_local)
                    , EXTRACT (HOUR FROM time_local) / {window_size}
                    , EXTRACT (YEAR FROM time_local)
            ) AS weather
            ON beta.year = weather.year
                AND beta.month = weather.month
                AND beta.day = weather.day
                AND beta.window = weather.window
                AND beta.nearest_weather_station = weather.station_id
        ) AS gamma
        LEFT JOIN (
            SELECT xdsegid 
                , year
                , month
                , day
                , hour / {window_size} AS window
                , SUM(total_killed) AS total_killed
                , SUM(total_injured) AS total_injured 
                , SUM(total_incapcitating_injuries) AS total_incapcitating_injuries
                , SUM(total_other_injuries) AS total_other_injuries
                , SUM(total_vehicles) AS total_vehicles
                , COUNT(DISTINCT case_number) AS count_incidents
            FROM {tdot_incident_table_name}
            WHERE year IN {year_range}
                AND DATE_PARSE(time_local,'%Y-%m-%d %H:%i:%s') >= DATE('{start_date}')
                AND DATE_PARSE(time_local,'%Y-%m-%d %H:%i:%s') <= DATE('{end_date}')
            GROUP BY xdsegid
                , year
                , month
                , day
                , hour / {window_size}
        ) AS incident
        ON gamma.year = incident.year
            AND gamma.month = incident.month
            AND gamma.day = incident.day
            AND gamma.window = incident.window
            AND gamma.xdsegid = incident.xdsegid
    );
    """
    print(query)
    return pd.read_sql(query, conn)
