def get_group_segments_query(group_name: str, view_name: str):
    return f"""
    SELECT {group_name}
        , FIRST(time_local) AS time_local
        , FIRST(time_utc) AS time_utc
        , window
        , day
        , FIRST(is_weekend) AS is_weekend
        , MIN(speed_min) AS speed_min
        , AVG(speed_mean) AS speed_mean
        , MAX(speed_max) AS speed_max
        , MIN(average_speed_min) AS average_speed_min
        , AVG(average_speed_mean) AS average_speed_mean
        , MAX(average_speed_max) AS average_speed_max
        , MIN(reference_speed_min) AS reference_speed_min
        , AVG(reference_speed_mean) AS reference_speed_mean
        , MAX(reference_speed_max) AS reference_speed_max
        , MIN(congestion_min) AS congestion_min
        , AVG(congestion_mean) AS congestion_mean
        , MAX(congestion_max) AS congestion_max
        , FIRST(county) AS county
        , FIRST(nearest_weather_station) AS nearest_weather_station
        , SUM(miles) AS miles
        , SUM(length_m) AS length_m
        , AVG(lanes) AS lanes
        , AVG(isf_length) AS isf_length
        , MIN(isf_length_min) AS isf_length_min
        , AVG(slope) AS slope 
        , AVG(slope_median) AS slope_median 
        , AVG(ends_ele_diff) AS ends_ele_diff
        , MAX(max_ele_diff) AS max_ele_diff
        , MIN(temp_min) AS temp_min
        , AVG(temp_mean) AS temp_mean
        , MAX(temp_max) AS temp_max
        , MIN(wind_spd_min) AS wind_spd_min
        , AVG(wind_spd_mean) AS wind_spd_mean
        , MAX(wind_spd_max) AS wind_spd_max
        , MIN(vis_min) AS vis_min
        , AVG(vis_mean) AS vis_mean
        , MAX(vis_max) AS vis_max
        , MIN(precip_min) AS precip_min
        , AVG(precip_mean) AS precip_mean
        , MAX(precip_max) AS precip_max
        , SUM(total_killed) AS total_killed
        , SUM(total_injured) AS total_injured
        , SUM(total_incapcitating_injuries) AS total_incapcitating_injuries
        , SUM(total_other_injuries) AS total_other_injuries 
        , SUM(total_vehicles) AS total_vehicles
        , SUM(count_incidents) AS count_incidents
        , SUM(count_incidents_past_window) AS count_incidents_past_window
        , SUM(count_incidents_exact_yesterday) AS count_incidents_exact_yesterday
        , SUM(count_incidents_exact_last_week) AS count_incidents_exact_last_week
        , SUM(count_incidents_exact_last_month) AS count_incidents_exact_last_month
        , SUM(mean_incidents_last_24_hours) AS mean_incidents_last_24_hours
        , SUM(mean_incidents_last_7_days) AS mean_incidents_last_7_days
        , SUM(mean_incidents_last_4_weeks) AS mean_incidents_last_4_weeks
        , SUM(mean_incidents_over_all_windows) AS mean_incidents_over_all_windows
        , year
        , month
    FROM {view_name}
    GROUP BY {group_name}
        , year
        , month
        , day
        , window
    """
