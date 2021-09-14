def get_historical_query(window_size: int, view_name: str):
    size_of_day: int = int(24 / window_size)
    return f"""
    SELECT {view_name}.*
        , LAG(count_incidents, 1) 
            OVER (
                PARTITION BY xdsegid 
                ORDER BY time_local ASC) AS count_incidents_past_window
        , AVG(count_incidents) 
            OVER (
                PARTITION BY xdsegid 
                ORDER BY time_local ASC
                ROWS BETWEEN {size_of_day} PRECEDING AND 1 PRECEDING) AS mean_incidents_last_24_hours
        , LAG(count_incidents, {size_of_day})
            OVER (
                PARTITION BY xdsegid 
                ORDER BY time_local ASC) AS count_incidents_exact_yesterday
        , AVG(count_incidents)
           OVER (
               PARTITION BY xdsegid 
               ORDER BY time_local ASC
               ROWS BETWEEN ({size_of_day} * 7) PRECEDING AND 1 PRECEDING) AS mean_incidents_last_7_days
        , LAG(count_incidents, ({size_of_day} * 7))
           OVER (
               PARTITION BY xdsegid 
               ORDER BY time_local ASC) AS count_incidents_exact_last_week
        , AVG(count_incidents)
           OVER (
               PARTITION BY xdsegid 
               ORDER BY time_local ASC
               ROWS BETWEEN ({size_of_day} * 7 * 4) PRECEDING AND 1 PRECEDING) AS mean_incidents_last_4_weeks
        , LAG(count_incidents, ({size_of_day} * 7 * 4))
           OVER (
               PARTITION BY xdsegid 
               ORDER BY time_local ASC) AS count_incidents_exact_last_month
    FROM {view_name}"""
