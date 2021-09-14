# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:50:15 2020

@author: vaziris
This scripts is  written by Sayyed Mohsen Vazirizade.  s.m.vazirizade@gmail.com
"""


from pprint import pprint
import pandas as pd
import geopandas as gpd
import numpy as np
import pickle
import time
from statresp.datajoin.cleaning.utils import Save_DF
from datetime import datetime, timedelta
import shapely.geometry as sg
from shapely.ops import nearest_points





def convert_to_camel_case(string: str):
    pieces = string.lower().replace(" ", " ").replace("_", " ").strip().split()
    return pieces[0] + "".join(i.capitalize() for i in pieces[1:])


def call_weatherbit_api(begin, end, station_id_list, key):
    CALLS_PER_SECOND_MAX = 50

    total_calls = 0
    # add 14 days to end so that our incrementation is guaranteed to cover our end date.
    api_end = end + timedelta(14)

    # Initialize empty Dataframe
    df_weather = pd.DataFrame({"A": []})

    # used to check if sleep needs to be used.
    sleep_counter = 0

    # For each station in our station list
    for station_id in station_id_list:
        single_date = begin

        # For each day in range, incrementing by
        while single_date < api_end:
            start_date = single_date.strftime("%Y-%m-%d")
            end_date = (single_date + timedelta(1)).strftime("%Y-%m-%d")

            # Make API call for historical weather data for a day
            resp = requests.get(
                "https://api.weatherbit.io/v2.0/history/hourly?station={}&start_date={}&end_date={}&tz=local&key={}".format(
                    station_id, start_date, end_date, key
                )
            )
            resp_json = resp.json()
            data = resp_json["data"]
            total_calls = total_calls + 1
            # Adjust sleep counter
            if sleep_counter >= CALLS_PER_SECOND_MAX:
                time.sleep(1)
                sleep_counter = 0
            else:
                sleep_counter = sleep_counter + 1

            # Convert the JSON into a DF
            new_entry = pd.json_normalize(data)
            new_entry["GPS Coordinate Latitude"] = resp_json["lat"]
            new_entry["GPS Coordinate Longitude"] = resp_json["lon"]
            new_entry["spatial_id"] = resp_json["city_name"]
            new_entry["station_id"] = resp_json["station_id"]

            # Edge case (first instance)
            if not df_weather.empty:
                df_weather = df_weather.append(new_entry, ignore_index=True)
            else:
                df_weather = new_entry
    try:
        print("Total API Calls: " + str(total_calls))
    except:
        print("issues printing total calls")
    return df_weather


def merge_old_and_new_data(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    old_df.columns = old_df.columns.str.lower().str.replace(' ', '_')
    new_df.columns = new_df.columns.str.lower().str.replace(' ', '_')
    df = pd.concat([old_df, new_df])
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], format="%Y-%m-%dT%H:%M:%S"
    )
    df["timestamp_local"] = pd.to_datetime(
        df["timestamp_local"], format="%Y-%m-%dT%H:%M:%S"
    )
    df["month"] = df["timestamp_local"].dt.month
    df["year"] = df["timestamp_local"].dt.year
    df.columns     = df.columns.str.lower().str.replace(' ', '_')
    return df


def calculate_nearest(row, destination):
    """
    it calulates the distacne from a shape (all weather stations) to a point (beggining of each segment) and find the closest shape id
    @param row: a row of a pandas dataframe (beggining of each segment) that we want to find the closest station to
    @param destination: a gepandas datafrane including the shapes (all weather stations)
    @return: match_value:   the closest station_id to the row
    """
    # 1 - create unary union
    dest_unary = destination["geometry"].unary_union
    # 2 - find closest point
    nearest_geom = nearest_points(row["geometry"], dest_unary)
    # 3 - Find the corresponding geom
    match_geom = destination.loc[destination.geometry == nearest_geom[1]]
    # 4 - get the corresponding value
    match_value = match_geom["station_id"].to_numpy()[0]
    return match_value


def Weather_Query_from_Raw(weather_df_Address, Weather_station_df_Address_csv, DF):
    """
    This function queries the weather information for a point and time from the raw weather df.

    Parameters
    ----------
    weather_df_Address : str
        The location of the weather df.
    Weather_station_df_Address_csv : str
        The location of the csv file that includes the location of the weather stations.
    DF : dataframes
        Is a df that includes, lat, long, an id, and time.

    Returns
    -------
    DF : dataframes
        weather information for each row of DF.

    """

    """
    #example of the input
    weather_df_Address='data/weather/weatherbit_1_hr_01-01-2017_06-06-2020.pk'
    Weather_station_df_Address_csv='data/weather/relevant_stations_df.csv'
    DF=pd.DataFrame({'Point':1,'time_local':pd.Timestamp(year=2019, month=1, day=1, hour=0) ,'Lon':[-90],'Lat':[-36.5]})
    """
    Geom = DF.apply(lambda ROW: sg.Point(ROW.Lon, ROW.Lat), axis=1)
    crs = {"init": "epsg:4326"}
    DF = gpd.GeoDataFrame(DF, geometry=Geom, crs=crs).copy()

    Weather_station_df = pd.read_csv(Weather_station_df_Address_csv)
    crs = {"init": "epsg:4326"}
    Weather_station_gdf = (
        gpd.GeoDataFrame(
            Weather_station_df,
            geometry=gpd.points_from_xy(
                Weather_station_df["lon"], Weather_station_df["lat"]
            ),
            crs=crs,
        )
    ).copy()
    DF["Nearest_Weather_Station"] = DF.apply(
        calculate_nearest, destination=Weather_station_gdf, axis=1
    )

    weather_df = pd.read_pickle(weather_df_Address)
    weather_df["time_local"] = pd.to_datetime(
        weather_df["timestamp_local"], format="%Y-%m-%dT%H:%M:%S"
    )

    DF = pd.merge(
        DF,
        weather_df,
        how="left",
        left_on=["time_local", "Nearest_Weather_Station"],
        right_on=["time_local", "station_id"],
    )

    return DF


def Read_Weather(weather_df_Address, MetaData):
    """
    Reading and cleaning the weather dataframe and adding some features to it
    @param weather_df_Address: the local address of the incident file
    @param MetaData:
    @return: the cleaned weather dataframe
    """
    # Define the features you want
    Weather_Feature_List = [
        "timestamp_local",
        "timestamp_utc",
        "temp",
        "wind_spd",
        "vis",
        "precip",
        "snow",
        "station_id",
    ]  #'weather.description'

    weather_df = pd.read_pickle(weather_df_Address)
    # weather_df=pd.read_pickle(MetaData['weather_df_Address'])
    weather_df = weather_df[Weather_Feature_List]
    weather_df["time_local"] = pd.to_datetime(
        weather_df["timestamp_local"], format="%Y-%m-%dT%H:%M:%S"
    )
    weather_df["time_utc"] = pd.to_datetime(
        weather_df["timestamp_utc"], format="%Y-%m-%dT%H:%M:%S"
    )
    weather_df.drop(["timestamp_local"], axis=1)
    # weather_df=weather_df[(weather_df['time_local']>= MetaData['Beg_Time']) & (weather_df['time_local']<= MetaData['End_Time'])  ]
    weather_df = weather_df.reset_index().drop("index", axis=1)
    weather_df["temp"] = weather_df["temp"].astype(float)
    weather_df["wind_spd"] = weather_df["wind_spd"].astype(float)
    weather_df["vis"] = weather_df["vis"].astype(float)
    weather_df["precip"] = weather_df["precip"].astype(float)

    weather_df_cleaned = weather_df.copy()
    weather_df_cleaned["year"] = weather_df_cleaned["time_local"].dt.year
    weather_df_cleaned["month"] = weather_df_cleaned["time_local"].dt.month
    weather_df_cleaned["day"] = weather_df_cleaned["time_local"].dt.day
    weather_df_cleaned["hour"] = weather_df_cleaned["time_local"].dt.hour
    weather_df_cleaned["window"] = (
        np.floor(
            (
                weather_df_cleaned["time_local"].dt.hour
                + weather_df_cleaned["time_local"].dt.minute / 60
            )
            / MetaData["Window_Size"]
        )
    ).astype("int64")

    if MetaData["Window_Size"] < 1:
        DF = pd.DataFrame()
        for Staion_ID in weather_df["station_id"].drop_duplicates():
            # print(Staion_ID)
            DF = DF.append(
                weather_df[weather_df["station_id"] == Staion_ID][
                    ["station_id", "time_utc", "time_local"]
                ]
                .set_index("time_utc")
                .resample(str(MetaData["Window_Size"] * 60) + "min")
                .ffill()
            )  # .ffill())
        DF = DF.reset_index()
        DF = DF.sort_values("time_utc")
        # DF[ DF['station_id']=='999999-63894'].head(20)
        weather_df = weather_df.sort_values("time_utc")
        weather_df = pd.merge_asof(
            DF,
            weather_df,
            on="time_utc",
            direction="nearest",
            by=["station_id", "time_local"],
            tolerance=pd.Timedelta("1h"),
        )
        weather_df["time_local"] = weather_df["time_local"] + pd.to_timedelta(
            weather_df["time_utc"].dt.minute, unit="minute"
        )
        # weather_df[weather_df['station_id']=='999999-63894'].head(20)

    weather_df["year"] = weather_df["time_local"].dt.year
    weather_df["month"] = weather_df["time_local"].dt.month
    weather_df["day"] = weather_df["time_local"].dt.day
    weather_df["hour"] = weather_df["time_local"].dt.hour
    weather_df["window"] = (
        np.floor(
            (weather_df["time_local"].dt.hour + weather_df["time_local"].dt.minute / 60)
            / MetaData["Window_Size"]
        )
    ).astype("int64")

    weather_df = weather_df.reset_index().drop("index", axis=1)

    if 'timestamp_local' in weather_df.columns:
        weather_df = weather_df.drop('timestamp_local', axis=1)
    if 'timestamp_utc' in weather_df.columns:
        weather_df = weather_df.drop('timestamp_utc', axis=1)


    weather_df_agg = weather_df.groupby(
        ["year", "month", "day", "window", "station_id"]
    ).agg(MetaData["Agg_Dic_Weather"])
    weather_df_agg.columns = [
        "_".join(col).strip() if col[-1] != "first" else col[0]
        for col in weather_df_agg.columns.values
    ]
    weather_df_agg = weather_df_agg.sort_values(["time_local", "station_id"])
    weather_df_agg = weather_df_agg.reset_index()
    return weather_df_agg, weather_df_cleaned


def Prepare_Weather(weather_df_Address=None, MetaData=None):
    """
    This function conducts preprocessing and cleaning analyses on the weather data set. It also adds/removes some features.

    Input
    ----------
    weather_df : String or DataFrame, optional
        The location of weather DataFrame or the DataFrame itself. If user provides None, the code automatically looks at the default location for this file. The default is
        MetaData : Dictionary
            It includes the meta data. The default is None.

    Returns
    -------
    weather_df : TYPE
        This is the cleaned version of the weather data set in a DataFrame format.

    """
    MetaData["Agg_Dic_Weather"] = {
        "time_local": ["first"],
        "time_utc": ["first"],
        "temp": ["min", "max", "mean"],
        "wind_spd": ["min", "max", "mean"],
        "vis": ["min", "max", "mean"],
        "precip": ["min", "max", "mean"],
    }

    start_time = time.time()
    old_df = pd.read_pickle(weather_df_Address)
    old_df.columns = old_df.columns.str.lower().str.replace(' ', '_')
    _ = pd.to_datetime(old_df["timestamp_local"], format="%Y-%m-%dT%H:%M:%S")
    Beg_of_data = _.max()
    End_of_data = _.max()
    if MetaData["End_Time"].date() > End_of_data.date():
        print(
            "The end of the requested date is beyond the available data. The module tries to collect data from weatherbit.io"
        )
        Weather_station_df = pd.read_csv(MetaData["Weather_station_df_Address_csv"])
        station_id_list = Weather_station_df["station_id"].tolist()
        new_df = call_weatherbit_api(
            MetaData["End_Time"].date(),
            End_of_data.date(),
            station_id_list,
            MetaData["weatherbit_apikey"],
        )
        df = merge_old_and_new_data(old_df, new_df)
        df.pickle('/'.join(weather_df_Address).split('/')[0:-1] + "weather_1h_" + Beg_of_data.date().strftime("%Y-%m-%d") + "_to_" + End_of_data.date().strftime("%Y-%m-%d") +'pkl.')
        weather_df_agg, weather_df_cleaned = Read_Weather(
            weather_df_Address=weather_df_Address, MetaData=MetaData
        )

    else:
        weather_df_agg, weather_df_cleaned = Read_Weather(
            weather_df_Address=weather_df_Address, MetaData=MetaData
        )

    Save_DF(
        weather_df_cleaned,
        MetaData["destination"]
        + "weather/"
        + "weather_cleaned_1h_"
        + Beg_of_data.date().strftime("%Y-%m-%d")
        + "_to_"
        + End_of_data.date().strftime("%Y-%m-%d"),
        Format="pkl",
        gpd_tag=False,
    )  # weather_df = pd.read_pickle('data/cleaned/weather.pkl')
    # Save_DF(weather_df_agg, MetaData['destination']+'weather/'+'weather_aggregated',Format='pkl', gpd_tag=False)  #weather_df = pd.read_pickle('data/cleaned/weather.pkl')
    print("Reading Weather Time: --- %s seconds ---" % (time.time() - start_time))
    return weather_df_cleaned

    # df=pd.read_pickle('D:\\inrix_new\\inrix_pipeline\\data\\cleaned\Line\\weather\\weather_cleaned.pkl')
    # df=pd.read_pickle('D:\\inrix_new\\inrix_pipeline\\data\\cleaned\Line\\weather\\weather_aggregated.pkl')
