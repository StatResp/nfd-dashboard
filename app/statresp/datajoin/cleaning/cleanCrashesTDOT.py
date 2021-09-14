import pandas as pd
import numpy as np
import shapely.geometry as sg
import pytz
from datetime import datetime
import os
# from data_join_pipeline.cleanTDOT.timeFloatConverter import convertTime
from statresp.datajoin.cleaning.timeFloatConverter import convertTime
from statresp.datajoin.cleaning.utils import Save_DF,Read_DF




def get_combined(combinedcsv: pd.DataFrame) -> pd.DataFrame:
    
    combinedcsv.columns=combinedcsv.columns.str.lower().str.replace(' ','_')
    combinedcsv["point"] = combinedcsv.apply(lambda row: sg.Point(row["gps_coordinate_longitude"],
                                                                  row["gps_coordinate_latitude"]),
                                             axis=1)
    combinedcsv.point = combinedcsv.point.astype(str)
    return combinedcsv





def clean_null_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    removes null points
    """
    df = df[~(df["gps_coordinate_latitude"].isnull() & df["gps_coordinate_longitude"].isnull())]
    return df[~(df["gps_coordinate_latitude"].isnull() | df["gps_coordinate_longitude"].isnull())]


def clean_gps_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(lambda row: fix_gps_coordinates(row), axis=1)
    df = df.dropna(subset=["gps_coordinate_latitude", "gps_coordinate_longitude"])
    df["geometry"] = df.apply(lambda row: sg.Point(row["gps_coordinate_longitude"],
                                                row["gps_coordinate_latitude"]),
                           axis=1)
    
    df["geometry_str"] = df['geometry'].apply(str)
    return df


def point_in_tn(lat: float, lon: float) -> bool:
    LEFT_LONG_TN = -90.5
    RIGHT_LONG_TN = -81.5
    TOP_LAT_TN = 36.7
    BOT_LAT_TN = 34.9
    if (BOT_LAT_TN < lat < TOP_LAT_TN) and (LEFT_LONG_TN < lon < RIGHT_LONG_TN):
        return True
    else:
        return False


def fix_gps_coordinates(row):
    """
    Used in a lambda function to convert rows of a DataFrame to the right
    coordinates
    """
    lat = row["gps_coordinate_latitude"]
    lon = row["gps_coordinate_longitude"]
    if point_in_tn(lat, lon):
        pass
    elif point_in_tn(-lat,-lon):
        row["gps_coordinate_latitude"] = -lat
        row["gps_coordinate_longitude"] = -lon
    elif point_in_tn(-lat, lon):
        row["gps_coordinate_latitude"] = -lat
    elif point_in_tn(lat, -lon):
        row["gps_coordinate_longitude"] = -lon
    elif point_in_tn(lon, lat):
        row["gps_coordinate_latitude"] = lon
        row["gps_coordinate_longitude"] = lat
    elif point_in_tn(-lon, lat):
        row["gps_coordinate_latitude"] = -lon
        row["gps_coordinate_longitude"] = lat
    elif point_in_tn(lon, -lat):
        row["gps_coordinate_latitude"] = lon
        row["gps_coordinate_longitude"] = -lat
    elif point_in_tn(-lon, -lat):
        row["gps_coordinate_latitude"] = -lon
        row["gps_coordinate_longitude"] = -lat
    else:
        row["gps_coordinate_latitude"] = None
        row["gps_coordinate_longitude"] = None
    return row


def format_time(df: pd.DataFrame) -> pd.DataFrame:
    df["timeLocal"] = df.apply(lambda row: add_local_timezone(row), axis=1)
    df["timeUTC"] = df.apply(lambda row: row["timeLocal"].astimezone("utc"), axis=1)
    return df


def add_local_timezone(row):
    """
    A side effect is county's are lowercase
    """
    row["county"] = row["county"].lower()
    central=['bedford', 'benton', 'bledsoe', 'cannon', 'carroll', 'cheatham', 'chester', 
             'clay', 'coffee', 'crockett', 'cumberland', 'davidson', 'decatur', 'dekalb', 
             'dickson', 'dyer', 'fayette', 'fentress', 'franklin', 'gibson', 'giles', 
             'grundy', 'hardeman', 'hardin', 'haywood', 'henderson', 'henry', 'hickman', 
             'houston', 'humphreys', 'jackson', 'lake', 'lauderdale', 'lawrence', 'lewis', 
             'lincoln', 'mcnairy', 'macon', 'madison', 'marion', 'marshall', 'maury', 
             'montgomery', 'moore', 'obion', 'overton', 'perry', 'pickett', 'putnam', 
             'robertson', 'rutherford', 'sequatchie', 'shelby', 'smith', 'stewart', 
             'sumner', 'tipton', 'trousdale', 'van', 'warren', 'wayne', 'weakley', 
             'white', 'williamson', 'wilson']
    eastern= ['anderson', 'blount', 'bradley', 'campbell', 'carter', 'claiborne', 
              'cocke', 'grainger', 'greene', 'hamblen', 'hamilton', 'hancock', 'hawkins', 
              'jefferson', 'johnson', 'knox', 'loudon', 'mcminn', 'meigs', 'monroe', 
              'morgan', 'polk', 'rhea', 'roane', 'scott', 'sevier', 'sullivan', 'unicoi', 
              'union', 'washington']
    if row["county"] in central:
        timeZone = pytz.timezone('CST6CDT')
    elif row["county"] in eastern:
        timeZone = pytz.timezone('EST5EDT')
    return timeZone.localize(row["time_of_crash"])

def get_incident_datetime(x):
    try:
        return datetime.strptime(x,'%d-%b-%y %H:%M').strftime('%Y-%m-%d %H:%M:%S')
    except:
        return None
    
def get_incident_timestamp(x):
    try:
        return datetime.strptime(x,'%Y-%m-%d %H:%M:%S').timestamp()
    except:
        return None

def format_to_old_pkl():
    df = pd.read_csv("historical-crashes/clean_historical_crashes.csv")
    # ... do stuff here


def stats_by_year(df: pd.DataFrame, MetaData: dict) -> None:
    stats = pd.DataFrame()
    basicStats = pd.DataFrame()
    for year in df["year_of_crash"].sort_values(ascending=True).unique():
        ydf = df.loc[df["year_of_crash"]==year]
        ysize = ydf.shape[0]
        ylatlongnull = ydf[ydf["gps_coordinate_latitude"].isnull() & ydf["gps_coordinate_longitude"].isnull()]
        ydf = ydf[~(ydf["gps_coordinate_latitude"].isnull() & ydf["gps_coordinate_longitude"].isnull())]
        ylatnull = ydf[ydf["gps_coordinate_latitude"].isnull()]
        ylongnull = ydf[ydf["gps_coordinate_longitude"].isnull()]
        ydf = ydf[~(ydf["gps_coordinate_latitude"].isnull() | ydf["gps_coordinate_longitude"].isnull())]
        
        countTN = 0
        countFlipLat = 0
        countFlipLong = 0
        countFlipBoth = 0
        countBad = 0
        countZero = 0
        wrongLatLongOrder = 0
        wlloFlipLat = 0
        wlloFlipLong = 0
        wlloFlipBoth = 0

        for index, row in ydf.iterrows():
            lat = row["gps_coordinate_latitude"]
            lon = row["gps_coordinate_longitude"]
            
            if point_in_tn(lat, lon):
                countTN = countTN + 1
            elif point_in_tn(-lat,-lon):
                countFlipBoth = countFlipBoth + 1
            elif point_in_tn(-lat, lon):
                countFlipLat = countFlipLat + 1
            elif point_in_tn(lat, -lon):
                countFlipLong = countFlipLong + 1
            elif point_in_tn(lon, lat):
                wrongLatLongOrder = wrongLatLongOrder + 1
            elif point_in_tn(lon, -lat):
                wlloFlipLat = wlloFlipLat + 1
            elif point_in_tn(-lon, lat):
                wlloFlipLong = wlloFlipLong + 1
            elif point_in_tn(-lon, -lat):
                wlloFlipBoth = wlloFlipBoth + 1
            else:
                if lat == 0 or lon == 0:
                    countZero = countZero + 1
                else: 
                    countBad = countBad + 1
        try: 
            countCorrupted = countBad + countZero + ylatlongnull.shape[0] + ylatnull.shape[0] + \
                             ylongnull.shape[0]
            new_row = {"year":year,
                    "count":ysize,
                    "%bothLatLongNull":(ylatlongnull.shape[0]/ysize) * 100,
                    "countLatLongNull":ylatlongnull.shape[0],
                    "%latNull":(ylatnull.shape[0]/ysize) * 100,
                    "countLatNull":ylatnull.shape[0],
                "%longNull":ylongnull.shape[0]/ysize * 100,
                "countLongNull":ylongnull.shape[0],
                "pointInTN":countTN,
                "bothLatLongFlipped":countFlipBoth,
                "latFlipped":countFlipLat,
                "longFlipped":countFlipLong,
                "countOutOfTN":countBad,
                "countZero":countZero,
                "%bad":(countCorrupted) / ysize * 100,
                "wrongLatLongOrder":wrongLatLongOrder, 
                "wrongOrderFlipLat":wlloFlipLat,
                "wrongOrderFlipLong":wlloFlipLong,
                "wrongOrderFlipBoth":wlloFlipBoth
        }
            basic_row = {"year":year,
                        "countTotalCrashes":ysize,
                        "countCorrupted":countCorrupted}
            stats = stats.append(new_row, ignore_index=True)
            basicStats = basicStats.append(basic_row, ignore_index=True)
        except:
            print(f"exception doing stats in year {year}")

        Address=('/').join(MetaData['incident_df_Address'].split('/')[0:-1])+'/logs'
        if not os.path.exists(Address):
            os.makedirs(Address)
        stats.to_csv(Address+"/crashDataStats.csv")
        basicStats.to_csv(Address+"/basicCrashDataStats.csv")





COMBINED_HISTORICAL_INCIDENTS_PATH =  'data/raw/incidents_tdot'+"/all_crash.csv"#  "tdot-incidents-data/all_crash_2.csv"
PKL_NAME = 'data/raw/incidents_tdot'+ 'tdot_testing_incidents_2006-2021_allTN.pkl'
def get_cleaned_picke_form_all_csvs(MetaData):
    
    import glob


    all_files = glob.glob(r'{}'.format(MetaData['incident_df_Address']+"/c*.csv"))
    
    
    all_files
    df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
    df   = pd.concat(df_from_each_file, ignore_index=True)
    COMBINED_HISTORICAL_INCIDENTS_PATH = ('/').join(MetaData['incident_df_Address'].split('/')[0:-1])+"/all_crash.csv"#  
    df.to_csv(COMBINED_HISTORICAL_INCIDENTS_PATH)    
    
    df = get_combined(df)
    #COMBINED_HISTORICAL_INCIDENTS_PATH = ('/').join(MetaData['incident_df_Address'].split('/')[0:-1])+"/all_crash.csv"#  "tdot-incidents-data/all_crash_2.csv"
    ## df.to_csv(COMBINED_HISTORICAL_INCIDENTS_PATH)


    stats_by_year(df,MetaData)


    df = clean_null_point(df)
    df = clean_gps_coordinates(df)
    df = df.sort_values(by=["case_number","time_of_crash"])
    df = df.drop_duplicates(subset=['case_number'], keep='last')
    df = df.fillna({'total_killed':-1,
                    "total_injured":-1,
                    "total_incapcitating_injuries":-1,
                    "total_other_injuries":-1,
                    "total_vehicles":-1,
                    "year_of_crash":-1})
    df.total_killed = df.total_killed.astype(int)
    df.total_injured = df.total_injured.astype(int)
    df.total_incapcitating_injuries = df.total_incapcitating_injuries.astype(int)
    df.total_other_injuries= df.total_other_injuries.astype(int)
    df.total_vehicles= df.total_vehicles.astype(int)
    df.year_of_crash = df.year_of_crash.astype(int)
    df["time_of_crash"] = df["time_of_crash"].apply(convertTime)
    df["time"] = df["date_of_crash"] + " " + df["time_of_crash"]
    df["time"] = df["time"].apply(get_incident_datetime)
    df["timestamp"] = df["time"].apply(get_incident_timestamp)
    #df.to_csv("cleanTDOT/historical_crashes.csv", index=False)
    return df
    


