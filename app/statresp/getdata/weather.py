# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:22:15 2021

@author: vaziris
"""


import json
from datetime import timedelta, date
from pandas.io.json import json_normalize
import requests
import pandas as pd
import numpy as np

# Initialize empty Dataframe

def get_weather_weatherbit(metadata,station_id_list):
    '''
    station_id_list=['723280-03894',
     '722322-99999',
     '724117-63802',
     '721031-00348',
     '723346-03811',
     '722279-53852',
     '723250-03847',
     '746720-99999',
     '723347-03809',
     '720924-99999',
     '723274-99999',
     '720353-63875',
     '999999-53877',
     '999999-23803',
     '720168-99999',
     '746716-93808',
     '723350-13877',
     '722165-63808',
     '724270-53868',
     '723249-99999',
     '723240-13882',
     '720768-00274',
     '723235-13896',
     '999999-63868',
     '722781-99999',
     '723270-13897',
     '720171-99999',
     '720907-99999',
     '720974-99999',
     '722154-53885',
     '723271-99999',
     '720307-63804',
     '720447-00437',
     '999999-63855',
     '720379-63882',
     '723273-13827',
     '722177-63811',
     '999999-63894',
     '723260-13891',
     '723264-99999']
    ''' 
    #key = 'b0b59fd3cba74279b744b036e87a7b5c'
    key  = metadata['weatherbit_key']
    
    
    #inrix_df=pd.read_pickle(metadata['inrix_pickle_address'])
    #station_id=inrix_df[inrix_df['MyGrouping_3_x'].isin(df[metadata['unit_name']].tolist())]['Nearest_Weather_Station'].drop_duplicates().tolist()

    # Initialize empty Dataframe
    df_weather = pd.DataFrame()
    # Time window
    #Beg = df['time_local'].min()
    #End = df['time_local'].max()+pd.DateOffset(hours=metadata['window_size']/3600)

    Beg = metadata['start_time_predict']
    End = metadata['end_time_predict']+pd.DateOffset(hours=metadata['window_size']/3600)

    # For each station in our station list
    for station_id in station_id_list:

    
        resp = requests.get('https://api.weatherbit.io/v2.0/forecast/hourly?station={}&key={}&hours={}'.format(station_id,key,24))
        #resp = requests.get('https://api.weatherbit.io/v2.0/forecast/hourly?city={}&key={}&hours={}'.format('Nashville,TN',key,48))
        resp_json = resp.json()
        data = resp_json['data']
    
        # Convert the JSON into a DF
        new_entry = pd.json_normalize(data)
        new_entry['GPS Coordinate Latitude'] = resp_json['lat']
        new_entry['GPS Coordinate Longitude'] = resp_json['lon']
        new_entry['city_name'] = resp_json['city_name']
        new_entry['state_code'] = resp_json['state_code']
        new_entry['country_code'] = resp_json['country_code']
        new_entry['timezone'] = resp_json['timezone']
        try:
            new_entry['sources'] = [resp_json['sources']]*len(new_entry)
        except:
            new_entry['sources'] = None 
        
        try:
            new_entry['station_id'] = resp_json['station_id']
        except:
            new_entry['station_id'] = None 
    
        # Edge case (first instance)
        if not df_weather.empty:
          df_weather = df_weather.append(new_entry,ignore_index=True)
        else:
          df_weather = new_entry


        
        Weather_Feature_List=['timestamplocal','timestamputc','temp','windspd','vis','precip','snow','stationid']
        weather_df=df_weather[Weather_Feature_List]
        weather_df['time_local']=pd.to_datetime(weather_df['timestamplocal'],format="%Y-%m-%dT%H:%M:%S")
        weather_df['time_utc']=pd.to_datetime(weather_df['timestamputc'],format="%Y-%m-%dT%H:%M:%S")
        weather_df.drop(['timestamplocal'], axis=1)
        #weather_df=weather_df[(weather_df['time_local']>= Beg) & (weather_df['time_local']< End)  ]
        weather_df=weather_df.reset_index().drop('index',axis=1)
        #%%
        weather_df['temp'] = weather_df['temp'].astype(float)
        weather_df['windspd'] = weather_df['windspd'].astype(float)
        weather_df['vis'] = weather_df['vis'].astype(float)
        weather_df['precip'] = weather_df['precip'].astype(float)
        
        
        weather_df["year"]  =weather_df['time_local'].dt.year
        weather_df["month"] =weather_df['time_local'].dt.month
        weather_df["day"]   =weather_df['time_local'].dt.day
        weather_df["hour"]  =weather_df['time_local'].dt.hour
        weather_df['window']=(np.floor((weather_df['time_local'].dt.hour+weather_df['time_local'].dt.minute/60)/metadata['Window_Size'])).astype('int64')   
        
          
        Agg_Dic_Weather={'time_local':['first'],
                        'time_utc':['first'],
                        'temp':['min','max','mean'],
                        'windspd':['min','max','mean'],
                        'vis':['min','max','mean'] ,
                        'precip':['min','max','mean']}   
        
        weather_df_agg=weather_df.groupby(['year','month','day','window','stationid']).agg(Agg_Dic_Weather)
        weather_df_agg.columns = ['_'.join(col).strip() if col[-1]!='first' else col[0] for col in weather_df_agg.columns.values]
        weather_df_agg=weather_df_agg.sort_values(['time_local','stationid'])
        weather_df_agg=weather_df_agg.reset_index()    
        weather_df_agg=weather_df_agg[metadata['features_weather']]
        weather_df_agg=weather_df_agg.reanme(columns={'stationid':'Nearest_Weather_Station'})
        return weather_df_agg


