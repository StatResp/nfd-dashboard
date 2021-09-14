# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 20:21:11 2021

@author: vaziris
"""
from statresp.getdata.weather import get_weather_weatherbit
from datetime import timedelta
from itertools import product
import pandas as pd
import numpy as np

def MainFrame(List_segments,Beg,End,Win_Hour,metadata,Other_Columns=None ):
        """
        Reading and cleaning the incident dataframe and adding some features to it
        @param List_segments: List of segments we want to use as index
        @param Beg: the beggining fo the time range we want to use as index
        @param End: the end fo the time range we want to use as index
        @param Win_Hour: the time interval for indexing
        @return Seg_win_DF: a dataframe including features regarding spatial temporal resolution
        """
  
        #Beg=pd.Timestamp(year=2017, month=1, day=1, hour=0)
        #End=pd.Timestamp(year=2020, month=5, day=1, hour=0)   
        #Beg=MetaData['Beg_Time']
        #End=MetaData['End_Time']
        #Win_Hour=1
        Dask_Tag=False

        #Making the big Regression DF__________________________________________________
        #creating the Window_series
        Window_Num=np.ceil((End-Beg).total_seconds()/(Win_Hour*3600))
        Window_series=Beg+timedelta(seconds=(Win_Hour*3600))*np.arange(Window_Num)
        Window_series
        #Time_id=range(len(Window_series))
        #0 creating the Seg_win_DF DF
        Seg_win_DF=pd.DataFrame()
        #Seg_win_DF=pd.DataFrame(columns=['time_local','Time_id','Space_id','ID_Seg_win_DF','year','month','day','window','XDSegID']+Other_Columns)
        Seg_win_DF['time_local']=Window_series
        #Seg_win_DF["Time_id"]=range(len(Window_series))
        Seg_win_DF["year"]=Seg_win_DF['time_local'].dt.year
        Seg_win_DF["month"]=Seg_win_DF['time_local'].dt.month
        Seg_win_DF["day"]=Seg_win_DF['time_local'].dt.day
        Seg_win_DF['window']=(np.floor((Seg_win_DF['time_local'].dt.hour+Seg_win_DF['time_local'].dt.minute/60)/Win_Hour)).astype('int64')   
        Seg_win_DF["dayofweek"]=Seg_win_DF['time_local'].dt.dayofweek
        Seg_win_DF['isweekend'] = (Seg_win_DF["dayofweek"]// 5 == 1).astype(int)
        
        
        
        Seg_win_DF.dtypes
        #SegIDs
        
        #List of segments we want to use as index
        Seg_win_DF=pd.concat([Seg_win_DF]*len(List_segments), ignore_index=True)
        Seg_win_DF[metadata['unit_name']]=np.repeat(List_segments,Window_Num)
        #Seg_win_DF['Space_id']=np.repeat(range(len(List_segments)),Window_Num)
        Seg_win_DF=Seg_win_DF.sort_values('time_local').reset_index().drop('index',axis=1)
        #Seg_win_DF['ID_Seg_win_DF']=range(len(Seg_win_DF))
        Seg_win_DF
        
        Seg_win_DF[metadata['unit_name']]=Seg_win_DF[metadata['unit_name']].astype('int64') 
        
        return Seg_win_DF

def get_inrix(metadata,df_train):
    #inrix_df=pd.read_pickle(metadata['inrix_pickle_address'])
    #inrix_df=inrix_df[inrix_df['MyGrouping_3_x'].isin(metadata['segment_list_pred'])]
    inrix_df=df_train[df_train[metadata['unit_name']].isin(metadata['segment_list_pred'])][metadata['features_static']+[metadata['unit_name']]+['Nearest_Weather_Station','cluster_label']].drop_duplicates()
    return inrix_df
def get_weather(metadata,df_train,station_id_list) :
    try:
        weather_df=get_weather_weatherbit(metadata,station_id_list)
    except:
        weather_df=df_train[['Nearest_Weather_Station']+metadata['features_weather']].groupby('Nearest_Weather_Station').mean()
        Beg = pd.Timestamp(metadata['start_time_predict'])
        End = pd.Timestamp(metadata['end_time_predict'])+pd.DateOffset(hours=metadata['window_size']/3600)
        (End-Beg).seconds/3600
        Window_Num=np.ceil((((End-Beg).seconds)/metadata['window_size']))
        Window_series=Beg+timedelta(seconds=(metadata['window_size']))*np.arange(Window_Num)
        weather_df_base=pd.DataFrame(list(product(station_id_list, Window_series)), columns=['Nearest_Weather_Station', 'time_local'])
        weather_df=pd.merge(weather_df_base, weather_df, left_on='Nearest_Weather_Station', right_on='Nearest_Weather_Station', how='left')
    return weather_df


def get_traffic(metadata,df_train):
    df_traffic=df_train[df_train[metadata['unit_name']].isin(metadata['segment_list_pred'])][[metadata['unit_name']]+metadata['features_traffic']].groupby(metadata['unit_name']).mean()
    #df_traffic['congestion_mean']=(df_traffic['reference_speed_mean']-df_traffic['average_speed_mean'])/df_traffic['reference_speed_mean']
    #df_traffic['congestion_mean'] = df_traffic['congestion_mean'].mask(df_traffic['congestion_mean']<0, 0)
    return df_traffic.reset_index()

def get_incident(metadata,df_train):
    df_incident=df_train[df_train[metadata['unit_name']].isin(metadata['segment_list_pred'])][[metadata['unit_name']]+metadata['features_incident']].groupby(metadata['unit_name']).mean()
    return df_incident.reset_index()
    

def get_future_data(metadata,df_train):
    
    inrix_df=get_inrix(metadata,df_train)
    weather_df=get_weather(metadata,df_train,inrix_df['Nearest_Weather_Station'].unique().tolist())
    df_traffic= get_traffic(metadata,df_train)
    df_incident= get_incident(metadata,df_train)
    
    df_predict=MainFrame(inrix_df[metadata['unit_name']].unique().tolist(),metadata['start_time_predict'],metadata['end_time_predict'],metadata['window_size']/3600,metadata,Other_Columns=None )
    df_predict=pd.merge(df_predict, inrix_df, left_on=metadata['unit_name'], right_on=metadata['unit_name'], how='left')
    df_predict=pd.merge(df_predict, weather_df, left_on=['time_local','Nearest_Weather_Station'], right_on=['time_local','Nearest_Weather_Station'], how='left')
    df_predict=pd.merge(df_predict, df_traffic, left_on=metadata['unit_name'], right_on=metadata['unit_name'], how='left')
    df_predict=pd.merge(df_predict, df_incident, left_on=metadata['unit_name'], right_on=metadata['unit_name'], how='left')
    #adding inrix
    return df_predict