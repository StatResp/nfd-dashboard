# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:59:55 2020

@author: vaziris
"""
#Importing Packages
from pprint import pprint
import pandas as pd
import numpy as np
import pickle
import _pickle as cPickle
import os
from copy import deepcopy
import bz2
from datetime import datetime
from datetime import timedelta
import geopandas as gpd
import pytz
import shapely.geometry as sg
import json
#import pygeoj
import pyproj
import seaborn as sns
import matplotlib.pyplot as plt
import dask.dataframe as dd
#import swifter
#import dataextract
from datetime import timezone, datetime
import time
from utils import Save_DF,Read_DF







#%%
def MainFrame(List_segments,Beg,End,Win_Hour ):
        """
        Reading and cleaning the incident dataframe and adding some features to it
        @param List_segments: List of segments we want to use as index
        @param Beg: the beggining fo the time range we want to use as index
        @param End: the end fo the time range we want to use as index
        @param Win_Hour: the time interval for indexing
        @return Seg_win_DF: a dataframe including features regarding spatial temporal resolution
        """



        #Making the big Regression DF__________________________________________________
        #creating the Window_series
        Window_Num=np.ceil((End-Beg).total_seconds()/(Win_Hour*3600))
        Window_series=Beg+timedelta(seconds=(Win_Hour*3600))*np.arange(Window_Num)
        Window_series
        #Time_id=range(len(Window_series))
        #0 creating the Seg_win_DF DF
        #Seg_win_DF=pd.DataFrame(Window_series,Time_id)
        Seg_win_DF=pd.DataFrame()
        Seg_win_DF['time_local']=Window_series
        Seg_win_DF["Time_id"]=range(len(Window_series))
        Seg_win_DF["year"]=Seg_win_DF['time_local'].dt.year
        Seg_win_DF["month"]=Seg_win_DF['time_local'].dt.month
        Seg_win_DF["day"]=Seg_win_DF['time_local'].dt.day
        Seg_win_DF['window']=(np.floor((Seg_win_DF['time_local'].dt.hour+Seg_win_DF['time_local'].dt.minute/60)/Win_Hour)).astype('int64')   

        Seg_win_DF
        #SegIDs
        
        #List of segments we want to use as index
        Seg_win_DF=pd.concat([Seg_win_DF]*len(List_segments), ignore_index=True)
        Seg_win_DF['XDSegID']=np.repeat(List_segments,Window_Num)
        Seg_win_DF['Space_id']=np.repeat(range(len(List_segments)),Window_Num)
        Seg_win_DF=Seg_win_DF.sort_values('time_local').reset_index().drop('index',axis=1)
        Seg_win_DF['ID_Seg_win_DF']=range(len(Seg_win_DF))
        Seg_win_DF
        
        return Seg_win_DF






def All_DF_Maker(inrix_df=None,weather_df=None,incident_df=None,traffic_df=None,List_segments=None, MetaData=None ):
        """
        This function brings all the DFs together
        @param inrix_df:
        @param weather_df:    
        @param incident_df:    
        @param traffic_df:
        @param MetaData:    
        @return Seg_win_DF: a dataframe including features regarding spatial temporal resolution
        """
    

        inrix_df=Read_DF(inrix_df=inrix_df,Reading_Tag='inrix_df',MetaData=MetaData )
        weather_df=Read_DF(weather_df=weather_df, Reading_Tag='weather_df',MetaData=MetaData )
        incident_df=Read_DF(incident_df=incident_df, Reading_Tag='incident_df',MetaData=MetaData )
        traffic_df=Read_DF(traffic_df=traffic_df, Reading_Tag='traffic_df',MetaData=MetaData )    

        
        #%%Preparation
        #List_segments=incident_df['XDSegID'].unique()
        #List_segments=set(traffic_df[traffic_df['xd_id'].notna()]['xd_id'].astype(int).unique().tolist()+ incident_df[incident_df['XDSegID'].notna()]['XDSegID'].astype(int).unique().tolist())
        #Windows: Defining the width of the time window, unit is hour
        MetaData['Window_Size']
        #creating the Window_series
        print('Time Range:', incident_df['time_local'].min(), ' to ' , incident_df['time_local'].max() )
        #Beg_Time=incident_df['time_local'].min()+timedelta(days=-1)
        #End_Time=incident_df['time_local'].max()+timedelta(days=1)
        #Beg_Time=pd.Timestamp(year=2017, month=1, day=1, hour=0)
        #End_Time=pd.Timestamp(year=2020, month=5, day=2, hour=0)
        Beg_Time=MetaData['Beg_Time']
        End_Time=MetaData['End_Time']
        print('Extracted Time: ', Beg_Time, ' to ', End_Time)
        Pivot_Columns=['year','month','day','window','XDSegID','Nearest_Weather_Station']
        
        #%%inrix
        List_Features_Inrix=['geometry','PreviousXD','NextXDSegI','FRC','County_inrix','Miles','Lanes',
                             'SlipRoad','XDGroup','iSF_length','iSF_length_min','grouped_XDSegID','grouped_XDSegID_id','grouped_XDSegID_id_miles']
        inrix_df=inrix_df[Pivot_Columns[4:6]+List_Features_Inrix]
        if set(Pivot_Columns[4:6]).issubset(inrix_df.columns):
            print('All required columns exist in inrix_df!')
        else:
            print('At leaset one colunm in inrix_df missing!')
        #%%weather
        List_Features_Weather=['temp_min','temp_max','temp_mean',  #'time_local',
                               'windspd_min','windspd_max','windspd_mean',
                               'vis_min','vis_max','vis_mean',
                               'precip_min','precip_max','precip_mean']
        weather_df=weather_df.rename(columns={"station_id": Pivot_Columns[5]})
        weather_df=weather_df[Pivot_Columns[0:4]+Pivot_Columns[5:]+List_Features_Weather]
        if set(Pivot_Columns[0:4]+Pivot_Columns[5:]).issubset(weather_df.columns):
            print('All required columns exist in weather_df!')
        else:
            print('At leaset one colunm in weather_df missing!')
        
        #%%incident
        #List_Features_Incident=['Incident_ID','Total Killed','Total Inj','Total Incap Injuries','Total Other Injuries','Total Veh']
        List_Features_Incident=['Incident_ID']
        incident_df=incident_df[Pivot_Columns[0:5]+List_Features_Incident]
        #incident_df=incident_df.rename(columns={"ID": "Incident_ID"})
        incident_df=incident_df[incident_df['XDSegID'].notna()]    
        if set(Pivot_Columns[0:5]).issubset(incident_df.columns):
            print('All required columns exist in incident_df!')
        else:
            print('At leaset one colunm in incident_dfis missing!')
        #%%traffic
        List_Features_Traffic=['congestion_min','congestion_max','congestion_mean','congestion_std',
                               'speed_mean', 'average_speed_mean', 'reference_speed_mean','cvalue_mean', 'confidence_score_mean']
        traffic_df=traffic_df.rename(columns={"xd_id": "XDSegID"})
        traffic_df=traffic_df[Pivot_Columns[0:5]+List_Features_Traffic]
        if set(Pivot_Columns[0:5]).issubset(traffic_df.columns):
            print('All required columns exist in traffic_df!')
        else:
            print('At leaset one colunm in traffic_df is missing!')
            
        #%%Making the Main DF
        Seg_win_DF=MainFrame(List_segments=List_segments,Beg=Beg_Time,End=End_Time,Win_Hour=MetaData['Window_Size']  )
        print('Main DF successfully created.')
        #%%Adding Traffic
        Seg_win_DF=pd.merge(Seg_win_DF,traffic_df, left_on=Pivot_Columns[0:5],right_on=Pivot_Columns[0:5], how='left' )
        del traffic_df 
        print('Traffic DF successfully added.')
        #%%Adding Inrix
        Seg_win_DF=pd.merge(Seg_win_DF,inrix_df, left_on=Pivot_Columns[4],right_on=Pivot_Columns[4], how='left' )
        del inrix_df
        print('Inrix DF successfully added.')
        #%%Adding Weather
        Seg_win_DF=pd.merge(Seg_win_DF,weather_df, left_on=Pivot_Columns[0:4]+Pivot_Columns[5:],right_on=Pivot_Columns[0:4]+Pivot_Columns[5:], how='left' )
        '''
        if MetaData['Window_Size']>=1:
            weather_df=weather_df.drop('time_local',axis=1)
            Seg_win_DF=pd.merge(Seg_win_DF,weather_df, left_on=Pivot_Columns[0:4]+Pivot_Columns[5:],right_on=Pivot_Columns[0:4]+Pivot_Columns[5:], how='left' )
        else:
            Seg_win_DF=pd.merge_asof(Seg_win_DF, weather_df, on="time_local",direction="nearest",
                by=Pivot_Columns[0:3]+Pivot_Columns[5:],
                tolerance=pd.Timedelta('0.5h')).drop('window_y',axis=1)
            Seg_win_DF=Seg_win_DF.rename(columns={Pivot_Columns[3]+'_x':Pivot_Columns[3]})
        '''
        del weather_df 
        print('Weather DF successfully added.')
        #%%Adding incident  
        Dic_for_Merge={}   #it will be the aggregation policy
        for i in Seg_win_DF.columns:
            if (i in Pivot_Columns[0:5])==False:
                Dic_for_Merge[i]= 'first'  
        Dic_for_Merge['Incident_ID']= 'count'   
        List_Features_Incident.remove('Incident_ID')
        while len(List_Features_Incident)>0:
            Dic_for_Merge[List_Features_Incident[0]]= 'sum'
            List_Features_Incident.remove(List_Features_Incident[0])

        '''
        Dic_for_Merge['Total Killed']= 'sum'
        Dic_for_Merge['Total Inj']= 'sum'
        Dic_for_Merge['Total Incap Injuries']= 'sum'
        Dic_for_Merge['Total Other Injuries']= 'sum'
        Dic_for_Merge['Total Veh']= 'sum'
        Dic_for_Merge['Total Incap Injuries']= 'sum'
        '''
        
        Dic_for_Merge
        #Merging  
        Seg_win_DF=pd.merge(Seg_win_DF,incident_df, left_on=Pivot_Columns[0:5],right_on=Pivot_Columns[0:5], how='left' ).reset_index()
        #Groupby: some cells may have multiple accidents so we should merge them
        del incident_df 
        Seg_win_DF=Seg_win_DF.groupby(Pivot_Columns[0:5]).agg(Dic_for_Merge)  
        Seg_win_DF=Seg_win_DF.rename(columns={"Incident_ID": "Total_Number_Incidents"})   
        
        Seg_win_DF=Seg_win_DF.reset_index()
        print('Incident DF successfully added.') 
        
        
        #adding the columns time and time_utc
        central=['bedford', 'benton', 'bledsoe', 'cannon', 'carroll', 'cheatham', 'chester', 'clay', 'coffee', 'crockett', 'cumberland', 'davidson', 'decatur', 'dekalb', 'dickson', 'dyer', 'fayette', 'fentress', 'franklin', 'gibson', 'giles', 'grundy', 'hardeman', 'hardin', 'haywood', 'henderson', 'henry', 'hickman', 'houston', 'humphreys', 'jackson', 'lake', 'lauderdale', 'lawrence', 'lewis', 'lincoln', 'mcnairy', 'macon', 'madison', 'marion', 'marshall', 'maury', 'montgomery', 'moore', 'obion', 'overton', 'perry', 'pickett', 'putnam', 'robertson', 'rutherford', 'sequatchie', 'shelby', 'smith', 'stewart', 'sumner', 'tipton', 'trousdale', 'van', 'warren', 'wayne', 'weakley', 'white', 'williamson', 'wilson']
        eastern= ['anderson', 'blount', 'bradley', 'campbell', 'carter', 'claiborne', 'cocke', 'grainger', 'greene', 'hamblen', 'hamilton', 'hancock', 'hawkins', 'jefferson', 'johnson', 'knox', 'loudon', 'mcminn', 'meigs', 'monroe', 'morgan', 'polk', 'rhea', 'roane', 'scott', 'sevier', 'sullivan', 'unicoi', 'union', 'washington']
        #Seg_win_DF['County_inrix']=Seg_win_DF['County_inrix'].str.lower()
        
        Seg_win_DF['time'] = Seg_win_DF['time_local'].mask(Seg_win_DF['County_inrix'].isin(eastern), Seg_win_DF['time_local'].dt.tz_localize(tz='US/Eastern', nonexistent='NaT'))
        Seg_win_DF['time'] = Seg_win_DF['time'].mask(Seg_win_DF['County_inrix'].isin(central), Seg_win_DF['time_local'].dt.tz_localize(tz='US/Central', nonexistent='NaT'))
        Seg_win_DF['time_C'] = Seg_win_DF['time_local'].dt.tz_localize(tz='US/Central', nonexistent='NaT')
        Seg_win_DF['time_E'] = Seg_win_DF['time_local'].dt.tz_localize(tz='US/Eastern', nonexistent='NaT')
        Seg_win_DF['time_utc']=Seg_win_DF['time_C'].dt.tz_convert(None)
        Seg_win_DF['time_utc'] = Seg_win_DF['time_utc'].mask(Seg_win_DF['County_inrix'].isin(eastern), Seg_win_DF['time_E'].dt.tz_convert(None))
        #Seg_win_DF['time_utc']=Seg_win_DF['time'].dt.tz_convert(None)
        Seg_win_DF=Seg_win_DF.drop(['time_C','time_E'],axis=1)
        Seg_win_DF[['time','time_local','time_utc']]
        
        
        
        print('Combining process is done.')        
        
        
        return Seg_win_DF






def Prepare_DF_All(inrix_df=None,weather_df=None,incident_df=None,traffic_df=None,List_segments=None,MetaData=None ):
    start_time = time.time()   
    


 

    DF_All=All_DF_Maker(inrix_df=inrix_df,weather_df=weather_df,incident_df=incident_df,traffic_df=traffic_df,List_segments=List_segments,MetaData=MetaData )
    #DF_All=DF_All.sort_values(["Time_id","Space_id"], ascending = (True, True))
    #DF_All=DF_All.reset_index().drop(['index'],axis=1)
    #print('sorting is done!')    
    Save_DF(DF_All, Destination_Address=MetaData['destination']+'ALL_'+str(MetaData['Window_Size'])+'h_DF',Format='pkl', gpd_tag=False)    #DF_All = pd.read_pickle('D:/inrix_new/data/cleaned/ALL_4h_DF.pkl')         
    len(DF_All['XDSegID'].unique())
    len(DF_All['time_local'].unique())
    print("Combining all DFs Time: --- %s seconds ---" % (time.time() - start_time))     
        

    return DF_All


















'''



central=['bedford', 'benton', 'bledsoe', 'cannon', 'carroll', 'cheatham', 'chester', 'clay', 'coffee', 'crockett', 'cumberland', 'davidson', 'decatur', 'dekalb', 'dickson', 'dyer', 'fayette', 'fentress', 'franklin', 'gibson', 'giles', 'grundy', 'hardeman', 'hardin', 'haywood', 'henderson', 'henry', 'hickman', 'houston', 'humphreys', 'jackson', 'lake', 'lauderdale', 'lawrence', 'lewis', 'lincoln', 'mcnairy', 'macon', 'madison', 'marion', 'marshall', 'maury', 'montgomery', 'moore', 'obion', 'overton', 'perry', 'pickett', 'putnam', 'robertson', 'rutherford', 'sequatchie', 'shelby', 'smith', 'stewart', 'sumner', 'tipton', 'trousdale', 'van', 'warren', 'wayne', 'weakley', 'white', 'williamson', 'wilson']
eastern= ['anderson', 'blount', 'bradley', 'campbell', 'carter', 'claiborne', 'cocke', 'grainger', 'greene', 'hamblen', 'hamilton', 'hancock', 'hawkins', 'jefferson', 'johnson', 'knox', 'loudon', 'mcminn', 'meigs', 'monroe', 'morgan', 'polk', 'rhea', 'roane', 'scott', 'sevier', 'sullivan', 'unicoi', 'union', 'washington']
DF['County_inrix']=DF['County_inrix'].str.lower()
DF['time'] = DF['time_local'].mask(DF['County_inrix'].isin(eastern), DF['time_local'].dt.tz_localize('US/Eastern'))
DF[['time','time_local','Space_id']]
DF['time'] = DF['time'].mask(DF['County_inrix'].isin(central), DF['time_local'].dt.tz_localize('US/Central'))
DF['time_utc']=DF['time'].dt.tz_convert(None)



DF_=MainFrame( [156047823.0, 1363492802.0],pd.Timestamp(year=2017, month=1, day=1, hour=0),pd.Timestamp(year=2018, month=1, day=1, hour=0),4 )
DF_['County_inrix']='bedford'
DF_['County_inrix'] = DF_['County_inrix'].mask(DF_['XDSegID'].isin([156047823]), 'anderson')
DF_
DF_['time'] = DF_['time_local'].mask(DF_['County_inrix'].isin(eastern), DF_['time_local'].dt.tz_localize('US/Eastern'))
DF_['time'] = DF_['time'].mask(DF_['County_inrix'].isin(central), DF_['time_local'].dt.tz_localize('US/Central'))
DF_['time_C'] = DF_['time_local'].dt.tz_localize('US/Central')
DF_['time_E'] = DF_['time_local'].dt.tz_localize('US/Eastern')
DF_['time_utc']=DF_['time_C'].dt.tz_convert(None)
DF_['time_utc'] = DF_['time_utc'].mask(DF_['County_inrix'].isin(eastern), DF_['time_E'].dt.tz_convert(None))
#DF_['time_utc']=DF_['time'].dt.tz_convert(None)
DF_=DF_.drop(['time_C','time_E'],axis=1)
DF_[['time','time_local','time_utc']]


DF_=DF
DF_['time'] = DF_['time_local'].mask(DF_['County_inrix'].isin(eastern), DF_['time_local'].dt.tz_localize('US/Eastern'))
DF_['time'] = DF_['time'].mask(DF_['County_inrix'].isin(central), DF_['time_local'].dt.tz_localize('US/Central'))
DF_['time_C'] = DF_['time_local'].dt.tz_localize('US/Central')
DF_['time_E'] = DF_['time_local'].dt.tz_localize('US/Eastern')
DF_['time_utc']=DF_['time_C'].dt.tz_convert(None)
DF_['time_utc'] = DF_['time_utc'].mask(DF_['County_inrix'].isin(eastern), DF_['time_E'].dt.tz_convert(None))
#DF_['time_utc']=DF_['time'].dt.tz_convert(None)
DF_=DF_.drop(['time_C','time_E'],axis=1)
DF_[['time','time_local','time_utc']]



DF=All_DF_Maker(inrix_df=None,weather_df=None,incident_df=None,traffic_df=None,MetaData=MetaData )
DF
























DF_=MainFrame( [156047823.0, 1363492802.0],pd.Timestamp(year=2017, month=1, day=1, hour=0),pd.Timestamp(year=2018, month=1, day=1, hour=0),4 )
DF_=DF_.sort_values('time_local')
DF_['time_C'] = DF_['time_local'].dt.tz_localize('US/Central')
DF_['time_E'] = DF_['time_local'].dt.tz_localize('US/Eastern')
DF_['time_utc']=DF_['time_C'].dt.tz_convert(None)
#DF_['time_utc'] = DF_['time_local'].mask(DF_['XDSegID'].isin([156047823.0]), DF_['time_C'].dt.tz_convert(None))
DF_['time_utc'] = DF_['time_utc'].mask(DF_['XDSegID'].isin([1363492802.0]), DF_['time_E'].dt.tz_convert(None))
DF_
DF_['time'] = DF_['time_local'].mask(DF_['XDSegID'].isin([156047823.0]), DF_['time_C'])
DF_['time'] = DF_['time'].mask(DF_['XDSegID'].isin([1363492802.0]), DF_['time_E'])

DF_=DF_.drop(['time_C','time_E'],axis=1)
DF_=DF



DF_['time_C'] = DF_['time_local'].dt.tz_localize('US/Central')
DF_['time_E'] = DF_['time_local'].dt.tz_localize('US/Eastern')
DF_[['time_C','time_E','time_local']]
DF_['time_utc']=DF_['time_C'].dt.tz_convert(None)
DF_['time_utc'] = DF_['time_utc'].mask(DF_['County_inrix'].isin(eastern), DF_['time_E'].dt.tz_convert(None))
DF_[['time_C','time_E','time_local','time_utc']]


DF_['time'] = DF_['time_local'].mask(DF_['County_inrix'].isin(eastern), DF_['time_local'].dt.tz_localize('US/Eastern'))
DF_['time'] = DF_['time'].mask(DF_['County_inrix'].isin(central), DF_['time_local'].dt.tz_localize('US/Central'))
DF_=DF_.drop(['time_C','time_E'],axis=1)
DF_[['time','time_local','time_utc']]

DF_[['time','time_local']]


DF_['time'] = DF_['time_C'].mask(DF_['XDSegID'].isin([156047823.0]), DF_['time_E'])
DF_
DF_['time'] = DF_['time_local'].dt.tz_localize('US/Central')
DF_.dtypes
type(DF.time)
type(DF.time_local)
DF_['time_utc']=DF_['time'].dt.tz_convert(None)
DF_

'''





















