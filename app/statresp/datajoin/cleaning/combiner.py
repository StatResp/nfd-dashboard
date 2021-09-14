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
from statresp.datajoin.cleaning.utils import Save_DF,Read_DF




Dask_Tag=False


#%%
def base_df_generator(List_segments,Beg,End,Win_Hour, MetaData):
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
        Seg_win_DF['time_local']=Window_series
        #Seg_win_DF["time_id"]=range(len(Window_series))
        Seg_win_DF["year"]=Seg_win_DF['time_local'].dt.year
        Seg_win_DF["month"]=Seg_win_DF['time_local'].dt.month
        #Seg_win_DF["day"]=Seg_win_DF['time_local'].dt.day
        #Seg_win_DF['window']=(np.floor((Seg_win_DF['time_local'].dt.hour+Seg_win_DF['time_local'].dt.minute/60)/Win_Hour)).astype('int64')

        Seg_win_DF.dtypes
        #SegIDs
        
        #List of segments we want to use as index
        Seg_win_DF=pd.concat([Seg_win_DF]*len(List_segments), ignore_index=True)
        Seg_win_DF['xdsegid']=np.repeat(List_segments,Window_Num)
        #Seg_win_DF['Space_id']=np.repeat(range(len(List_segments)),Window_Num)
        Seg_win_DF=Seg_win_DF.sort_values('time_local').reset_index().drop('index',axis=1)
        #Seg_win_DF=Seg_win_DF.reset_index().drop('index',axis=1)
        #Seg_win_DF['ID_Seg_win_DF']=range(len(Seg_win_DF))
        Seg_win_DF
        
        #Seg_win_DF['xdsegid']=Seg_win_DF['xdsegid'].astype('int64')
        Seg_win_DF['xdsegid']=Seg_win_DF['xdsegid'].astype('float')
        if Dask_Tag==True:
            Seg_win_DF = dd.from_pandas(Seg_win_DF, npartitions=2 )

        #Save_DF(Seg_win_DF, Destination_Address='data/cleaned/Line/'+'merged/'+'Seg_win_DF',Format='pkl', gpd_tag=False) #inrix_df = pd.read_pickle('data/cleaned/inrix_grouped.pkl')
        #Seg_win_DF.to_pickle('data/cleaned/Line/'+'merged/'+'Seg_win_DF.pkl')  
        #Seg_win_DF.to_parquet('data/cleaned/Line/'+'merged/'+'Seg_win_DF.pqt',engine='pyarrow')  
        Save_DF(Seg_win_DF, 'data/merged_templates/base_df',  Format='pkl', gpd_tag=False)
        
        return Seg_win_DF










