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
import json
import pygeoj
import pyproj
import shapely.geometry as sg
import swifter
from shapely.ops import nearest_points
from google.cloud import bigquery
import dask.dataframe as dd
import multiprocessing
from datetime import timedelta
from zipfile import ZipFile
import time
import shutil
import re
import matplotlib.pyplot as plt
from utils import Save_DF,Read_DF

#%%
#1) Narrow down the traffic data to just the segments we need
#2) Narrow down the time range to the time we need







    #%%


def Read_Traffic_No_agg(List_segments,Beg_Time,End_Time, traffic_df_Address_csv,MetaData, RAM_Saving_Tag='Dask_1'):
    """
    Reading and cleaning the traffic dataframe and adding some features to it
    @param List_segments: the list of the segments that they have at least one accident
    @param Beg_Time: the beggining of the time range we want to collect the traffic data
    @param End_Time: the end of the time range we want to collect the traffic data
    @param incident_df_Address: the local address of the incident file
    @param MetaData: 
    @param RAM_Saving_Tag: which method do you want to use for saving up RAM if the traffic data is very huge.  Use Dask_2 if the data is super huge
    @return: the cleaned incident dataframe save as pkl, csv, or gzip
    """    
    #if the traffic data is not extracted we should do it here_____________________________________________________________________________________________________________________
    #RAM_Saving_Tag='Dask_1'
    #traffic_df_Address_csv='data\\traffic\\Tennessee-report---for-traffic_smaller.csv'  #D:\inrix\data_inrix_traffic\Tennessee-report---for-traffic.csv
    
    #calling traffic data
    #We look through the incident df and make a list of the segments having at least one incident.
    #This process make the traffic df much smaller
    if RAM_Saving_Tag=='CSV_Chunk':
        chunksize = 10e7
        print('if you run into memory error change the size of chunksize. The current value is:', chunksize, 'rows')
        i=0
        df=pd.DataFrame()
        for chunk in pd.read_csv(traffic_df_Address_csv, chunksize=chunksize):
            i=i+1
            print(i,'th chunk')
            print(chunk.columns)
            df=df.append(chunk[chunk['xd_id'].isin(List_segments)],ignore_index = True )
            
        df['time_local']=pd.to_datetime(df['measurement_tstamp'],format="%Y-%m-%d  %H:%M:%S")
        df=df[(df['time_local']<=End_Time) & (df['time_local']>=Beg_Time)].reset_index().drop(['index','measurement_tstamp'],axis=1)
        df.to_pickle("data/traffic/traffic_just_incidents.pkl")
        
    
    elif RAM_Saving_Tag[0:4]=='Dask':
        df=pd.DataFrame()
        chunk=dd.read_csv(traffic_df_Address_csv)
        df=chunk[chunk['xd_id'].isin(List_segments)]
        df['time_local']=dd.to_datetime(df['measurement_tstamp'],format="%Y-%m-%d  %H:%M:%S")
        df=df[(df['time_local']<=End_Time) & (df['time_local']>=Beg_Time)].reset_index().drop(['index','measurement_tstamp'],axis=1)
        
        
        if RAM_Saving_Tag=='Dask_1':
            df.compute().to_pickle("data/traffic/traffic_just_incidents.pkl")
        elif RAM_Saving_Tag=='Dask_2':    
            dd.to_parquet(df,"data/traffic/traffic_just_incidents.gzip",compression='gzip').compute()

        
    print('Done cleaning traffic!')    
    
    '''
    df.to_csv("data\traffic\traffic_just_incidents.csv").compute()
    dd.to_parquet(df,"data\traffic\traffic_just_incidents.csv").compute()
    df.to_parquet('data\\traffic\\dfparquet.gzip',compression='gzip') 
    df1=df.drop(['time_local'],axis=1)
    df.to_csv("data\\traffic\\Tennessee-report---for-traffic.csv")
    '''






def Read_Traffic(List_segments,Beg_Time,End_Time, traffic_df_Address_csv,MetaData, RAM_Saving_Tag='Dask_1',Save_File=None):
    """
    Reading and cleaning the traffic dataframe and adding some features to it
    @param List_segments: the list of the segments that they have at least one accident
    @param Beg_Time: the beggining of the time range we want to collect the traffic data
    @param End_Time: the end of the time range we want to collect the traffic data
    @param incident_df_Address: the local address of the incident file
    @param MetaData: 
    @param RAM_Saving_Tag: which method do you want to use for saving up RAM if the traffic data is very huge. 
                            CSV_Chunk: 1) Reads the traffic_csv data chunk by chunk: pandas ---> 2) Filter, aggregate, and combine all the chunks: pandas --> 3) save it as pkl
                            Dask:      1) Reads the traffic_csv data chunk by chunk: Dask   ---> 2) Filter, aggregate, and combine all the chunks: pandas 
                                                                                                                                               Dask_01    --> 3) save it as pkl
                                                                                                                                               Dask_02    --> 3) save it as parquet
    Dask_1: csv, Dask_2: parquet. Use Dask_2 if the data is super huge
    @return: the cleaned aggreegated traffic dataframe save as pkl, csv, or gzip
    """    
    
    
    
    if RAM_Saving_Tag=='CSV_Chunk':
        chunksize = 10e7
        print('if you run into memory error change the size of chunksize. The current value is:', chunksize, 'rows')
        i=0
        df=pd.DataFrame()
        for chunk in pd.read_csv(traffic_df_Address_csv, chunksize=chunksize):
            i=i+1
            print(i,'th chunk')
            print(chunk.columns)
            df=df.append(chunk[chunk['xd_id'].isin(List_segments)],ignore_index = True )
            
        df['time_local']=pd.to_datetime(df['measurement_tstamp'],format="%Y-%m-%d  %H:%M:%S")
        df=df[(df['time_local']<=End_Time) & (df['time_local']>=Beg_Time)].reset_index().drop(['index','measurement_tstamp'],axis=1)
    
        df["year"]=df['time_local'].dt.year
        df["month"]=df['time_local'].dt.month
        df["day"]=df['time_local'].dt.day
        df["hour"]=df['time_local'].dt.hour
        #df['window']=(np.floor(df['time_local'].dt.hour/MetaData['Window_Size'])).astype(int)     
        df['window']=(np.floor((df['time_local'].dt.hour+df['time_local'].dt.minute/60)/MetaData['Window_Size'])).astype('int64')     
        df['Congestion']=(df['reference_speed_mean']-df['speed_mean'])/df['reference_speed_mean']
        df['congestion'] = df['congestion'].mask(df['congestion']<0, 0) 
  
        df=df.groupby(['xd_id','year','month','day','window']).agg(MetaData['Agg_Dic_Traffic'])
        
        df.columns = ['_'.join(col).strip() if col[-1]!='first' else col[0] for col in df.columns.values]    
        df=df.reset_index()
        df=df.sort_values(['time_local','xd_id'])
        df.to_pickle(traffic_df_Address_csv+'analyzed'+'/'+Save_File+'.pkl')
        
    
    elif RAM_Saving_Tag[0:4]=='Dask':
        df=pd.DataFrame()
        chunk=dd.read_csv(MetaData['getcwd']+'/' +MetaData['Traffic_File_Names'])  #traffic_df_Address_csv    
        if 'xd_id' in chunk.columns.tolist():
            df=chunk[chunk['xd_id'].isin(List_segments)]
        df['time_local']=dd.to_datetime(df['measurement_tstamp'],format="%Y-%m-%d  %H:%M:%S")
        df=df[(df['time_local']<=End_Time) & (df['time_local']>=Beg_Time)].reset_index().drop(['index','measurement_tstamp'],axis=1)
        
        df["year"]=df['time_local'].dt.year
        df["month"]=df['time_local'].dt.month
        df["day"]=df['time_local'].dt.day
        df["hour"]=df['time_local'].dt.hour
        df['window']=(np.floor((df['time_local'].dt.hour+df['time_local'].dt.minute/60)/MetaData['Window_Size'])).astype('int64')     
        
        df['congestion']=(df['reference_speed']-df['speed'])/df['reference_speed']
        df['congestion'] = df['congestion'].astype(float)
        #df.loc[df['congestion']<0,'congestion']=0  
        df['congestion'] = df['congestion'].mask(df['congestion']<0, 0) 
  
        df=df.groupby(['xd_id','year','month','day','window']).agg(MetaData['Agg_Dic_Traffic'])
        
        df.columns = ['_'.join(col).strip() if col[-1]!='first' else col[0] for col in df.columns.values]    
        df=df.reset_index()
        #df=df.sort_values(['xd_id','time_local'])
        
       
        if RAM_Saving_Tag=='Dask_1':
            df.compute().to_pickle(traffic_df_Address_csv+'analyzed'+'/'+Save_File+'.pkl')
            print(RAM_Saving_Tag,' is done and the results are exctraced from csv and saved in pkl')
        elif RAM_Saving_Tag=='Dask_2':    
            
            #df=df.repartition(npartitions=df.npartitions * 10)
            dd.to_parquet(df,traffic_df_Address_csv+'analyzed'+'/'+Save_File+'.gzip',compression='gzip')
            print(RAM_Saving_Tag,' is done and the results are exctraced from csv and saved in gzip')
        del df
        del chunk
        
    print('Done cleaning traffic!')    





def Read_Traffic_Zip(List_segments,Beg_Time,End_Time, traffic_df_Address_zip, file_list_zip,destdir,MetaData, RAM_Saving_Tag='Dask_1', Test_Tag=True, Save_File=None):
    """
    Reading and cleaning the traffic dataframe and adding some features to it
    @param List_segments: the list of the segments that they have at least one accident
    @param Beg_Time: the beggining of the time range we want to collect the traffic data
    @param End_Time: the end of the time range we want to collect the traffic data
    @param traffic_df_Address_zip: the local address that zip files are located
    @param file_list_zip: list of the zipfiles we want to the explored
    @param destdir: the local address that the zip files will be extraced 
    @param MetaData: 
    @param RAM_Saving_Tag: which method do you want to use for saving up RAM if the traffic data is very huge. 
                            Dask:      1) Reads the traffic_csv data chunk by chunk:
                                                                                  Dask_01   ---> 2) Filter, aggregate, and combine all the chunks: Dask  --> 3) save it as pkl
                                                                                  Dask_02   ---> 2) Filter, aggregate, and combine all the chunks: Dask  --> 3) save it as parquet
    @param Test_Tag: Changed the tag to true if you just want to make sure the pipeline works
    @return: the cleaned aggreegated traffic dataframe save as pkl, csv, or gzip
    """    
    
              
                
    if RAM_Saving_Tag[0:4]=='Dask':
        if Test_Tag==True:
            df = dd.from_pandas(pd.DataFrame(),npartitions=1)
            #ddf = pd.DataFrame()
            i=0
            for f in file_list_zip:
                i=i+1
                print(traffic_df_Address_zip + f)
                with ZipFile(traffic_df_Address_zip + f, mode = 'r', allowZip64 = True) as zip:
                    print(zip.namelist())
                    Members=['XD_Identification.csv']  #['Readings.csv']
                    zip.extractall(path=destdir+f[0:-4] , members=Members, pwd=None)
                    print(destdir+f[0:-4] )
                    #chunk = dd.read_csv(destdir+f[0:-4] +'\\'+'XD_Identification.csv')
                    chunk = dd.read_csv(destdir+f[0:-4] +'\\'+Members[0])
                    #print(chunk.columns,'\n', len(chunk),len(chunk[chunk['xd'].isin(List_segments)]))   #Index(['xd_id', 'measurement_tstamp', 'speed', 'average_speed', 'reference_speed', 'travel_time_seconds', 'confidence_score', 'cvalue'], dtype='object')
                    #print(len(chunk),len(chunk[chunk['xd'].isin(List_segments)]))
                    df = df.append(chunk[chunk['xd'].isin(List_segments)])                                          
            df['zip'] = df['zip'].mask(df['zip']==37762, 0)
            
            df=df[['xd','miles','county']].groupby(['xd']).agg({ 'miles':['sum'],'county':['first']})
            df.columns = ['_'.join(col).strip() if col[-1]!='first' else col[0] for col in df.columns.values]    
            df=df.reset_index()
            dd.to_parquet(df,traffic_df_Address_zip+'analyzed'+'/'+Save_File+'.gzip',compression='gzip')
            df=df.compute()
            df.to_pickle(traffic_df_Address_zip+'analyzed'+'/'+Save_File+'.pkl')
            #df.to_pickle('D:/inrix_new/data/traffic/analyzed/traffic_just_incidents_Final_test.pkl')
            print(RAM_Saving_Tag,' is done using Test_Tag and the results are exctraced from zip files and saved in both pkl and gzip')
            return df                
                
            
        elif Test_Tag==False:  
            
            df = dd.from_pandas(pd.DataFrame(),npartitions=4)
            for f in file_list_zip:
                print(traffic_df_Address_zip + f)
                with ZipFile(traffic_df_Address_zip + f, mode = 'r', allowZip64 = True) as zip:
                    print(zip.namelist())
                    #Members=[(zip.namelist()[1])]
                    
                    try:
                        Members=['Readings.csv']
                        zip.extractall(path=destdir+f[0:-4], members=Members, pwd=None)
                    except:
                        try:
                            Members=['47-Counties-2020-5min.csv'] 
                            zip.extractall(path=destdir+f[0:-4], members=Members, pwd=None)
                        except:
                            Members=['48-Counties-2020-5min.csv'] 
                            zip.extractall(path=destdir+f[0:-4], members=Members, pwd=None)
                    
                    #df = dd.read_csv(destdir+f[0:-4] +'\\'+'XD_Identification.csv')
                    chunk = dd.read_csv(destdir+f[0:-4] +'\\'+Members[0])
                    df = df.append(chunk[chunk['xd_id'].isin(List_segments)])       #xd_id: the segment id that the filteraion is based on
            print(df.columns)        
            df['time_local']=dd.to_datetime(df['measurement_tstamp'],format="%Y-%m-%d  %H:%M:%S")
            df=df[(df['time_local']<=End_Time) & (df['time_local']>=Beg_Time)].reset_index().drop(['index','measurement_tstamp'],axis=1)
            
            df["year"]=df['time_local'].dt.year
            df["month"]=df['time_local'].dt.month
            df["day"]=df['time_local'].dt.day
            df["hour"]=df['time_local'].dt.hour
            df['window']=(np.floor((df['time_local'].dt.hour+df['time_local'].dt.minute/60)/MetaData['Window_Size'])).astype('int64')     
            
            df['congestion']=(df['reference_speed']-df['speed'])/df['reference_speed']
            df['congestion'] = df['congestion'].astype(float)
            #df.loc[df['congestion']<0,'congestion']=0
            df['congestion'] = df['congestion'].mask(df['congestion']<0, 0)              #congestion cannot be negative so it requires correction. 
      
            df=df.groupby(['xd_id','year','month','day','window']).agg(MetaData['Agg_Dic_Traffic'])
            
            df.columns = ['_'.join(col).strip() if col[-1]!='first' else col[0] for col in df.columns.values]    
            df=df.reset_index()
            #df=df.sort_values(['xd_id','time_local'])
            
           
        
            
            
            if RAM_Saving_Tag=='Dask_1':
                df.compute().to_pickle(traffic_df_Address_zip+'analyzed'+'/'+Save_File+'.pkl')
                print(RAM_Saving_Tag,' is done and the results are exctraced from zip files and saved in pkl')
            elif RAM_Saving_Tag=='Dask_2':    
                dd.to_parquet(df,traffic_df_Address_zip+'analyzed'+'/'+Save_File+'.gzip',compression='gzip')
                print(RAM_Saving_Tag,' is done and the results are exctraced from zip files and saved in gzip')
        
            #return df
            
            
            
            
            
def Traffic_Final(traffic_df_Address=None,inrix_df=None,incident_df=None,List_segments=None,MetaData=None, Tag='ZIP'):
        """
        This function brings all the DFs together
        @param incident_df:    
        @param MetaData:    
        @param Tag:        it is a tag to choose you want to extract the traffic data, ZIP: if they are located in multiple zip files, CSV: if it is located in just one big CSV file, BQ: if you collect them using big querry gcp
        @return Seg_win_DF: a dataframe including features regarding spatial temporal resolution
        """    

        #traffic_df_Address=getcwd+'\\data\\traffic\\'
        
        
        
    
        
        #traffic_df_Address=MetaData['traffic_df_Address']
        #destdir=getcwd+'\\data\\traffic\\extracted\\'    

        
        
        '''#A segment might not have an accident but be a part of segment. We want to included that
        if inrix_df.columns.isin(['grouped_XDSegID']).sum()==1:  
            List_groups=inrix_df[inrix_df['XDSegID'].isin(List_segments)]['MyGrouping_3'].astype(int).unique()
            List_segments=inrix_df[inrix_df['MyGrouping_3'].isin(List_groups)]['XDSegID'].astype(int).unique()
        '''
        #Beg_Time=incident_df['time_local'].min()+timedelta(days=-1)
        #End_Time=incident_df['time_local'].max()+timedelta(days=1)
        Beg_Time=MetaData['Beg_Time']
        End_Time=MetaData['End_Time']
    
        if Tag=='ZIP':
            #%% if you want to use Read_Traffic_Zip becuase the files are in multiple zipfiles
            destdir=MetaData['getcwd']+'/' +MetaData['traffic_df_Address']+'extracted/'     #MetaData['getcwd']+'/data/traffic/extracted/'         #location of the file that includes the incidents
            #0 List of the zipfiles including the traffic data:
            file_list_All=MetaData['Traffic_File_Names']
            '''
            ['47countiesJan2017-Dec2017.zip',
            '48countiesJan2017-Dec2017.zip',
            '47countiesJan2018-Dec2018.zip',
            '48countiesJan2018-Dec2018.zip',
            '47countiesJan2019-Dec2019.zip',
            '48countiesJan2019-Dec2019.zip',
            '47countiesJan2020-Dec2020.zip',
            '48countiesJan2020-Dec2020.zip',
            ]        
            '''
 
            #1 extracting each year from zip file seperately and filter and aggregate it:
            Save_File_List=[]
            for i,year in enumerate(range(MetaData['Beg_Time'].year,1+MetaData['End_Time'].year)):
                print(i,year)
                pattern=re.compile(r'.*%d.*.zip'%year)
                file_list_zip=[re.findall(pattern,i)[0] for i in file_list_All if len(re.findall(pattern,i))>0]
                print(file_list_zip)
                try:
                    shutil.rmtree(destdir[:-1])
                except:
                    print(destdir,'doesnt exist!')       
                Save_File='traffic_just_Segment_aggregated_5m-'+str(MetaData['Window_Size'])+'h_'+str(year)
                #Save_File='traffic_just_incidents_Final_BiggerTraffic_'+str(year)
                Save_File_List.append(Save_File)
                #Read_Traffic_Zip(List_segments,Beg_Time,End_Time, traffic_df_Address, file_list_zip,destdir,MetaData, RAM_Saving_Tag='Dask_1',Test_Tag=True,Save_File=Save_File)
                Read_Traffic_Zip(List_segments,Beg_Time,End_Time, traffic_df_Address, file_list_zip,destdir,MetaData, RAM_Saving_Tag='Dask_2',Test_Tag=False,Save_File=Save_File)
            
            #2 combining all the years and move the final results to the cleaned folder,MetaData['destination'] :    
            df=pd.DataFrame()
            for Name in Save_File_List:
                Address_to_Read=traffic_df_Address+'analyzed'+'\\'+Name+'.gzip'
                df_=pd.read_parquet(Address_to_Read, engine='auto')
                df=df.append(df_)
     
            #Save_File='traffic_just_Segment_aggregated_5m-'+str(MetaData['Window_Size'])+'h'
            Save_File='traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)
            #df.to_pickle(MetaData['destination']+Save_File+'.pkl')
            df.to_parquet(MetaData['destination']+Save_File+'.gzip',compression='gzip')
        elif Tag=='CSV':
            #%% if you want to use Read_Traffic since it it just one csv file
        
            #1 filter and aggregate:
            #Save_File='traffic_just_Segment_aggregated_5m-'+str(MetaData['Window_Size'])+'h'
            Save_File='traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)
            Read_Traffic(List_segments,Beg_Time,End_Time, traffic_df_Address,MetaData, RAM_Saving_Tag='Dask_2',Save_File=Save_File)
            #2 moving the data to the clean folder 
            #traffic_df_Address_zip+'analyzed'+'/'+Save_File+'.pkl'
            try:
                try:
                    shutil.rmtree(MetaData['destination']+Save_File+'.gzip')
                    shutil.copytree(traffic_df_Address+'analyzed'+'/'+Save_File+'.gzip', MetaData['destination']+Save_File+'.gzip')
                except:
                    shutil.copytree(traffic_df_Address+'analyzed'+'/'+Save_File+'.gzip', MetaData['destination']+Save_File+'.gzip')
                print(traffic_df_Address+'analyzed'+'/'+Save_File+'.gzip', 'copied to ', MetaData['destination'])
            except:
                shutil.copy(traffic_df_Address+'analyzed'+'/'+Save_File+'.pkl', MetaData['destination'])
                print(traffic_df_Address+'analyzed'+'/'+Save_File+'.pkl', 'copied to ', MetaData['destination'])
                
                
        elif Tag=='BQ':  
            try: 
                Save_File_BQ='traffic_just_Segment_aggregated_5m-60m_2017-04-01-to-2020-03-01_BigQuery' 
                df=pd.read_parquet(traffic_df_Address+Save_File_BQ+'.gzip', engine='auto')
                print('Big Query is not employed since the results could be loaded from: ',traffic_df_Address+Save_File_BQ+'.gzip')
                
            except:
                client=bigquery.Client()           
                Querry=[       'SELECT xd_id, '
                                   'min(measurement_tstamp) as measurement_tstamp, '
                                   'avg(speed) as speed, '
                                   'avg(average_speed) as average_speed, '
                                   'avg(reference_speed) as reference_speed, '
                                   'avg(cvalue) as cvalue, '
                                   'avg(confidence_score) as confidence_score, '
                                   'extract (YEAR from measurement_tstamp) as year, '
                                   'extract (MONTH from measurement_tstamp) as month, '
                                   'extract (DAY from measurement_tstamp) as day, '
                                   'extract (HOUR from measurement_tstamp) as hour '
                            'FROM `mystic-impulse-228617.traffic.traffic_2017_2020` ',
                            'WHERE DATE(measurement_tstamp) >= ', '"', Beg_Time.strftime("%Y-%m-%d"), '" ',
                            'and DATE(measurement_tstamp) <= ', '"', End_Time.strftime("%Y-%m-%d"), '" and ',
                            'xd_id in ', str(tuple(List_segments)), ' ',
                             'group by xd_id, '
                                     'extract (YEAR from measurement_tstamp), '
                                     'extract (MONTH from measurement_tstamp), '
                                     'extract (DAY from measurement_tstamp), '
                                     'extract (HOUR from measurement_tstamp)']
                Querry_str=''
                for i in Querry:
                    Querry_str=Querry_str+i
                    
                query_job = client.query(Querry_str)
                results = query_job.result()
                df=results.to_dataframe()
                Save_File_BQ='traffic_just_Segment_aggregated_5m-1h_BigQuery' 
                df.to_parquet(traffic_df_Address+Save_File_BQ+'.gzip',compression='gzip')
                print('Big Query is done and the results are saved in: ',traffic_df_Address+Save_File_BQ+'.gzip')

            
            df['time_local']=df['measurement_tstamp'].dt.tz_convert(None)
            df=df.drop('measurement_tstamp', axis=1)
            df=df[(df['time_local']> MetaData['Beg_Time']) & (df['time_local']< MetaData['End_Time'])  ]
            df['window']=(np.floor((df['time_local'].dt.hour+df['time_local'].dt.minute/60)/MetaData['Window_Size'])).astype('int64')     
            df['congestion']=(df['reference_speed']-df['speed'])/df['reference_speed']
            df['congestion'] = df['congestion'].astype(float)
            #df.loc[df['congestion']<0,'congestion']=0
            df['congestion'] = df['congestion'].mask(df['congestion']<0, 0)              #congestion cannot be negative so it requires correction. 
      
            df=df.groupby(['xd_id','year','month','day','window']).agg(MetaData['Agg_Dic_Traffic'])
            
            df.columns = ['_'.join(col).strip() if col[-1]!='first' else col[0] for col in df.columns.values]    
            df=df.reset_index()
            #Save_File='traffic_just_Segment_aggregated_5m-'+str(MetaData['Window_Size'])+'h'
            Save_File='traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)
            #df.to_pickle(MetaData['destination']+Save_File+'.pkl')
            df.to_parquet(MetaData['destination']+Save_File+'.gzip',compression='gzip')
            
        elif Tag=='Athena':  
            try: 
                Save_File_Athena='traffic_just_Segment_aggregated_5m-60m_2017-04-01-to-2020-03-01_Athena' 
                df=pd.read_parquet(traffic_df_Address+Save_File_Athena+'.gzip', engine='auto')
                print('Athena Query is not employed since the results could be loaded from: ',traffic_df_Address+Save_File_Athena+'.gzip')
            except:
                print('Athena Query is not implemented yet!')
                
            #MetaData=dict()
            #MetaData['Window_Size']=4
            #Beg_Time=pd.Timestamp(year=2019, month=12, day=30, hour=0)
            #End_Time=pd.Timestamp(year=2020, month=1, day=2, hour=0)
            #List_segments=[156185911, 1524646417]    
                
            Querry=[       'SELECT xd_id, '
                               'extract (YEAR from DATE(measurement_tstamp)) as year, '
                               'extract (MONTH from DATE(measurement_tstamp)) as month, '
                               'extract (DAY from DATE(measurement_tstamp)) as day, '
                               'extract(HOUR FROM measurement_tstamp)/',str(MetaData['Window_Size']),' AS window, '
                               'min(DATE_FORMAT(measurement_tstamp,\'%Y-%m-%d %H:%i:%s\')) as measurement_tstamp, '
                               'avg(speed) as speed_mean, max(speed) as speed_max, min(speed) as speed_min, '
                               'avg(average_speed) as average_speed_mean, max(average_speed) as average_speed_max, min(average_speed) as average_speed_min, '
                               'avg(reference_speed) as reference_speed_mean, max(reference_speed) as reference_speed_max, min(reference_speed) as reference_speed_min, '
                               'avg(cvalue) as cvalue_mean, max(cvalue) as cvalue_max, min(cvalue) as cvalue_min, '
                               'avg(confidence_score) as confidence_score_mean, max(confidence_score) as confidence_score_max, min(confidence_score) as confidence_score_min, '
                               'avg(speed) as speed_mean, max(speed) as speed_max, min(speed) as speed_min '
                               #'extract (HOUR from measurement_tstamp) as hour '
                        'FROM "trafficdb"."speeds" ',
                        'WHERE '
                        'year in (2019,2020)  AND month in (12,1) AND countyid=1 AND '
                        'DATE(measurement_tstamp) >= ', 'DATE(\'', Beg_Time.strftime("%Y-%m-%d"), '\') ',
                        'and DATE(measurement_tstamp) <= ', 'DATE(\'', End_Time.strftime("%Y-%m-%d"), '\') and ',
                        'xd_id in ', str(tuple(List_segments)), ' ',
                        'group by xd_id, '
                                 'extract (YEAR from DATE(measurement_tstamp)), '
                                 'extract (MONTH from DATE(measurement_tstamp)), '
                                 'extract (DAY from DATE(measurement_tstamp)), '
                                 'extract(HOUR FROM measurement_tstamp)/',str(MetaData['Window_Size'])
                        ]
            
            Querry_str=''
            for i in Querry:
                Querry_str=Querry_str+i
            print(Querry_str.replace("\'","'")+' limit 10')
            

                
            df['time_local']=df['measurement_tstamp'].dt.tz_convert(None)
            df=df.drop('measurement_tstamp', axis=1)
            df=df[(df['time_local']> MetaData['Beg_Time']) & (df['time_local']< MetaData['End_Time'])  ]
            df['window']=(np.floor((df['time_local'].dt.hour+df['time_local'].dt.minute/60)/MetaData['Window_Size'])).astype('int64')     
            df['congestion']=(df['reference_speed']-df['speed'])/df['reference_speed']
            df['congestion'] = df['congestion'].astype(float)
            #df.loc[df['congestion']<0,'congestion']=0
            df['congestion'] = df['congestion'].mask(df['congestion']<0, 0)              #congestion cannot be negative so it requires correction. 
      
            df=df.groupby(['xd_id','year','month','day','window']).agg(MetaData['Agg_Dic_Traffic'])
            
            df.columns = ['_'.join(col).strip() if col[-1]!='first' else col[0] for col in df.columns.values]    
            df=df.reset_index()
            #Save_File='traffic_just_Segment_aggregated_5m-'+str(MetaData['Window_Size'])+'h'
            Save_File='traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)
            #df.to_pickle(MetaData['destination']+Save_File+'.pkl')
            df.to_parquet(MetaData['destination']+Save_File+'.gzip',compression='gzip')            

                                                
        #%% 
        print('The final traffic extraction, filtering, and aggregation is done and uploaded in the clean folder.')
        #df=pd.read_parquet(MetaData['destination']+Save_File+'.gzip', engine='auto')
        #return df
        
        
        
def Prepare_Traffic(traffic_df_Address=None,inrix_df=None,incident_df=None,List_segments=None,MetaData=None ):   
    
    '''
    *Prepare_Traffic:* This function conducts preprocessing and cleaning analyses on the traffic data set. It also adds/removes some features. For more information please refer to *Pipeline.pptx *
    
        Input
        ----------
        traffic_df_Address : String
            The location of the traffic file. The default is None.
        inrix_df : String or DataFrame, optional
            The location of inrix DataFrame or the DataFrame itself. If user provides None, the code automatically looks at the default location for this file. The default is None. 
        incident_df : String or DataFrame, optional
            The location of incident DataFrame or the DataFrame itself. If user provides None, the code automatically looks at the default location for this file. The default is None. 
        List_segments: List
            It includes the list of the roadway segments we want to extract the traffic information for
        MetaData : Dictionary
            It includes the meta data. The default is None. 
            
        Returns
        -------
        traffic_df : DataFrame
            This is the cleaned version of the traffic data set in a DataFrame format. For more information, please refer to Pipeline.pptx
    '''    
    
    MetaData['Agg_Dic_Traffic']={'time_local':['first'],       
                                'congestion':['min','max','mean','std'],
                                'speed':['mean'],
                                'average_speed':['mean'] ,
                                'reference_speed':['mean']  ,
                                'cvalue':['mean'],
                                'confidence_score':['mean']}      
    
    
    start_time = time.time()   
    
    inrix_df=Read_DF(inrix_df=inrix_df,Reading_Tag='inrix_df',MetaData=MetaData )
    incident_df=Read_DF(incident_df=incident_df, Reading_Tag='incident_df',MetaData=MetaData )

    
    traffic_df=None
    Traffic_Final(traffic_df_Address=traffic_df_Address,inrix_df=inrix_df, incident_df=incident_df,List_segments=List_segments,MetaData=MetaData,Tag=MetaData['Traffic_Source_Tag'] )
    
    '''
    try:
        #traffic_df=pd.read_parquet(MetaData['destination']+'traffic_just_Segment_aggregated_5m-'+str(MetaData['Window_Size'])+'h'+'.gzip', engine='auto')
        traffic_df=pd.read_parquet(MetaData['destination']+'traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)+'.gzip', engine='auto')
        print('traffic_df loaded from',MetaData['destination']+'traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)+'.gzip' )
    except:
        #traffic_df=pd.read_pickle(MetaData['destination']+'traffic_just_Segment_aggregated_5m-'+str(MetaData['Window_Size'])+'h'+'.pkl')
        traffic_df=pd.read_pickle(MetaData['destination']+'traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)+'.pkl')
        print('traffic_df loaded from',MetaData['destination']+'traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)+'.pkl')
    '''
    print("Reading Traffic Time: --- %s seconds ---" % (time.time() - start_time))        
        
    
    return traffic_df