# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:36:00 2020

@author: vaziris
This scripts is  written by Sayyed Mohsen Vazirizade.  s.m.vazirizade@gmail.com
"""


import pandas as pd
import geopandas as gpd 
import numpy as np   
import pickle
#import geoparquet as gpq
import dask.dataframe as dd
import networkx as nx
import os
from statresp.datajoin.cleaning.inrix_cleaning import County_ID_Allocator


def Read_DF(inrix_df=None,weather_df=None,incident_df=None,traffic_df=None,List_segments=None,MetaData=None, Reading_Tag=None,DF_All=None, base_df=None, Dask=False ):
    '''
    *Read_DF*:This function is used to read the files from hard disk drive.
    you can choose one of the inrix_df, weather_df, incident_df, traffic_df, or DF_All. 
    They can be either string or None. If it is string it should include the address. If it is None then the code tries to find it in the defautl repo.
    
    

    Input
    ----------
    inrix_df,weather_df,incident_df,traffic_df,DF_All : string, DataFrame, or None
        The location of DataFrame or the DataFrame itself. If user provides None, the code automatically looks at the default location for this file. The default is None. 
    Reading_Tag : String
        You should choose one of the aformentioned tags.     
    List_segments : TYPE, optional
        List of the required segments (not being used yet). The default is None.
    MetaData : Dict
        It includes information such as time range. The default is None.
        If MetaData includes Beg_Time and End_Time, the code automatically filters the data based on time. 
    Dask : Boolian
        If we want to use Dask for reading the dfs. The default is False.


    -------
    return: DataFrame
    one of the dataframes from inrix_df,weather_df,incident_df,traffic_df,DF_All 

    '''
    
    
    #%%
    if Reading_Tag=='inrix_df':
        #inrix           
        if inrix_df is None:
            try:
                inrix_df = pd.read_pickle(MetaData['destination']+'inrix/'+'inrix_weather_zslope_grouped_renamedcolumns.pkl')
                inrix_df.columns = inrix_df.columns.str.lower().str.replace(' ', '_')
                print('inrix_df is loaded from default address: ', MetaData['destination']+'inrix/'+'inrix_weather_zslope_grouped_renamedcolumns.pkl')
            except:
                inrix_df = pd.read_pickle(MetaData['destination']+'inrix/'+'inrix.pkl')
                inrix_df.columns = inrix_df.columns.str.lower().str.replace(' ', '_')
                print('inrix_df is loaded from default address: ', MetaData['destination']+'inrix/'+'inrix.pkl')
        elif isinstance(inrix_df,str):
            print('inrix_df is loaded from ', inrix_df)
            inrix_df = pd.read_pickle(inrix_df) 
        elif isinstance(inrix_df, pd.core.frame.DataFrame):
            print('inrix_df is already loaded')
        
        print('Shape:',inrix_df.shape) 
        return inrix_df
         
    #%%
    elif Reading_Tag=='weather_df':
        #weather           
        if weather_df is None:
            weather_df = pd.read_pickle(MetaData['destination']+'weather/'+'weather.pkl')
            print('weather_df is loaded from default address: ', MetaData['destination']+'weather/'+'weather_df.pkl')
        elif isinstance(weather_df,str):
            print('weather_df is loaded from ', weather_df)
            weather_df = pd.read_pickle(weather_df) 
        elif isinstance(weather_df, pd.core.frame.DataFrame):
            print('weather_df is already loaded')
        
        if 'Beg_Time' in MetaData.keys()  and  'End_Time' in MetaData.keys():
            weather_df=weather_df[(weather_df['time_local']>= MetaData['Beg_Time']) & (weather_df['time_local']<= MetaData['End_Time'])  ]

        print('Shape:',weather_df.shape,'Start Time:',weather_df.shape,weather_df['time_local'].min(),'End Time:',weather_df['time_local'].max() ) 
        return weather_df
    
    #%%
    elif Reading_Tag=='incident_df':
        #incident           
        if incident_df is None:
            incident_df = pd.read_pickle(MetaData['destination']+'incident/'+'incident_xdsegid.pkl')
            incident_df.columns = incident_df.columns.str.lower().str.replace(' ', '_')
            print('incident_df is loaded from default address: ', MetaData['destination']+'incident/'+'incident_xdsegid.pkl')
         
        elif isinstance(incident_df,str):
            print('incident_df is loaded from ', incident_df)
            incident_df = pd.read_pickle(incident_df) 
        elif isinstance(incident_df, pd.core.frame.DataFrame):
            print('incident_df is already loaded')
            
        if 'Beg_Time' in MetaData.keys()  and  'End_Time' in MetaData.keys():
            incident_df=incident_df[(incident_df['time_local']>= MetaData['Beg_Time']) & (incident_df['time_local']<= MetaData['End_Time'])  ]    
            print('Shape:',incident_df.shape,'Start Time:',incident_df['time_local'].min(),'End Time:',incident_df['time_local'].max()  ) 
        if (MetaData['incident_filter'] is None)==False:
            incident_df=incident_df[incident_df[MetaData['incident_filter']['Col_name']].astype(str).str.contains(MetaData['incident_filter']['Value'])]
            print('Shape:',incident_df.shape,'incident_filter:', MetaData['incident_filter'] )
        return incident_df
    #%%
    elif Reading_Tag=='traffic_df':
        #traffic           
        if traffic_df is None:
            try: 
                #Address=MetaData['destination']+'traffic_just_SegmentwithIncident_aggregated_5m-'+str(MetaData['Window_Size'])+'h'+'.gzip'
                Address=MetaData['destination']+'traffic/'+'traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)+'.gzip'
                if Dask==False:
                    traffic_df=pd.read_parquet(Address, engine='auto')
                elif Dask==True:
                    traffic_df=dd.read_parquet(Address, engine='auto')
            except: 
                #Address=MetaData['destination']+'traffic_just_SegmentwithIncident_aggregated_5m-'+str(MetaData['Window_Size'])+'h'+'.pkl'
                Address=MetaData['destination']+'traffic/'+'traffic_just_Segment_aggregated_5m-'+str(int(MetaData['Window_Size']*60))+'m_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)+'.pkl'
                traffic_df=pd.read_pickle(Address)            
            print('traffic_df is loaded from default address: ', Address)


        elif isinstance(traffic_df,str):
            print('traffic_df is loaded from ', traffic_df)
            if traffic_df[-5:]=='.gzip':
                traffic_df = pd.read_parquet(traffic_df, engine='auto') 
            elif DF_All[-4:]=='.pkl':
                traffic_df = pd.read_pickle(traffic_df)         
        
        
        elif isinstance(traffic_df, pd.core.frame.DataFrame):
            print('traffic_df is already loaded')
            #traffic_df=dd.read_parquet(traffic_df_Address+'analyzed'+'\\'+'traffic_just_incidents_Final_BiggerTraffic_all'+'.gzip', engine='auto')   
     
        if 'Beg_Time' in MetaData.keys()  and  'End_Time' in MetaData.keys():
            traffic_df=traffic_df[(traffic_df['time_local']>= MetaData['Beg_Time']) & (traffic_df['time_local']<= MetaData['End_Time'])  ]
            print('Shape:', traffic_df.shape,'Num of unq Seg',len(traffic_df['xd_id'].drop_duplicates()),
                  'Num of unq Time',len(traffic_df['time_local'].drop_duplicates()) ,
                  'Start Time:', traffic_df['time_local'].min(),
                  'End Time:',traffic_df['time_local'].max()  ) 
        return traffic_df



    elif Reading_Tag=='DF_All':
        #traffic           
        if DF_All is None:
            try: 
                #Address=MetaData['destination']+'ALL_'+str(MetaData['Window_Size'])+'h_DF'+'.gzip'
                #Address=MetaData['destination']+'ALL_'+str(MetaData['Window_Size'])+'h_DF_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)+'.gzip'
                Address=MetaData['destination']+'merged/'+'ALL_'+str(int(MetaData['Window_Size']*60))+'m_DF_'+str(MetaData['Beg_Time'].year)+'_'+str(MetaData['Beg_Time'].month)+'_'+str(MetaData['Beg_Time'].day)+'.gzip'
                #DF_All=pd.read_parquet('data/cleaned/Line/ALL_0.08333333333333333h_DF_2019_5_1.gzip', engine='auto')
                DF_All=pd.read_parquet(Address, engine='auto')
                
            except: 
                Address=MetaData['destination']+'merged/'+'ALL_'+str(MetaData['Window_Size'])+'h_DF'+'.pkl'
                DF_All=pd.read_pickle(Address)            
            print('DF_All is loaded from default address: ', Address)
        elif isinstance(DF_All,str):
            print('DF_All is loaded from ', DF_All)
            if DF_All[-5:]=='.gzip':
                DF_All = pd.read_parquet(DF_All, engine='auto') 
            elif DF_All[-4:]=='.pkl':
                DF_All = pd.read_pickle(DF_All) 
        elif isinstance(DF_All, pd.core.frame.DataFrame):
            print('DF_All is already loaded')
  
     
        if 'Beg_Time' in MetaData.keys()  and  'End_Time' in MetaData.keys(): 
            DF_All=DF_All[(DF_All['time_local']>= MetaData['Beg_Time']) & (DF_All['time_local']<= MetaData['End_Time'])  ]
        #print(DF_All.shape,DF_All['time_local'].min(),DF_All['time_local'].max()  )
        #print(DF_All)
        print('Size of DF_All:',DF_All.shape,
              '|notna in traffic :',len(DF_All[DF_All['speed_mean'].notna()]),
              '|notna in weather :',len(DF_All[DF_All['temp_mean'].notna()]),
              '|notna in incident:',len(DF_All[DF_All['Total_Number_Incidents']>0]),
              )
        print(DF_All.shape)
        return DF_All

    elif Reading_Tag == 'base_df':
        # incident
        if base_df is None:
            base_df = pd.read_pickle(MetaData['destination'] + 'base_df/' + 'base_df.pkl')
            base_df.columns = base_df.columns.str.lower().str.replace(' ', '_')
            print('incident_df is loaded from default address: ',
                  MetaData['destination'] + 'base_df/' + 'base_df.pkl')

        elif isinstance(base_df, str):
            print('base_df is loaded from ', base_df)
            base_df = pd.read_pickle(base_df)
        elif isinstance(base_df, pd.core.frame.DataFrame):
            print('base_df is already loaded')
        return base_df




#%%
def Save_DF(DF, Destination_Address,Format='pkl', gpd_tag=False):
    """
    Saving the results as dataframe or geodataframe with the intended format :
        
    @param DF: the dataframe we want to save 
    @param Destination_Address: the local address and name for saving the file
    @param Format: the format you want to save the file: pickle=pkl, ESRI shape files= shp  geojson=geojson, CSV=csv,  parquet=gzip
    """

    if (gpd_tag==True) | (Format=='shp') | (Format=='geojson') | (Format=='geogzip'):
            crs = {'init': 'epsg:4326'}
            DF = (gpd.GeoDataFrame(DF, geometry=DF['geometry'], crs=crs)).copy()
    
    if not os.path.exists('/'.join(Destination_Address.split('/')[0:-1])):
        os.makedirs('/'.join(Destination_Address.split('/')[0:-1]))
    
    Name=Destination_Address+'.'+Format
    if Format=='pkl':
        DF.to_pickle(Name)  
    elif Format=='csv':
        DF.to_csv(Name)     
    elif Format=='shp':
        try:
            DF.to_file(Name)    
        except KeyError:
            print('shp doesnt support time format. Please remove it and try it again.')   
    elif Format=='geojson':       
            DF.to_file(Name, driver='GeoJSON')  
    elif Format=='json':       
            DF.to_json(Name)              
    elif Format=='gzip':
            DF.to_parquet(Name, compression='gzip')    
            #dd.to_parquet(DF,Name)  #compression='pyarrow'
            #DF.to_csv(Name)  #compression='pyarrow'
    elif Format=='geogzip':
            print('geogzip')
            DF.to_geoparquet(Name, compression='geoparquet')  
    elif Format=='gpickle':
            print('gpickle')            
            nx.write_gpickle(DF, Name)
    else:
            print('The format is not recognizable.')
    print('Saving is done! ',Name )
    
    
    
    
def Call_Back(DF=None, DF_Tag='incident_df',     Time_beg=None,  Time_end=None,  Segment=None):
    '''
    *Call_Back:* This is a look up function. You can extract a segment from inrix DataFrame based on XDSegID or you can extract an incident between a specific time frames and location (XDSegID). The output may not be unique.
    
    
    Input
    ----------
    DF : string, DataFrame, or None
        The location of DataFrame or the DataFrame itself. If user provides None, the code automatically looks at the default location for this file. The default is None.     
    DF_Tag : string
        You can choose between 'incident_df', if you want to look up a segment, or 'incident_df' if you want to look up an incidnet. 
    Time_beg, Time_end : pandas._libs.tslibs.timestamps.Timestamp
        beggining an end of the time frame for temporal filtering.  
        DESCRIPTION. The default is None.
    Segment : List
        List of the inrix roadway segments (XDSegID) for spatial filtering. 
    
    Returns
    -------
    DF : DataFrame
        Filtered DataFrame.
    
    '''
        #Time_beg=pd.Timestamp(year=2019, month=1, day=0, hour=0), 
        #Time_end=pd.Timestamp(year=2019, month=1, day=0, hour=0),     
     
    Meta=dict()
    Meta['destination']= 'data/cleaned/Line/'
    Meta['time_local']= None 
     
    if  DF_Tag== 'inrix_df' :
        DF=Read_DF(inrix_df=DF, MetaData=Meta, Reading_Tag=DF_Tag, Dask=False )
        if (Segment is not None):
            DF=DF[DF['XDSegID'].isin(Segment)]
            
            
            
    elif DF_Tag== 'incident_df' :
        DF=Read_DF(incident_df=DF, MetaData=Meta, Reading_Tag=DF_Tag, Dask=False )
        if (Time_beg is not None) & (Time_end is not None):
            DF=DF[(DF['time_local']>=Time_beg) & (DF['time_local']<=Time_end)]
        if (Segment is not None):
            DF=DF[DF['XDSegID'].isin(Segment)]
    return DF
     


def convert_base_to_parquet(df, address):
    """
    This converts a segment windows pkl file to a parquet which can be uploaded onto s3
    """
    df.to_parquet(path=address, compression='snappy', engine='pyarrow',
                  partition_cols=['year', 'month'])
    print('Saving base dataframe as parquet is done.')


def convert_weather_to_parquet(df, address):
    """
    This converts a segment windows pkl file to a parquet which can be uploaded onto s3
    """
    if 'timestamp_local' in df.columns:
        df = df.drop('timestamp_local', axis=1)
    if 'timestamp_utc' in df.columns:
        df = df.drop('timestamp_utc', axis=1)
    df.to_parquet(path=address, compression='snappy', engine='pyarrow',
                  partition_cols=['year', 'month'])
    print('Saving weather dataframe as parquet is done.')


def convert_incident_to_parquet(df, address):
    """
    This converts an incident pkl file into a parquet which can be uploaded onto s3
    """

    df.time_local = df.time_local.astype(str)
    df.time_utc = df.time_utc.astype(str)
    df.geometry = df.geometry.astype(str)
    if 'time' in df.columns:
        df=df.drop('time',axis=1)
    '''
    df = df.fillna({'totalKilled':-1,
                    "totalInjured":-1,
                    "totalIncapcitatingInjuries":-1,
                    "totalOtherInjuries":-1,
                    "totalVehicles":-1})
    
    df.total_killed = df.total_killed.astype(int)
    df.total_injured = df.total_injured.astype(int)
    df.total_incapcitating_injuries = df.total_incapcitating_injuries.astype(int)
    df.total_other_injuries= df.total_other_injuries.astype(int)
    df.total_vehicles= df.total_vehicles.astype(int)
    
    df.to_parquet(path=INCIDENT_PARQUET_NAME,
              partition_cols=['year', 'month'],
              compression='snappy',
              engine='pyarrow')
    '''
    df.to_parquet(path=address, compression='snappy', engine='pyarrow',
                  partition_cols=['year', 'month'])
    print('Saving incident dataframe as parquet is done.')

def convert_inrix_to_parquet(df, address, Partition_Tag=True):
    """
    This converts an incident pkl file into a parquet which can be uploaded onto s3
    """
    uncompatible_list= ['beg', 'end', 'center', 'geometry', 'geometry3d', 'geometry_highres']
    for i in uncompatible_list:
        if i in df.columns:
            df[i] = df[i].astype(str)
            #from shapely import wkt
            #df[i] = df[i].apply(wkt.loads)


    for i in df.columns:
        if isinstance(df[i].iloc[0], list):
            df = df.drop(i, axis=1)



    '''
    df = df.fillna({'totalKilled':-1,
                    "totalInjured":-1,
                    "totalIncapcitatingInjuries":-1,
                    "totalOtherInjuries":-1,
                    "totalVehicles":-1})

    df.total_killed = df.total_killed.astype(int)
    df.total_injured = df.total_injured.astype(int)
    df.total_incapcitating_injuries = df.total_incapcitating_injuries.astype(int)
    df.total_other_injuries= df.total_other_injuries.astype(int)
    df.total_vehicles= df.total_vehicles.astype(int)

    df.to_parquet(path=INCIDENT_PARQUET_NAME,
              partition_cols=['year', 'month'],
              compression='snappy',
              engine='pyarrow')
    '''
    if Partition_Tag==True:
        if not 'countyid' in df.columns:
            from statresp.datajoin.cleaning.inrix_cleaning import County_ID_Allocator
            df=County_ID_Allocator(df)
        df.to_parquet(path=address, compression='snappy', engine='pyarrow',
                      partition_cols=['countyid'])
        print('Number of columns is {}: ', len(df.columns))
        print('Saving inrix dataframe as parquet is done.')
    elif Partition_Tag==False:
        df.to_parquet(path=address,compression='snappy', engine='pyarrow')
        print('Number of columns is {}: ', len(df.columns))
        print('Saving inrix dataframe as parquet without partitions is done.')


