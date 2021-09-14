# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:36:00 2020

@author: vaziris
This scripts is  written by Sayyed Mohsen Vazirizade.  s.m.vazirizade@gmail.com
"""


from pprint import pprint
import pandas as pd
import geopandas as gpd 
import shapely.geometry as sg
import numpy as np   
import pickle
import os
from datetime import datetime
import pytz
import time
import swifter
import time
from utils import Save_DF



def Timefixer(DF,Time_Col='time_local', Location_Col='County_Incident'): 
    central=['bedford', 'benton', 'bledsoe', 'cannon', 'carroll', 'cheatham', 'chester', 'clay', 'coffee', 'crockett', 'cumberland', 'davidson', 'decatur', 'dekalb', 'dickson', 'dyer', 'fayette', 'fentress', 'franklin', 'gibson', 'giles', 'grundy', 'hardeman', 'hardin', 'haywood', 'henderson', 'henry', 'hickman', 'houston', 'humphreys', 'jackson', 'lake', 'lauderdale', 'lawrence', 'lewis', 'lincoln', 'mcnairy', 'macon', 'madison', 'marion', 'marshall', 'maury', 'montgomery', 'moore', 'obion', 'overton', 'perry', 'pickett', 'putnam', 'robertson', 'rutherford', 'sequatchie', 'shelby', 'smith', 'stewart', 'sumner', 'tipton', 'trousdale', 'van', 'warren', 'wayne', 'weakley', 'white', 'williamson', 'wilson']
    eastern= ['anderson', 'blount', 'bradley', 'campbell', 'carter', 'claiborne', 'cocke', 'grainger', 'greene', 'hamblen', 'hamilton', 'hancock', 'hawkins', 'jefferson', 'johnson', 'knox', 'loudon', 'mcminn', 'meigs', 'monroe', 'morgan', 'polk', 'rhea', 'roane', 'scott', 'sevier', 'sullivan', 'unicoi', 'union', 'washington']
    if DF[Location_Col].lower() in central:
        ZONE = pytz.timezone('CST6CDT')
    elif DF[Location_Col].lower() in eastern:
        ZONE = pytz.timezone('EST5EDT')
    #print(DF['County'].lower() )    
    TimewithZone = ZONE.localize(DF[Time_Col])
    TimewithZone_utc = TimewithZone.astimezone('utc')
    return TimewithZone, TimewithZone_utc


'''
incident_df.iloc[0]['time'].astimezone('utc')
incident_df['time_utc_2']=incident_df['time_utc'].dt.tz_localize(None)
incident_df.dtypes
incident_df
'''

def Read_Incident(incident_df_Address,MetaData, Swifter_Tag=False):
    """
    Reading and cleaning the incident dataframe and adding some features to it
    @param incident_df_Address: the local address of the incident file
    @param MetaData: 
    @return: the cleaned incident dataframe
    """
    
    try:
        incident_df = pd.read_pickle(incident_df_Address)   #loading the incident the data from TDOT
    except KeyError:
        print('Either the address or fromat is wrong.')
    
    #adding features and cleaning regarding time
    incident_df=incident_df.reset_index().drop('index',axis=1)
    incident_df['County']=incident_df['County'].str.lower()
    incident_df=incident_df.rename(columns={'County':'County_Incident'})
    incident_df=incident_df.rename(columns={'time':'time_local'})
    if Swifter_Tag==True:
        incident_df['time'], incident_df['time_utc'] =zip(*incident_df.swifter.apply(lambda ROW: Timefixer(ROW), axis=1) ) #adding the time zones
        print('Swifter is working for correcting the incident time!')
    else:
        incident_df['time'], incident_df['time_utc'] =zip(*incident_df.apply(lambda ROW: Timefixer(ROW), axis=1) ) #adding the time zones
    incident_df['time_utc']=incident_df['time_utc'].dt.tz_localize(None)
    incident_df["year"]=incident_df['time_local'].dt.year
    incident_df["month"]=incident_df['time_local'].dt.month
    incident_df["day"]=incident_df['time_local'].dt.day
    incident_df["day_of_week"]=incident_df['time_local'].dt.dayofweek
    incident_df['weekend_or_not'] = (incident_df["day_of_week"]// 5 == 1).astype(int)
    incident_df["hour"]=incident_df['time_local'].dt.hour
    incident_df['window']=(np.floor(incident_df['time_local'].dt.hour/MetaData['Window_Size'])).astype(int)

    
    #adding features and cleaning regarding space
    #remove coords with NA Values
    NotNaCoords=(incident_df.notna()['GPS Coordinate Longitude']) & (incident_df.notna()['GPS Coordinate Latitude'])  
    #remove the points that are very far away
    ReasonableCoords_1=incident_df['GPS Coordinate Longitude']<0   
    incident_df=incident_df[NotNaCoords & ReasonableCoords_1]
    
    
    incident_df['geometry']=incident_df.apply(lambda ROW: sg.Point(ROW['GPS Coordinate Longitude'], ROW['GPS Coordinate Latitude']), axis=1)

    
    #the index in the orignal file is not unique so we correct that here
    incident_df=incident_df.reset_index().drop('index',axis=1)
    incident_df['Incident_ID']=range(len(incident_df))
        
    
    incident_df=incident_df.rename(columns={'ID NUMBER': 'ID_Original' })
    incident_df=incident_df.drop(['unit_segment_id', 'Time of Crash', 'Date of Crash', 'Co Seq', 'timestamp'], axis=1)   #Removing the infromation that are not useful. 
    
    incident_df['Dist_to_Seg']=np.nan
    incident_df['XDSegID']=np.nan
    
    
    incident_df=incident_df[(incident_df['time_local']> MetaData['Beg_Time']) & (incident_df['time_local']< MetaData['End_Time'])  ]
    incident_df=incident_df.reset_index().drop('index',axis=1)
    
    return incident_df


def min_distance(point, lines,Cap):
    '''
        Finding the closest segment to an accident:
    
        @param point: the incident
        @param lines: the segment geodataframe
        @param Cap: the maximum allowbale distance
        @return: the distance and the id of the closest segemnt to each accident
    '''
    if lines.empty:
        Dist=np.nan
        XDSegID=np.nan
    elif len(lines)==1:
        Dist=lines.distance(point).iloc[0]
        XDSegID=lines['XDSegID'].iloc[0]
    else:
        Dist=lines.distance(point).min()
        XDSegID=lines.iloc[lines.distance(point).argmin()]['XDSegID']          
    
    if Dist>Cap:  
        Dist=np.nan
        XDSegID=np.nan
        
    #print()    
    return  Dist, XDSegID


def Find_CLosest_Segment(incident_df, Line_df,MetaData):
    """
    Finding the closest segment to each accident:
        
    @param incident_df: the incident dataframe
    @param inrix_df_Address: the local address of the segment file
    @param MetaData: 
    @return: the distance and the id of the closest segemnt to each accident
    """
    if Line_df is None:
        try: 
            Line_df = pd.read_pickle('data/cleaned/inrix_grouped.pkl')
            print('inrix is loaded. It has the group column.')
        except:
            Line_df = pd.read_pickle('data/cleaned/inrix.pkl')
            print('inrix is loaded. It does not have the group column.')
    
    #Points
    #building a geopanda file for incidents
    incident_gdf_4326 = (gpd.GeoDataFrame(incident_df, geometry=incident_df['geometry'], crs={'init': 'epsg:4326'} )).copy()
    incident_gdf_3857 = incident_gdf_4326.to_crs(epsg = 3857)
    #Lines/Inrix        
    Line_gdf_4326 = (gpd.GeoDataFrame(Line_df, geometry=Line_df['geometry'], crs={'init': 'epsg:4326'} )).copy()
    Line_gdf_3857 = Line_gdf_4326.to_crs(epsg = 3857)
    
    #Buffer to find the zone search
    incident_gdf_3857_buffered=incident_gdf_3857.copy()
    incident_gdf_3857_buffered['Center']=incident_gdf_3857['geometry']
    incident_gdf_3857_buffered['geometry']=incident_gdf_3857.buffer(MetaData['Max_Alocation_Dist'])
    
    JOINT_DF = gpd.sjoin(Line_gdf_3857[['XDSegID','geometry']],
                         incident_gdf_3857_buffered[['Incident_ID','geometry','Center']],
                         how="inner", op='intersects').sort_values('Incident_ID').reset_index().drop(['index','index_right'],axis=1)

    incident_df['Dist_to_Seg'],incident_df['XDSegID']=zip(*incident_gdf_3857.apply(lambda ROW: min_distance(ROW['geometry'],
                                                                                                            JOINT_DF[JOINT_DF['Incident_ID']==ROW.Incident_ID],
                                                                                                            MetaData['Max_Alocation_Dist']),
                                                                                   axis=1))
    incident_df = pd.merge(incident_df, Line_df[['XDSegID', 'grouped_XDSegID']], left_on='XDSegID', right_on='XDSegID', how='left')
    return(incident_df)



def Prepare_Incident(incident_df_Address=None,inrix_df=None, MetaData=None,Swifter_Tag=False):
    start_time = time.time()   
    
    #inrix           
    if inrix_df is None:
        try:
            inrix_df = pd.read_pickle(MetaData['destination']+'inrix_grouped.pkl')
            print('inrix_df is loaded from default address: ', MetaData['destination']+'inrix_grouped.pkl')
        except:
            inrix_df = pd.read_pickle(MetaData['destination']+'inrix.pkl')
            print('inrix_df is loaded from default address: ', MetaData['destination']+'inrix.pkl')
    elif isinstance(inrix_df,str):
        print('inrix_df is loaded from ', inrix_df)
        inrix_df = pd.read_pickle(inrix_df) 
    elif isinstance(inrix_df, pd.core.frame.DataFrame):
        print('inrix_df is already loaded')
    
    
    
    incident_df=Read_Incident(incident_df_Address=MetaData['incident_df_Address'],MetaData=MetaData,Swifter_Tag=False)   
    #Save_DF(incident_df, Destination_Address=MetaData['destination']+'incident',Format='pkl', gpd_tag=False)  #incident_df = pd.read_pickle('data/cleaned/incident.pkl')
    #incident_df= pd.read_pickle("data/incidents/1-2017_5-2020_incidents_inrix_pd_net_TN.pk")     #should be removed later
    incident_df=Find_CLosest_Segment(incident_df, inrix_df[inrix_df['FRC']==0],MetaData)   
    Save_DF(incident_df, Destination_Address=MetaData['destination']+'incident_XDSegID',Format='pkl', gpd_tag=False)    #incident_df = pd.read_pickle('data/cleaned/incident_XDSegID.pkl')
    print('Total Numbe of accidents is:', len(incident_df))        
    print('Total Numbe of allocated accidents is:', len(incident_df['XDSegID'].notna()))            
    print("Reading Incident Time: --- %s seconds ---" % (time.time() - start_time)) 


    return incident_df

