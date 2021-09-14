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
from statresp.datajoin.cleaning.utils import Save_DF
import matplotlib.pyplot as plt
import shutil


'''
A=MetaData['incident_df_Address']
A = pd.read_csv(A) 
len(A)/1000
#1
A[(A['arrivalDateTime']=='None')][['arrivalDateTime']].iloc[0]
A=A[~(A['arrivalDateTime']=='None')][['arrivalDateTime']]
len(A)/1000
#2
A[(A['arrivalDateTime'].str[-1]=='Z')][['arrivalDateTime']].iloc[0]
A=A[~(A['arrivalDateTime'].str[-1]=='Z')][['arrivalDateTime']]
len(A)/1000
#3
A[(A['arrivalDateTime'].str[-6:]=='-06:00')][['arrivalDateTime']].iloc[0]
A=A[~(A['arrivalDateTime'].str[-6:]=='-06:00')][['arrivalDateTime']]
len(A)/1000
#4
A[(A['arrivalDateTime'].str[-6:]=='-05:00')][['arrivalDateTime']].iloc[0]
A=A[~(A['arrivalDateTime'].str[-6:]=='-05:00')][['arrivalDateTime']]
len(A)/1000
#5
A[(A['arrivalDateTime'].str[-7]==',')][['arrivalDateTime']].iloc[0]
A=A[~(A['arrivalDateTime'].str[-7]==',')][['arrivalDateTime']]
len(A)/1000




(incident_df[(incident_df['arrivalDateTime']=='None')]['response_time_sec']/3600).hist(bins=50);plt.xlabel('response time (hour)')

(incident_df[(A['arrivalDateTime'].str[-1]=='Z')]['response_time_sec']/3600).hist(bins=50);plt.xlabel('response time (hour)')
incident_df[(A['arrivalDateTime'].str[-1]=='Z')][['response_time_sec','arrivalDateTime','arrivalDateTime_Original']].sort_values('response_time_sec')

(incident_df[(A['arrivalDateTime'].str[-6:]=='-06:00')]['response_time_sec']/3600).hist(bins=50);plt.xlabel('response time (hour)')
incident_df[(A['arrivalDateTime'].str[-6:]=='-06:00')][['response_time_sec','arrivalDateTime','arrivalDateTime_Original']].sort_values('response_time_sec')

(incident_df[(A['arrivalDateTime'].str[-6:]=='-05:00')]['response_time_sec']/3600).hist(bins=50);plt.xlabel('response time (hour)')
incident_df[(A['arrivalDateTime'].str[-6:]=='-05:00')][['response_time_sec','arrivalDateTime','arrivalDateTime_Original']].sort_values('response_time_sec')

(incident_df[(A['arrivalDateTime'].str[-7]==',')]['response_time_sec']/3600).hist(bins=50);plt.xlabel('response time (hour)')
incident_df[(A['arrivalDateTime'].str[-7]==',')][['response_time_sec','arrivalDateTime','arrivalDateTime_Original']].sort_values('response_time_sec')


incident_df[['response_time_sec','arrivalDateTime','arrivalDateTime_Original']].sort_values('response_time_sec')
'''







'''
A=MetaData['incident_df_Address']
A = pd.read_csv(A) 
len(A)
#1
A[(A['arrivalDateTime']=='None')][['arrivalDateTime']].iloc[0]
A=A[~(A['arrivalDateTime']=='None')][['arrivalDateTime']]
len(A)/len(incident_df)



#2
B=A[(A['arrivalDateTime'].str[-1]=='Z')][['arrivalDateTime']]
B['test']=1
B['arrivalDateTime_2']=pd.to_datetime(B['arrivalDateTime'].str[:-1], format="%Y-%m-%dT%H:%M:%S.%f")-pd.DateOffset(hours=6)
B.dtypes
type(B.iloc[0]['arrivalDateTime_2'])
for i in range(len(B)):
    if isinstance( B.iloc[i]['arrivalDateTime_2'], pd._libs.tslibs.timestamps.Timestamp)==False:
        print(i)
        
        
A[(A['arrivalDateTime'].str[-1]=='Z')][['arrivalDateTime']].dtypes
A=A[~(A['arrivalDateTime'].str[-1]=='Z')][['arrivalDateTime']]
len(A)/len(incident_df)



#3
C=A[(A['arrivalDateTime'].str[-6:]=='-06:00')][['arrivalDateTime']]
C['test']=1
C['arrivalDateTime_2']=pd.to_datetime(C['arrivalDateTime'].str[0:19],format="%Y-%m-%dT%H:%M:%S")#-pd.DateOffset(hours=6)
C["arrivalDateTime_3"]= pd.to_datetime(C["arrivalDateTime_2"]) 
C.dtypes
type(C.iloc[0]['arrivalDateTime_2'])
for i in range(len(C)):
    if isinstance( C.iloc[i]['arrivalDateTime_2'], pd._libs.tslibs.timestamps.Timestamp)==False:
        print(i)
for i in range(len(C)):
    try:
        D=pd.to_datetime(C.iloc[i]["arrivalDateTime_2"]) 
    except:
        print(i)


A=A[~(A['arrivalDateTime'].str[-6:]=='-06:00')][['arrivalDateTime']]
len(A)/len(incident_df)
#4
B=A[(A['arrivalDateTime'].str[-6:]=='-05:00')][['arrivalDateTime']]
B['test']=1
B['arrivalDateTime_2']=pd.to_datetime(B['arrivalDateTime'].str[0:21],format="%Y-%m-%dT%H:%M:%S.%f")-pd.DateOffset(hours=5)
B.dtypes

A=A[~(A['arrivalDateTime'].str[-6:]=='-05:00')][['arrivalDateTime']]
len(A)/len(incident_df)




#5
B=A[(A['arrivalDateTime'].str[-7]==',')][['arrivalDateTime']]
B['test']=1
B['arrivalDateTime_2']=pd.to_datetime(B['arrivalDateTime'], format="%Y,%m,%d,%H,%M,%S,%f")-pd.DateOffset(hours=6)
B.dtypes

A=A[~(A['arrivalDateTime'].str[-7]==',')][['arrivalDateTime']]
len(A)/len(incident_df)

A.iloc[41873]['arrivalDateTime']


'''







def Timefixer_utc_to_local(DF,Time_Col='time_utc', Location_Col='County_Incident'): 
    '''
    This function is used to convert the utc time to local time based on the county the incident is located. 

    Parameters
    ----------
    DF : TYPE
        DESCRIPTION.
    Time_Col : TYPE, optional
        DESCRIPTION. The default is 'time_utc'.
    Location_Col : TYPE, optional
        DESCRIPTION. The default is 'County_Incident'.

    Returns
    -------
    TimewithZone : TYPE
        DESCRIPTION.
    TimewithZone_local : TYPE
        DESCRIPTION.

    '''
    central=['bedford', 'benton', 'bledsoe', 'cannon', 'carroll', 'cheatham', 'chester', 'clay', 'coffee', 'crockett', 'cumberland', 'davidson', 'decatur', 'dekalb', 'dickson', 'dyer', 'fayette', 'fentress', 'franklin', 'gibson', 'giles', 'grundy', 'hardeman', 'hardin', 'haywood', 'henderson', 'henry', 'hickman', 'houston', 'humphreys', 'jackson', 'lake', 'lauderdale', 'lawrence', 'lewis', 'lincoln', 'mcnairy', 'macon', 'madison', 'marion', 'marshall', 'maury', 'montgomery', 'moore', 'obion', 'overton', 'perry', 'pickett', 'putnam', 'robertson', 'rutherford', 'sequatchie', 'shelby', 'smith', 'stewart', 'sumner', 'tipton', 'trousdale', 'van', 'warren', 'wayne', 'weakley', 'white', 'williamson', 'wilson']
    eastern= ['anderson', 'blount', 'bradley', 'campbell', 'carter', 'claiborne', 'cocke', 'grainger', 'greene', 'hamblen', 'hamilton', 'hancock', 'hawkins', 'jefferson', 'johnson', 'knox', 'loudon', 'mcminn', 'meigs', 'monroe', 'morgan', 'polk', 'rhea', 'roane', 'scott', 'sevier', 'sullivan', 'unicoi', 'union', 'washington']
    if DF[Location_Col].lower() in central:
        ZONE = pytz.timezone('CST6CDT')
    elif DF[Location_Col].lower() in eastern:
        ZONE = pytz.timezone('EST5EDT')
    #print(DF['County'].lower() )    
    TimewithZone = pytz.timezone('UTC').localize(DF[Time_Col])
    TimewithZone_local = TimewithZone.astimezone(ZONE)
    return TimewithZone, TimewithZone_local


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
        incident_df = pd.read_csv(incident_df_Address)   #loading the incident from NFD
    except KeyError:
        print('Either the address or fromat is wrong.')
    
    #%%
    #if you want all incidents or just traffic
    if (MetaData['incident_filter'] is None)==False:
        incident_df=incident_df[incident_df[MetaData['incident_filter']['Col_name']].str.contains(MetaData['incident_filter']['Value'])]
    
    #%%
    
    
    #adding features and cleaning regarding time
    incident_df=incident_df.reset_index().drop('index',axis=1)
    
    
    #%%Fixing time
    #alarmDateTime
    incident_df['alarmDateTime']=pd.to_datetime(incident_df['alarmDateTime'],format="%Y-%m-%dT%H:%M:%S.%fz")
    incident_df['alarmDateTime']=incident_df['alarmDateTime']-pd.DateOffset(hours=0)
    incident_df['arrivalDateTime_Original']=incident_df['arrivalDateTime'].copy()
    #arrivalDateTime
    #1
    incident_df['arrivalDateTime']=incident_df['arrivalDateTime'].mask(incident_df['arrivalDateTime']=='None',None)
    #2
    incident_df['arrivalDateTime']=incident_df['arrivalDateTime'].mask(incident_df['arrivalDateTime'].str[-1]=='Z',
                                                                        (pd.to_datetime(incident_df[incident_df['arrivalDateTime'].str[-1]=='Z']['arrivalDateTime'].str[0:19],
                                                                                       format="%Y-%m-%dT%H:%M:%S")-pd.DateOffset(hours=0)))
    #3
    incident_df['arrivalDateTime']=incident_df['arrivalDateTime'].mask(incident_df['arrivalDateTime'].str[-6:]=='-06:00',
                                                                        (pd.to_datetime(incident_df[incident_df['arrivalDateTime'].str[-6:]=='-06:00']['arrivalDateTime'].str[0:19],
                                                                                       format="%Y-%m-%dT%H:%M:%S")+pd.DateOffset(hours=6)))  
    #4
    incident_df['arrivalDateTime']=incident_df['arrivalDateTime'].mask(incident_df['arrivalDateTime'].str[-6:]=='-05:00',
                                                                        pd.to_datetime(incident_df[incident_df['arrivalDateTime'].str[-6:]=='-05:00']['arrivalDateTime'].str[0:19],
                                                                                       format="%Y-%m-%dT%H:%M:%S")+pd.DateOffset(hours=5))  
    #5
    incident_df['arrivalDateTime']=incident_df['arrivalDateTime'].mask(incident_df['arrivalDateTime'].str[-7]==',',
                                                                        pd.to_datetime(incident_df[incident_df['arrivalDateTime'].str[-7]==',']['arrivalDateTime'].str[0:19],
                                                                                       format="%Y,%m,%d,%H,%M,%S")-pd.DateOffset(hours=0))    
    
    '''
    incident_df['arrivalDateTime']=incident_df['arrivalDateTime'].mask((incident_df['arrivalDateTime'].astype(str).str.isdigit()==False) & 
                                                                       (incident_df['arrivalDateTime'].notna()),
                                                                        pd.to_datetime(incident_df[(incident_df['arrivalDateTime'].astype(str).str.isdigit()==False) & (incident_df['arrivalDateTime'].notna()) ]['arrivalDateTime'].str[0:19],
                                                                                       format="%Y-%m-%d %H:%M:%S"))    
    '''
    
    incident_df['arrivalDateTime']=pd.to_datetime(incident_df['arrivalDateTime'])
    
    
    

    #%%
    #adding features and cleaning regarding space
    #remove coords with NA Values
    NotNaCoords=(incident_df.notna()['longitude']) & (incident_df.notna()['latitude'])  
    #remove the points that are very far away
    ReasonableCoords_1=incident_df['longitude']<0   
    incident_df=incident_df[NotNaCoords & ReasonableCoords_1]    
    
    #%% saving cleaned data
    incident_df.to_csv(incident_df_Address[:-4]+'_cleaned.csv')    
    
    #%%
    
    incident_df['County_Incident']='davidson'
    incident_df=incident_df.rename(columns={'alarmDateTime':'time_utc'})
    
    
    if Swifter_Tag==True:
        incident_df['time'], incident_df['time_utc'] =zip(*incident_df.swifter.apply(lambda ROW: Timefixer_utc_to_local(ROW), axis=1) ) #adding the time zones
        print('Swifter is working for correcting the incident time!')
    else:
        incident_df['time'], incident_df['time_local'] =zip(*incident_df.apply(lambda ROW: Timefixer_utc_to_local(ROW), axis=1) ) #adding the time zones
    incident_df['time_local']=incident_df['time_local'].dt.tz_localize(None)
    incident_df['response_time_sec']=(incident_df['arrivalDateTime']-incident_df['time_utc']).astype('timedelta64[s]')
    incident_df=incident_df.drop('arrivalDateTime', axis=1)     
    
    incident_df[['time_local','time_utc','time','response_time_sec']]
    incident_df=incident_df.rename(columns={'alarmDateTime':'time_local'})
    incident_df["year"]=incident_df['time_local'].dt.year
    incident_df["month"]=incident_df['time_local'].dt.month
    incident_df["day"]=incident_df['time_local'].dt.day
    incident_df["day_of_week"]=incident_df['time_local'].dt.dayofweek
    incident_df['weekend_or_not'] = (incident_df["day_of_week"]// 5 == 1).astype(int)
    incident_df["hour"]=incident_df['time_local'].dt.hour
    incident_df['window']=(np.floor((incident_df['time_local'].dt.hour+incident_df['time_local'].dt.minute/60)/MetaData['Window_Size'])).astype('int64')   
    
    

    
    
    incident_df['geometry']=incident_df.apply(lambda ROW: sg.Point(ROW['longitude'], ROW['latitude']), axis=1)

    
    #the index in the orignal file is not unique so we correct that here
    incident_df=incident_df.sort_values('time_local').reset_index().drop('index',axis=1)
    incident_df['Incident_ID']=range(len(incident_df))
        
    
    incident_df=incident_df.rename(columns={'_id': 'ID_Original' })
    incident_df=incident_df.drop(['incidentNumber', 'weather', 'alarm_datetime', 
                                  'alarm_date', 'alarm_time', 'alarm_year', 'location.coordinates',
                                  'location.type','respondingVehicles','arrivalDateTime_Original','County_Incident'], axis=1)   #Removing the infromation that are not useful. 
    
    incident_df['Dist_to_Seg']=np.nan
    incident_df['XDSegID']=np.nan
    
    
    #incident_df=incident_df[(incident_df['time_local']> MetaData['Beg_Time']) & (incident_df['time_local']< MetaData['End_Time'])  ]
    incident_df=incident_df.reset_index().drop('index',axis=1)

    (incident_df['response_time_sec']/3600).hist(bins=50);plt.xlabel('response time (hour)')
    
    return incident_df

#%%
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

#%%
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
    incident_gdf_3310 = incident_gdf_4326.to_crs(epsg = 3310)
    print('incidents gpd is done')
    #Lines/Inrix        
    Line_gdf_4326 = (gpd.GeoDataFrame(Line_df, geometry=Line_df['geometry'], crs={'init': 'epsg:4326'} )).copy()
    Line_gdf_3310 = Line_gdf_4326.to_crs(epsg = 3310)
    print('roadways gpd is done')
    #Buffer to find the zone search
    incident_gdf_3310_buffered=incident_gdf_3310.copy()
    incident_gdf_3310_buffered['Center']=incident_gdf_3310['geometry']
    incident_gdf_3310_buffered['geometry']=incident_gdf_3310.buffer(MetaData['Max_Alocation_Dist'])
    print('incident buffer is done')
    JOINT_DF = gpd.sjoin(Line_gdf_3310[['XDSegID','geometry']],
                         incident_gdf_3310_buffered[['Incident_ID','geometry','Center']],
                         how="inner", op='intersects').sort_values('Incident_ID').reset_index().drop(['index','index_right'],axis=1)
    print('joint of incidents and roadways is done',
          len(JOINT_DF),len(JOINT_DF['Incident_ID'].unique()),len(JOINT_DF['XDSegID'].unique()))
    incident_gdf_3310.iloc[0]
    
    incident_df['Dist_to_Seg'],incident_df['XDSegID']=zip(*incident_gdf_3310.apply(lambda ROW: min_distance(ROW['geometry'],
                                                                                                            JOINT_DF[JOINT_DF['Incident_ID']==ROW.Incident_ID],
                                                                                                            MetaData['Max_Alocation_Dist']),
                                                                                   axis=1))
    incident_df = pd.merge(incident_df, Line_df[['XDSegID', 'grouped_XDSegID']], left_on='XDSegID', right_on='XDSegID', how='left')
    print('finding the closest is done')    
    #%%drawing
    fig, ax = plt.subplots(figsize=(20,5))
    incident_gdf_3310_buffered.plot(color='red',ax=ax )
    incident_gdf_3310.plot(color='green',ax=ax )
    Line_gdf_3310.plot(color='red',ax=ax )
    #plt.xlim(2.9*1000000,3*1000000)  	
    #plt.ylim(3*100000,3.5*100000)  	 
    plt.show()
    
    return(incident_df)
    #%%


def Prepare_Incident_NFD(incident_df_Address=None,inrix_df=None, MetaData=None,Swifter_Tag=False):
    '''
    *Prepare_Incident_TDOT/Prepare_Incident_NFD:* These functions conduct preprocessing and cleaning analyses on the inrix roadway segment data set. It also adds/removes some features. For more information please refer to *Pipeline.pptx *
    choose Prepare_Incident_TDOT if your accident data set is collected from TDOT and use Prepare_Incident_NFD if your accident data set is collected from NFD 
    
        Input
        ----------
        incident_df_Address : String
            The location of incident file. The default is None.
        inrix_df : String or DataFrame, optional
            The location of inrix DataFrame or the DataFrame itself. If user provides None, the code automatically looks at the default location for this file. The default is None. 
        MetaData : Dictionary
            It includes the meta data. The default is None.
        Swifter_Tag : boolean
            A switch for using Swifter library or not. The default is False.  
            
        Returns
        -------
        incident_df : DataFrame
            This is the cleaned version of the incident data set in a DataFrame format. For more information, please refer to Pipeline.pptx
    '''     
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
    Save_DF(incident_df, Destination_Address='/'.join(MetaData['incident_df_Address'].split('/')[:-1])+'/incident_NFD',Format='pkl', gpd_tag=False)  #incident_df = pd.read_pickle('data/cleaned/incident.pkl')
    #incident_df= pd.read_pickle("data/incidents/1-2017_5-2020_incidents_inrix_pd_net_TN.pk")     #should be removed later
    
    
    if (MetaData['inrix_filter'] is None)==False:
        incident_df=incident_df[incident_df[MetaData['inrix_filter']['Col_name']]<=MetaData['inrix_filter']['Value']]

    incident_df=Find_CLosest_Segment(incident_df, inrix_df,MetaData)   
    #incident_df=Find_CLosest_Segment(incident_df, inrix_df[inrix_df['FRC']==0],MetaData)   
    
    Save_DF(incident_df, Destination_Address='/'.join(MetaData['incident_df_Address'].split('/')[:-1])+'/incident_NFD_XDSegID',Format='pkl', gpd_tag=False)    #incident_df = pd.read_pickle('data/cleaned/incident_XDSegID.pkl')
    print('Total Numbe of accidents is:', len(incident_df))        
    print('Total Numbe of allocated accidents is:', len(incident_df['XDSegID'].notna()))            
    print("Reading Incident Time: --- %s seconds ---" % (time.time() - start_time)) 

    shutil.copyfile('/'.join(MetaData['incident_df_Address'].split('/')[:-1])+'/incident_NFD_XDSegID.pkl', MetaData['destination']+'incident_XDSegID.pkl')

    return incident_df

