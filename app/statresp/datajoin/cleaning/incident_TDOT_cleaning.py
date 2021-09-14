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
from statresp.datajoin.cleaning.utils import Save_DF,Read_DF
from statresp.datajoin.cleaning.cleanCrashesTDOT import get_cleaned_picke_form_all_csvs
import shutil
import os


def Timefixer_local_to_utc(DF,Time_Col='time_local', Location_Col='county_incident'): 
    '''
    This function is used to convert the local time to utc time based on the county the incident is located. 

    Parameters
    ----------
    DF : TYPE
        DESCRIPTION.
    Time_Col : TYPE, optional
        DESCRIPTION. The default is 'time_local'.
    Location_Col : TYPE, optional
        DESCRIPTION. The default is 'county_incident'.

    Returns
    -------
    TimewithZone : TYPE
        DESCRIPTION.
    TimewithZone_utc : TYPE
        DESCRIPTION.

    '''
    central=['bedford', 'benton', 'bledsoe', 'cannon', 'carroll', 'cheatham', 'chester', 'clay', 'coffee', 'crockett', 'cumberland', 'davidson', 'decatur', 'dekalb', 'dickson', 'dyer', 'fayette', 'fentress', 'franklin', 'gibson', 'giles', 'grundy', 'hardeman', 'hardin', 'haywood', 'henderson', 'henry', 'hickman', 'houston', 'humphreys', 'jackson', 'lake', 'lauderdale', 'lawrence', 'lewis', 'lincoln', 'mcnairy', 'macon', 'madison', 'marion', 'marshall', 'maury', 'montgomery', 'moore', 'obion', 'overton', 'perry', 'pickett', 'putnam', 'robertson', 'rutherford', 'sequatchie', 'shelby', 'smith', 'stewart', 'sumner', 'tipton', 'trousdale', 'van', 'van buren', 'warren', 'wayne', 'weakley', 'white', 'williamson', 'wilson']
    eastern= ['anderson', 'blount', 'bradley', 'campbell', 'carter', 'claiborne', 'cocke', 'grainger', 'greene', 'hamblen', 'hamilton', 'hancock', 'hawkins', 'jefferson', 'johnson', 'knox', 'loudon', 'mcminn', 'meigs', 'monroe', 'morgan', 'polk', 'rhea', 'roane', 'scott', 'sevier', 'sullivan', 'unicoi', 'union', 'washington']
    if DF[Location_Col].lower() in central:
        ZONE = pytz.timezone('CST6CDT')
    elif DF[Location_Col].lower() in eastern:
        ZONE = pytz.timezone('EST5EDT')
    #print(DF['county'].lower() )    
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
        #incident_df_2 = pd.read_pickle('data/incidents_TDOT/tdot_testing_incidents_2017-2020_allTN.pk') 
        #incident_df_3 = pd.read_pickle('data/incidents_TDOT/tdot_testing_incidents_2006-2021_allTN_old.pkl') 
    except KeyError:
        print('Either the address or fromat is wrong.')



    if 'point' in incident_df.columns:
        incident_df=incident_df.drop('point',axis=1)
    #adding features and cleaning regarding time
    incident_df=incident_df.reset_index().drop('index',axis=1)
    incident_df['county']=incident_df['county'].str.lower()
    incident_df=incident_df.rename(columns={'county':'county_incident'})
    incident_df=incident_df.rename(columns={'time':'time_local'})
    if isinstance(incident_df['time_local'].iloc[0],str):
        incident_df['time_local']=pd.to_datetime(incident_df['time_local'],format="%Y-%m-%d  %H:%M:%S")
    #removing the incident with no time
    incident_df=incident_df[incident_df['time_local'].notna()]

    if Swifter_Tag==True:
        incident_df['time'], incident_df['time_utc'] =zip(*incident_df.swifter.apply(lambda ROW: Timefixer_local_to_utc(ROW,Time_Col='time_local', Location_Col='county_incident'), axis=1) ) #adding the time zones
        print('Swifter is working for correcting the incident time!')
    else:
        incident_df['time'], incident_df['time_utc'] =  zip(*incident_df.apply(lambda ROW: Timefixer_local_to_utc(DF=ROW,Time_Col='time_local', Location_Col='county_incident'), axis=1) ) #adding the time zones
    incident_df['time_utc']=incident_df['time_utc'].dt.tz_localize(None)
    incident_df["year"]=incident_df['time_local'].dt.year
    incident_df["month"]=incident_df['time_local'].dt.month
    incident_df["day"]=incident_df['time_local'].dt.day
    incident_df["day_of_week"]=incident_df['time_local'].dt.dayofweek
    incident_df['weekend_or_not'] = (incident_df["day_of_week"]// 5 == 1).astype(int)
    incident_df["hour"]=incident_df['time_local'].dt.hour
    incident_df['window']=(np.floor((incident_df['time_local'].dt.hour+incident_df['time_local'].dt.minute/60)/MetaData['Window_Size'])).astype('int64')   


    #adding features and cleaning regarding space
    #remove coords with NA Values
    NotNaCoords=(incident_df.notna()['gps_coordinate_longitude']) & (incident_df.notna()['gps_coordinate_latitude'])  
    #remove the points that are very far away
    ReasonableCoords_1=incident_df['gps_coordinate_longitude']<0   
    incident_df=incident_df[NotNaCoords & ReasonableCoords_1]
    
    if not 'geometry' in incident_df.columns:
        incident_df['geometry']=incident_df.apply(lambda ROW: sg.Point(ROW['gps_coordinate_longitude'], ROW['gps_coordinate_latitude']), axis=1)
    if not 'geometry_string' in incident_df.columns:
        incident_df['geometry_string'] = incident_df['geometry'].apply(str)    
    
    #the index in the orignal file is not unique so we correct that here
    incident_df=incident_df.sort_values('time_local').reset_index().drop('index',axis=1)
    incident_df['incident_id']=range(len(incident_df))
        
    
    incident_df=incident_df.rename(columns={'id_number': 'id_original' })
    incident_df=incident_df.drop(['time_of_crash', 'date_of_crash', 'year_of_crash', 'county_sequence', 'timestamp'], axis=1)   #Removing the infromation that are not useful. 
    try:
        incident_df=incident_df.drop(['unit_segment_id'], axis=1)   #Removing the infromation that are not useful. 
    except:
        pass
  
    incident_df['dist_to_seg']=np.nan
    incident_df['xdsegid']=np.nan
    
    
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
        xdsegid=np.nan
    elif len(lines)==1:
        Dist=lines.distance(point).iloc[0]
        xdsegid=lines['xdsegid'].iloc[0]
    else:
        Dist=lines.distance(point).min()
        xdsegid=lines.iloc[lines.distance(point).argmin()]['xdsegid']
        
    if Dist>Cap:  
        Dist=np.nan
        xdsegid=np.nan

        
    #print()    
    return  Dist, xdsegid


def Find_CLosest_Segment(incident_df, Line_df,MetaData):
    """
    Finding the closest segment to each accident:
        
    @param incident_df: the incident dataframe
    @param inrix_df_Address: the local address of the segment file
    @param MetaData: 
    @return: the distance and the id of the closest segemnt to each accident
    """
    print('Finding the closest segment to each incident is started:')

    
    #Points
    #building a geopanda file for incidents
    incident_gdf_4326 = (gpd.GeoDataFrame(incident_df, geometry=incident_df['geometry'], crs={'init': 'epsg:4326'} )).copy()
    incident_gdf_3310 = incident_gdf_4326.to_crs(epsg = 3310)
    #Lines/Inrix        
    Line_gdf_4326 = (gpd.GeoDataFrame(Line_df, geometry=Line_df['geometry'], crs={'init': 'epsg:4326'} )).copy()
    Line_gdf_3310 = Line_gdf_4326.to_crs(epsg = 3310)
    
    #Buffer to find the zone search
    print('Adding Buffer:')
    incident_gdf_3310_buffered=incident_gdf_3310.copy()
    incident_gdf_3310_buffered['Center']=incident_gdf_3310['geometry']
    incident_gdf_3310_buffered['geometry']=incident_gdf_3310.buffer(MetaData['Max_Alocation_Dist'])
    
    
    print('Finding Joint:')
    JOINT_DF = gpd.sjoin(Line_gdf_3310[['xdsegid','geometry','frc']],
                         incident_gdf_3310_buffered[['incident_id','geometry','Center']],
                         how="inner", op='intersects').sort_values('incident_id').reset_index().drop(['index','index_right'],axis=1)
    print('JOINT_DF:',len(JOINT_DF))
    print('Finding Closest:')
    incident_df['dist_to_seg'],incident_df['xdsegid']=zip(*incident_gdf_3310.apply(lambda ROW: min_distance(ROW['geometry'],
                                                                                                            JOINT_DF[JOINT_DF['incident_id']==ROW.incident_id],
                                                                                                            MetaData['Max_Alocation_Dist']),
                                                                                   axis=1))
    incident_df=pd.merge(incident_df,Line_df[['xdsegid','grouped_xdsegid','frc','county_inrix']], left_on='xdsegid', right_on='xdsegid', how='left' )
    return(incident_df)



def Prepare_Incident_TDOT(incident_df_Address=None,inrix_df=None, MetaData=None,Swifter_Tag=False):
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
    inrix_df = Read_DF(inrix_df=inrix_df, Reading_Tag='inrix_df',MetaData=MetaData)     

    
    #if the input is a folder of csv files, it the following lines collect them and put them in one csv file. 
    if os.path.isdir(MetaData['incident_df_Address']):
       incident_df=get_cleaned_picke_form_all_csvs(MetaData)
       MetaData['incident_df_Address']=('/').join(MetaData['incident_df_Address'].split('/')[0:-1])+'/tdot_incidents_TN_'+str(incident_df['year_of_crash'].min())+'_to_'+str(incident_df['year_of_crash'].max())
       Save_DF(incident_df, MetaData['incident_df_Address'],Format='pkl', gpd_tag=False)
     
       
    incident_df=Read_Incident(incident_df_Address=MetaData['incident_df_Address'],MetaData=MetaData,Swifter_Tag=False)   
    Save_DF(incident_df, Destination_Address='/'.join(MetaData['incident_df_Address'].split('/')[:-1])+'/incident_TDOT',Format='pkl', gpd_tag=False) 
    

    
    #incident_df=incident_df.iloc[0:10000]
    start_time = time.time() 
    if not MetaData['inrix_filter'] is None: 
        incident_df=Find_CLosest_Segment(incident_df, inrix_df[inrix_df[MetaData['inrix_filter']['Col_name']].isin(MetaData['inrix_filter']['Value'])],MetaData)   
    else:
        incident_df=Find_CLosest_Segment(incident_df, inrix_df,MetaData) 
    print("Reading Incident Time: --- %s seconds ---" % (time.time() - start_time))     
    #Save_DF(incident_df, Destination_Address='/'.join(MetaData['incident_df_Address'].split('/')[:-1])+'/incident_TDOT_xdsegid',Format='pkl', gpd_tag=False)    #incident_df = pd.read_pickle('data/cleaned/incident_xdsegid.pkl')
    #shutil.copyfile('/'.join(MetaData['incident_df_Address'].split('/')[:-1])+'/incident_TDOT_xdsegid.pkl', MetaData['destination']+'incident/'+'incident_xdsegid.pkl')
    incident_df.columns=incident_df.columns.str.lower().str.replace(' ','_')
    Save_DF(incident_df,MetaData['destination']+'incident/'+'incident_xdsegid',Format='pkl', gpd_tag=False)
    #incident_df=pd.read_pickle(MetaData['destination']+'incident/'+'incident_xdsegid.pkl')
    ##saving as shp
    incident_df_shp=incident_df.drop(['time_local','time_utc','time','geometry_string','log_mle','hazmat_involved','updated_by','updated_on','special_case'],axis=1).copy()
    incident_df_shp['frc']=incident_df_shp['frc'].fillna(5)
    Save_DF(incident_df_shp, MetaData['destination']+'incident/'+'incident_xdsegid/' + 'incidents_ESRI.shp',Format='shp', gpd_tag=False)
    
    print('Total Numbe of accidents is:', len(incident_df))        
    print('Total Numbe of allocated accidents is:', len(incident_df['xdsegid'].notna()))            
    print("Reading Incident Time: --- %s seconds ---" % (time.time() - start_time))
    if not 'grouped_xdsegid' in incident_df.columns:
        print('Warning! grouped_xdsegid does not exist among the columns.')

    return incident_df

#incident_df = Read_DF(incident_df=MetaData['destination']+'incident/'+'incident_xdsegid.pkl', #Reading_Tag='incident_df',MetaData=MetaData)  

