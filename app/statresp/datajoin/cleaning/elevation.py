# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:04:34 2021

@author: vaziris
"""
import requests
import time
import json
import urllib
import pandas as pd
import geopandas as gpd
import numpy as np   
import pickle
import shapely.geometry as sg
from statresp.datajoin.cleaning.utils import Save_DF,Read_DF
from shapely.ops import nearest_points



def Nearest_Finder(point, pts,Points_df_3310):
    '''
    This function finds the nearest from pts (the reference points which have elevation data) to intented points.

    Parameters
    ----------
    point : dataframe
        it is a dataframe that includes all the the intented points.
    pts : shapely.geometry.multipoint.MultiPoint
        This is a shapely file includes all the reference points which have elevation data.
    Points_df_3310 : geopandas/pandas
        This dataframes includes the elevation data. In other words, pts was generated from this.

    Returns
    -------
    The function returns the closest point ID, its distance, and its elevation
    '''
    # find the nearest point and return the corresponding Place value
    nearest_geom = nearest_points(point['geometry'], pts)
    Nearest_ID=Points_df_3310[Points_df_3310.geometry == nearest_geom[1]]['ID'].values[0]
    Distance=nearest_geom[1].distance(point['geometry'])
    Elevation=Points_df_3310['Elevation'].values[0]
    if Distance<=10:
        return (Nearest_ID,Distance,Elevation)
    else:
        return (np.nan,np.nan,np.nan)

def request_elevation_usgs(lat,
                      lon,
                      units='Meters',
                      max_tries=10,
                      sec_btw_tries=1):
    '''
    
    This function is used to query elevation from usgs. The query should be one by one. 
    Parameters
    ----------
    lat : float
        The latitude of the point.
    lon : float
        The longitude of the point.
    units : string, optional
        The unit of interest for the elevation. The default is 'Meters'.
    max_tries : TYPE, optional
        The maximum number of iteration. The default is 10.
    sec_btw_tries : TYPE, optional
        The time interval between each iteration. The default is 1.

    Returns
    -------
    elevation : float
        The elevation of the point.

    '''

    usgs_url = r'https://nationalmap.gov/epqs/pqs.php?'
    usgs_params = {'output': 'json', 'x': lon, 'y': lat, 'units': units}
    for i in range(max_tries):
        try:
            usgs_request = requests.get(url=usgs_url,
                                        params=usgs_params)
            elevation = float(usgs_request.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
            break
        except Exception as e:
            print(e)
            elevation = None
            time.sleep(sec_btw_tries)
    return elevation



def request_elevation_gcp(List_Points):
    '''
    
    This function is used to query elevation from google maps. The query can be one by one or in batch. The unit of the output result is meter.
    To see how you can create and api key and setup your GCP please see the following links
    https://developers.google.com/maps/documentation/elevation/overview
    https://developers.google.com/maps/gmp-get-started
    
    Parameters
    ----------
    List_Points List or String: TYPE
        if list: it is a pair of longitude and latitude. For example [-90, 24]
        if string: it is seperated multiple pairs of longitude and latitude. For example "40.714728,-73.998672|-34.397,150.644"

    Returns
    -------
    elevationArray : List
        It is a list including the elevation of the requested points.

    '''
  
    
    apikey = ""
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    
    
    if isinstance(List_Points,str):
        print('string (batch) mode')
        request_ = urllib.request.urlopen(url+"?locations="+List_Points+"&key="+apikey)
    
    elif len(List_Points)==2:
        print('single mode')
        lat=List_Points[1]
        lng=List_Points[0]
        request_ = urllib.request.urlopen(url+"?locations="+str(lat)+","+str(lng)+"&key="+apikey)
    else:
        'The input is not correct.'
    try:
        results = json.load(request_).get('results')
        if 0 < len(results):
            elevationArray = []
            for resultset in results:
                elevationArray.append(resultset['elevation'])
                #elevation = results[0].get('elevation')
            # ELEVATION
            return elevationArray
        else:
            print('HTTP GET Request failed.')
    except ValueError:
        print('JSON decode failed: '+str(request_))



def Points_Builder(ROW, Point_Dict,Counter,Counter_unique):
    '''
    This function builds a dictionary usuing all points from all of the segments
    '''
    Points=list(ROW.geometry.coords)
    for P in Points:
        if P in Point_Dict.keys():
            Point_Dict[P]['XDSegID']=Point_Dict[P]['XDSegID']+[ROW.XDSegID]
            #print(Point_Dict[P]['ID'])
        else:
            Point_Dict[P]=dict()
            Point_Dict[P]['Elevation']=None
            Point_Dict[P]['ID']=Counter_unique
            Point_Dict[P]['Lon']=P[0]
            Point_Dict[P]['Lat']=P[1]
            Point_Dict[P]['XDSegID']=[ROW.XDSegID]
            Counter_unique=Counter_unique+1
        Counter=Counter+1
    return Point_Dict,Counter,Counter_unique




#Adding Elevation column to the inrix dataframe
def Elevation_Finder(ROW,Point_Dict):
    List_X=[]
    List_Y=[]
    List_Z=[]
    for Point in list(ROW.coords):
        try:
            Elevation=Point_Dict[Point]['Elevation']
        except:
            Elevation=np.nan
        #Elevation.append(Point[0])
        List_X.append(Point[0])
        List_Y.append(Point[1])
        List_Z.append(Elevation)
        
        #print(Point_Dict[Point])
    geometry3D=sg.LineString(zip(List_X,List_Y,List_Z))
    return List_X,List_Y, List_Z,geometry3D

#%%
def Elevation_Builder(inrix_df,MetaData):
    '''

    Parameters
    ----------
    inrix_df : TYPE
        DESCRIPTION.
    MetaData : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    
    '''
    Example
    #The following part is an example to run it
    #inrix_df: is a dataframe that includes a geometry column based on that the function goes to find the coordinates of the points. 
    #MetaData: is a dictionary includes some metadata information
    MetaData=dict()
    MetaData['Elevation_Tag']=='GCP'   #you can choose a tag between GCP or USGS
    inrix_df=Elevation_Builder(inrix_df,MetaData)
    '''
    
    Point_Dict=dict()
    Counter=0    
    Counter_unique=0    
    for i,ROW in inrix_df.iterrows():
        Point_Dict,Counter,Counter_unique=Points_Builder(ROW, Point_Dict,Counter,Counter_unique)
    print('Total Number of the points       : {} \nTotal Number of the unique points: {}'.format(Counter,Counter_unique))
        
        
    

    if MetaData['Elevation_Tag']=='USGS':
        print('The process of collecting elevation data from USGS has been starterd.')
        '''
        #One Point
        P=list(Point_Dict.keys())[0]
        print(P)
        Elevation=request_elevation_usgs(P[1], P[0])
        print(Elevation)
        '''
        for P in Point_Dict:
            #print(P)
            Point_Dict[P]['Elevation'] = request_elevation_usgs(P[1], P[0])
        #Point_Dict
    
    elif MetaData['Elevation_Tag']=='GCP':
        print('The process of collecting elevation data from GCP has been starterd.')
        
        #Since google api can call elevation data in batch, this code divides the points in to batch of 512-sized.
        SIZE=300
        Point_List=list(Point_Dict.keys())
        Point_Dict_Batch=dict()
        Counter=0
        while len(Point_List)>0:
            if len(Point_List)>=SIZE:
                Point_Dict_Batch[Counter]=dict()
                Point_Dict_Batch[Counter]['Points']=Point_List[0:SIZE]
                Point_List=Point_List[SIZE:]
                Counter=Counter+1
            elif len(Point_List)<SIZE:
                Point_Dict_Batch[Counter]=dict()
                Point_Dict_Batch[Counter]['Points']=Point_List
                Point_List=[]
        print('Total number of the batches: {}'.format(Counter+1))        
        
        #adding a string to each batch, which is the format google api needs for the input
        for Counter in Point_Dict_Batch.keys():
            STR=''     
            #print(B[b]['Points'])
            for P in Point_Dict_Batch[Counter]['Points']:
                STR=STR+str(P[1])+','+str(P[0])+'|'
            Point_Dict_Batch[Counter]['STR']=  STR[0:-1]
        #print(Point_Dict_Batch)       
        ''' 
        #One Point
        P=Point_Dict_Batch[0]['Points'][0] #or P=list(Point_Dict.keys())[0]
        print(P)
        Elevation=request_elevation_gcp(P)
        print(Elevation)        
        '''
        
        #Batch Mode
        for Counter in Point_Dict_Batch.keys():
            #print(Counter)
            Point_Dict_Batch[Counter]['Elevation']=request_elevation_gcp(Point_Dict_Batch[Counter]['STR'])        
                
        #Adding elevation to the original data
        for Counter in Point_Dict_Batch.keys():
            #print('Batch Number:',Counter)
            for i in range(len(Point_Dict_Batch[Counter]['Points'])):
                #print(' ',i)
                Point_Dict[Point_Dict_Batch[Counter]['Points'][i]]['Elevation'] =Point_Dict_Batch[Counter]['Elevation'][i]
        #Point_Dict
                
    else: 
        print("MetaData['Elevation_Tag'] is not defined properly.")
    
    
    
    
    
    inrix_df['Lon'],inrix_df['Lat'],inrix_df['Elevation'],inrix_df['geometry3D']=zip(*inrix_df['geometry'].apply(lambda ROW: Elevation_Finder(ROW,Point_Dict)))   
    return inrix_df,Point_Dict



def get_Elevation(qeury_point_df,Points_df=None):
    '''
    This function goes throught the qeury_point_df (a dataframe includes the points we want to know their eleivation) 
    row by row to find the elevation.

    Parameters
    ----------
    qeury_point_df : dataframe
        It is the dataframe includes the points we want to know their eleivation.
    Points_df : geopandas
        It is the dataframe includes the reference points with their elevation values.

    Returns
    -------
    qeury_point_df : dataframe
        The function adds 3 columns to the original dataframe. The function adds the closest point ID, its distance, and its elevation

    '''
    '''
    Example:
        #%% Query Elevation for a point(points)
        # unary union of the gpd2 geomtries 
        #you should create qeury_point_df, which includes the points and their id. 
        #Then you should load the Points_df, which includes a lot of points with their eleivation.
        #If Points_df=None, the function goes to load it from the default directory. 
        qeury_point_df=pd.DataFrame({'id':[0,1], 'geometry': [sg.Point(-84 ,35),sg.Point(-84.08076 ,35.86515) ]})
        qeury_point_df=get_Elevation(qeury_point_df,Points_df   )
        print(qeury_point_df.head())

    '''
    
    
    
    if Points_df is None:
        Points_df=pd.read_pickle('output/Points_df.pkl')
        
    qeury_point_df=gpd.GeoDataFrame(qeury_point_df, geometry=qeury_point_df['geometry'], crs='epsg:4326' )
    qeury_point_df_3310= qeury_point_df.to_crs('EPSG:3310')
    
    
    Points_df_3310= Points_df.copy()
    Points_df_3310= Points_df_3310.to_crs('EPSG:3310')
    pts = Points_df_3310.geometry.unary_union 
    
    qeury_point_df['Nearest_ID'], qeury_point_df['Distance_to_Nearest'], qeury_point_df['Elevation']= \
          zip(*qeury_point_df_3310.apply(lambda row: Nearest_Finder(row, pts,Points_df_3310), axis=1))
    return qeury_point_df

