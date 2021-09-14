

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:50:15 2020

@author: vaziris
This scripts is  written by Sayyed Mohsen Vazirizade.  s.m.vazirizade@gmail.com
"""

from statresp.datajoin.cleaning.utils import Save_DF,Read_DF
from statresp.datajoin.cleaning.graph.graph import Prepare_Graph
from statresp.datajoin.cleaning.grouping_segments.Grouping_Functions import GroupMaker
from statresp.datajoin.cleaning.elevation import Elevation_Builder
import pandas as pd
import geopandas as gpd 
import numpy as np
import pyproj
import pygeoj
import shapely.geometry as sg
from shapely.ops import nearest_points
import time
import matplotlib.pyplot as plt



def County_ID_Allocator(inrix_df):
   Dict= {'countyid': {'knox': 2,
  'washington': 8,
  'carter': 47,
  'jefferson': 25,
  'claiborne': 61,
  'loudon': 27,
  'greene': 19,
  'sevier': 28,
  'roane': 29,
  'blount': 13,
  'sullivan': 6,
  'hawkins': 34,
  'monroe': 36,
  'anderson': 18,
  'hamblen': 41,
  'grainger': 68,
  'campbell': 49,
  'union': 80,
  'scott': 87,
  'hancock': 73,
  'morgan': 77,
  'cumberland': 23,
  'cocke': 37,
  'johnson': 82,
  'hamilton': 3,
  'rhea': 59,
  'bradley': 14,
  'unicoi': 71,
  'mcminn': 24,
  'polk': 56,
  'meigs': 84,
  'obion': 20,
  'weakley': 22,
  'lake': 93,
  'marion': 15,
  'lincoln': 33,
  'lawrence': 50,
  'sumner': 7,
  'smith': 51,
  'williamson': 4,
  'davidson': 1,
  'rutherford': 5,
  'lewis': 79,
  'putnam': 16,
  'wilson': 12,
  'sequatchie': 70,
  'bledsoe': 89,
  'maury': 11,
  'giles': 26,
  'dickson': 21,
  'bedford': 53,
  'hickman': 42,
  'montgomery': 10,
  'franklin': 48,
  'marshall': 39,
  'overton': 55,
  'robertson': 17,
  'cannon': 83,
  'grundy': 57,
  'warren': 40,
  'perry': 81,
  'houston': 90,
  'jackson': 85,
  'van buren': 86,
  'macon': 75,
  'cheatham': 66,
  'humphreys': 64,
  'dekalb': 74,
  'stewart': 60,
  'clay': 88,
  'coffee': 43,
  'wayne': 52,
  'fentress': 65,
  'shelby': 0,
  'decatur': 69,
  'chester': 76,
  'madison': 9,
  'dyer': 31,
  'gibson': 30,
  'hardeman': 46,
  'henry': 44,
  'pickett': 94,
  'carroll': 38,
  'hardin': 54,
  'mcnairy': 45,
  'henderson': 58,
  'crockett': 63,
  'benton': 72,
  'lauderdale': 67,
  'tipton': 62,
  'haywood': 35,
  'fayette': 32,
  'white': 78,
  'moore': 91,
  'trousdale': 92}}

   if 'County_inrix' in inrix_df.columns:
       CountyID_df=pd.DataFrame.from_dict(Dict).reset_index().rename(columns={'index': 'County_inrix'})
       inrix_df=pd.merge(inrix_df, CountyID_df, left_on='County_inrix',right_on='County_inrix', how='left')
   elif 'County' in inrix_df.columns:
       CountyID_df=pd.DataFrame.from_dict(Dict).reset_index().rename(columns={'index': 'County'})
       inrix_df=pd.merge(inrix_df, CountyID_df, left_on='County',right_on='County', how='left')
   elif 'county_inrix' in inrix_df.columns:
       CountyID_df=pd.DataFrame.from_dict(Dict).reset_index().rename(columns={'index': 'county_inrix'})
       inrix_df=pd.merge(inrix_df, CountyID_df, left_on='county_inrix',right_on='county_inrix', how='left')
   
   '''
   inrix_df=pd.read_pickle('D:\\inrix_new\\inrix_pipeline\\data\cleaned\\Line\\inrix\\inrix_weather_zslope_grouped.pkl')
   DF=pd.read_parquet('D:\\inrix_new\\inrix_pipeline\\data\cleaned\\Line\\inrix\\tn2021segment.parquet')
   DF['county']=DF['county'].str.lower()
   DF=DF[['county','countyid']].drop_duplicates().set_index('county')
   Dict=DF.T.to_dict('index');
   Dict
   CountyID_df=pd.DataFrame.from_dict(Dict).reset_index().rename(columns={'index': 'County_inrix'})
   inrix_df=pd.merge(inrix_df, CountyID_df, left_on='County_inrix',right_on='County_inrix', how='left')
   '''
   return inrix_df

def Beg_End_Finder(ROW):
    '''
    This function finds the beggining and end points of a line segment
    '''
    Beg=sg.Point(ROW.coords[0])
    End=sg.Point(ROW.coords[-1])
    return (Beg,End)


def Spatial_Filtering_of_Segments(inrix_df, incident_df,MetaData ):
    '''
    *Spatial_Filtering_of_Segments:*This Function use MetaData['Segment_Filtering_Tag'] to spatially filter the the roadway segments. 
    You can select just the segments with at least one incident: 'At_Least_One_Acc'
    You can select segments in a region or bounding box: 'Region'
    You can select all segmetns: 'All'

    Input
    ----------
    inrix_df : String or DataFrame, optional
        The location of inrix DataFrame or the DataFrame itself. If user provides None, the code automatically looks at the default location for this file. The default is None. 
    incident_df : String or DataFrame, optional
        The location of incident DataFrame or the DataFrame itself. If user provides None, the code automatically looks at the default location for this file. The default is None. 
        MetaData : Dictionary
    It includes the meta data. The default is None. 

    Raises
    ------
    NameError
        If you define something except At_Least_One_Acc,Region, or All.

    Returns
    -------
    List_segments : List
        List of inrix roadway segments based on the filter.

    '''
    
    
    if MetaData['Segment_Filtering_Tag']=='At_Least_One_Acc':
        
        List_segments=incident_df[incident_df['xdsegid'].notna()]['xdsegid'].astype('int64').unique()
        #adding neighbors
        '''
        List_segments=incident_df[incident_df['xdsegid'].notna()]['xdsegid'].astype('int64').unique()
        List_segments_group=inrix_df[inrix_df['xdsegid'].isin(List_segments)]['grouped_xdsegid'].astype('int64').unique()
        List_segments=inrix_df[inrix_df['grouped_xdsegid'].isin(List_segments_group)]['xdsegid'].astype('int64').unique()
        '''
    elif MetaData['Segment_Filtering_Tag'] == 'At_Least_One_Acc_and_frc_0':

        List_segments = incident_df[(incident_df['xdsegid'].notna()) & (incident_df['frc'].isin([0]))]['xdsegid'].astype('int64').unique()
        # adding neighbors


    elif MetaData['Segment_Filtering_Tag']=='Region': 
        Region = gpd.read_file(MetaData['Segment_Filtering_Region_Address'])
        
        '''
        #if you want the convex region
        Region=Region.iloc[0:1]
        Region_Convex=Region.convex_hull.reset_index()
        Region_Convex=Region_Convex.rename(columns={0:'geometry' })
        fig, ax = plt.subplots(figsize = (5,5)) 
        Region_Convex.plot(color='blue',ax=ax,label='Convex')
        Region.plot(color='red',ax=ax,label='Region')
        ax.legend()
        Region_Convex.to_file(MetaData['Segment_Filtering_Region_Address']+'/convex.shp')
        '''
        
        
        
        #incident_gdf_4326 = (gpd.GeoDataFrame(incident_df, geometry=incident_df['geometry'], crs={'init': 'epsg:4326'} )).copy()
        inrix_gdf_4326 = (gpd.GeoDataFrame(inrix_df, geometry=inrix_df['Center'], crs={'init': 'epsg:4326'} )).copy()
        Region_4326 =    (gpd.GeoDataFrame(Region, geometry=Region['geometry'],     crs={'init': 'epsg:4326'} )).copy()
    
        '''
        Region=pd.DataFrame({'District':['A'],
                           'lng':[[-86.9,-86.6,-86.6,-86.9]],
                           'lat':[[36.1,36.1,36.25,36.25]]})
        Region['All']=Region.apply(lambda ROW: list(zip(ROW.lng, ROW.lat)), axis=1)
        Region['Poly']=Region.apply(lambda ROW: sg.Polygon(ROW.All), axis=1)
        Region_4326 = gpd.GeoDataFrame(Region, geometry=Region.Poly, crs={'init': 'epsg:4326'} )
        '''
    
    
        JOINT_DF = gpd.sjoin(inrix_gdf_4326[['xdsegid','Center']],
                             Region_4326[['geometry']],
                             how="inner", op='intersects').sort_values('xdsegid').reset_index().drop(['index','index_right'],axis=1)
        List_segments=JOINT_DF[JOINT_DF['xdsegid'].notna()]['xdsegid'].astype('int64').unique()
        
    elif MetaData['Segment_Filtering_Tag']=='All' or len(MetaData['Segment_Filtering_Tag']==0):
        List_segments=inrix_df[inrix_df['xdsegid'].notna()]['xdsegid'].astype('int64').unique()
    
    else: 
        raise NameError('No Spatial filtering is defined!')
        
    inrix_List_segments_df=inrix_df[inrix_df['xdsegid'].isin(List_segments)]
    inrix_List_segments_gdf_4326 = (gpd.GeoDataFrame(inrix_List_segments_df, geometry=inrix_List_segments_df['geometry'], crs={'init': 'epsg:4326'} )).copy()    
        
    fig, ax = plt.subplots(figsize=(20,5))    
    inrix_List_segments_gdf_4326.plot(ax=ax,color='red')  
    
    
    return List_segments.tolist()

def iSFCalculator(Shape):
    """
    it calculates the inverse Stretch Factor (iSF) 
    @param Shape: the shape file you want to calculate the iSF for
    @return: the iSF value:   iSF_length: This is the iSF caluclated using the beggining and end of the shape  
                              iSF_length_min: This is the minimum of all iSF values which are calculated by considering all points 
    """
    #print(Shape)
    geod = pyproj.Geod(ellps='WGS84') #geod = pyproj.Geod(ellps='sphere')
    Points=list((Shape.coords))
    XYZ = list(zip(*Points))
    List=[]
    if len(XYZ[0])==1:
        return([np.nan,np.nan])
    elif len(XYZ[0])==2:
        return([1,1,sum(geod.inv(XYZ[0][0:-1],XYZ[1][0:-1],XYZ[0][1:],XYZ[1][1:])[2])])
    else:
        
        for i in np.arange(0,len(XYZ[0])-1,1):
            for j in np.arange(len(XYZ[0])-1,i+1,-1):
                #print(j,i)
                List.append(geod.inv(XYZ[0][i],XYZ[1][i],
                                     XYZ[0][j],XYZ[1][j])[2]/
                        sum(geod.inv(XYZ[0][i:j]        ,XYZ[1][i:j],
                                     XYZ[0][(i+1):(j+1)],XYZ[1][(i+1):(j+1)])[2]))
        Length_m=sum(geod.inv(XYZ[0][0:-1],XYZ[1][0:-1],XYZ[0][1:],XYZ[1][1:])[2])                
        return([List[0],min(List),Length_m])    #List[0] is from the first point to the last point




def SlopCalculator(ROW):
    """
    it calculates the slope
    @param Shape: the shape file you want to calculate the iSF for
    @return: the iSF value:   Slope: This is the iSF caluclated using the beggining and end of the shape  
                              Slope_median: This is the median of all iSF values which are calculated by considering all points 
    """
    
    Shape=ROW.geometry3D
    Length_m=ROW.Length_m
    Elevation=ROW.Elevation
    geod = pyproj.Geod(ellps='WGS84') #geod = pyproj.Geod(ellps='sphere')
    Points=list((Shape.coords))
    XYZ = list(zip(*Points))
    if len(XYZ[0])==1:
        return([np.nan,np.nan])
    else:
        Slope=(np.array(XYZ[2][-1])-np.array(XYZ[2][0]))/Length_m #np.array(geod.inv(XYZ[0][-1],XYZ[1][-1],XYZ[0][0],XYZ[1][0])[2])   
        Slopes=(np.array(XYZ[2][1:])-np.array(XYZ[2][:-1]))/np.array(geod.inv(XYZ[0][1:],XYZ[1][1:],XYZ[0][:-1],XYZ[1][:-1])[2])
        
        Threshold=10/100
        if (abs(Slope)>Threshold):
            Slope=np.nan
        for i in range(len(Slopes)):
            if (abs(Slopes[i])>Threshold):
                Slopes[i]=np.nan            
        
        if len(Slopes)>1:
            for i in range(len(Slopes)-1):
                if (Slopes[i] is np.nan) & (Slopes[i+1] is np.nan):
                    Elevation[i+1]=np.nan 
                
        
        Ends_Ele_Diff=np.array(XYZ[2][-1])-np.array(XYZ[2][0])
        Max_Ele_Diff=np.nanmax(np.array(XYZ[2]))-np.min(np.array(XYZ[2]))
        return([Slope, np.nanmedian(Slopes),Ends_Ele_Diff,Max_Ele_Diff,list(Slopes)] )  
        #return([Slope, np.median(Slopes),Ends_Ele_Diff,Max_Ele_Diff] )    
    
#%%

def calculate_nearest(row, destination):
    """
    it calulates the distacne from a shape (all weather stations) to a point (beggining of each segment) and find the closest shape id
    @param row: a row of a pandas dataframe (beggining of each segment) that we want to find the closest station to 
    @param destination: a gepandas datafrane including the shapes (all weather stations)
    @return: match_value:   the closest station_id to the row
    """
    # 1 - create unary union    
    dest_unary = destination["geometry"].unary_union
    # 2 - find closest point
    nearest_geom = nearest_points(row["geometry"], dest_unary)
    # 3 - Find the corresponding geom
    match_geom = destination.loc[destination.geometry == nearest_geom[1]]
    # 4 - get the corresponding value
    match_value = match_geom["station_id"].to_numpy()[0]
    return match_value
    
#%%
def Closest_weather_station_finder(inrix_df, Weather_station_df_Address_csv='data/weather/relevant_stations_df.csv',Swifter_Tag=False):
    """
    find the id of the closest weather station to each inrix segment
    @param inrix_df: dataframe that includes the information regarding inrix
    @param Weather_station_df_Address_csv: the local address of the weather station file. This file should be in csv format and includes the location and station id
    @return: inrix_df:   the original dataframe which now has another column showing the closest weather station id
    """    
    #Geom=inrix_df.apply(lambda ROW: sg.Point(ROW['geometry'].coords[0][0], ROW['geometry'].coords[0][1]), axis=1)
    Geom=inrix_df.apply(lambda ROW: sg.Point(ROW['Center'].coords[0][0], ROW['Center'].coords[0][1]), axis=1)
    crs = {'init': 'epsg:4326'}
    inrix_gdf_beg= inrix_df[['XDSegID']].copy()
    inrix_gdf_beg = gpd.GeoDataFrame(inrix_df[['XDSegID']], geometry=Geom, crs=crs).copy()
    
    Weather_station_df=pd.read_csv(Weather_station_df_Address_csv)
    crs = {'init': 'epsg:4326'}
    Weather_station_gdf = (gpd.GeoDataFrame(Weather_station_df, geometry=gpd.points_from_xy(Weather_station_df['lon'], Weather_station_df['lat']), crs=crs)).copy()
    


    if Swifter_Tag==True:
        inrix_gdf_beg["Nearest_Weather_Station"] = inrix_gdf_beg.swifter.apply(calculate_nearest, destination=Weather_station_gdf, axis=1)
        print('Swifter is working for finding the closest weather station to each segment!')    
    else:
        inrix_gdf_beg["Nearest_Weather_Station"] = inrix_gdf_beg.apply(calculate_nearest, destination=Weather_station_gdf, axis=1)
    
    inrix_df=pd.merge(inrix_df, inrix_gdf_beg[['XDSegID','Nearest_Weather_Station']], left_on='XDSegID', right_on='XDSegID',  how='left')
    
    return inrix_df
    

def UNIQUE(list1): 
    '''
    Some of the segments has multiple points with the exact same coordinates. This function removes dubplicate points. 

    Parameters
    ----------
    list1 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
      # intilize a null list 
    unique_list = []  
    # traverse for all elements 
    print(list1)
    print(type(list1))
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    # print list 
    return(unique_list)  

def Resolution_Increaseer(LINE, Est_new_Len=10):
    '''
    This function adds extra points to a line string to increase the resolution.
    '''
    #print(LINE.length)
    if LINE.length>Est_new_Len:
        distances = np.linspace(0, LINE.length, int(np.ceil(LINE.length/Est_new_Len)))
        POINTS = [LINE.interpolate(distance) for distance in distances]
        New_LINE = sg.LineString(POINTS)  # or new_line = LineString(points)
    else:
        New_LINE=LINE
    return New_LINE

#%%
def Read_Inrix(inrix_df_Address_json,inrix_df_Address_osm_csv=None,MetaData=None, Swifter_Tag=False):
    """
    Reading and cleaning the inrix dataframe and adding some features to it
    @param inrix_df_Address_json: the local address of the incident file in json format which includes the shapefiles
    @param inrix_df_Address_osm_csv: the local address of the incident file in csv format which includes the osm mapping
    @param MetaData: 
    @return: the cleaned inrix dataframe
    """
    
    #inrix_df_Address_osm_csv='data/inrix/MapSegments/osmconflation/USA_Tennessee_OSM_Mapping.csv'
    #inrix_df_Address_json='data/inrix/MapSegments/geojson/USA_Tennessee.geojson'
    
    
    inrix2=pygeoj.load(inrix_df_Address_json)
    #print(inrix2[0].properties.keys())
    data = []
    #bringing the data to a Dataframe
    for d in inrix2:
      data.append({'geometry': sg.shape(d.geometry),#sg.shape(UNIQUE(d.geometry)),
                   'XDSegID':d.properties['XDSegID'],
                   'PreviousXD':d.properties['PreviousXD'],               
                   'NextXDSegI':d.properties['NextXDSegI'],               
                   'FRC':d.properties['FRC'],                             
                   'County_inrix':d.properties['County'],
                   'Miles':d.properties['Miles'],
                   'Lanes':d.properties['Lanes'],
                   'SlipRoad':d.properties['SlipRoad'],               
                   'StartLat':d.properties['StartLat'],               
                   'StartLong':d.properties['StartLong'],               
                   'EndLat':d.properties['EndLat'],    
                   'EndLong':d.properties['EndLong'],    
                   'Bearing':d.properties['Bearing'],    
                   'XDGroup':d.properties['XDGroup'],                   
                   'RoadNumber':d.properties['RoadNumber'],               
                   'RoadName':d.properties['RoadName'], 
                   'RoadList':d.properties['RoadList']})
    inrix_df = pd.DataFrame(data)
    #fixing the data type
    inrix_df['XDSegID'] = inrix_df['XDSegID'].astype(float)
    inrix_df['PreviousXD'] = inrix_df['PreviousXD'].astype(float)#[None if i is None else int(i) for i in inrix_df['PreviousXD'] ]
    inrix_df['NextXDSegI'] = inrix_df['NextXDSegI'].astype(float)#[None if i is None else int(i) for i in inrix_df['NextXDSegI'] ]
    inrix_df['FRC'] = inrix_df['FRC'].astype(int)
    inrix_df['County_inrix'] = inrix_df['County_inrix'].astype(str)
    inrix_df['County_inrix']=inrix_df['County_inrix'].str.lower()
    inrix_df=County_ID_Allocator(inrix_df) #adding countyid
    inrix_df['Miles'] = inrix_df['Miles'].astype(float)
    inrix_df['Lanes'] = inrix_df['Lanes'].astype(float)
    inrix_df['SlipRoad'] = inrix_df['SlipRoad'].astype(int).astype(bool)
    inrix_df['StartLat'] = inrix_df['StartLat'].astype(float)
    inrix_df['StartLong'] = inrix_df['StartLong'].astype(float)
    inrix_df['EndLat'] = inrix_df['EndLat'].astype(float)
    inrix_df['EndLong'] = inrix_df['EndLong'].astype(float)
    inrix_df['Bearing'] = inrix_df['Bearing'].astype(str)
    inrix_df['XDGroup'] = inrix_df['XDGroup'].astype(float)#[None if i is None else int(i) for i in inrix_df['XDGroup'] ]
    inrix_df['RoadNumber'] = inrix_df['RoadNumber'].astype(str)
    inrix_df['RoadName'] = inrix_df['RoadName'].astype(str)
    inrix_df['RoadList'] = inrix_df['RoadList'].astype(str)
    inrix_df.dtypes
    
    
    
    
    
    inrix_df['Beg'],inrix_df['End']=zip(*inrix_df['geometry'].apply(lambda ROW: Beg_End_Finder(ROW)))  
    inrix_df['Beg_string'] = inrix_df['Beg'].apply(str)
    inrix_df['End_string'] = inrix_df['End'].apply(str)
    inrix_df['Center']=inrix_df.apply(lambda row: row['geometry'].centroid, axis=1)
    inrix_df['CenterEnd_string'] = inrix_df['Center'].apply(str)
    
    
    if (MetaData['inrix_increase_resolution'] is None) | (MetaData['inrix_increase_resolution']=={None} ):
        print('No spatial resolution change is required.')
        inrix_df=inrix_df.rename(columns={'geometry':'geometry_original'})
        inrix_df['geometry_original_string'] = inrix_df['geometry_original'].apply(str)  
    elif (isinstance(MetaData['inrix_increase_resolution'], int)) :
        print('Spatial resolution is changed to {}m.'.format(MetaData['inrix_increase_resolution'] ))
        #the following lines change the geometry column to geometry_original and add the high resolution geometry as geometry
        inrix_df['geometry_original']=inrix_df['geometry'].copy()
        inrix_gdf= (gpd.GeoDataFrame(inrix_df, geometry=inrix_df['geometry'], crs='epsg:4326' ))
        inrix_gdf = inrix_gdf.to_crs('EPSG:3310')
        inrix_gdf['geometry_3310']=inrix_gdf['geometry'].copy()
        
        inrix_gdf['geometry_HighRes_3310']=inrix_gdf.apply(lambda row: Resolution_Increaseer(row.geometry,MetaData['inrix_increase_resolution']) , axis=1)
        inrix_gdf= (gpd.GeoDataFrame(inrix_gdf, geometry=inrix_gdf['geometry_HighRes_3310'], crs='epsg:3310' ))
        inrix_gdf = inrix_gdf.to_crs('EPSG:4326')
        inrix_gdf['geometry_HighRes']=inrix_gdf['geometry'].copy()    
        
        inrix_df = pd.DataFrame(inrix_gdf)
        inrix_df = inrix_df.drop(['geometry_3310','geometry_HighRes_3310','geometry'],axis=1)
        
        inrix_df['geometry_original_string'] = inrix_df['geometry_original'].apply(str)  
        inrix_df['geometry_HighRes_string'] = inrix_df['geometry_HighRes'].apply(str)    
        
    inrix_df=inrix_df.rename(columns={'geometry_original':'geometry'})
    
    
    #Merging OSM
    try:
        inrix_df_osm=pd.read_csv(inrix_df_Address_osm_csv)
        inrix_df=pd.merge(inrix_df, inrix_df_osm, left_on='XDSegID', right_on='XDSegID',  how='left')
    except :
        print('No OSM mapping is employed!')
        
    '''    
    #Sainity Check    Index 12243, the beggining and end are the same
    for i in range(len(inrix_df)):
        if inrix_df.iloc[i]['geometry'].geom_type != "LineString":
            print(i, 'Warning: This index is not a LineString')
        if len(inrix_df.iloc[i]['geometry'].xy[0])<2:
            print(i, 'Warning: This index is does not have enough points')    
    '''       
    '''
    def Func(inrix_df):
            inrix_df['iSF_length'],inrix_df['iSF_length_min']=zip(*inrix_df.apply(lambda ROW: iSFCalculator(ROW.geometry), axis=1)) 
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Done')
            return inrix_df
    inrix_df['iSF_length']=np.nan
    inrix_df['iSF_length_min']=np.nan
    inrix_df=dd.from_pandas(inrix_df, npartitions=4*multiprocessing.cpu_count()).map_partitions(Func,meta=inrix_df).compute(scheduler='processes')  #processes     single-threaded
    '''
    
    
    
    
    
    if Swifter_Tag==True:
        inrix_df['iSF_length'],inrix_df['iSF_length_min']=zip(*inrix_df.swifter.apply(lambda ROW: iSFCalculator(ROW.geometry), axis=1))
        #dd.from_pandas(inrix_df, npartitions=4*multiprocessing.cpu_count()).map_partitions(lambda df: df.apply((lambda ROW: iSFCalculator(ROW.geometry)), axis=1)).compute(scheduler='processes')
        print('Swifter is working for calculating iSF!')
    else:
        inrix_df['iSF_length'],inrix_df['iSF_length_min'],inrix_df['Length_m']=zip(*inrix_df.apply(lambda ROW: iSFCalculator(ROW.geometry), axis=1))            
    print('inverse Scale Factor is added.')        
    
    return inrix_df





def Elevation_Analysis(inrix_df,MetaData,Swifter_Tag=False):
    Save_GCP_z_DF=MetaData['destination']+'inrix/elevation/'+'inrix_gcp_z.pkl'
    Save_GCP_z_Points=MetaData['destination']+'inrix/elevation'+'Points_z_df.pkl'
    try:
        inrix_df=pd.read_pickle(Save_GCP_z_DF)
        print('Elevation from GCP is already available. It is loaded from: ',Save_GCP_z_DF)
       
    except:
        print('Warning! The elevation data is being extracted from GCP and you will be charged for it.')
        x
        inrix_df,Points_Dic=Elevation_Builder(inrix_df,MetaData)
        inrix_df.to_pickle(Save_GCP_z_DF)
        #Points_df
        Points_df=pd.DataFrame.from_dict(Points_Dic, orient='index').reset_index().drop(['level_0','level_1'],axis=1)
        Points_df['geometry']=Points_df.apply(lambda ROW: sg.Point(ROW.Lon, ROW.Lat), axis=1)
        Points_df= gpd.GeoDataFrame(Points_df, geometry=Points_df['geometry'], crs='epsg:4326' )
        Points_df['geometry_string'] = Points_df['geometry'].apply(str)
        Points_df.to_pickle(Save_GCP_z_Points)
                
    #adding slope
    print('The input data is 3D')
    if Swifter_Tag==True:
        inrix_df['Slope'],inrix_df['Slope_median'],inrix_df['Ends_Ele_Diff'],inrix_df['Max_Ele_Diff'],inrix_df['Slopes']=zip(*inrix_df.swifter.apply(lambda ROW: SlopCalculator(ROW), axis=1))
        print('Swifter is working for calculating Slope!')
    else:
      
        inrix_df['Slope'],inrix_df['Slope_median'],inrix_df['Ends_Ele_Diff'],inrix_df['Max_Ele_Diff'],inrix_df['Slopes']=zip(*inrix_df.apply(lambda ROW: SlopCalculator(ROW), axis=1))
    return inrix_df


            
            
      


def Prepare_Inrix(inrix_df_Address_json=None,inrix_df_Address_osm_csv=None,
                       Weather_station_df_Address_csv=None,
                       MetaData=None,Swifter_Tag=False):
    '''
    *Prepare_Inrix:* This function conducts preprocessing and cleaning analyses on the inrix roadway segment data set. It also adds/removes some features. For more information please refer to *Pipeline.pptx *
    
        Input
        ----------
        inrix_df_Address_json : String
            The location of inrix json files. The default is None.
        inrix_df_Address_osm_csv : String, optional
            The location of inrix to osm maping file. The default is None.
        Weather_station_df_Address_csv : String
            The location of the file including the weather station coordinates. The default is None.
        MetaData : Dictionary
            It includes the meta data. The default is None.
        Swifter_Tag : boolean
            A switch for using Swifter library or not. The default is False.  
            
        Returns
        -------
        inrix_df : DataFrame
            This is the cleaned version of the inrix data set in a DataFrame format. For more information, please refer to Pipeline.pptx
    '''
    
    
    start_time = time.time()    
    inrix_df=Read_Inrix(inrix_df_Address_json=inrix_df_Address_json,inrix_df_Address_osm_csv=inrix_df_Address_osm_csv,MetaData=MetaData,Swifter_Tag=False)
    #Save_DF(inrix_df, Destination_Address=MetaData['destination']+'inrix/'+'inrix',Format='pkl', gpd_tag=False) #inrix_df = pd.read_pickle('data/cleaned/inrix.pkl')
    print("Reading Inrix Time: --- %s seconds ---" % (time.time() - start_time))
    inrix_df=Closest_weather_station_finder(inrix_df, Weather_station_df_Address_csv=Weather_station_df_Address_csv,Swifter_Tag=False)
    print('Closest weather station to Inrix is found.')
    #Save_DF(inrix_df, Destination_Address=MetaData['destination']+'inrix/'+'inrix_weather',Format='pkl', gpd_tag=False) #inrix_df = pd.read_pickle('data/cleaned/inrix.pkl')
    print("Adding Weather to Inrix Time: --- %s seconds ---" % (time.time() - start_time)) 
    if MetaData['Elevation_Tag'] in ['GCP','USGS']:
        inrix_df=Elevation_Analysis(inrix_df,MetaData,Swifter_Tag=False)
        #Save_DF(inrix_df, Destination_Address=MetaData['destination']+'inrix/'+'inrix_weather_zslope',Format='pkl', gpd_tag=False) 
    print("Reading Elevation Time: --- %s seconds ---" % (time.time() - start_time)) 
    ## Inrix Grouping
    inrix_df,DF_grouped=GroupMaker(inrix_df,MetaData,Type_Tag=MetaData['Grouping']['Type_Tag'])
    
    if 'MyGrouping_1_id' in inrix_df.columns:
        inrix_df = inrix_df.drop(['MyGrouping_1_id', 'MyGrouping_1'], axis=1)
    #Save_DF(inrix_df, Destination_Address=MetaData['destination']+'inrix/'+'inrix_weather_zslope_grouped',Format='pkl', gpd_tag=False) #inrix_df = pd.read_pickle('data/cleaned/inrix_grouped.pkl')
    print("Grouping Inrix Time: --- %s seconds ---" % (time.time() - start_time)) 
    #%%Graph
    DF_adj, L_XDSegID_Nodes, G_XDSegID_Edges=Prepare_Graph(inrix_df,MetaData)
    Save_DF(L_XDSegID_Nodes, Destination_Address=MetaData['destination']+'inrix/graph/'+'inrix_graph',Format='gpickle', gpd_tag=False) 
    #Save_DF(G_XDSegID_Edges, Destination_Address=MetaData['destination']+'inrix/graph/'+'inrix_graph_XDSegID_Edges',Format='gpickle', gpd_tag=False)
    #nx.write_gpickle(L_XDSegID, "inrix_graph.gpickle")
    #L_XDSegID = nx.read_gpickle("inrix_graph.gpickle")
    Save_DF(DF_adj, Destination_Address=MetaData['destination']+'inrix/graph/'+'inrix_graph_adj',Format='pkl', gpd_tag=False)
    #Save_DF(DF_adj, Destination_Address=MetaData['destination']+'inrix/graph/'+'inrix_graph_adj',Format='json', gpd_tag=False)     
    print("Inrix Building Graph Time: --- %s seconds ---" % (time.time() - start_time)) 
    #%%
    inrix_df = Read_DF(inrix_df=None, Reading_Tag='inrix_df',MetaData=MetaData) 
    inrix_df_newcol=inrix_df.copy()
    inrix_df_newcol.columns=inrix_df_newcol.columns.str.lower().str.replace(' ','_')
    Save_DF(inrix_df_newcol, Destination_Address=MetaData['destination']+'inrix/'+'inrix_weather_zslope_grouped_renamedcolumns',Format='pkl', gpd_tag=False) #inrix_df = pd.read_pickle('data/cleaned/inrix_grouped.pkl')

    return inrix_df_newcol,DF_grouped
    

    #inrix=pd.read_pickle('data/cleaned/Grid/inrix.pkl')
    #inrix[(inrix['County_inrix']=='davidson') &  (inrix['FRC']==0)]['XDSegID'].iloc[-1]
