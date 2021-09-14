# -*- coding: utf-8 -*-
"""
@Author - Sayyed Mohsen Vazirizade  s.m.vazirizade@vanderbilt.edu
"""
import pandas as pd
import shapely.geometry as sg
import geopandas as gpd

def Spatial_Filtering(df,metadata):
    '''
    modified the segment_list_pred based on the county and polygon


    Parameters
    ----------
    df : dataframe
        The df we want to filter basd on segment_list_pred.
    metadata : TYPE
        DESCRIPTION.

    Raises
    ------
    NameError
        DESCRIPTION.

    Returns
    -------
    df: df
        Filtered df.
    segment_list_pred : TYPE
        if the original segment_list_pred is a list of segments, the modified segment_list_pred includes just the qualified segments.
        if the original segment_list_pred is a list of counties, the modfieid segment_list_pred includes the qualified segments in those counties

    '''
    #Counties=['davidson', 'shelby','knox','hamilton']
    segment_list_pred = df[metadata['unit_name']].drop_duplicates().tolist()

    if not ((len(metadata['county_list_pred'])==0) or (metadata['county_list_pred'] is None)):
        df = df[df['county'].isin(metadata['county_list_pred'])]
        segment_list_pred = df[metadata['unit_name']].drop_duplicates().tolist()
        if len(segment_list_pred)==0:
                raise NameError('Spatial filtering results in zero segments. Please modify your filter')

    if not ((len(metadata['segment_list_pred'])==0) or (metadata['segment_list_pred'] is None)):
        df = df[df[metadata['unit_name']].isin(metadata['segment_list_pred'])]
        segment_list_pred = df[metadata['unit_name']].drop_duplicates().tolist()
        if len(segment_list_pred) == 0:
            raise NameError('Spatial filtering results in zero segments. Please modify your filter')

    if not ((len(metadata['polygon'])==0) or (metadata['polygon'] is None)):
        if len(metadata['polygon'])<3:
            raise NameError('Not enough points to make a polygon. ')
        else:
            #making the polygon
            DF_polygon=pd.DataFrame(metadata['polygon'],columns=['lon','lat'])
            DF_polygon['Point']=DF_polygon.apply(lambda ROW: sg.Point(ROW.lon, ROW.lat), axis=1)
            DF_polygon = gpd.GeoDataFrame(pd.DataFrame({'Polygon':[1]}), geometry=[sg.Polygon(DF_polygon.Point)])
            #making the line segments
            if not 'geometry' in df.columns:
                if metadata['inrix_pickle_address'][-4:] == '.pkl':
                    DF_seg_geom = pd.read_pickle(metadata['inrix_pickle_address'])
                else:
                    DF_seg_geom = pd.read_parquet(metadata['inrix_pickle_address'])
                    uncompatible_list = ['beg', 'end', 'center', 'geometry', 'geometry3d', 'geometry_highres']
                    for i in uncompatible_list:
                        if i in DF_seg_geom.columns:
                            # df[i] = df[i].astype(str)
                            from shapely import wkt
                            DF_seg_geom[i] = DF_seg_geom[i].apply(wkt.loads)
                #DF=pd.read_pickle('sample_data/data/cleaned/Line/grouped/grouped_3.pkl')
                DF_seg_geom = DF_seg_geom[DF_seg_geom[metadata['unit_name']].isin(segment_list_pred)]
            else:
                DF_seg_geom = df[[metadata['unit_name'], 'geometry']].copy()
            DF_seg_geom['Center']=DF_seg_geom.apply(lambda row: row['geometry'].centroid, axis=1)
            DF_seg_geom = gpd.GeoDataFrame(DF_seg_geom, geometry=DF_seg_geom.Center)

            JOINT_DF = gpd.sjoin(DF_seg_geom,
                             DF_polygon[['geometry']],
                             how="inner", op='intersects')     
            segment_in_polygon=JOINT_DF[JOINT_DF[metadata['unit_name']].notna()][metadata['unit_name']].astype('int64').unique()
            df=df[df[metadata['unit_name']].isin(segment_in_polygon)]
            segment_list_pred=df[metadata['unit_name']].drop_duplicates().tolist()

        if len(segment_list_pred)==0:
                raise NameError('Spatial filtering results in zero segments. Please modify your filter')

    return df, segment_list_pred

