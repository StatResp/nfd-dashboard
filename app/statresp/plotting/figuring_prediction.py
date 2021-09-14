"""
@Author - Sayyed Mohsen Vazirizade
Plotting methods for forecasting models
"""

#packages______________________________________________________________
from scipy.interpolate import UnivariateSpline
import numpy as np
import os
import pandas as pd
import pyproj
import math
import matplotlib.pyplot as plt
import shapely.geometry as sg
import folium
import geopandas as gpd
import random
import pprint
import seaborn as sns
from datetime import timedelta
from math import ceil, floor
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from copy import deepcopy
import geopandas as gpd
import folium
import plotly.graph_objects as go
from shapely import wkt
from plotly import graph_objects as go
from shapely.geometry import LineString, MultiLineString
import plotly.express as px


def Figure_Generator(Rolling_Window_Num_str,learn_results,metadata,Type='df_test',Address=None, Save_Tag=True):
    '''
    This fuction draws all the figures back to back

    Parameters
    ----------
    Rolling_Window_Num_str : str
        We may have multiple test windows. This parameter defines which Test window we want to draw the figures for.
    learn_results : Dict
        This dictionary contains all of the results. 
    metadata : Dict
        DESCRIPTION.
    Type : String, optional
        It is usually either df_test or df_predict based on what you want to draw. The default is 'df_test'.

    Returns
    -------
    None.

    '''
    #df=pd.read_pickle('data/grouped_inrix.pkl')
    print('Generating Figures...')
    if metadata['inrix_pickle_address'][-4:] == '.pkl':
        df_geometry = pd.read_pickle(metadata['inrix_pickle_address'])
    else:
        df_geometry = pd.read_parquet(metadata['inrix_pickle_address'])
        uncompatible_list = ['beg', 'end', 'center', 'geometry', 'geometry3d', 'geometry_highres']
        for i in uncompatible_list:
            if i in df_geometry.columns:
                # df[i] = df[i].astype(str)
                from shapely import wkt
                df_geometry[i] = df_geometry[i].apply(wkt.loads)
    #df_geometry = df_geometry.rename(columns={'grouped_type3':metadata['unit_name']})
    if not os.path.exists(Address):
        os.makedirs(Address)

    for m in  learn_results[Rolling_Window_Num_str]:   #This for loop goes over all of the models in the Rolling_Window_Num_str
        model_type= learn_results[Rolling_Window_Num_str][m]['model'].metadata['model_type'] #it is the type of the model: NN, LR, Naive, etc.
        model_i   = learn_results[Rolling_Window_Num_str][m]['model'].metadata['Name']       #it is the name of the model: such as LR+RUS1.0+KM2
        df        = learn_results[Rolling_Window_Num_str][m][Type]                           #this df includes all the segments and time_windows we want to draw for
        #df       = learn_results[Rolling_Window_Num_str][model_i]['df_predict']
        #time_range=df['time_local'].drop_duplicates().iloc[1:4].tolist()   #for drawing on the map, we have to specify a time frame
        time_range=df['time_local'].drop_duplicates().sort_values().tolist()  #for drawing on the map, we have to specify a time frame
        #1) Figure with x-xis: time, y-axis=segment, color= the value such as likehihood we want to show (defined by metadata['pred_name_TF'])
        if (model_type in metadata['TF_Models']) | (model_type=='Naive'):    
            #Drawing the prediction
            plt1=Heatmap(        df,metadata,model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+': Actual Data', COLNAME=metadata['pred_name_TF'],SeedNum=metadata['seed_number'])
            if Save_Tag==True:
                plt1.savefig(Address+'/spatial_temporal_'+model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+'Actual Data.png')
            #Drawing the actual (for comparision)
            plt2=Heatmap(        df,metadata,model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+': Prediction',  COLNAME='predicted',SeedNum=metadata['seed_number'])
            if Save_Tag == True:
                plt2.savefig(Address+'/spatial_temporal_'+model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+'Prediction.png')
        elif model_type in metadata['Count_Models']:
            #Drawing the prediction
            plt3=Heatmap(        df,metadata,model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+'Actual Data', COLNAME=metadata['pred_name_Count'],SeedNum=metadata['seed_number'],maxrange=max(1, df[[metadata['pred_name_Count'],'predicted_Count']].max().max()))
            if Save_Tag == True:
                plt3.savefig(Address+'/spatial_temporal_'+model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+': Actual Data.png')
            #Drawing the actual (for comparision)
            plt4=Heatmap(        df,metadata,model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+'Prediction',  COLNAME='predicted_Count',SeedNum=metadata['seed_number'],maxrange=max(1, df[[metadata['pred_name_Count'],'predicted_Count']].max().max()))
            if Save_Tag == True:
                plt4.savefig(Address+'/spatial_temporal_'+model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+': Prediction.png')
        
        #2) Figure on the map  x-xis: long, y-axis lat,  color= the aggregated value for the time range such as likehihood we want to show (defined by metadata['pred_name_TF'])   
        #Map=Heatmap_on_Map_TimeRange(df,df_geometry,time_range,metadata,Feature_List=[metadata['pred_name_TF']]+['predicted','cluster_label'],Name=model_i) ;   #using folium
        #Map.save(Address+'/Map_'+model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+time_range[0].strftime('%Y-%m-%d %H')+'.html')
        #plt.show()
        print(Address)
        Map1 = Heatmap_on_Map_TimeRange_plotly(df, df_geometry, metadata, time_range[0:min(41, len(time_range))],Name=model_i)
        Map2 = Cluster_on_Map_plotly(df, df_geometry, metadata, time_range[0:1], Name=model_i)
        if Save_Tag == True:
            Map1.write_html(Address+'/Heatmap_'+model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+time_range[0].strftime('%Y-%m-%d %H')+'.html')
            Map2.write_html(Address + '/Cluster_' + model_i + '_' + 'rollingwindow(' + Rolling_Window_Num_str + ')_' + time_range[
                0].strftime('%Y-%m-%d %H') + '.html')
            #Map1.to_json(Address+'/Heatmap_'+model_i+'_'+'rollingwindow('+Rolling_Window_Num_str+')_'+time_range[0].strftime('%Y-%m-%d %H')+'.html')
            #Map2.to_json(Address + '/Cluster_' + model_i + '_' + 'rollingwindow(' + Rolling_Window_Num_str + ')_' + time_range[0].strftime('%Y-%m-%d %H') + '.html')
        #print(learn_results[model_i]['model'].model_params[0].summary())
        print('Generating Figures is done.')
        if Save_Tag==False:
            return plt1, plt2, Map1, Map2


def Table_Generator(Rolling_Window_Num_str,learn_results,metadata,Address=None):
    '''
    This fuction summerizes all of the resutls in a dictionary called  results[model_i]

    Parameters
    ----------
    Rolling_Window_Num_str : str
        We may have multiple test windows. This parameter defines which Test window we want to make the table for.
    learn_results : Dict
        This dictionary contains all of the results. 
    metadata : Dict
        DESCRIPTION.
    Type : String, optional
        It is usually either df_test or df_predict based on what you want to draw. The default is 'df_test'.
    Address : TYPE, str
        Where you want the figures to be saved. The default is None.

    Returns
    -------
    None.

    '''        
    for m in  learn_results[Rolling_Window_Num_str]:   #This for loop goes over all of the models in the Rolling_Window_Num_str
        model_i   = learn_results[Rolling_Window_Num_str][m]['model'].metadata['Name']       #it is the name of the model: such as LR+RUS1.0+KM2, LR
        results[model_i]=learn_results[Rolling_Window_Num_str][m]['results']   
    generate_report(results,metadata['cluster_number']+[''],Address+'/Report_'+metadata['model_type'][0]+'_'+'_rollingwindow('+Rolling_Window_Num_str+')'+'.html' )   




def MatrixMaker(df,metadata,NROWS, NCOLS,SeedNum,COLNAME):
    '''
    This fucntion convers a df to a matrix shape using 3 columns of df
    1)metadata['unit_name'] is the row in matrix
    2)metadata['time_local'] is the column in matrix
    3)COLNAME is the value in matrix

    Parameters
    ----------
    df : dataframe
        The input dataframe that included the data we want to draw
    metadata : TYPE
        DESCRIPTION.
    NROWS : int
        maximum number of rows you want to show.
    NCOLS : int
        maximum number of columns you want to show.
    SeedNum : TYPE
        DESCRIPTION.
    COLNAME : str
        The feature in the df that you want to be the value in the matrix.

    Returns
    -------
    HeatmapMatrix : TYPE
        DESCRIPTION.

    '''
    np.random.seed(seed=SeedNum)           
    HeatmapMatrix=df.pivot(index=metadata['unit_name'], columns='time_local')[COLNAME]
    HeatmapMatrix=HeatmapMatrix.sample(min(NROWS,len(HeatmapMatrix)),replace=False).sort_values(by=metadata['unit_name'])
    np.random.seed(seed=SeedNum) 
    random.seed(SeedNum)
    ListofCols=sorted(random.sample(list(np.arange(0,    len(HeatmapMatrix.columns))), min(NCOLS,len(HeatmapMatrix.columns))     ))
    HeatmapMatrix=HeatmapMatrix[HeatmapMatrix.columns[ListofCols]]
    try: 
        HeatmapMatrix.columns=HeatmapMatrix.columns.strftime('%Y-%m-%d %H')        #('%Y-%m-%d %H:%M:%S')
    except:
        HeatmapMatrix.columns=pd.DataFrame(HeatmapMatrix.columns).time_local.apply(lambda x: x.strftime('%Y-%m-%d %H')) 
        #HeatmapMatrix.columns=pd.DataFrame(HeatmapMatrix.columns).time.apply(lambda x: x.strftime('%Y-%m-%d %H'))        #('%Y-%m-%d %H:%M:%S') 
    HeatmapMatrix.index=[int(i) for i in HeatmapMatrix.index] 
    return HeatmapMatrix
    



def Heatmap(df,metadata,model_i,COLNAME=None,maxrange=1,NROWS=100, NCOLS=100,SeedNum=0 ):
    '''
    It draws a 2D (space and time) graph to show the predicted/actual incident. 

    Parameters
    ----------
    df : dataframe
         The input dataframe that included the data we want to draw.
    metadata : dict
        DESCRIPTION.
    model_i : str
        Just name of the model which is used for saving and caption.
    COLNAME : string, optional
        The feature we want to show. It is usally the predicted likelihood or the actual incident_TF.
    maxrange : int, optional
        For the heatmap, what is the maximum number. If it is T/F, it should be 1. If it is a count model, it can be more than 1. The default is 1.
    NROWS : int
        maximum number of rows you want to show.
    NCOLS : int
        maximum number of columns you want to show.
    SeedNum : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    '''
    CMAP='rocket'
    if not COLNAME in df.columns:
        df[COLNAME]=None  
        df=df.fillna(0)
        maxrange=0
        CMAP='Blues'
        
    HeatmapMatrix=MatrixMaker(df,metadata,NROWS=NROWS, NCOLS=NCOLS,SeedNum=SeedNum,COLNAME=COLNAME)
    fig = plt.figure(figsize=(10,10))
    plt.suptitle(model_i, fontsize=16)
    plt.tight_layout()
    Subplot_size=10
    ax1 = plt.subplot2grid((Subplot_size,Subplot_size), (1,2),  colspan=Subplot_size-3, rowspan=Subplot_size-1) # middle
    sns.heatmap(data=HeatmapMatrix,vmin=0, vmax=maxrange, cmap=CMAP,facecolor='y', cbar=False, ax=ax1)  #sns.cm.rocket viridis
    #sns.heatmap(data=HeatmapMatrix_backup,vmin=0, vmax=maxrange, cmap=CMAP,facecolor='y', cbar=False, ax=ax1) 
    ax1.set_facecolor('blue')

    ax2 = plt.subplot2grid((Subplot_size,Subplot_size), (0,2),  colspan=Subplot_size-3, rowspan=1)            # top
    sns.heatmap(data=HeatmapMatrix.mean(axis=0).values.reshape(1,HeatmapMatrix.shape[1]), vmin=0, vmax=maxrange, cmap=CMAP,facecolor='y', cbar=False, ax=ax2)
    ax2.set_facecolor('blue'); ax2.set_ylabel(''); ax2.set_xlabel('');ax2.set(xticklabels=[]);ax2.set(yticklabels=[])  

    ax3 = plt.subplot2grid((Subplot_size,Subplot_size), (1,Subplot_size-1),  colspan=1, rowspan=Subplot_size-1)   #right
    sns.heatmap(data=HeatmapMatrix.mean(axis=1).values.reshape(HeatmapMatrix.shape[0],1),vmin=0, vmax=maxrange, cmap=CMAP,facecolor='y', cbar=True, ax=ax3)
    ax3.set_facecolor('blue'); ax3.set_ylabel(''); ax3.set_xlabel('');ax3.set(xticklabels=[]);ax3.set(yticklabels=[])  

    HeatmapMatrix=pd.DataFrame({'Mean': [HeatmapMatrix.mean(axis=0).mean()]})
    ax5 = plt.subplot2grid((Subplot_size,Subplot_size), (0, Subplot_size-1),  colspan=1, rowspan=1)    #top right
    sns.heatmap(data=HeatmapMatrix,vmin=0, vmax=maxrange, cmap=CMAP,facecolor='y', cbar=False, ax=ax5)
    ax5.set_facecolor('blue'); ax5.set_ylabel(''); ax5.set_xlabel('');ax5.set(xticklabels=[]);ax5.set(yticklabels=[])   
    
    
    if 'cluster_label' in df.columns:
        HeatmapMatrix=MatrixMaker(df,metadata,NROWS=NROWS, NCOLS=NCOLS,SeedNum=SeedNum,COLNAME='cluster_label') #left
        ax4 = plt.subplot2grid((Subplot_size,Subplot_size), (1,0),  colspan=1, rowspan=Subplot_size-1)   
        sns.heatmap(data=HeatmapMatrix.mean(axis=1).values.reshape(HeatmapMatrix.shape[0],1),cmap='viridis',facecolor='y', cbar=False, ax=ax4)
        ax4.set_facecolor('blue'); ax4.set_ylabel(''); ax4.set_xlabel('Clusters');ax4.set(xticklabels=[]);ax4.set(yticklabels=[])  

    return fig



def Heatmap_on_Map_TimeRange(df, df_geometry, time_range, metadata, Feature_List=None, Name=' ',number=5000):
    '''
    

    Parameters
    ----------
    df : dataframe
         The input dataframe that included the data we want to draw.
    time_range : pandas timestampe
        If it is more than 1 window, the function draws the aggregate. 
    metadata : TYPE
        DESCRIPTION.
    Feature_List : list, optional
        The feature we want to show. It is usally the predicted likelihood or the actual incident_TF..
    Name : TYPE, optional
        DESCRIPTION. The default is ' '.
    number : TYPE, optional
        DESCRIPTION. The default is 5000.

    Returns
    -------
    None.

    '''

    Predict_df=df[df['time_local'].isin(time_range)].copy()
    Predict_df=Predict_df[[metadata['unit_name']]+Feature_List].groupby(metadata['unit_name']).mean().reset_index()

    #Predict_df=pd.merge(Predict_df[['time','time_local',metadata['pred_name_TF'],'cluster_label','predicted','MyGrouping_3']], df_geometry[['XDSegID','MyGrouping_3']], left_on='MyGrouping_3', right_on='MyGrouping_3', how='right')
    Predict_df=pd.merge(Predict_df, df_geometry[[metadata['unit_name'],'geometry' ]], left_on=metadata['unit_name'], right_on=metadata['unit_name'], how='inner')
    


    
    #maping XDSegID to geometry
    #static_df = pd.read_pickle(metadata['static_pickle_address'])
    #static_df=pd.read_pickle("D:/inrix/inrix/other/inrix_pd_2D.pk")
    #Predict_df=pd.merge(Predict_df, static_df[['geometry','XDSegID']], left_on='XDSegID', right_on='XDSegID', how='left')
    
    IDs=range(0,min(number,len(Predict_df)))
    Predict_df['geometry']= [i.buffer(0.025) for i in Predict_df['geometry']]  
    if not metadata['pred_name_TF'] in Predict_df.columns:
        Predict_df[metadata['pred_name_TF']]=0
        
    
    plot_area= gpd.GeoDataFrame(data=Predict_df.iloc[IDs][[metadata['unit_name'],'geometry']])
    plot_area = plot_area.to_json()
    Map = folium.Map(location = [35.151, -86.852], opacity=0.5, tiles='cartodbdark_matter', zoom_start = 8)   #'Stamen Toner'
    #title_html = '''     <h3 align="center" style="font-size:16px"><b>{}</b></h3>  '''.format(time_range[0].strftime('%Y-%m-%d %H:%M:%S %Z'))
    Title=[i.strftime('%Y-%m-%d %H:%M:%S %Z') for i in [time_range[0],time_range[-1] ]]
    Title=[Title[0]+' to '+Title[1] + Name]
    #print(Title[0])
    title_html = '''     <h3 align="center" style="font-size:16px"><b>{}</b></h3>  '''.format(Title[0])
    Map.get_root().html.add_child(folium.Element(title_html))
    #Min=min(df['predicted'].min(),df[metadata['pred_name_TF']].min())
    #Max=max(df['predicted'].max(),df[metadata['pred_name_TF']].max())
    Range=[0, 0.1, 0.25, 0.5 , 0.75,1]
    
    Key_on_tag='feature.properties.'+metadata['unit_name']
    #'feature.properties.unit_segment_id'
    for Feature in Feature_List:
            folium.Choropleth(
                geo_data=plot_area,
                name=Feature,#+time_point.strftime('%Y-%m-%d %H'),
                data=Predict_df,
                columns=[metadata['unit_name'],Feature],
                key_on=Key_on_tag,
                fill_color = 'Reds', line_color = 'white', fill_opacity = 1,line_opacity = 0.5 ,line_weight=0.001, #RdYlBu_r   #I coulndt find rocket cmap for folium!
                #threshold_scale=Range,
                #threshold_scale=[i/df[metadata['pred_name_TF']].max() for i in Range],
                legend_name = Feature).add_to(Map)         
            
    folium.LayerControl().add_to(Map)
    #    Map.save("Map_rate.html")      
    return(Map)  





def Graph_Metric(DF_summary,Title):
    #print(DF)

    import matplotlib
    from matplotlib import pyplot
    import seaborn as sns
    
    #plt.figure(figsize=[20,10])
    fig, axis = pyplot.subplots(4,1,figsize=(15,15))
    sns.set_palette("tab10")
    DF_summary_=DF_summary.set_index('time_local')
    metric_list=['accuracy','recall','precsion','F1']
    for i in range(4):
        DF=DF_summary_[[j for j in DF_summary_.columns if j.split('_')[-1]==metric_list[i]]]
        ax=axis[i]
        DF.plot(linewidth=1,ax=ax) 
        ax.set_xlabel('')
        ax.set_ylabel(metric_list[i])
        #ax.set_title(metric_list[i])
        ax.set_xticks([],minor=False)   #position

        for k,Value in enumerate(DF.mean(axis=0)):
            ax.axhline(Value, ls='--',linewidth=2,color=sns.color_palette("tab10")[k])

    #ax.set_xlabel('Time (Local)')
    if len(DF)>6:
        ax.set_xticks(DF.index[::6],minor=False)   #position
        ax.set_xticklabels(DF.index[::6], rotation=90,minor=False) 	  #label rotation, etc. 
    ax.set_xticks([],minor=True)   #position
    axis[0].set_title(Title)    
    plt.show()



def plot_me(geom, name='', old=None, color='blue'):
    if type(geom) == sg.LineString:
        coords = list(geom.coords)
        lat = [item[1] for item in coords]
        lon = [item[0] for item in coords]
    elif type(geom) == sg.MultiLineString:
        l = [list(line.coords) for line in geom]
        coords = [item for sublist in l for item in sublist]
        lat = [item[1] for item in coords]
        lon = [item[0] for item in coords]
    else:
        return None
    if old is None:
        fig = go.Figure(go.Scattermapbox(mode="markers+lines", lon=lon,
                                         lat=lat, name=name, text=name, marker={'size': 6, 'color': color}))
        fig = fig.add_trace(go.Scattermapbox(mode="markers", lon=[lon[0]], lat=[
                            lat[0]], name='start', text='start', marker={'size': 9, 'color': 'black'}))
        fig = fig.add_trace(go.Scattermapbox(mode="markers", lon=[
                            lon[-1]], lat=[lat[-1]], name='end', text='end', marker={'size': 9, 'color': 'red'}))
    else:
        fig = old.add_trace(go.Scattermapbox(mode="markers+lines", lon=lon,
                                             lat=lat, name=name, text=name, marker={'size': 6, 'color': color}))
    fig.update_layout(margin={'l': 0, 't': 30, 'b': 0, 'r': 0}, mapbox_style="open-street-map",
                      mapbox={'center': {'lon': lon[0], 'lat': lat[0]}, 'zoom': 15})
    return fig


#%%
def extract_coords(geom):
    if type(geom) == LineString:
        coords = list(geom.coords)
        lat = [item[1] for item in coords]
        lon = [item[0] for item in coords]
    elif type(geom) == MultiLineString:
        l = [list(line.coords) for line in geom]
        coords = [item for sublist in l for item in sublist]

    return coords


def Heatmap_on_Map_TimeRange_plotly(df, df_geometry, metadata, time_range, Name):
    if len(time_range) > 1:
        df = df[(df['time_local'] >= time_range[0]) & ((df['time_local'] < time_range[-1]))].copy()
    df = df.sort_values('time_local')

    if not 'geometry' in df.columns:
        # df=pd.merge(df,df_geometry, left_on='spaceid', right_on= 'spaceid', how='left' )
        df = pd.merge(df, df_geometry[['geometry',metadata['unit_name']]], left_on=metadata['unit_name'], right_on=metadata['unit_name'], how='left')

    df[metadata['unit_name']+'_str'] = df[metadata['unit_name']].apply(str)
    df['cluster_str'] = df['cluster_label'].apply(str)
    df['coords'] = df.geometry.apply(extract_coords)
    df_points = df[['predicted', 'coords', 'time_local', metadata['unit_name']+'_str', 'cluster_str']].reset_index().explode('coords')
    df_points['lat'] = df_points.coords.apply(lambda x: x[1])
    df_points['lon'] = df_points.coords.apply(lambda x: x[0])
    df_points['time'] = df_points.time_local.apply(str)

    Title = [i.strftime('%Y-%m-%d %H:%M:%S %Z') for i in [time_range[0], time_range[-1]]]
    Title = [Title[0] + ' to ' + Title[1] + Name]

    fig1 = px.scatter_mapbox(df_points, animation_frame='time', animation_group='index',
                             hover_data=[metadata['unit_name']+'_str', "cluster_str"], lat="lat", lon="lon", color="predicted",
                             mapbox_style='carto-darkmatter',  # mapbox_style="open-street-map",
                             title=Title[0],
                             color_continuous_scale=px.colors.sequential.Hot,
                             center=dict(lat=df_points['lat'].median(), lon=df_points['lon'].median()), size_max=15,
                             zoom=7, range_color=[0, 1])

    return fig1


def Cluster_on_Map_plotly(df, df_geometry, metadata, time_range, Name):
    '''
    This function draws the segment color coded based on their clusters on the map using plotly

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    df_geometry : TYPE
        DESCRIPTION.
    metadata : TYPE
        DESCRIPTION.
    time_range : TYPE
        DESCRIPTION.
    Name : TYPE
        DESCRIPTION.

    Returns
    -------
    fig1 : TYPE
        DESCRIPTION.

    '''

    df = df[df['time_local'] == time_range[0]].copy()
    df = df.sort_values('time_local')

    if not 'geometry' in df.columns:
        # df=pd.merge(df,df_geometry, left_on='spaceid', right_on= 'spaceid', how='left' )
        df = pd.merge(df, df_geometry[['geometry',metadata['unit_name']]], left_on=metadata['unit_name'], right_on=metadata['unit_name'], how='left')

    df[metadata['unit_name']+'_str'] = df[metadata['unit_name']].apply(str)
    df['cluster_str'] = df['cluster_label'].apply(str)
    df['coords'] = df.geometry.apply(extract_coords)
    df_points = df[['predicted', 'coords', 'time_local', metadata['unit_name']+'_str', 'cluster_str']].reset_index().explode('coords')
    df_points['lat'] = df_points.coords.apply(lambda x: x[1])
    df_points['lon'] = df_points.coords.apply(lambda x: x[0])
    df_points['time'] = df_points.time_local.apply(str)

    Title = [i.strftime('%Y-%m-%d %H:%M:%S %Z') for i in [time_range[0]]]
    Title = [Title[0] + Name]

    fig1 = px.scatter_mapbox(df_points, animation_frame='time', hover_data=["cluster_str"], lat="lat", lon="lon",
                             color="cluster_str", mapbox_style='carto-darkmatter',  # mapbox_style="open-street-map",
                             title=Name,
                             color_continuous_scale=px.colors.sequential.Hot,
                             center=dict(lat=df_points['lat'].median(), lon=df_points['lon'].median()), size_max=15,
                             zoom=7, range_color=[0, 1])
    return fig1

'''

# ('feature.properties.'+metadata['unit_name'])   
def Heatmap_on_Map(DF, time_point, metadata, number=5000):
    #DF=learn_results[model_i]['df_predict']
    #number=5000
    
    # Heatmap_on_Map(learn_results[m]['df_predict'],learn_results[m]['df_predict']['time'].iloc[0] ) 
    metadata['unit_name']
    Predict_df=DF[DF['time_local']==time_point].copy()
    Predict_df=Predict_df.rename(columns={'XDSegID':'MyGrouping_3'})
    
    
    #maping my MyGrouping_3 to actual XDSegID
    #FRC0 = pd.read_pickle('D:/inrix/inrix/other/inrix_FRC0_etrims_3D_curvature_df.pk')
    FRC0 = pd.read_pickle('sample_data/data_cleaned_inrix_grouped.pkl')
    #Predict_df=pd.merge(Predict_df[['time','time_local','count','cluster_label','predicted','MyGrouping_3']], FRC0[['XDSegID','MyGrouping_3']], left_on='MyGrouping_3', right_on='MyGrouping_3', how='right')
    Predict_df=pd.merge(Predict_df[['time','time_local','count','cluster_label','predicted','MyGrouping_3']], FRC0[['XDSegID','MyGrouping_3','geometry' ]], left_on='MyGrouping_3', right_on='MyGrouping_3', how='inner')
    
    #maping XDSegID to geometry
    #static_df = pd.read_pickle(metadata['static_pickle_address'])
    #static_df=pd.read_pickle("D:/inrix/inrix/other/inrix_pd_2D.pk")
    #Predict_df=pd.merge(Predict_df, static_df[['geometry','XDSegID']], left_on='XDSegID', right_on='XDSegID', how='left')
    
    IDs=range(0,min(number,len(Predict_df)))
    Predict_df['geometry']= [i.buffer(0.01) for i in Predict_df['geometry']]  
    
    plot_area= gpd.GeoDataFrame(data=Predict_df.iloc[IDs][[metadata['unit_name'],'geometry',metadata['pred_name'],'predicted']])
    plot_area = plot_area.to_json()
    Map = folium.Map(location = [36.151, -86.852], opacity=0.5, tiles='Stamen Toner', zoom_start = 12)
    title_html = '     <h3 align="center" style="font-size:16px"><b>{}</b></h3>  '.format(time_point.strftime('%Y-%m-%d %H:%M:%S %Z'))
    Map.get_root().html.add_child(folium.Element(title_html))
    #Min=min(DF['predicted'].min(),DF[metadata['pred_name']].min())
    #Max=max(DF['predicted'].max(),DF[metadata['pred_name']].max())
    Range=[0, 0.1, 0.25, 0.5 , 0.75,1]
    
    Key_on_tag='feature.properties.'+metadata['unit_name']
    #'feature.properties.unit_segment_id'
    if "count" in DF.columns:
        folium.Choropleth(
            geo_data=plot_area,
            name=metadata['pred_name']+time_point.strftime('%Y-%m-%d %H'),
            data=Predict_df,
            columns=[metadata['unit_name'],metadata['pred_name']],
            key_on=Key_on_tag,
            fill_color = 'Reds', line_color = 'black', fill_opacity = 1,line_opacity = 1,line_weight=0.001, #RdYlBu_r   #I coulndt find rocket cmap for folium!
            threshold_scale=Range,
            #threshold_scale=[i/DF[metadata['pred_name']].max() for i in Range],
            legend_name = 'Rate Count').add_to(Map)         
        
    else:
        print('No count column found!')
    if "predicted" in DF.columns:
        folium.Choropleth(
            geo_data=plot_area,
            name='Predict'+time_point.strftime('%Y-%m-%d %H'),
            data=Predict_df,
            columns=[metadata['unit_name'],'predicted'],
            key_on=Key_on_tag,
            fill_color = 'Reds', line_color = 'black', fill_opacity = 1,line_opacity = 1,line_weight=0.001,
            threshold_scale=Range,
            legend_name = 'Rate Prediction').add_to(Map) 
    else:
        print('No count column found!')
    folium.LayerControl().add_to(Map)
    #    Map.save("Map_rate.html")      
    return(Map)  



def Clusters_on_Map(DF, time_point, metadata, number=5000):
    # Heatmap_on_Map(learn_results[m]['df_predict'],learn_results[m]['df_predict']['time'].iloc[0] ) 
    metadata['unit_name']
    Predict_df=DF[DF['time_local']==time_point].copy()
    Predict_df=Predict_df.rename(columns={'XDSegID':'MyGrouping_3'})
    
    
    #maping my MyGrouping_3 to actual XDSegID
    #FRC0 = pd.read_pickle('D:/inrix/inrix/other/inrix_FRC0_etrims_3D_curvature_df.pk')
    FRC0 = pd.read_pickle('sample_data/data_cleaned_inrix_grouped.pkl')
    #Predict_df=pd.merge(Predict_df[['time','time_local','count','cluster_label','predicted','MyGrouping_3']], FRC0[['XDSegID','MyGrouping_3']], left_on='MyGrouping_3', right_on='MyGrouping_3', how='right')
    Predict_df=pd.merge(Predict_df[['time','time_local','count','cluster_label','predicted','MyGrouping_3']], FRC0[['XDSegID','MyGrouping_3','geometry' ]], left_on='MyGrouping_3', right_on='MyGrouping_3', how='inner')
    
    #maping XDSegID to geometry
    #static_df = pd.read_pickle(metadata['static_pickle_address'])
    #static_df=pd.read_pickle("D:/inrix/inrix/other/inrix_pd_2D.pk")
    #Predict_df=pd.merge(Predict_df, static_df[['geometry','XDSegID']], left_on='XDSegID', right_on='XDSegID', how='left')
    
    IDs=range(0,min(number,len(Predict_df)))
    Predict_df['geometry']= [i.buffer(0.01) for i in Predict_df['geometry']]  
    
    plot_area= gpd.GeoDataFrame(data=Predict_df.iloc[IDs][[metadata['unit_name'],'geometry',metadata['pred_name'],'predicted']])
    plot_area = plot_area.to_json()
    Map = folium.Map(location = [36.151, -86.852], opacity=0.5, tiles='Stamen Toner', zoom_start = 12)
    title_html = '     <h3 align="center" style="font-size:16px"><b>{}</b></h3>  '.format(time_point.strftime('%Y-%m-%d %H:%M:%S %Z'))
    Map.get_root().html.add_child(folium.Element(title_html))
    #Min=min(DF['predicted'].min(),DF[metadata['pred_name']].min())
    #Max=max(DF['predicted'].max(),DF[metadata['pred_name']].max())

    Key_on_tag='feature.properties.'+metadata['unit_name']
    #'feature.properties.unit_segment_id'
    if "cluster_label" in DF.columns:
        folium.Choropleth(
            geo_data=plot_area,
            name=metadata['pred_name']+time_point.strftime('%Y-%m-%d %H'),
            data=Predict_df,
            columns=[metadata['unit_name'],metadata['pred_name']],
            key_on=Key_on_tag,
            fill_color ='RdYlGn',line_color = 'black', fill_opacity = 1,line_opacity = 1,line_weight=0.001, #RdYlBu_r   #I coulndt find rocket cmap for folium!
            #threshold_scale=Range,
            legend_name = 'Rate Count').add_to(Map)    
    else:
        print('No count column found!')
    folium.LayerControl().add_to(Map)
    #    Map.save("Map_rate.html")      
    return(Map)  




'''












'''
    
#drawing segments in an order range on the map
#IDs=range(0,len(df))
Predict=learn_results[m]['df_predict']
Predict=Predict[Predict['time']==learn_results[m]['df_predict']['time'].iloc[0]]

static_df = pd.read_pickle(metadata['static_pickle_address'])

DF=pd.merge(Predict, static_df[['geometry','seg_id']], left_on='unit_segment_id', right_on='seg_id', how='inner')

#1
colors = ['red','blue','gray','darkred','lightred','orange','beige','green','darkgreen','lightgreen','darkblue','lightblue','purple','darkpurple','pink','cadetblue',    'lightgray']
#2
def get_color(props):
  Curve_Turn_Min = props['Curve_Turn_Min']
  if Curve_Turn_Min< 1:
    return 'red'
  elif Curve_Turn_Min <100 :
    return 'yellow'
  else:
    return 'blue'
#3
#import folium.colormap as cm
import branca.colormap as cm
linearcol = cm.LinearColormap(['green','yellow','red'], vmin=0., vmax=6)
linearcol(9)

IDs=range(0,300)
df1= gpd.GeoDataFrame(data=DF.iloc[IDs][['unit_segment_id','geometry','predicted',metadata['pred_name']]])
plot_data = df1.to_json()
m1 = folium.Map(location=[36.151, -86.852], tiles='Stamen Toner',zoom_start=14)
#style_function = lambda x: {"color": linearcol(x['properties']), "weight": 8} 
#style_function = lambda x: {"color": random.choice(colors),"weight":5}
style_function = lambda x: {"color": linearcol(x['properties'].predicted),"weight":5}
folium.GeoJson(plot_data, style_function=style_function, name='predicted').add_to(m1)
folium.LayerControl().add_to(m1)
m1  
m1.save("Map.html")      
    
    

    








    
            

NROWS=30
NCOLS=30
SeedNum=0 
maxrange=6
df=learn_results[m]['df_predict']
np.random.seed(seed=SeedNum)        
HeatmapMatrix=df.pivot(index=metadata['unit_name'], columns='time')[metadata['pred_name']]
HeatmapMatrix=HeatmapMatrix.sample(min(NROWS,len(HeatmapMatrix)),replace=False).sort_values(by=metadata['unit_name'])
ListofCols=sorted(random.sample(list(np.arange(0,    len(HeatmapMatrix.columns))), min(NCOLS,len(HeatmapMatrix.columns))     ))
HeatmapMatrix=HeatmapMatrix[HeatmapMatrix.columns[ListofCols]]
HeatmapMatrix.columns=HeatmapMatrix.columns.strftime('%Y-%m-%d %H')        #('%Y-%m-%d %H:%M:%S')
HeatmapMatrix.index=[int(i) for i in HeatmapMatrix.index]
plt.figure(figsize=(10,10))
g=sns.heatmap(data=HeatmapMatrix,vmin=0, vmax=maxrange, cmap='rocket')  #sns.cm.rocket 



def HeatmapTrain(df_train,metadata):    
    windows = ceil(int((metadata['end_time_train'] - metadata['time_train']).total_seconds()) / metadata['window_size'])
    temp_start = metadata['start_time_test']
    Alltimes=[]
    for i in range(windows):
        Alltimes.append(temp_start)
        temp_start += timedelta(seconds=metadata['window_size'])
        
    unit_name=metadata['unit_name']
    DrawingDF=pd.DataFrame()
    DrawingDF[unit_name]=df_train[unit_name]
    DrawingDF['Time']= np.repeat(Alltimes, len(metadata['units']))
    DrawingDF[metadata['pred_name']]=df_train[metadata['pred_name']]
    DrawingDF=DrawingDF.sort_values(by=['Time',unit_name])
    
    HeatmapMattest=DrawingDF.pivot(index=unit_name, columns='Time')[metadata['pred_name']]
    plt.figure(figsize=(10,10))
    g=sns.heatmap(data=HeatmapMattest)
        
    
def HeatmapSample(df_samples,metadata):    
    windows = ceil(int((metadata['end_time_test'] - metadata['start_time_test']).total_seconds()) / metadata['window_size'])
    temp_start = metadata['start_time_test']
    Alltimes=[]
    for i in range(windows):
        Alltimes.append(temp_start)
        temp_start += timedelta(seconds=metadata['window_size'])
        
    unit_name=metadata['unit_name']
    DrawingDF=pd.DataFrame()
    DrawingDF[unit_name]=df_samples[unit_name]
    DrawingDF['Time']= np.repeat(Alltimes, len(metadata['units']))
    DrawingDF['sample']=df_samples['sample']
    DrawingDF=DrawingDF.sort_values(by=['Time',unit_name])
    
    HeatmapMattest=DrawingDF.pivot(index=unit_name, columns='Time')['sample']
    plt.figure(figsize=(10,10))
    g=sns.heatmap(data=HeatmapMattest)
            
'''