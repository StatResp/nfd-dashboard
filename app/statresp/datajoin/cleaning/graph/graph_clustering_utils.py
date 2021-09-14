# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:11:41 2021

@author: vaziris
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
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


import dask.dataframe as dd
import multiprocessing
import matplotlib.pyplot as plt

import plotly.io as pio;
pio.renderers.default = "browser";
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import nxviz as nv
import networkx as nx
import scipy 


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    
    
def Matrix_FIG(DATA, Min, Max, Title):
    fig, axes=plt.subplots(figsize=[30,30])
    sns.heatmap(data=DATA,vmin=Min, vmax=Max,ax=axes,label=Title, cmap='YlOrBr_r', square=True, cbar=False)  #sns.cm.rocket viridis
    #sns.lineplot(data=DT, x='x', y=name, ax=ax, label=name, color=color,palette ={'BLUE': 'blue', 'RED':'red'})
    #sns.lineplot(data=df, x=df.index, y=name, ax=ax, color=color)
    axes.set_title(Title)    
    
    
    
def Color_Finder_Continues(x):
    #autumn_r, seismic
    Color= 'rgba'+str(plt.cm.autumn_r(x,Mmin_Color,Max_Color))
    #return Color[:-2]+'250)'
    return 'rgb'+str(plt.cm.seismic(x,Mmin_Color,Max_Color)[0:3])


def Plotly_plotter_Line(DF,Title):
    fig = go.Figure()
    for i in range(len(DF)):
        #print(i)
        fig.add_trace(
            go.Scattergeo(
                locationmode = 'USA-states',
                lon = [DF['source_Center'][i].xy[0][0], DF['targe_Center'][i].xy[0][0]],
                lat = [DF['source_Center'][i].xy[1][0], DF['targe_Center'][i].xy[1][0]],
                mode = 'lines',
                #line = dict(width = 10,color = Color_Finder_Continues(DF['Percent_diff_Average_Total_Number_Incidents'][i])), #'rgb({}, 200, 200)'.format(int(100+100*GDF_adj['Percent_diff_Average_Total_Number_Incidents'][i]/Max_Color))),
                opacity=0.5
            )
        )

    fig.update_layout(
        title_text = Title,
        showlegend = False,
        geo = dict(
            scope = 'north america',
            #showland = True,
            landcolor = 'rgb(50, 50, 50)',
            countrycolor = 'rgb(10, 10, 10)',
    		lonaxis = dict(range= [ -91, -82]),
            lataxis = dict(range= [ 35.0, 38.0 ] ),
            )
        )
    fig.show()
    
    
#%%
def Graph_Read(Filtering_DFadj_basedon_Incdient_Tag=True,
               Graph_Address='inrix/graph/inrix_graph.gpickle',
               Adj_Address= 'inrix/graph/inrix_graph_adj.pkl',
               Inrix_Address= 'inrix/inrix_weather_zslope_grouped.pkl',
               Incident_Address= 'incident/incident_XDSegID.pkl',   
               Main_address='D:/inrix_new/inrix_pipeline/data/cleaned/Line/'):
    Graph = nx.read_gpickle(Main_address+Graph_Address)
    #Graph = nx.read_gpickle('D:/inrix_new/inrix_pipeline/data/cleaned_main/inrix_cleaned/inrix_graph.gpickle')
    DF_adj=pd.read_pickle(     Main_address+Adj_Address) 
    DF_inrix=pd.read_pickle(   Main_address+Inrix_Address) 
    DF_incident=pd.read_pickle(Main_address+Incident_Address) 
    Number_of_Windows=int(np.floor((DF_incident['time_local'].max()-DF_incident['time_local'].min()).total_seconds()/3600/4))
    Total_Num_Accidents=len(DF_incident)
    Sparsity=Total_Num_Accidents/Number_of_Windows/len(DF_incident['XDSegID'].dropna().unique())
    print('Total Number of Windows is {} \nTotal Number of unique Segments is {} \nTotal Number of Incidents is {}'.format(Number_of_Windows,len(DF_incident['XDSegID'].dropna().unique()),len(DF_incident)))
    print('Total sparsity level without removing NA values is {:05.2f}%'.format(Sparsity*100))
    #%%
    #Adding 2 columns to the DF_inrix: total as well as average number of accidents
    DF_incident=DF_incident[['Incident_ID','XDSegID']].groupby('XDSegID').count().rename(columns={'Incident_ID':'Total_Number_Incidents'}).reset_index()
    DF_incident['Average_Total_Number_Incidents']=DF_incident['Total_Number_Incidents']/Number_of_Windows
    DF_incident.sort_values(['Total_Number_Incidents'])
    DF_incident['Average_Total_Number_Incidents'].mean()
    DF_inrix=pd.merge(DF_inrix,DF_incident[['XDSegID','Total_Number_Incidents','Average_Total_Number_Incidents']], left_on='XDSegID',right_on='XDSegID', how='left')
    #%%Modifying DF_adj
    if Filtering_DFadj_basedon_Incdient_Tag==True:
        #We want to shrink the DF_adj to the links that at least of the nodes has 1 accident or it is directly connected to a node with 1 accident
        XD_segID_with_Incidents=DF_inrix[DF_inrix['Total_Number_Incidents']>0]['XDSegID'].tolist()
        XD_segID_with_Incidents_Nbr_1=list(set(DF_adj[(DF_adj['source'].isin(XD_segID_with_Incidents)) & (DF_adj['target'].isin(XD_segID_with_Incidents)==False)]['target'].tolist()))
        XD_segID_with_Incidents_Nbr_2=list(set(DF_adj[(DF_adj['target'].isin(XD_segID_with_Incidents)) & (DF_adj['source'].isin(XD_segID_with_Incidents)==False)]['source'].tolist()))
        XD_segID_with_Incidents_Nbr=list(set(XD_segID_with_Incidents+XD_segID_with_Incidents_Nbr_1+XD_segID_with_Incidents_Nbr_2))
        DF_adj=DF_adj[(DF_adj['source'].isin(XD_segID_with_Incidents_Nbr)) & (DF_adj['target'].isin(XD_segID_with_Incidents_Nbr))].reset_index().drop('index',axis=1)
        print('Total Number of unique Segments (with at least one incident or being a neighbour of one) in DF_adj',len(set(list(DF_adj['source'])+list(DF_adj['target']))))

    #Adding  average number of accidents to the DF_adj
    DF_adj=pd.merge(DF_adj,DF_inrix[['XDSegID','Average_Total_Number_Incidents']], left_on='source',right_on='XDSegID', how='left').drop('XDSegID',axis=1)
    DF_adj=DF_adj.rename(columns={'Average_Total_Number_Incidents':'source_Average_Total_Number_Incidents'})
    DF_adj=pd.merge(DF_adj,DF_inrix[['XDSegID','Average_Total_Number_Incidents']], left_on='target',right_on='XDSegID', how='left').drop('XDSegID',axis=1)
    DF_adj=DF_adj.rename(columns={'Average_Total_Number_Incidents':'targe_Average_Total_Number_Incidents'})
    #DF_adj=DF_adj.dropna(thresh=3)  #Shrinks the DF_adj to the Links with with at least one of nodes (segments) having accidents
    DF_adj['diff_Average_Total_Number_Incidents']=-DF_adj['source_Average_Total_Number_Incidents']+DF_adj['targe_Average_Total_Number_Incidents']
    DF_adj['Percent_diff_Average_Total_Number_Incidents']=DF_adj['diff_Average_Total_Number_Incidents']*2/(DF_adj['source_Average_Total_Number_Incidents']+DF_adj['targe_Average_Total_Number_Incidents'])
    DF_adj['Percent_diff_Average_Total_Number_Incidents']=DF_adj['Percent_diff_Average_Total_Number_Incidents'].fillna(2, inplace = False) 
    DF_adj.sort_values('Percent_diff_Average_Total_Number_Incidents')
    #adding geometry features to DF_adj
    DF_adj=pd.merge(DF_adj,DF_inrix[['XDSegID','Center']], left_on='source',right_on='XDSegID', how='left').drop('XDSegID',axis=1)
    DF_adj=DF_adj.rename(columns={'Center':'source_Center'})
    DF_adj=pd.merge(DF_adj,DF_inrix[['XDSegID','Center']], left_on='target',right_on='XDSegID', how='left').drop('XDSegID',axis=1)
    DF_adj=DF_adj.rename(columns={'Center':'targe_Center'})
    DF_adj['Line']=DF_adj.apply(lambda ROW: sg.LineString([ROW.source_Center, ROW.targe_Center]), axis=1)
    return DF_inrix,DF_adj,Graph

#%%
def Adding_Degree(Graph,DF_inrix):
    Degree=pd.DataFrame.from_dict(nx.degree(Graph))
    Degree.columns=['XDSegID','Degree']
    DF_inrix=pd.merge(DF_inrix,Degree[['XDSegID','Degree']], left_on='XDSegID',right_on='XDSegID', how='left')
    return DF_inrix

#%% Creating Distance and DiffRate Matrix
##DiffRate_Matrix
def Difference_Matrix_Generator(DF_inrix_accident):
    DiffRate_Matrix = np.array( [[np.nan]*len(DF_inrix_accident)]*len(DF_inrix_accident))
    for i,SegI in DF_inrix_accident.iterrows():
        #print(i)
        DiffRate_Matrix[:,i]=(2*(DF_inrix_accident['Average_Total_Number_Incidents']-SegI['Average_Total_Number_Incidents'])/
                                (DF_inrix_accident['Average_Total_Number_Incidents']+SegI['Average_Total_Number_Incidents'])).abs().tolist()
    return DiffRate_Matrix


##Distance_Matrix
def Distance_Matrix_Generator(Graph):
    '''
    This can be used for the graph. You cannot use it for a some nodes in the Graph

    Parameters
    ----------
    DF_inrix_accident : TYPE
        DESCRIPTION.
    Graph : TYPE
        DESCRIPTION.

    Returns
    -------
    Distance_Matrix : TYPE
        DESCRIPTION.

    '''
    #Distance_Matrix= nx.algorithms.shortest_paths.dense.floyd_warshall_numpy(Graph,nodelist =DF_inrix_accident['XDSegID'].tolist())
    Distance_Matrix=nx.algorithms.shortest_paths.dense.floyd_warshall(Graph)
    Distance_Matrix=np.nan_to_num(Distance_Matrix, posinf=1000)
    return Distance_Matrix


def Distance_Dic_Generator(Segment_List,Graph,Method_Tag='simple', Threshold=100, weight_feature='Not_Defined'):
    start = time.time()
    Distance_Dic=dict()
    print('Length of Segment_List is:',len(Segment_List))
    print(Method_Tag, weight_feature)    
    if Method_Tag=='simple': 
        for Seg in Segment_List:
            Distance_Dic[Seg] = dict(nx.single_source_shortest_path_length(Graph,Seg,Threshold))
            #Distance_Dic[Seg][Seg]=0
    elif Method_Tag=='dijkstra': 
        for Seg in Segment_List:
            Distance_Dic[Seg] = dict(nx.single_source_dijkstra_path_length(Graph,Seg,Threshold))
    elif Method_Tag=='dijkstra_weighted': 
        for Seg in Segment_List:
            #print(Seg)
            Distance_Dic[Seg] = dict(nx.single_source_dijkstra_path_length(Graph,Seg,Threshold,weight=weight_feature))
    elif Method_Tag=='single_dijkstra_weighted': 
        for Seg_i in Segment_List:
            Distance_Dic[Seg_i]=dict()
            for Seg_j in Segment_List:
                print(Seg_i,Seg_j)
                Distance_Dic[Seg_i][Seg_j] = nx.dijkstra_path_length(Graph, Seg_i, Seg_j,weight_feature)        
    elif Method_Tag=='bellman_ford':
        for Seg in Segment_List:
            Distance_Dic[Seg] = dict(nx.single_source_bellman_ford_path_length(Graph,Seg,Threshold))
    end = time.time()
    print('Total time reuqired to calculate the distant dictionary:',end - start)
    return  Distance_Dic



def Dict_to_DF_to_Matrix(Dict, DF):
    df = pd.DataFrame.from_dict(Dict,orient='index',columns=DF['XDSegID'].tolist())
    df = df.loc[DF['XDSegID'].tolist()]
    print('Sanity Check:')
    if (df.index.tolist()==DF['XDSegID'].tolist()) and (df.columns.tolist()==DF['XDSegID'].tolist()):
             print('       OK')
    else :
             print('       Not OK. Warning!')
    df=np.nan_to_num(df, nan=1000,posinf=1000)         
    return df


def Physical_Neighbour_Matrix_Generator(DF_inrix_accident_graph,Distance_Threshold):
    DF_inrix_accident_graph_Physical=DF_inrix_accident_graph[['XDSegID','Center']].copy()
    DF_inrix_accident_graph_Physical= (gpd.GeoDataFrame(DF_inrix_accident_graph_Physical, geometry=DF_inrix_accident_graph_Physical['Center'], crs='epsg:4326' ))
    DF_inrix_accident_graph_Physical = DF_inrix_accident_graph_Physical.to_crs('EPSG:3310')
    DF_inrix_accident_graph_Physical['geometry']=DF_inrix_accident_graph_Physical.buffer(Distance_Threshold*1000/2)
    
    JOINT_DF = gpd.sjoin(DF_inrix_accident_graph_Physical[['XDSegID','geometry']],
                         DF_inrix_accident_graph_Physical[['XDSegID','geometry']],
                         how="inner", op='intersects').reset_index()
    print(len(JOINT_DF))
    JOINT_DF[['XDSegID_left','index','index_right']]
    
    PH_Distance_Matrix_modified=np.zeros((len(DF_inrix_accident_graph),len(DF_inrix_accident_graph)), dtype=int)
    for k in range(len(JOINT_DF)):
        #print(k)
        i=JOINT_DF['index'].iloc[k]
        j=JOINT_DF['index_right'].iloc[k]
        PH_Distance_Matrix_modified[i,j]=1
    return PH_Distance_Matrix_modified
#%%
def Max_Param_Estimator(Distance_Matrix_modified):
    Connected_Component=scipy.sparse.csgraph.connected_components(Distance_Matrix_modified)
    Connected_Component_df=pd.DataFrame({'Component_Index':Connected_Component[1]})
    Connected_Component_df_counts=Connected_Component_df.value_counts().sort_values(ascending=False)
    print('Total Number of Components:',Connected_Component[0],'or', len(Connected_Component_df_counts))
    print(Connected_Component_df_counts.head(10))
    #Connected_Component_df_counts
    return Connected_Component_df

#%%

def Clustering(DF,connectivity,Affinity_Matrix,Feature,linkage='ward', affinity='euclidean',n_clusters=20):

    #linkage='ward'  #'complete', 'average', 'single'
    DF=DF.copy()
    DF['Dummy']=0
    #1
    #Agglomerative_model = AgglomerativeClustering(n_clusters=n_clusters,affinity ='precomputed',linkage=linkage,compute_distances=True).fit(Affinity_Matrix)
    ##Agglomerative_model = AgglomerativeClustering(distance_threshold=distance_threshold,n_clusters=None,affinity ='precomputed',linkage=linkage,compute_distances=True).fit(Affinity_Matrix)
    #DF['Cluster']=Agglomerative_model.fit_predict(Affinity_Matrix)
    #2
    Agglomerative_model = AgglomerativeClustering(n_clusters=n_clusters,affinity=affinity,linkage=linkage,compute_distances=True,connectivity=connectivity).fit(DF[['Average_Total_Number_Incidents','Dummy']])
    DF['Cluster']=Agglomerative_model.fit_predict(DF[[Feature,'Dummy']])
    
    #adding the Cluster name to the DF_inrix
    #DF_inrix=pd.merge(DF_inrix,DF[['XDSegID','Cluster']], left_on='XDSegID',right_on='XDSegID', how='left')
    
    #figure
    list(px.colors.qualitative.Light24)
    from scipy.cluster import hierarchy
    hierarchy.set_link_color_palette(list(px.colors.qualitative.Light24))
    fig, axes = plt.subplots(figsize=(10, 5))
    #fig.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    plot_dendrogram(Agglomerative_model, truncate_mode='level', p=3,ax=axes)
    axes.set_xlabel("Number of points in node (or index of point if no parenthesis).")
    #plt.axhline(y=distance_threshold, color='r', linestyle='--')
    plt.suptitle(linkage)
    plt.show()
    #print(DF['Cluster'].value_counts())
    print(DF[['Cluster','Average_Total_Number_Incidents']].groupby('Cluster').agg(['mean','count']))
    Most_Freq_Clusters=DF['Cluster'].value_counts().sort_values()[-8:].index
    
    #DF=DF.rename(columns={'Cluster': Cluster_Col_Name})
    return DF, Most_Freq_Clusters




#%%Plotting Plotly
def Plot(DF_inrix_accident,DF_adj,Show_Links_Tag=False):
    '''
    This function plots
    1) the center of the segments
    2) color code based on the cluster the belong to
    3) optionally you can show the connection between the segments

    Parameters
    ----------
    DF_inrix_accident : TYPE
        DESCRIPTION.
    DF_adj : TYPE
        DESCRIPTION.
    Show_Links_Tag : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    #1) Plotting the center of the Segments using points
    #adding the Cluster name to the DF_inrix
    #DF=pd.merge(DF_inrix,DF_inrix_accident[['XDSegID','Cluster','index']], left_on='XDSegID',right_on='XDSegID', how='left')
    DF=DF_inrix_accident.copy()
    #DF=DF[DF['XDSegID'].isin([i for i in Sub_Graphs[0]])]
    #DF=DF[DF['Cluster'].isin(Most_Freq_Clusters)]
    DF['Center_lat'],DF['Center_lon']=zip(*DF.apply(lambda row: (row.Center.coords[0][1],row.Center.coords[0][0]),axis=1))
    DF=DF[DF['Total_Number_Incidents']>=1]
    color_discrete_map =dict(zip([str(i) for i in np.sort(DF['Cluster'].unique())],list(px.colors.qualitative.Light24)))
    #DF['Cluster']=DF['Cluster'].astype('int').astype('str')
    DF['Cluster']=DF['Cluster'].astype('str')
    fig = px.scatter_mapbox(DF, lat="Center_lat", lon="Center_lon", 
                            opacity=.5, color='Cluster',
                            size=[1.5]*len(DF),hover_name="XDSegID",
                            hover_data =['Degree','Total_Number_Incidents','index'], 
                            color_discrete_map=color_discrete_map)  
    lats = [];lons = [];colors= [];texts=[]
    
    if Show_Links_Tag:
        for i,row in DF_adj.iterrows():
            lats = np.append(lats, row.source_Center.coords[0][1]);    
            lats = np.append(lats, row.targe_Center.coords[0][1]);    
            lons = np.append(lons, row.source_Center.coords[0][0]);    
            lons = np.append(lons, row.targe_Center.coords[0][0])
            texts= np.append(texts, row.source)
            texts= np.append(texts, row.target)
            colors = np.append(colors, ['gray']*3)
            lats = np.append(lats, None);    lons = np.append(lons, None); texts = np.append(texts, None)
        fig.add_trace(go.Scattermapbox(lon=lons, 
                                    lat=lats ,
                                    mode = 'lines',
                                    line = dict(width = 1,color = 'gray'),
                                    hovertext     =texts))
        fig.update_layout(showlegend=False)
    fig.update_layout(mapbox_style="carto-darkmatter",mapbox_zoom=7.5, mapbox_center_lat = 36, mapbox_center_lon=-86)
    fig.show() 



    '''
    #%%Plotting Using Geopandas
    #1) Plotting the center of the Segments using points
    fig, ax = plt.subplots(figsize = (15,15)) 
    GDF_inrix=DF_inrix.copy()
    GDF_inrix=gpd.GeoDataFrame(DF_inrix, geometry=DF_inrix.Center)
    GDF_inrix.plot(column='Cluster',ax=ax)
    ax.set_title('Center of the Segments color coded by cluster number')
    
    
    
    #2)Plotting the links (the connection between the center of the segments) using Geopandas using lines
    fig, ax = plt.subplots(figsize = (15,15)) 
    GDF_adj=GDF_adj.copy()
    GDF_adj = gpd.GeoDataFrame(DF_adj, geometry=DF_adj.Line)
    GDF_adj.plot(column='Percent_diff_Average_Total_Number_Incidents',ax=ax)
    ax.set_title('Link between center of the segments color coded based on the gradiant of the rates')
    '''









