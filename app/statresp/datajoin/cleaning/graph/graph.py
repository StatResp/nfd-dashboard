# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 20:40:39 2021

@author: vaziris
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




import nxviz as nv
import networkx as nx
from networkx.readwrite import json_graph

import plotly.io as pio;
pio.renderers.default = "browser";
import plotly.express as px
import plotly.graph_objects as go


def Graph_Builder(df,Precision=4,Plotting_Tag=False):
    inrix_df=df.copy()
    inrix_df['Beg_']=inrix_df['Beg'].apply(lambda row: (round(row.coords[0][0],Precision),round(row.coords[0][1],Precision)))
    inrix_df['End_']=inrix_df['End'].apply(lambda row: (round(row.coords[0][0],Precision),round(row.coords[0][1],Precision)))
    inrix_df['Center_']=inrix_df['Center'].apply(lambda row: (round(row.coords[0][0],Precision),round(row.coords[0][1],Precision)))
    #inrix_df=inrix_df[(inrix_df['StartLat']>36.0) & (inrix_df['StartLat']<36.1) & (inrix_df['StartLong']>-87.95) & (inrix_df['StartLong']<-87.8)]
    
    
    Transfered_Features=['XDSegID','FRC','Miles']
    G=nx.from_pandas_edgelist(inrix_df,source ='Beg_', target ='End_', edge_attr=Transfered_Features, create_using=nx.MultiDiGraph())
    print('Number of the Nodes in the Graph:',len(list(G.nodes())))
    print('Number of the Edges in the Graph:',len(list(G.edges())))
    
    if Plotting_Tag==True:
        print('Graph: Edges are Segments')
        Fig1=nv.CircosPlot(G,figsize=(9, 9),title='test')
        Fig1.draw()    
        #plt.show() 
    
        print('Graph: Edges are Segments')
        Map_Graph(G, Title='Graph: Edges are Segments', Type='Graph')    
    return G
    

def Line_Graph(G,df,Plotting_Tag=False):
    L = nx.empty_graph(0, create_using=G.__class__)
    #DF_adj=pd.DataFrame(columns=['source','target'])
    source_list=[]
    target_list=[]
    for from_node in G.edges(None, keys=True,data=True):
        L.add_node(from_node[0:3],XDSegID=from_node[3]['XDSegID'],FRC=from_node[3]['FRC'],Miles=from_node[3]['Miles'] )
        for to_node in G.edges(from_node[1],keys=True,data=True):
            L.add_node(to_node[0:3],XDSegID=to_node[3]['XDSegID'],FRC=to_node[3]['FRC'],Miles=from_node[3]['Miles'])
            L.add_edge(from_node[0:3], to_node[0:3])
            #DF_adj=DF_adj.append(pd.DataFrame({'source':[from_node[3]['XDSegID']],'target':[to_node[3]['XDSegID']]}),ignore_index=True)
            source_list.append(from_node[3]['XDSegID'])
            target_list.append(to_node[3]['XDSegID'])
            
    DF_adj=pd.DataFrame({'source':source_list,'target':target_list})
    DF_adj=Inrix_info_Incorporation_Graph(L,df,DF_adj)
    
    
    print('Number of the Nodes in the Line_Graph:',len(list(L.nodes())))
    print('Number of the Edges in the Line_Graph:',len(list(L.edges())))
    
    
    L_XDSegID=nx.from_pandas_edgelist(DF_adj,source ='source', target ='target', create_using=nx.MultiDiGraph())
    print('Number of the Nodes in the Line_Graph_XDSegID:',len(list(L_XDSegID.nodes())))
    print('Number of the Edges in the Line_Graph_XDSegID:',len(list(L_XDSegID.edges())))
    inrix_df=df.copy()
    inrix_df['Center_']=inrix_df['Center'].apply(lambda row: (row.coords[0][0],row.coords[0][1]))
    XDSegID_Center = dict(zip(inrix_df.XDSegID, inrix_df.Center_))
    for nodes in L_XDSegID.nodes():
        L_XDSegID.nodes[nodes]['Center']=XDSegID_Center[nodes]      #inrix_df[inrix_df['XDSegID']==nodes]['Center']
        
    
    
    if Plotting_Tag==True:
        print('Graph: Nodes are Segments for Fig 1')
        Fig1=nv.CircosPlot(L_XDSegID,figsize=(9, 9),title='test')
        Fig1.draw()    
        #plt.show()
    
        print('Graph: Nodes are Segments for Fig 2')
        Map_Graph(L_XDSegID, Title='Graph: Nodes are Segments', Type='Line_Graph_Inrix')    
    

    return DF_adj,L_XDSegID





def Map_Graph(G,Title, Type):
    '''
    Type='Graph' or Type='Line_Graph'
    '''
    pos=dict()
    for i in list(G.nodes()):
        if Type=='Graph':
            pos[i]=i
        elif Type=='Line_Graph':
            pos[i]=(  (i[0][0]+i[1][0])/2, (i[0][1]+i[1][1])/2  )
        elif Type=='Line_Graph_Inrix':
            #print(G.nodes[i]['Center'] )
            pos[i]=G.nodes[i]['Center']  #i[1]['Center']            
            
    plt.figure(figsize =(27, 9))
    if Type=='Graph':
        nx.draw(G, pos,with_labels=False, font_weight='bold',width =1,node_size =.1)
    elif Type=='Line_Graph':
        nx.draw(G,pos,with_labels=False, font_weight='bold',width =2,node_size =.00001)  
    elif Type=='Line_Graph_Inrix':
        nx.draw(G,pos,with_labels=False, font_weight='bold',width =2,node_size =.00001)          
    plt.title(Title)


def Inrix_info_Incorporation_Graph(L,inrix_df,DF_adj):
    
    '''
    This functions tries to fix small offsets. 
    It will add an edges between the Segments that are defined to be back to back by inrix
    '''
    
    for i,row in inrix_df.iterrows():
        if (row['PreviousXD'] is None)  | (np.isnan(row['PreviousXD']) ) ==False:
            if row['PreviousXD'] in inrix_df['XDSegID'].tolist():
                if any((DF_adj['source']==row['PreviousXD']) & (DF_adj['target']==row['XDSegID']))==False:
                    #print(i, row['PreviousXD'], 'and', row['XDSegID'],'Edge Doesnt Exist')
                    DF_added_Edge=pd.DataFrame({'source':[row['PreviousXD']],'target':[row['XDSegID']]})
                    DF_adj=DF_adj.append(DF_added_Edge, ignore_index=True)
                
        if (row['NextXDSegI'] is None)  | (np.isnan(row['NextXDSegI']) ) ==False:
            if row['NextXDSegI'] in inrix_df['XDSegID'].tolist():
                if any((DF_adj['source']==row['XDSegID']) & (DF_adj['target']==row['NextXDSegI']))==False:
                    #print(i, row['XDSegID'], 'and', row['NextXDSegI'],'Edge Doesnt Exist')                
                    DF_added_Edge=pd.DataFrame({'source':[row['XDSegID']],'target':[row['NextXDSegI']]})
                    DF_adj=DF_adj.append(DF_added_Edge, ignore_index=True)                
    return DF_adj       
                
    

def Prepare_Graph(inrix_df,MetaData):
    print('Building graph process is stareted.')
    #Address_Inrix=MetaData['destination']+'inrix/'+'inrix_weather_zslope_grouped.pkl'
    #inrix_df=pd.read_pickle(Address_Inrix)
    G=Graph_Builder(inrix_df,Precision=4)
    DF_adj,L_XDSegID=Line_Graph(G,inrix_df)
    '''
    Save_DF(L_XDSegID, Destination_Address=MetaData['destination']+'inrix/graph/'+'inrix_graph',Format='gpickle', gpd_tag=False) 
    #nx.write_gpickle(L_XDSegID, "inrix_graph.gpickle")
    #L_XDSegID = nx.read_gpickle("inrix_graph.gpickle")
    Save_DF(DF_adj, Destination_Address=MetaData['destination']+'inrix/graph/'+'inrix_graph_adj',Format='pkl', gpd_tag=False) 
    Save_DF(DF_adj, Destination_Address=MetaData['destination']+'inrix/graph/'+'inrix_graph_adj',Format='json', gpd_tag=False) 
    '''
    return DF_adj, L_XDSegID, G
    
'''
inrix_df=pd.read_pickle(   'D:/inrix_new/inrix_pipeline/data/cleaned/inrix_cleaned/inrix_grouped.pkl') 
G=Graph_Builder(inrix_df,Precision=4,Plotting_Tag=False)
DF_adj,L_XDSegID=Line_Graph(G,inrix_df,Plotting_Tag=False)
'''

'''
DF_adj=pd.read_pickle('D:/inrix_new/inrix_pipeline/data/cleaned/inrix_cleaned/inrix_graph_adj.pkl') 
Graph = nx.read_gpickle('D:/inrix_new/inrix_pipeline/data/cleaned/inrix_cleaned/inrix_graph.gpickle')

'''




def Plot_Graph_ConnectedComponnet(DF_adj,inrix_df, Graph):
    '''
    This function plots
    1) the biggest connected component in grey
    2) and the rest in various colors

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
    def Connected_Component_Coloring(source,Sub_Graphs):
        #cmap =px.colors.qualitative.Light24.copy()
        #return ['#ff0000']
        
        for i in range(0,10):
            if source in Sub_Graphs[i]:
                return [i] #[cmap[i]]
        return [10] #[cmap[i+1]]


    #Graph = nx.read_gpickle(MetaData['destination']+'inrix/graph/inrix_graph.gpickle')
    #DF_adj = nx.read_gpickle(MetaData['destination']+'inrix/graph/inrix_graph_adj.pkl')
    #inrix_df=pd.read_pickle(MetaData['destination']+'inrix/inrix_weather_zslope_grouped.pkl')
    DF_adj=pd.merge(DF_adj,inrix_df[['XDSegID', 'Center']], left_on='source', right_on='XDSegID', how='left' ).rename(columns={'Center':'source_Center'})
    DF_adj=pd.merge(DF_adj,inrix_df[['XDSegID', 'Center']], left_on='target', right_on='XDSegID', how='left' ).rename(columns={'Center':'target_Center'})
    
    
    



    
    Sub_Graphs = sorted(nx.strongly_connected_components(Graph), key=len,reverse=True)
    Sub_Graphs_Len=[[i, len(j)] for i,j in enumerate(Sub_Graphs)]
    print('Subgraphs and their sizes:',Sub_Graphs_Len[0:50])
    
    
    fig=dict()
    cmap =px.colors.qualitative.Light24.copy()
    for j in range(0,21):
        if j<20:
            if j==0:
                color='grey'
            else:
                color=cmap[j]
            DF=DF_adj[DF_adj['source'].isin(Sub_Graphs[j])]
        else:
             color=cmap[20]
             DF=DF_adj[DF_adj['source'].isin([list(k)[0] for k in Sub_Graphs[20:]])]
             
        lats = [];lons = [];colors= [];texts=[]
        for i,row in DF.iterrows():
            lats = np.append(lats, row.source_Center.coords[0][1]);    
            lats = np.append(lats, row.target_Center.coords[0][1]);    
            lons = np.append(lons, row.source_Center.coords[0][0]);    
            lons = np.append(lons, row.target_Center.coords[0][0])
            texts= np.append(texts, row.source)
            texts= np.append(texts, row.target)
            #colors = np.append(colors, ['grey']*3)
            #colors = np.append(colors, 3*Connected_Component_Coloring(row.source,Sub_Graphs))
            lats = np.append(lats, None);    lons = np.append(lons, None); texts = np.append(texts, None)
            
        fig[j]=go.Scattermapbox(
                                    lon=lons, 
                                    lat=lats ,
                                    mode = 'lines',
                                    line = dict(width = 1,color = color),
                                    hovertext     =texts)
        
    Fig=go.Figure(fig[1])
    for i in range(2,21):
        Fig.add_trace(fig[i])
    #fig.update_layout(showlegend=False)
    Fig.update_layout(mapbox_style="carto-darkmatter",mapbox_zoom=7.5, mapbox_center_lat = 36, mapbox_center_lon=-86)
    #fig.update_layout(mapbox_style="stamen-terrain",mapbox_zoom=4, mapbox_center_lat = 35, mapbox_center_lon=-85)
    Fig.show()    

